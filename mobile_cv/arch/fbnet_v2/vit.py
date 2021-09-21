import torch
import torch.nn as nn

from .levit import trunc_normal_
from .visual_transformer import MatMul


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        qkv_mode="qkv",
        attn_proj=True,
        qkv_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        if qkv_dim is None:
            qkv_dim = dim
        self.qkv_dim = qkv_dim
        head_dim = qkv_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        assert qkv_mode in ("qkv", "qk", "none")
        if qkv_mode == "qkv":
            self.qkv = nn.Linear(dim, qkv_dim * 3, bias=qkv_bias)
        elif qkv_mode == "qk":
            self.qk = nn.Linear(dim, qkv_dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        if qkv_mode == "qkv":
            self.proj = nn.Linear(qkv_dim, dim) if attn_proj else nn.Identity()
        else:
            self.proj = nn.Linear(dim, dim) if attn_proj else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop)

        self.kq_matmul = MatMul()
        self.kqv_matmul = MatMul()

    def forward(self, x):
        B, N, C = x.shape
        if hasattr(self, "qkv"):
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, self.qkv_dim // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = (
                qkv[0],
                qkv[1],
                qkv[2],
            )  # make torchscript happy (cannot use tensor as tuple)
        elif hasattr(self, "qk"):
            qk = (
                self.qk(x)
                .reshape(B, N, 2, self.num_heads, self.qkv_dim // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k = qk[0], qk[1]  # make torchscript happy (cannot use tensor as tuple)
            v = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        else:
            x = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q, k, v = x, x, x

        attn = self.kq_matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = self.kqv_matmul(attn, v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        qkv_mode="qkv",
        attn_proj=True,
        qkv_dim=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            qkv_mode=qkv_mode,
            attn_proj=attn_proj,
            qkv_dim=qkv_dim,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.apply(_init_weights)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViT_Early(nn.Module):
    def __init__(self, num_patches, embed_dim, drop_rate, distillation):
        super().__init__()
        self.distillation = distillation

        additional_tokens = 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.distillation:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            additional_tokens += 1
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + additional_tokens, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        if self.distillation:
            trunc_normal_(self.dist_token, std=0.02)
        self.apply(_init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        if self.distillation:
            dist_token = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        return x


class Head(torch.nn.Module):
    def __init__(self, embed_dim, norm_layer, num_classes, distillation=True):
        super().__init__()
        self.distillation = distillation
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )
        if distillation:
            self.head_dist = (
                nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
            )
        self.apply(_init_weights)

    def forward(self, x):
        x = self.norm(x)
        if self.distillation:
            x_dist = self.head_dist(x[:, 1])
            x = self.head(x[:, 0])
            if self.training:
                return x, x_dist
            else:
                # during inference, return the average of both classifier predictions
                return (x + x_dist) / 2
        else:
            x = self.head(x[:, 0])
            return x
