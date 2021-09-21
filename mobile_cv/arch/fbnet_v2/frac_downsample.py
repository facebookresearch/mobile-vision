#!/usr/bin/env python3
"""
Attention-based fractional downsampling blocks. To use if in fbnet builder, it
can be defined as the following example:
    ( 'frac_ds', # op name
        64, # output channel size: should be the same as the previous block
        1, # stride: will be ignored. Keep it for compatibility.
        1, # number of repeats
        {
            'si': 3,  # number of pixels in each input patch
            'so': 2, # number of pixels in each output patch
            'key_dim_ratio': 0.25, # determines the dimension of the key vector
        }
    )
The downsampling ratio will be determined by si/so
"""

import torch
import torch.nn.functional as F

from . import irf_block


def conv1x1(in_planes, out_planes):
    """1x1 convolution"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


def _compute_padding(dim, stride):
    if dim % stride == 0:
        pad_1 = pad_2 = 0
    else:
        pad = (dim // stride + 1) * stride - dim
        pad_1 = pad // 2
        pad_2 = pad - pad_1

    return pad_1, pad_2


class FracDownSample(torch.nn.Module):
    expansion = 1

    def __init__(self, dim, si, so, key_dim_ratio=0.25):
        super(FracDownSample, self).__init__()
        key_dim = int(dim * key_dim_ratio)
        self.compute_key = conv1x1(dim, key_dim)
        self.compute_sim = conv1x1(key_dim, so ** 2)
        self.si = si
        self.so = so

    def forward(self, x):
        N, C, H, W = x.shape
        si, so = self.si, self.so

        # pad the input
        pad_t, pad_b = _compute_padding(H, si)
        pad_l, pad_r = _compute_padding(W, si)
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        N, C, H, W = x.shape

        # N, so**2, H, W
        kq = self.compute_sim(F.relu(self.compute_key(x)))

        # N, C, H, W -> N, C, H/si, W/si, si, si
        x = x.unfold(2, si, si).unfold(3, si, si).contiguous()
        # -> N, C, H/si, W/si, si**2
        # -> N, H/si, W/si, C, si**2
        x = x.view((N, C, H // si, W // si, si ** 2)).permute(0, 2, 3, 1, 4)

        # N, so**2, H, W -> N, so**2, H/si, W/si, si, si
        kq = kq.unfold(2, si, si).unfold(3, si, si).contiguous()
        # -> N, so**2, H/si, W/si, si**2
        # -> N, H/si, W/si, si**2, so**2
        kq = kq.view(N, so ** 2, H // si, W // si, si ** 2).permute(0, 2, 3, 4, 1)
        # normalize across each block that contains (si * si) elements
        kq = F.softmax(kq / si, dim=3)  # softmax(kq / sqrt(n_elements))

        # N, H/si, W/si, C, so**2
        x = torch.matmul(x, kq)

        # -> N, H/si, W/si, C, so, so -> N, C, H/si, so, W/si, so
        x = x.view(N, H // si, W // si, C, so, so).permute(0, 3, 1, 4, 2, 5)
        # -> N, C, H*so/si, W*so/si
        x = x.contiguous().view(N, C, H * so // si, W * so // si)

        return x


class FracDownSample2(torch.nn.Module):
    def __init__(self, dim, si, so):
        super(FracDownSample2, self).__init__()
        self.si = si
        self.so = so
        self.downsample = torch.nn.Conv2d(
            dim, dim * so ** 2, kernel_size=si, stride=si, groups=dim
        )

    def forward(self, x):
        N, C, H, W = x.shape
        x = self.downsample(x).view(N, C, self.so, self.so, H // self.si, W // self.si)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(N, C, self.so * H // self.si, self.so * W // self.si)

        return x


class AttDownsample(torch.nn.Module):
    def __init__(
        self,
        dim,
        stride,
        ds_kernel=3,
        compute_kq=False,
        kq_dim_ratio=0.25,
        min_kq_dim=8,
        dw_downsample=False,
        merge_mode="norm",
        merge_kernel=3,
        norm_epsilon=1e-6,
        norm_exp=1.0,
    ):
        super(AttDownsample, self).__init__()

        if compute_kq:
            kq_dim = max(int(dim * kq_dim_ratio), min_kq_dim)
            kq_dim = dim if merge_mode == "norm" else kq_dim
            kq_conv = torch.nn.Conv2d(
                dim, kq_dim * 2, kernel_size=1, stride=1, bias=False
            )
            kq_bn = torch.nn.BatchNorm2d(kq_dim * 2)
            self.compute_kq = torch.nn.Sequential(kq_conv, kq_bn)
            self.kq_dim = kq_dim
        else:
            self.compute_kq = None

        if dw_downsample:
            ds_dim = kq_dim if compute_kq else dim
            q_dw = torch.nn.Conv2d(
                ds_dim,
                ds_dim,
                kernel_size=ds_kernel,
                stride=stride,
                padding=(ds_kernel // 2),
                bias=False,
                groups=ds_dim,
            )
            q_bn = torch.nn.BatchNorm2d(ds_dim)
            self.downsample = torch.nn.Sequential(q_dw, q_bn)
        else:
            self.downsample = torch.nn.AvgPool2d(
                ds_kernel, stride=stride, padding=(ds_kernel // 2)
            )

        assert merge_mode in ("norm", "softmax")
        self.merge_mode = merge_mode
        self.merge_kernel = merge_kernel
        self.stride = stride
        self.norm_epsilon = norm_epsilon
        self.norm_exp = norm_exp

    def forward(self, x):
        if self.compute_kq:
            kq = self.compute_kq(x)
            k = kq[:, : self.kq_dim, :, :]
            q = kq[:, self.kq_dim :, :, :]
        else:
            k, q = x, x

        q = self.downsample(q)

        if self.merge_mode == "softmax":
            # N, H/S, W/S, 1, C
            q = self._unfold(q, kernel=1, stride=1, shape="NHWKC")
            # N, H/S, W/S, C, k^2
            k = self._unfold(
                k, kernel=self.merge_kernel, stride=self.stride, shape="NHWCK"
            )
            # -> N, H/S, W/S, 1, k^2
            qk = F.softmax(torch.matmul(q, k) / self.merge_kernel, dim=4)
            # N, H/S, W/S, k^2, C
            v = self._unfold(
                x, kernel=self.merge_kernel, stride=self.stride, shape="NHWKC"
            )
            # -> N, H/S, W/S, 1, C -> N, C, H/S, W/S
            y = torch.matmul(qk, v).squeeze().permute(0, 3, 1, 2)
        elif self.merge_mode == "norm":
            # N, H/S, W/S, C, 1
            q = self._unfold(q, kernel=1, stride=1, shape="NHWCK")
            # N, H/S, W/S, C, k^2
            k = self._unfold(
                k, kernel=self.merge_kernel, stride=self.stride, shape="NHWCK"
            )
            # -> N, H/S, W/S, C, k^2
            qk = torch.abs(k - q) ** self.norm_exp + self.norm_epsilon
            # N, H/S, W/S, C, k^2
            v = self._unfold(
                x, kernel=self.merge_kernel, stride=self.stride, shape="NHWCK"
            )
            # -> N, H/S, W/S, C
            y = (v * qk).sum(dim=4) / qk.sum(dim=4)
            # -> N, C, H/S, W/S
            y = y.permute(0, 3, 1, 2)

        return y

    def _unfold(self, x, kernel=1, stride=1, shape="NHWKC"):
        assert shape in ("NHWKC", "NHWCK")
        N, C, H, W = x.shape
        # N, C, H, W -> N, C, H/stride, W/stride, kernel, kernel
        padding = kernel // 2
        x = F.pad(x, (padding, padding, padding, padding))
        x = x.unfold(2, kernel, stride).unfold(3, kernel, stride).contiguous()
        # -> N, C, H/stride, W/stride, kernel^2
        x = x.view(N, C, H // stride, W // stride, kernel ** 2)
        if shape == "NHWKC":
            # -> N, H/stride, W/stride, kernel^2, C
            x = x.permute(0, 2, 3, 4, 1).contiguous()
        elif shape == "NHWCK":
            # -> N, H/stride, W/stride, C, kernel^2
            x = x.permute(0, 2, 3, 1, 4).contiguous()
        return x


class IRF_FS_Block(irf_block.IRFBlock):
    def __init__(self, in_channels, out_channels, si=1.0, so=1.0, *args, **kwargs):
        use_att_ds = kwargs.pop("use_att_ds", False)
        ad_compute_kq = kwargs.pop("ad_compute_kq", False)
        ad_dw_downsample = kwargs.pop("ad_dw_downsample", False)
        ad_merge_mode = kwargs.pop("ad_merge_mode", "norm")
        super(IRF_FS_Block, self).__init__(in_channels, out_channels, *args, **kwargs)
        self.si = si
        self.so = so
        self.stride = kwargs["stride"]
        if use_att_ds:
            # s = 2 -> k = 3, s = 3 -> k = 5, s = 4 -> k = 5
            kernel = (self.so // self.si + 1) // 2 * 2 + 1
            self.att_ds = AttDownsample(
                dim=self.dw.out_channels,
                stride=(self.so // self.si),
                ds_kernel=kernel,
                compute_kq=ad_compute_kq,
                kq_dim_ratio=0.25,
                dw_downsample=ad_dw_downsample,
                merge_mode=ad_merge_mode,
                merge_kernel=kernel,
            )
        else:
            self.att_ds = None

    def forward(self, x):
        y = x
        if self.pw:
            y = self.pw(y)
        if self.stride == 1:
            y = F.interpolate(
                y,
                scale_factor=(self.so / self.si, self.so / self.si),
                mode="bilinear",
                align_corners=True,
            )
        if self.dw:
            y = self.dw(y)
        if self.se:
            y = self.se(y)
        if self.stride == 1:
            if self.att_ds is None:
                y = F.interpolate(
                    y,
                    scale_factor=(self.si / self.so, self.si / self.so),
                    mode="bilinear",
                    align_corners=True,
                )
            else:
                y = self.att_ds(y)
        if self.pwl:
            y = self.pwl(y)
        if self.res_conn:
            y = self.res_conn(y, x)
        return y


class IRF_AD_Block(irf_block.IRFBlock):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        use_att_ds = kwargs.pop("use_att_ds", False)
        ad_compute_kq = kwargs.pop("ad_compute_kq", False)
        ad_dw_downsample = kwargs.pop("ad_dw_downsample", False)
        ad_merge_mode = kwargs.pop("ad_merge_mode", "norm")
        stride = kwargs["stride"]
        if use_att_ds:
            kwargs["stride"] = 1

        super(IRF_AD_Block, self).__init__(in_channels, out_channels, *args, **kwargs)

        if use_att_ds and stride > 1:
            # s = 2 -> k = 3, s = 3 -> k = 5, s = 4 -> k = 5
            kernel = (stride + 1) // 2 * 2 + 1
            self.att_ds = AttDownsample(
                dim=self.dw.out_channels,
                stride=stride,
                ds_kernel=kernel,
                compute_kq=ad_compute_kq,
                kq_dim_ratio=0.25,
                dw_downsample=ad_dw_downsample,
                merge_mode=ad_merge_mode,
                merge_kernel=kernel,
            )
        else:
            self.att_ds = None

    def forward(self, x):
        y = x
        if self.pw:
            y = self.pw(y)
        if self.dw:
            y = self.dw(y)
        if self.att_ds is not None:
            y = self.att_ds(y)
        if self.se:
            y = self.se(y)
        if self.pwl:
            y = self.pwl(y)
        if self.res_conn:
            y = self.res_conn(y, x)
        return y
