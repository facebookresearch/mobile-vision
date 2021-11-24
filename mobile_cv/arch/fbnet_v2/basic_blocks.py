#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
FBNet model basic building blocks
"""

import logging
import numbers
from typing import List

import mobile_cv.arch.utils.helper as hp
import mobile_cv.arch.utils.misc as utils_misc
import mobile_cv.common.misc.iter_utils as iu
import mobile_cv.common.misc.registry as registry
import torch
import torch.fx
import torch.nn as nn
from mobile_cv.arch.layers import (
    FrozenBatchNorm2d,
    GroupNorm,
    NaiveSyncBatchNorm,
    NaiveSyncBatchNorm1d,
    NaiveSyncBatchNorm3d,
)
from torch.nn.quantized.modules import FloatFunctional

from .blur_pool import BlurPool2d as BlurPool

# needed for SE module with fx tracing
torch.fx.wrap("len")


BN_REGISTRY = registry.Registry("bn")
CONV_REGISTRY = registry.Registry("conv")
RELU_REGISTRY = registry.Registry("relu")
RESIDUAL_REGISTRY = registry.Registry("residual_connect")
UPSAMPLE_REGISTRY = registry.Registry("upsample")


logger = logging.getLogger(__name__)


class Identity(nn.Module):
    def __init__(self, in_channels, out_channels, stride, **kwargs):
        super().__init__()
        self.conv = None
        if in_channels != out_channels or stride != 1:
            self.conv = ConvBNRelu(
                in_channels,
                out_channels,
                **hp.merge(
                    conv_args={
                        "kernel_size": 1,
                        "stride": stride,
                        "bias": False,
                    },
                    kwargs=kwargs,
                ),
            )
        self.out_channels = out_channels

    def forward(self, x):
        out = x
        if self.conv is not None:
            out = self.conv(x)
        return out


class TorchNoOp(nn.Module):
    """An operator used to cut certain edges"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return None


class TorchAdd(nn.Module):
    """Wrapper around torch.add so that all ops can be found at build"""

    def __init__(self):
        super().__init__()
        self.add_func = FloatFunctional()

    def forward(self, x, y):
        return self.add_func.add(x, y)


class TorchNLengthAdd(nn.Module):
    """Wrapper around torch.add so that all ops can be found at build"""

    def __init__(self, num_inputs=2):
        super().__init__()
        self.num_inputs = num_inputs
        self.add_funcs = nn.ModuleList([TorchAdd() for _ in range(self.num_inputs - 1)])

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        ret = x[0]
        for idx, add_func in enumerate(self.add_funcs):
            ret = add_func(x[idx + 1], ret)
        return ret


class TorchAddScalar(nn.Module):
    """Wrapper around torch.add so that all ops can be found at build
    y must be a scalar, needed for quantization
    """

    def __init__(self, scalar):
        super().__init__()
        self.add_func = FloatFunctional()
        self.scalar = scalar

    def forward(self, x):
        return self.add_func.add_scalar(x, self.scalar)


class TorchMultiply(nn.Module):
    """Wrapper around torch.mul so that all ops can be found at build"""

    def __init__(self):
        super().__init__()
        self.mul_func = FloatFunctional()

    def forward(self, x, y):
        return self.mul_func.mul(x, y)


class TorchMulScalar(nn.Module):
    """Wrapper around torch.mul so that all ops can be found at build
    y must be a scalar, needed for quantization
    """

    def __init__(self, scalar):
        super().__init__()
        self.mul_func = FloatFunctional()
        self.scalar = scalar

    def forward(self, x):
        return self.mul_func.mul_scalar(x, self.scalar)


class TorchCat(nn.Module):
    """Wrapper around torch.cat so that all ops can be found at build"""

    def __init__(self):
        super().__init__()
        self.cat_func = FloatFunctional()

    def forward(self, tensors: List[torch.Tensor], dim: int):
        return self.cat_func.cat(tensors, dim)


class TorchCat2T(nn.Module):
    """Concatenate two tensors in channel dimension"""

    def __init__(self):
        super().__init__()
        self.cat_func = FloatFunctional()

    def forward(self, x, y):
        return self.cat_func.cat([x, y], dim=1)


class TorchUnsqueeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.unsqueeze(x, self.dim)


class TorchWhere(nn.Module):
    def forward(self, condition, x, y):
        return torch.where(condition, x, y)


class IgnoreWhereSelectX1(nn.Module):
    """Uses the same interface as TorchWhere but returns x1"""

    def forward(self, condition, x1, x2):
        return x1


class ChooseRightPath(nn.Module):
    """Given x, y inputs: returns y input"""

    def forward(self, x, y):
        return y


class ChannelShuffle1d(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle1d, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,D] -> [N,g,C/g,D] -> [N,C/g,g,D] -> [N,C,D]"""
        N, C, D = x.size()
        g = self.groups
        assert C % g == 0, "Incompatible group size {} for input channel {}".format(
            g, C
        )
        return (
            x.view(N, g, int(C / g), D).permute(0, 2, 1, 3).contiguous().view(N, C, D)
        )


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,W] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        return nn.functional.channel_shuffle(x, self.groups)


class HSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU6(inplace=True)
        self.add_scalar = TorchAddScalar(3.0)
        self.mul_scalar = TorchMulScalar(1.0 / 6.0)

    def forward(self, x):
        # return self.relu(x + 3.0) / 6.0
        return self.mul_scalar(self.relu(self.add_scalar(x)))


HSwish = torch.nn.Hardswish


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sig = nn.Sigmoid()
        self.mul = TorchMultiply()

    def forward(self, x):
        return self.mul(x, self.sig(x))


def _init_conv_weight(op, weight_init="kaiming_normal"):
    if weight_init is None:
        return

    elif weight_init == "kaiming_normal":
        nn.init.kaiming_normal_(op.weight, mode="fan_out", nonlinearity="relu")
        if hasattr(op, "bias") and op.bias is not None:
            nn.init.constant_(op.bias, 0.0)

    elif weight_init == "orthogonal":
        nn.init.orthogonal_(op.weight.data, gain=0.6)
        if hasattr(op, "bias") and op.bias is not None:
            nn.init.constant_(op.bias.data, 0.01)

    else:
        raise AssertionError(f"Unsupported init type {weight_init}")


def build_conv(
    name="conv",
    in_channels=None,
    out_channels=None,
    weight_init="kaiming_normal",
    **conv_args,
):
    if name is None or name == "none":
        return None

    def _create_conv(conv_class, conv_args):
        conv_args = hp.filter_kwargs(conv_class, conv_args)
        if "kernel_size" not in conv_args:
            conv_args["kernel_size"] = 1
        ret = conv_class(in_channels, out_channels, **conv_args)
        _init_conv_weight(ret, weight_init)
        return ret

    CONV_DEFAULT_MAPS = {
        "conv1d": lambda: _create_conv(nn.Conv1d, conv_args),
        "conv": lambda: _create_conv(nn.Conv2d, conv_args),
        "conv2d": lambda: _create_conv(nn.Conv2d, conv_args),
        "conv3d": lambda: _create_conv(nn.Conv3d, conv_args),
        "linear": lambda: nn.Linear(in_channels, out_channels),
    }

    if name in CONV_DEFAULT_MAPS:
        return CONV_DEFAULT_MAPS[name]()

    return CONV_REGISTRY.get(name)(in_channels, out_channels, **conv_args)


def build_bn(name, num_channels, zero_gamma=None, gamma_beta=None, **kwargs):
    def _create_bn(bn_class):
        bn_op = bn_class(num_channels, **kwargs)
        if zero_gamma is True:
            nn.init.constant_(bn_op.weight, 0.0)
        if gamma_beta is not None:
            assert isinstance(gamma_beta, tuple)
            nn.init.constant_(bn_op.weight, gamma_beta[0])
            nn.init.constant_(bn_op.bias, gamma_beta[1])
        return bn_op

    BN_DEFAULT_MAPS = {
        # 2d
        "bn": lambda: _create_bn(nn.BatchNorm2d),
        "sync_bn": lambda: _create_bn(NaiveSyncBatchNorm),
        "naiveSyncBN": lambda: _create_bn(NaiveSyncBatchNorm),
        # 3d
        "bn3d": lambda: _create_bn(nn.BatchNorm3d),
        "naiveSyncBN3d": lambda: _create_bn(NaiveSyncBatchNorm3d),
        # 1d
        "bn1d": lambda: _create_bn(nn.BatchNorm1d),
        "naiveSyncBN1d": lambda: _create_bn(NaiveSyncBatchNorm1d),
        # any dimension
        "sync_bn_torch": lambda: _create_bn(nn.SyncBatchNorm),
        # others
        "gn": lambda: GroupNorm(num_channels=num_channels, **kwargs),
        "instance": lambda: nn.InstanceNorm2d(num_channels, **kwargs),
        "frozen_bn": lambda: FrozenBatchNorm2d(num_channels, **kwargs),
    }

    if name is None or name == "none":
        return None
    if name in BN_DEFAULT_MAPS:
        return BN_DEFAULT_MAPS[name]()

    return BN_REGISTRY.get(name)(num_channels, zero_gamma=zero_gamma, **kwargs)


def build_relu(name=None, num_channels=None, **kwargs):
    inplace = kwargs.pop("inplace", True)
    if name is None or name == "none":
        return None
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    if name == "relu6":
        return nn.ReLU6(inplace=inplace)
    if name == "leakyrelu":
        return nn.LeakyReLU(inplace=inplace, **kwargs)
    if name == "prelu":
        return nn.PReLU(num_parameters=num_channels, **kwargs)
    if name == "hswish":
        return HSwish()
    if name == "swish":
        return Swish()
    if name in ["sig", "sigmoid"]:
        return nn.Sigmoid()
    if name in ["hsig", "hsigmoid"]:
        return HSigmoid()

    return RELU_REGISTRY.get(name)(**kwargs)


def build_residual_connect(
    name, in_channels, out_channels, stride, drop_connect_rate=None, **res_args
):
    if name is None or name == "none":
        return None
    if name == "default" or name == "projection":
        assert isinstance(stride, (numbers.Number, tuple, list))
        if isinstance(stride, (tuple, list)):
            stride_one = all(x == 1 for x in stride)
        else:
            stride_one = stride == 1
        add_f = (
            TorchAdd()
            if drop_connect_rate is None
            else AddWithDropConnect(drop_connect_rate)
        )
        if in_channels == out_channels and stride_one:
            return add_f
        if name == "projection":
            return AddWithResProj(in_channels, out_channels, stride, add_f, **res_args)
        return None
    return RESIDUAL_REGISTRY.get(name)(in_channels, out_channels, stride, **res_args)


class ConvNormAct(nn.Module):
    def __init__(self, conv, norm, act=None):
        super().__init__()
        self.conv = conv
        self.norm = norm
        self.act = act

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ConvBNRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv_args="conv",
        bn_args="bn",
        relu_args="relu",
        upsample_args="default",
        # additional arguments for conv
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_args = conv_args
        self.bn_args = bn_args
        self.relu_args = relu_args
        self.upsample_args = upsample_args
        self.kwargs = kwargs

        conv_full_args = hp.merge_unify_args(conv_args, kwargs)
        conv_stride = conv_full_args.pop("stride", 1)
        # build upsample op if stride is negative
        upsample_op, conv_stride = build_upsample_neg_stride(
            stride=conv_stride, **hp.unify_args(upsample_args)
        )
        # build conv
        conv_op = build_conv(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=conv_stride,
            **conv_full_args,
        )

        # register in order
        self.conv = conv_op

        self.bn = (
            build_bn(num_channels=out_channels, **hp.unify_args(bn_args))
            if bn_args is not None and len(bn_args) != 0
            else None
        )
        self.relu = (
            build_relu(num_channels=out_channels, **hp.unify_args(relu_args))
            if relu_args is not None
            else None
        )
        self.upsample = upsample_op
        self.stride = conv_stride
        self.conv_full_args = conv_full_args

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


def antialiased_conv_bn_relu(
    in_channels,
    out_channels,
    conv_args="conv",
    bn_args="bn",
    relu_args="relu",
    upsample_args="default",
    # additional arguments for conv
    **kwargs,
):
    conv_full_args = hp.merge_unify_args(conv_args, kwargs).copy()
    conv_stride = conv_full_args.pop("stride", 1)

    if conv_stride > 1:
        blur_args = conv_full_args.pop("blur_args", {"name": "default"})
        blur_args = hp.unify_args(blur_args)

        apply_blur_before_conv = blur_args.pop("apply_blur_before_conv", False)
        if apply_blur_before_conv:
            blur = build_blur(num_channels=in_channels, stride=1, **blur_args)
            # keep conv stride intact
        else:
            blur = build_blur(
                num_channels=out_channels, stride=conv_stride, **blur_args
            )
            if blur is not None:
                # override conv stride to be 1
                conv_stride = 1

        # remove 'stride' from kwargs if it exists
        kwargs.pop("stride", None)
        conv_full_args["stride"] = conv_stride

        conv_bn_relu = ConvBNRelu(
            in_channels,
            out_channels,
            conv_args=conv_full_args,
            bn_args=bn_args,
            relu_args=relu_args,
            upsample_args=upsample_args,
            **kwargs,
        )
    else:
        conv_bn_relu = ConvBNRelu(
            in_channels,
            out_channels,
            conv_args=conv_args,
            bn_args=bn_args,
            relu_args=relu_args,
            upsample_args=upsample_args,
            **kwargs,
        )
        blur = None
    if blur is not None:
        if apply_blur_before_conv:
            return nn.Sequential(blur, conv_bn_relu)
        else:
            return nn.Sequential(conv_bn_relu, blur)
    else:
        return conv_bn_relu


def _se_op_fc(in_channels, mid_channels, relu_args, sigmoid_type):
    conv1_relu = ConvBNRelu(
        in_channels,
        mid_channels,
        conv_args="linear",
        bn_args=None,
        relu_args=relu_args,
    )
    conv2 = nn.Linear(mid_channels, in_channels, bias=True)
    sig = build_relu(sigmoid_type)
    ret = nn.Sequential(conv1_relu, conv2, sig)
    return ret


def _se_op_conv(in_channels, mid_channels, relu_args, sigmoid_type, conv_type="conv2d"):
    conv1_relu = ConvBNRelu(
        in_channels,
        mid_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        conv_args=conv_type,
        bn_args=None,
        relu_args=relu_args,
    )
    conv2 = build_conv(
        conv_type, mid_channels, in_channels, kernel_size=1, weight_init=None
    )
    sig = build_relu(sigmoid_type)
    ret = nn.Sequential(conv1_relu, conv2, sig)
    return ret


class SEModule(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        fc=False,
        sigmoid_type="sigmoid",
        relu_args="relu",
    ):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if not fc:
            self.se = _se_op_conv(
                in_channels,
                mid_channels,
                relu_args=relu_args,
                sigmoid_type=sigmoid_type,
                conv_type="conv2d",
            )
        else:
            self.se = _se_op_fc(
                in_channels,
                mid_channels,
                relu_args=relu_args,
                sigmoid_type=sigmoid_type,
            )
        self.use_fc = fc
        self.mul = TorchMultiply()

    def forward(self, x):
        torch._assert(len(x.shape) == 4, "input must have dimension 4")
        n, c = x.shape[:2]
        y = self.avg_pool(x)
        if self.use_fc:
            y = y.view(n, c)
        y = self.se(y)
        if self.use_fc:
            y = y.view(n, c, 1, 1).expand_as(x)
        return self.mul(x, y)


class SE3DModule(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        fc=False,
        sigmoid_type="sigmoid",
        relu_args="relu",
    ):
        super(SE3DModule, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        if not fc:
            self.se = _se_op_conv(
                in_channels,
                mid_channels,
                relu_args=relu_args,
                sigmoid_type=sigmoid_type,
                conv_type="conv3d",
            )
        else:
            self.se = _se_op_fc(
                in_channels,
                mid_channels,
                relu_args=relu_args,
                sigmoid_type=sigmoid_type,
            )
        self.use_fc = fc
        self.mul = TorchMultiply()

    def forward(self, x):
        torch._assert(len(x.shape) == 5, "input must have dimension 5")
        n, c = x.shape[:2]
        y = self.avg_pool(x)
        if self.use_fc:
            y = y.view(n, c)
        y = self.se(y)
        if self.use_fc:
            y = y.view(n, c, 1, 1, 1).expand_as(x)
        return self.mul(x, y)


def build_se(name=None, in_channels=None, mid_channels=None, width_divisor=1, **kwargs):
    if name is None:
        return None
    mid_channels = hp.get_divisible_by(mid_channels, width_divisor)
    if name == "se":
        return SEModule(in_channels, mid_channels, **kwargs)
    if name == "se_fc":
        return SEModule(in_channels, mid_channels, fc=True, **kwargs)
    if name == "se_hsig":
        return SEModule(in_channels, mid_channels, sigmoid_type="hsigmoid", **kwargs)
    if name == "se3d":
        return SE3DModule(in_channels, mid_channels, **kwargs)
    if name == "se3d_hsig":
        return SE3DModule(in_channels, mid_channels, sigmoid_type="hsigmoid", **kwargs)
    raise Exception(f"Invalid SEModule arugments {name}")


class Upsample(nn.Module):
    def __init__(
        self, size=None, scale_factor=None, mode="nearest", align_corners=None
    ):
        super(Upsample, self).__init__()
        self.size = size
        self.scale = scale_factor
        self.mode = mode
        self.align_corners = align_corners

        # scripting requires float instead of int
        if isinstance(self.scale, int):
            self.scale = float(self.scale)
        elif isinstance(self.scale, list):
            self.scale = [float(x) for x in self.scale]

    def forward(self, x):
        return torch.nn.functional.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale,
            mode=self.mode,
            align_corners=self.align_corners,
        )

    def __repr__(self):
        ret = []
        attr_list = ["size", "scale", "mode", "align_corners"]
        for x in attr_list:
            val = getattr(self, x, None)
            if val is not None:
                ret.append(f"{x}={val}")
        return f"Upsample({', '.join(ret)})"


def build_upsample(name=None, scales=None, **kwargs):
    if name is None or scales is None:
        return None

    if all(x == 1 for x in iu.recursive_iterate(scales)):
        return None

    if name == "default":
        ret = Upsample(scale_factor=scales, **kwargs)
    else:
        ret = UPSAMPLE_REGISTRY.get(name)(scales, **kwargs)

    return ret


def build_blur(name=None, num_channels=None, stride=None, **kwargs):
    """
    Create blur
    """
    if name is None:
        return None

    if name == "default":
        ret = BlurPool(num_channels, stride=stride, **kwargs)
    else:
        raise ValueError(f"Unknown blur name: {name}")

    return ret


def is_neg_stride(stride):
    return all(x < 0 for x in iu.recursive_iterate(stride))


def get_neg_stride(stride):
    iters = iu.recursive_iterate(stride)
    for ss in iters:
        assert ss is not None
        iters.send(-ss)
    return iters.value


def build_upsample_neg_stride(name=None, stride=None, **kwargs):
    """Use negative stride to represent scales, i.e., stride=-2 means scale=2
    Return upsample op if the stride is negative, return None otherwise
    Reset and return the stride to 1 if it is negative
    """
    if name is None:
        return None, stride

    if not is_neg_stride(stride):
        return None, stride

    scales = get_neg_stride(stride)
    ret = build_upsample(name, scales=scales, **kwargs)

    return ret, 1


class AddWithDropConnect(nn.Module):
    """Apply drop connect on x before adding with y"""

    def __init__(self, drop_connect_rate):
        super().__init__()
        self.drop_connect_rate = drop_connect_rate
        self.add = TorchAdd()

    def forward(self, x, y):
        xx = utils_misc.drop_connect_batch(x, self.drop_connect_rate, self.training)
        return self.add(xx, y)

    def extra_repr(self):
        return f"drop_connect_rate={self.drop_connect_rate}"


class AddWithResProj(nn.Module):
    """Apply pw conv on x before adding with y"""

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        add_f,
        bias=False,
        conv_args="conv",
        bn_args="bn",
    ):
        super().__init__()
        self.add_f = add_f
        self.proj = ConvBNRelu(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_args={
                "kernel_size": 1,
                "stride": stride,
                "padding": 0,
                "groups": 1,
                "bias": bias,
                **hp.unify_args(conv_args),
            },
            bn_args=hp.unify_args(bn_args),
            relu_args=None,
        )

    def forward(self, y, x):
        x = self.proj(x)
        return self.add_f(y, x)
