#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
FBNet model building blocks factory
"""

import mobile_cv.arch.utils.helper as hp
import mobile_cv.common.misc.registry as registry
from torch import nn

from . import basic_blocks as bb, irf_block, res_block, sg_block  # noqa


PRIMITIVES = registry.Registry("blocks_factory")


_PRIMITIVES = {
    "noop": lambda in_channels, out_channels, stride, **kwargs: bb.TorchNoOp(),
    "unsqueeze": lambda in_channels, out_channels, stride, dim, **kwargs: bb.TorchUnsqueeze(
        dim
    ),
    "upsample": lambda in_channels, out_channels, stride, **kwargs: bb.Upsample(
        scale_factor=stride, mode="nearest"
    ),
    "downsample": lambda in_channels, out_channels, stride, mode="bicubic", **kwargs: bb.Upsample(  # noqa
        scale_factor=(1.0 / stride), mode=mode
    ),
    "dc_k3": lambda in_channels, out_channels, stride, **kwargs: irf_block.DepthConvBNRelu(
        in_channels, out_channels, kernel_size=3, stride=stride, **kwargs
    ),
    "skip": lambda in_channels, out_channels, stride, **kwargs: bb.Identity(
        in_channels, out_channels, stride
    ),
    "maxpool": lambda in_channels, out_channels, stride, kernel_size=3, padding=0, **kwargs: nn.MaxPool2d(  # noqa
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        **hp.filter_kwargs(func=nn.MaxPool2d, kwargs=kwargs)
    ),
    "res_k3": lambda in_channels, out_channels, stride, **kwargs: res_block.BasicBlock(
        in_channels,
        out_channels,
        **hp.merge(
            conv_args={"stride": stride, "kernel_size": 3, "padding": 1},
            kwargs=kwargs,
        )
    ),
    "res_block_k3": lambda in_channels, out_channels, stride, **kwargs: res_block.Bottleneck(
        in_channels,
        out_channels,
        **hp.merge(
            conv_args={"stride": stride, "kernel_size": 3, "padding": 1},
            kwargs=kwargs,
        )
    ),
    "res_block_3D_k133": lambda in_channels, out_channels, stride, **kwargs: res_block.Bottleneck(
        in_channels,
        out_channels,
        **hp.merge(
            conv_args={
                "name": "conv3d",
                "stride": (1, stride, stride),
                "kernel_size": (1, 3, 3),
                "padding": (0, 1, 1),
            },
            kwargs=kwargs,
        )
    ),
    "res_block_3D_k155": lambda in_channels, out_channels, stride, **kwargs: res_block.Bottleneck(
        in_channels,
        out_channels,
        **hp.merge(
            conv_args={
                "name": "conv3d",
                "stride": (1, stride, stride),
                "kernel_size": (1, 5, 5),
                "padding": (0, 2, 2),
            },
            kwargs=kwargs,
        )
    ),
    "conv": lambda in_channels, out_channels, stride, **kwargs: bb.ConvBNRelu(
        in_channels,
        out_channels,
        **hp.merge(conv_args={"stride": stride}, kwargs=kwargs)
    ),
    "conv_k1": lambda in_channels, out_channels, stride, **kwargs: bb.ConvBNRelu(
        in_channels,
        out_channels,
        **hp.merge(
            conv_args={"stride": stride, "kernel_size": 1, "padding": 0},
            kwargs=kwargs,
        )
    ),
    "conv_k3": lambda in_channels, out_channels, stride, **kwargs: bb.ConvBNRelu(
        in_channels,
        out_channels,
        **hp.merge(
            conv_args={"stride": stride, "kernel_size": 3, "padding": 1},
            kwargs=kwargs,
        )
    ),
    "aa_conv_k3": lambda in_channels, out_channels, stride, **kwargs: bb.antialiased_conv_bn_relu(
        in_channels,
        out_channels,
        **hp.merge(
            conv_args={
                "stride": stride,
                "kernel_size": 3,
                "padding": 1,
                "blur_args": {"name": "default"},
            },
            kwargs=kwargs,
        )
    ),
    "conv_k5": lambda in_channels, out_channels, stride, **kwargs: bb.ConvBNRelu(
        in_channels,
        out_channels,
        **hp.merge(
            conv_args={"stride": stride, "kernel_size": 5, "padding": 2},
            kwargs=kwargs,
        )
    ),
    "conv_k7": lambda in_channels, out_channels, stride, **kwargs: bb.ConvBNRelu(
        in_channels,
        out_channels,
        **hp.merge(
            conv_args={"stride": stride, "kernel_size": 7, "padding": 3},
            kwargs=kwargs,
        )
    ),
    "avgpool": lambda in_channels, out_channels, stride, kernel_size=2, padding=0, **kwargs: nn.AvgPool2d(  # noqa
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        **hp.filter_kwargs(func=nn.AvgPool2d, kwargs=kwargs)
    ),
    "adaptive_avg_pool": lambda in_channels, out_channels, stride, output_size, **kwargs: nn.AdaptiveAvgPool2d(  # noqa
        output_size=output_size,
        **hp.filter_kwargs(func=nn.AdaptiveAvgPool2d, kwargs=kwargs)
    ),
    "conv_hs": lambda in_channels, out_channels, stride, **kwargs: bb.ConvBNRelu(
        in_channels,
        out_channels,
        **hp.merge(conv_args={"stride": stride}, relu_args="hswish", kwargs=kwargs)
    ),
    "conv_k1_hs": lambda in_channels, out_channels, stride, **kwargs: bb.ConvBNRelu(
        in_channels,
        out_channels,
        **hp.merge(
            conv_args={"stride": stride, "kernel_size": 1, "padding": 0},
            relu_args="hswish",
            kwargs=kwargs,
        )
    ),
    "conv_k3_hs": lambda in_channels, out_channels, stride, **kwargs: bb.ConvBNRelu(
        in_channels,
        out_channels,
        **hp.merge(
            conv_args={"stride": stride, "kernel_size": 3, "padding": 1},
            relu_args="hswish",
            kwargs=kwargs,
        )
    ),
    "conv_k5_hs": lambda in_channels, out_channels, stride, **kwargs: bb.ConvBNRelu(
        in_channels,
        out_channels,
        **hp.merge(
            conv_args={"stride": stride, "kernel_size": 5, "padding": 2},
            relu_args="hswish",
            kwargs=kwargs,
        )
    ),
    "irf": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRFBlock(
        in_channels, out_channels, stride=stride, **kwargs
    ),
    "ir_k3": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRFBlock(
        in_channels, out_channels, stride=stride, kernel_size=3, **kwargs
    ),
    "ir_k3_g2": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRFBlock(
        in_channels, out_channels, stride=stride, kernel_size=3, pw_groups=2, **kwargs
    ),
    "ir_k5": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRFBlock(
        in_channels, out_channels, stride=stride, kernel_size=5, **kwargs
    ),
    "ir_k5_g2": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRFBlock(  # noqa
        in_channels, out_channels, stride=stride, kernel_size=5, pw_groups=2, **kwargs
    ),
    "aa_ir_k3": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRFBlock(
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=3,
        is_antialiased=True,
        **kwargs
    ),
    "aa_ir_k5": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRFBlock(
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=5,
        is_antialiased=True,
        **kwargs
    ),
    "ir_k3_hs": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRFBlock(
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=3,
        relu_args="hswish",
        **kwargs
    ),
    "ir_k5_hs": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRFBlock(
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=5,
        relu_args="hswish",
        **kwargs
    ),
    "ir_k3_se": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRFBlock(
        in_channels, out_channels, stride=stride, kernel_size=3, se_args="se", **kwargs
    ),
    "ir_k5_se": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRFBlock(
        in_channels, out_channels, stride=stride, kernel_size=5, se_args="se", **kwargs
    ),
    "ir_k3_sehsig": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRFBlock(  # noqa
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=3,
        se_args="se_hsig",
        **kwargs
    ),
    "ir_k5_sehsig": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRFBlock(  # noqa
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=5,
        se_args="se_hsig",
        **kwargs
    ),
    "ir_k3_sehsig_hs": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRFBlock(  # noqa
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=3,
        relu_args="hswish",
        se_args="se_hsig",
        **kwargs
    ),
    "ir_k5_sehsig_hs": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRFBlock(  # noqa
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=5,
        relu_args="hswish",
        se_args="se_hsig",
        **kwargs
    ),
    "ir_pool": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRPoolBlock(  # noqa
        in_channels,
        out_channels,
        stride=stride,
        **hp.filter_kwargs(irf_block.IRPoolBlock, kwargs)
    ),
    "ir_pool_hs": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRPoolBlock(  # noqa
        in_channels,
        out_channels,
        stride=stride,
        relu_args="hswish",
        **hp.filter_kwargs(irf_block.IRPoolBlock, kwargs)
    ),
}
PRIMITIVES.register_dict(_PRIMITIVES)
