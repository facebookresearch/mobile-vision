#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
GhostNet: More Features from Cheap Operations
"""


import math

import mobile_cv.arch.utils.helper as hp
import torch
import torch.nn as nn

from . import basic_blocks as bb
from .blocks_factory import PRIMITIVES


PRIMITIVES.register_dict(
    {
        "gb_k3_r2": lambda in_channels, out_channels, **kwargs: GhostBottleneckBlock(
            in_channels, out_channels, ratio=2, **kwargs
        )
    }
)


class GhostModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        ratio=2,
        stride=1,
        bias=False,
        conv_args=None,
        pw_args=None,
        dw_args=None,
        bn_args="bn",
        relu_args="relu",
        dw_skip_bnrelu=False,
    ):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = bb.ConvBNRelu(
            in_channels=in_channels,
            out_channels=init_channels,
            conv_args={
                "kernel_size": 1,
                "stride": 1,
                "padding": 0,
                "bias": bias,
                "groups": 1,
                **hp.merge_unify_args(conv_args, pw_args),
            },
            bn_args=bn_args,
            relu_args=relu_args,
        )

        self.cheap_operation = bb.ConvBNRelu(
            in_channels=init_channels,
            out_channels=new_channels,
            conv_args={
                "kernel_size": kernel_size,
                "stride": 1,
                "padding": kernel_size // 2,
                "groups": init_channels,
                "bias": bias,
                **hp.merge_unify_args(conv_args, dw_args),
            },
            bn_args=bn_args if not dw_skip_bnrelu else None,
            relu_args=relu_args if not dw_skip_bnrelu else None,
        )

    def forward(self, inputs):
        pw = self.primary_conv(inputs)
        dw = self.cheap_operation(pw)
        output = torch.cat([pw, dw], dim=1)
        return output[:, : self.out_channels, :, :]


class GhostBottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        expansion=6,
        kernel_size=3,
        ratio=2,
        stride=1,
        bias=False,
        conv_args="conv",
        bn_args="bn",
        relu_args="relu",
        se_args=None,
        res_conn_args="default",
        upsample_args="default",
        width_divisor=8,
        pw_args=None,
        dw_args=None,
        pwl_args=None,
        dw_skip_bnrelu=False,
        pw_groups=1,
        dw_group_ratio=1,  # dw_group == mid_channels // dw_group_ratio
        pwl_groups=1,
        always_pw=False,
        less_se_channels=False,
        zero_last_bn_gamma=False,
        drop_connect_rate=None,
    ):
        super(GhostBottleneckBlock, self).__init__()
        assert stride <= 2, f"stride = {stride}"
        exp_channels = hp.get_divisible_by(in_channels * expansion, ratio)
        out_channels = hp.get_divisible_by(out_channels, ratio)

        res_conn = bb.build_residual_connect(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            drop_connect_rate=drop_connect_rate,
            **hp.unify_args(res_conn_args),
        )

        self.pw = None
        self.shuffle = None
        self.pw = GhostModule(
            in_channels=in_channels,
            out_channels=exp_channels,
            kernel_size=kernel_size,
            ratio=ratio,
            stride=stride,
            bias=bias,
            conv_args=conv_args,
            pw_args=pw_args,
            dw_args=dw_args,
            bn_args=bn_args,
            relu_args=relu_args,
            dw_skip_bnrelu=False,
        )
        if pw_groups > 1:
            self.shuffle = bb.ChannelShuffle(pw_groups)
        self.upsample, dw_stride = bb.build_upsample_neg_stride(
            stride=stride, **hp.unify_args(upsample_args)
        )
        self.dw = bb.ConvBNRelu(
            in_channels=exp_channels,
            out_channels=exp_channels,
            conv_args={
                "kernel_size": kernel_size,
                "stride": dw_stride,
                "padding": kernel_size // 2,
                "groups": exp_channels // dw_group_ratio,
                "bias": bias,
                **hp.merge_unify_args(conv_args, dw_args),
            },
            bn_args=bn_args if not dw_skip_bnrelu else None,
            relu_args=relu_args if not dw_skip_bnrelu else None,
        )
        se_ratio = 0.25
        if less_se_channels:
            se_ratio /= expansion
        self.se = bb.build_se(
            in_channels=exp_channels,
            mid_channels=int(exp_channels * se_ratio),
            width_divisor=width_divisor,
            **hp.merge(relu_args=relu_args, kwargs=hp.unify_args(se_args)),
        )
        self.pwl = GhostModule(
            in_channels=exp_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            ratio=ratio,
            stride=stride,
            bias=bias,
            conv_args=conv_args,
            pw_args=pw_args,
            dw_args=dw_args,
            bn_args=bn_args,
            relu_args=relu_args,
            dw_skip_bnrelu=False,
        )
        self.res_conn = res_conn
        self.out_channels = out_channels

    def forward(self, x):
        y = x
        if self.pw is not None:
            y = self.pw(y)
        if self.shuffle is not None:
            y = self.shuffle(y)
        if self.upsample is not None:
            y = self.upsample(y)
        if self.dw is not None:
            y = self.dw(y)
        if self.se is not None:
            y = self.se(y)
        if self.pwl is not None:
            y = self.pwl(y)
        if self.res_conn is not None:
            y = self.res_conn(y, x)
        return y
