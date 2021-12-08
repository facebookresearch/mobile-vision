#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
FBNet model inverse residual building block
"""

import mobile_cv.arch.utils.helper as hp
import torch.nn as nn

from . import basic_blocks as bb


class IRFBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        expansion=6,
        kernel_size=3,
        stride=1,
        bias=False,
        conv_args="conv",
        bn_args="bn",
        relu_args="relu",
        se_args=None,
        round_se_channels=False,
        res_conn_args="default",
        upsample_args="default",
        width_divisor=8,
        pw_args=None,
        pw_bn_args=None,
        skip_dw=False,
        dw_args=None,
        dw_bn_args=None,
        skip_pwl=False,
        pwl_args=None,
        pwl_bn_args=None,
        dw_skip_bnrelu=False,
        skip_pwl_bn=False,
        pw_groups=1,
        dw_group_ratio=1,  # dw_group == mid_channels // dw_group_ratio
        pwl_groups=1,
        always_pw=False,
        skip_shuffle=False,
        less_se_channels=False,
        zero_last_bn_gamma=False,
        drop_connect_rate=None,
        mid_expand_out=False,  # mid_channels = out_channels * expansion if mid_expand_out=True
        last_relu=False,  # apply relu after res_conn
        is_antialiased=False,  # apply antialiasing (BlurPool) for strided convolutions?
    ):
        super().__init__()

        conv_args = hp.unify_args(conv_args)
        bn_args = hp.unify_args(bn_args)
        relu_args = hp.unify_args(relu_args)

        mid_channels_base = out_channels if mid_expand_out else in_channels
        mid_channels = hp.get_divisible_by(mid_channels_base * expansion, width_divisor)

        res_conn = bb.build_residual_connect(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            drop_connect_rate=drop_connect_rate,
            **hp.unify_args(res_conn_args)
        )

        self.pw = None
        self.shuffle = None
        if in_channels != mid_channels or always_pw:
            self.pw = bb.ConvBNRelu(
                in_channels=in_channels,
                out_channels=mid_channels,
                conv_args={
                    "kernel_size": pw_args.pop("kernel_size", 1)
                    if pw_args is not None
                    else 1,
                    "stride": pw_args.pop("stride", 1) if pw_args is not None else 1,
                    "padding": pw_args.pop("padding", 0) if pw_args is not None else 0,
                    "bias": bias,
                    "groups": pw_groups,
                    **hp.merge_unify_args(conv_args, pw_args),
                },
                bn_args=hp.merge_unify_args(bn_args, pw_bn_args),
                relu_args=relu_args,
            )
        if pw_groups > 1 and not skip_shuffle:
            self.shuffle = bb.ChannelShuffle(pw_groups)
        # use negative stride for upsampling
        self.upsample, dw_stride = bb.build_upsample_neg_stride(
            stride=stride, **hp.unify_args(upsample_args)
        )

        dw_padding = (
            kernel_size // 2
            if not (dw_args and "padding" in dw_args)
            else dw_args.pop("padding")
        )
        self.dw = (
            (bb.ConvBNRelu if not is_antialiased else bb.antialiased_conv_bn_relu)(
                in_channels=mid_channels,
                out_channels=mid_channels,
                conv_args={
                    "kernel_size": kernel_size,
                    "stride": dw_stride,
                    "padding": dw_padding,
                    "groups": mid_channels // dw_group_ratio,
                    "bias": bias,
                    **hp.merge_unify_args(conv_args, dw_args),
                },
                bn_args=hp.merge_unify_args(bn_args, dw_bn_args)
                if not dw_skip_bnrelu
                else None,
                relu_args=relu_args if not dw_skip_bnrelu else None,
            )
            if not skip_dw
            else None
        )
        se_ratio = 0.25
        if less_se_channels:
            se_ratio /= expansion
        self.se = bb.build_se(
            in_channels=mid_channels,
            mid_channels=(
                int(mid_channels * se_ratio)
                if not round_se_channels
                else round(mid_channels * se_ratio)
            ),
            width_divisor=width_divisor,
            **hp.merge(relu_args=relu_args, kwargs=hp.unify_args(se_args))
        )

        self.pwl = (
            (bb.ConvBNRelu if not is_antialiased else bb.antialiased_conv_bn_relu)(
                in_channels=mid_channels,
                out_channels=out_channels,
                conv_args={
                    "kernel_size": pwl_args.pop("kernel_size", 1)
                    if pwl_args is not None
                    else 1,
                    "stride": pwl_args.pop("stride", 1) if pwl_args is not None else 1,
                    "padding": pwl_args.pop("padding", 0)
                    if pwl_args is not None
                    else 0,
                    "bias": bias,
                    "groups": pwl_groups,
                    **hp.merge_unify_args(conv_args, pwl_args),
                },
                bn_args={
                    **hp.merge_unify_args(bn_args, pwl_bn_args),
                    **{
                        "zero_gamma": (
                            zero_last_bn_gamma if res_conn is not None else False
                        )
                    },
                }
                if not skip_pwl_bn
                else None,
                relu_args=None,
            )
            if not skip_pwl
            else None
        )

        self.res_conn = res_conn
        self.relu = (
            bb.build_relu(num_channels=out_channels, **relu_args) if last_relu else None
        )
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
        if self.relu is not None:
            y = self.relu(y)
        return y


class IRPoolBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        expansion=6,
        kernel_size=-1,
        stride=1,
        bias=False,
        pw_args="conv",
        pwl_args="conv",
        pool_args=None,
        bn_args="bn",
        relu_args="relu",
        se_args=None,  # se after pool
        pw_se_args=None,  # se after pw conv
        res_conn_args="default",
        width_divisor=8,
        always_pw=False,
        less_se_channels=False,
    ):
        super().__init__()

        mid_channels = hp.get_divisible_by(in_channels * expansion, width_divisor)

        self.pw = None
        if in_channels != mid_channels or always_pw:
            self.pw = bb.ConvBNRelu(
                in_channels=in_channels,
                out_channels=mid_channels,
                conv_args={
                    "kernel_size": 1,
                    "stride": 1,
                    "padding": 0,
                    "bias": bias,
                    **hp.unify_args(pw_args),
                },
                bn_args=bn_args,
                relu_args=relu_args,
            )

        pw_se_ratio = 0.25
        if less_se_channels:
            pw_se_ratio /= expansion
        self.pw_se = bb.build_se(
            in_channels=mid_channels,
            mid_channels=(mid_channels * pw_se_ratio),
            width_divisor=width_divisor,
            **hp.merge_unify_args({"relu_args": relu_args}, pw_se_args)
        )

        if kernel_size == -1:
            self.dw = nn.AdaptiveAvgPool2d(1)
        else:
            self.dw = nn.AvgPool2d(
                kernel_size, stride=stride, **hp.unify_args(pool_args)
            )

        se_ratio = 0.25
        if less_se_channels:
            se_ratio /= expansion
        self.se = bb.build_se(
            in_channels=mid_channels,
            mid_channels=(mid_channels * se_ratio),
            width_divisor=width_divisor,
            **hp.merge_unify_args({"relu_args": relu_args}, se_args)
        )
        self.pwl = bb.ConvBNRelu(
            in_channels=mid_channels,
            out_channels=out_channels,
            conv_args={
                "kernel_size": 1,
                "stride": 1,
                "padding": 0,
                "bias": bias,
                **hp.merge_unify_args(pwl_args),
            },
            # no bn
            bn_args=None,
            # has relu
            relu_args=relu_args,
        )
        self.res_conn = bb.build_residual_connect(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            **hp.unify_args(res_conn_args)
        )
        self.out_channels = out_channels

    def forward(self, x):
        y = x
        if self.pw is not None:
            y = self.pw(y)
        if self.pw_se is not None:
            y = self.pw_se(y)
        if self.dw is not None:
            y = self.dw(y)
        if self.se is not None:
            y = self.se(y)
        if self.pwl is not None:
            y = self.pwl(y)
        if self.res_conn is not None:
            y = self.res_conn(y, x)
        return y


class DepthConvBNRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        bias=False,
        conv_args="conv",
        bn_args="bn",
        relu_args="relu",
        width_divisor=8,
        act_first=False,
        dw_skip_bnrelu=True,
    ):
        super().__init__()

        conv_args = hp.unify_args(conv_args)
        bn_args = hp.unify_args(bn_args)
        relu_args = hp.unify_args(relu_args)

        self.act = (
            bb.build_relu(num_channels=in_channels, **relu_args) if act_first else None
        )

        self.dw = bb.ConvBNRelu(
            in_channels=in_channels,
            out_channels=in_channels,
            conv_args={
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": kernel_size // 2,
                "groups": in_channels,
                "bias": bias,
                **conv_args,
            },
            bn_args=bn_args if not dw_skip_bnrelu else None,
            relu_args=relu_args if not dw_skip_bnrelu else None,
        )
        self.pw = bb.ConvBNRelu(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_args={
                "kernel_size": 1,
                "stride": 1,
                "padding": 0,
                "groups": 1,
                "bias": bias,
                **conv_args,
            },
            bn_args=bn_args,
            relu_args=relu_args if not act_first else None,
        )
        self.out_channels = out_channels

    def forward(self, x):
        y = x
        if self.act is not None:
            y = self.act(y)
        y = self.dw(y)
        y = self.pw(y)
        return y
