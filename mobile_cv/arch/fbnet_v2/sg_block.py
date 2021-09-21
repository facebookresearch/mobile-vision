#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
FBNet builder sandglass building block
"""

import mobile_cv.arch.utils.helper as hp
import torch.nn as nn

from . import basic_blocks as bb


class SGBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        expansion=6,
        kernel_size=3,
        stride=1,
        bias=False,
        *,
        conv_args="conv",
        bn_args="bn",
        relu_args="relu",
        se_args=None,
        res_conn_args="default",
        upsample_args="default",
        width_divisor=8,
        dw_args=None,
        dw_bn_args=None,
        pwl_args=None,
        pwl_bn_args=None,
        pw_args=None,
        pw_bn_args=None,
        dw_skip_bnrelu=False,
        dw_group_ratio=1,  # dw_group == mid_channels // dw_group_ratio
        less_se_channels=False,
        zero_last_bn_gamma=False,
        drop_connect_rate=None,
    ):
        super().__init__()

        conv_args = hp.unify_args(conv_args)
        bn_args = hp.unify_args(bn_args)
        relu_args = hp.unify_args(relu_args)

        mid_channels = hp.get_divisible_by(
            in_channels // expansion,
            width_divisor
            # max(in_channels // expansion, out_channels // 6), 16
        )

        res_conn = bb.build_residual_connect(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            drop_connect_rate=drop_connect_rate,
            **hp.unify_args(res_conn_args),
        )

        dw_padding = (
            kernel_size // 2
            if not (dw_args and "padding" in dw_args)
            else dw_args.pop("padding")
        )
        self.dw1 = None
        if stride == 1:
            self.dw1 = bb.ConvBNRelu(
                in_channels=in_channels,
                out_channels=in_channels,
                conv_args={
                    "kernel_size": kernel_size,
                    "stride": 1,
                    "padding": dw_padding,
                    "groups": in_channels // dw_group_ratio,
                    "bias": bias,
                    **hp.merge_unify_args(conv_args, dw_args),
                },
                bn_args=hp.merge_unify_args(bn_args, dw_bn_args)
                if not dw_skip_bnrelu
                else None,
                relu_args=relu_args if not dw_skip_bnrelu else None,
            )

        self.pwl = None
        if in_channels != mid_channels:
            self.pwl = bb.ConvBNRelu(
                in_channels=in_channels,
                out_channels=mid_channels,
                conv_args={
                    "kernel_size": 1,
                    "stride": 1,
                    "padding": 0,
                    "bias": bias,
                    "groups": 1,
                    **hp.merge_unify_args(conv_args, pwl_args),
                },
                bn_args=hp.merge_unify_args(bn_args, pwl_bn_args),
                relu_args=None,
            )

        self.pw = bb.ConvBNRelu(
            in_channels=mid_channels,
            out_channels=out_channels,
            conv_args={
                "kernel_size": 1,
                "stride": 1,
                "padding": 0,
                "bias": bias,
                "groups": 1,
                **hp.merge_unify_args(conv_args, pwl_args),
            },
            bn_args=hp.merge_unify_args(bn_args, pwl_bn_args),
            relu_args=relu_args,
        )

        # use negative stride for upsampling
        self.upsample, dw_stride = bb.build_upsample_neg_stride(
            stride=stride, **hp.unify_args(upsample_args)
        )
        self.dw2 = bb.ConvBNRelu(
            in_channels=out_channels,
            out_channels=out_channels,
            conv_args={
                "kernel_size": kernel_size,
                "stride": dw_stride,
                "padding": dw_padding,
                "groups": out_channels // dw_group_ratio,
                "bias": bias,
                **hp.merge_unify_args(conv_args, dw_args),
            },
            bn_args={
                **hp.merge_unify_args(bn_args, dw_bn_args),
                **{
                    "zero_gamma": (
                        zero_last_bn_gamma if res_conn is not None else False
                    )
                },
            }
            if not dw_skip_bnrelu
            else None,
            relu_args=None,
        )

        se_ratio = 0.25
        if less_se_channels:
            se_ratio /= expansion
        self.se = bb.build_se(
            in_channels=out_channels,
            mid_channels=(round(out_channels * se_ratio)),
            width_divisor=width_divisor,
            **hp.merge(relu_args=relu_args, kwargs=hp.unify_args(se_args)),
        )

        self.res_conn = res_conn
        self.out_channels = out_channels

    def forward(self, x):
        y = x
        if self.dw1 is not None:
            y = self.dw1(y)
        if self.pwl is not None:
            y = self.pwl(y)
        if self.pw is not None:
            y = self.pw(y)
        if self.upsample is not None:
            y = self.upsample(y)
        if self.dw2 is not None:
            y = self.dw2(y)
        if self.se is not None:
            y = self.se(y)
        if self.res_conn is not None:
            y = self.res_conn(y, x)
        return y
