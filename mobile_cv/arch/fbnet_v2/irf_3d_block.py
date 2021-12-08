#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
FBNet model inverse residual building block for video
"""

import mobile_cv.arch.utils.helper as hp
import torch.nn as nn

from . import basic_blocks as bb
from .blocks_factory import PRIMITIVES


PRIMITIVES.register_dict(
    {
        "ir2dp1_k3": lambda in_channels, out_channels, stride, **kwargs: IRF2dP1Block(
            in_channels, out_channels, stride=stride, kernel_size=3, **kwargs
        ),
        "ir2dp1_k5": lambda in_channels, out_channels, stride, **kwargs: IRF2dP1Block(
            in_channels, out_channels, stride=stride, kernel_size=5, **kwargs
        ),
        "ir3d": lambda in_channels, out_channels, stride, **kwargs: IRF3dBlock(
            in_channels, out_channels, stride=stride, **kwargs
        ),
        "ir3d_k3": lambda in_channels, out_channels, stride, **kwargs: IRF3dBlock(
            in_channels, out_channels, stride=stride, kernel_size=3, **kwargs
        ),
        "ir3d_k133": lambda in_channels, out_channels, stride, **kwargs: IRF3dBlock(
            in_channels,
            out_channels,
            stride=stride,
            kernel_size=3,
            kernel_size_temporal=1,
            **hp.filter_kwargs(func=IRF3dBlock, kwargs=kwargs),
        ),
        "ir3d_k5": lambda in_channels, out_channels, stride, **kwargs: IRF3dBlock(
            in_channels, out_channels, stride=stride, kernel_size=5, **kwargs
        ),
        "ir3d_k155": lambda in_channels, out_channels, stride, **kwargs: IRF3dBlock(
            in_channels,
            out_channels,
            stride=stride,
            kernel_size=5,
            kernel_size_temporal=1,
            **hp.filter_kwargs(func=IRF3dBlock, kwargs=kwargs),
        ),
        "ir3d_pool": lambda in_channels, out_channels, stride, **kwargs: IR3DPoolBlock(
            in_channels,
            out_channels,
            stride=stride,
            **hp.filter_kwargs(func=IR3DPoolBlock, kwargs=kwargs),
        ),
    }
)


class IRF2dP1Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        expansion=6,
        kernel_size=3,
        stride=1,
        bias=False,
        conv_args="conv3d",
        bn_args="bn3d",
        relu_args="relu",
        se_args=None,
        kernel_size_temporal=3,
        stride_temporal=1,
        res_conn_args="default",
        # upsample_args="default",
        width_divisor=8,
        pw_args=None,
        dw_args=None,
        tw_args=None,
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
        super().__init__()

        conv_args = hp.unify_args(conv_args)
        bn_args = hp.unify_args(bn_args)
        relu_args = hp.unify_args(relu_args)

        mid_channels = hp.get_divisible_by(in_channels * expansion, width_divisor)

        res_conn = bb.build_residual_connect(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            drop_connect_rate=drop_connect_rate,
            **hp.unify_args(res_conn_args),
        )

        self.pw = None
        # self.shuffle = None
        if in_channels != mid_channels or always_pw:
            self.pw = bb.ConvBNRelu(
                in_channels=in_channels,
                out_channels=mid_channels,
                conv_args={
                    "kernel_size": 1,
                    "stride": 1,
                    "padding": 0,
                    "bias": bias,
                    "groups": pw_groups,
                    **hp.merge_unify_args(conv_args, pw_args),
                },
                bn_args=bn_args,
                relu_args=relu_args,
            )
        # if pw_groups > 1:
        #     self.shuffle = bb.ChannelShuffle(pw_groups)
        # use negative stride for upsampling
        # self.upsample, dw_stride = bb.build_upsample_neg_stride(
        #     stride=stride, **hp.unify_args(upsample_args)
        # )
        dw_stride = stride
        self.dw = bb.ConvBNRelu(
            in_channels=mid_channels,
            out_channels=mid_channels,
            conv_args={
                "kernel_size": (1, kernel_size, kernel_size),
                "stride": (1, dw_stride, dw_stride),
                "padding": (0, kernel_size // 2, kernel_size // 2),
                "groups": mid_channels // dw_group_ratio,
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
            in_channels=mid_channels,
            mid_channels=int(mid_channels * se_ratio),
            width_divisor=width_divisor,
            **hp.merge(relu_args=relu_args, kwargs=hp.unify_args(se_args)),
        )

        # conv in temporal dimension
        self.tw = bb.ConvBNRelu(
            in_channels=mid_channels,
            out_channels=mid_channels,
            conv_args={
                "kernel_size": (kernel_size_temporal, 1, 1),
                "stride": (stride_temporal, 1, 1),
                "padding": (kernel_size_temporal // 2, 0, 0),
                "bias": bias,
                **hp.merge_unify_args(conv_args, tw_args),
            },
            bn_args=bn_args,
            relu_args=relu_args,
        )

        self.pwl = bb.ConvBNRelu(
            in_channels=mid_channels,
            out_channels=out_channels,
            conv_args={
                "kernel_size": 1,
                "stride": 1,
                "padding": 0,
                "bias": bias,
                "groups": pwl_groups,
                **hp.merge_unify_args(conv_args, pwl_args),
            },
            bn_args={
                **bn_args,
                **{
                    "zero_gamma": (
                        zero_last_bn_gamma if res_conn is not None else False
                    )
                },
            },
            relu_args=None,
        )

        self.res_conn = res_conn
        self.out_channels = out_channels

    def forward(self, x):
        y = x
        if self.pw is not None:
            y = self.pw(y)
        # if self.shuffle is not None:
        # y = self.shuffle(y)
        # if self.upsample is not None:
        # y = self.upsample(y)
        if self.dw is not None:
            y = self.dw(y)
        if self.se is not None:
            y = self.se(y)
        if self.tw is not None:
            y = self.tw(y)
        if self.pwl is not None:
            y = self.pwl(y)
        if self.res_conn is not None:
            y = self.res_conn(y, x)
        return y


class IRF3dBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        expansion=6,
        kernel_size=3,
        stride=1,
        bias=False,
        conv_args="conv3d",
        bn_args="bn3d",
        relu_args="relu",
        se_args=None,
        kernel_size_temporal=3,
        stride_temporal=1,
        res_conn_args="default",
        # upsample_args="default",
        width_divisor=8,
        pw_args=None,
        dw_args=None,
        pwl_args=None,
        skip_dw=False,
        skip_pwl=False,
        dw_skip_bnrelu=False,
        pw_groups=1,
        dw_group_ratio=1,  # dw_group == mid_channels // dw_group_ratio
        pwl_groups=1,
        always_pw=False,
        less_se_channels=False,
        zero_last_bn_gamma=False,
        drop_connect_rate=None,
        mid_expand_out=False,  # mid_channels = out_channels * expansion if mid_expand_out=True
        last_relu=False,  # apply relu after res_conn
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
            **hp.unify_args(res_conn_args),
        )

        self.pw = None
        # self.shuffle = None
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
                bn_args=bn_args,
                relu_args=relu_args,
            )
        # if pw_groups > 1:
        #     self.shuffle = bb.ChannelShuffle(pw_groups)
        # use negative stride for upsampling
        # self.upsample, dw_stride = bb.build_upsample_neg_stride(
        #     stride=stride, **hp.unify_args(upsample_args)
        # )
        dw_stride = stride
        dw_strides = (stride_temporal, dw_stride, dw_stride)
        kernel_sizes = (kernel_size_temporal, kernel_size, kernel_size)
        paddings = tuple(x // 2 for x in kernel_sizes)
        self.dw = (
            bb.ConvBNRelu(
                in_channels=mid_channels,
                out_channels=mid_channels,
                conv_args={
                    "kernel_size": kernel_sizes,
                    "stride": dw_strides,
                    "padding": paddings,
                    "groups": mid_channels // dw_group_ratio,
                    "bias": bias,
                    **hp.merge_unify_args(conv_args, dw_args),
                },
                bn_args=bn_args if not dw_skip_bnrelu else None,
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
            mid_channels=int(mid_channels * se_ratio),
            width_divisor=width_divisor,
            **hp.merge(relu_args=relu_args, kwargs=hp.unify_args(se_args)),
        )

        self.pwl = (
            bb.ConvBNRelu(
                in_channels=mid_channels,
                out_channels=out_channels,
                conv_args={
                    "kernel_size": 1,
                    "stride": 1,
                    "padding": 0,
                    "bias": bias,
                    "groups": pwl_groups,
                    **hp.merge_unify_args(conv_args, pwl_args),
                },
                bn_args={
                    **bn_args,
                    **{
                        "zero_gamma": (
                            zero_last_bn_gamma if res_conn is not None else False
                        )
                    },
                },
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
        # if self.shuffle is not None:
        # y = self.shuffle(y)
        # if self.upsample is not None:
        # y = self.upsample(y)
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


class IR3DPoolBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        expansion=6,
        kernel_size=-1,
        stride=1,
        bias=False,
        se_args=None,
        res_conn_args="default",
        width_divisor=8,
        pw_args="conv3d",
        pool_args=None,
        pwl_args="conv3d",
        bn_args="bn3d",
        relu_args="relu",
        always_pw=False,
        less_se_channels=False,
        drop_connect_rate=None,
    ):
        super().__init__()

        bn_args = hp.unify_args(bn_args)
        relu_args = hp.unify_args(relu_args)
        mid_channels = hp.get_divisible_by(in_channels * expansion, width_divisor)

        res_conn = bb.build_residual_connect(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            drop_connect_rate=drop_connect_rate,
            **hp.unify_args(res_conn_args),
        )

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

        if kernel_size == -1:
            self.dw = nn.AdaptiveAvgPool3d(1)
        else:
            self.dw = nn.AvgPool3d(
                kernel_size, stride=stride, **hp.unify_args(pool_args)
            )

        se_ratio = 0.25
        if less_se_channels:
            se_ratio /= expansion
        self.se = bb.build_se(
            in_channels=mid_channels,
            mid_channels=int(mid_channels * se_ratio),
            width_divisor=width_divisor,
            **hp.merge(relu_args=relu_args, kwargs=hp.unify_args(se_args)),
        )

        self.pwl = bb.ConvBNRelu(
            in_channels=mid_channels,
            out_channels=out_channels,
            conv_args={
                "kernel_size": 1,
                "stride": 1,
                "padding": 0,
                "bias": bias,
                **hp.unify_args(pwl_args),
            },
            # no bn
            bn_args=None,
            # has relu
            relu_args=relu_args,
        )

        self.res_conn = res_conn
        self.out_channels = out_channels

    def forward(self, x):
        y = x
        if self.pw is not None:
            y = self.pw(y)
        if self.dw is not None:
            y = self.dw(y)
        if self.se is not None:
            y = self.se(y)
        if self.pwl is not None:
            y = self.pwl(y)
        if self.res_conn is not None:
            y = self.res_conn(y, x)
        return y
