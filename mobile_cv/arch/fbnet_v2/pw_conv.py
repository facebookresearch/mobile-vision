#!/usr/bin/env python3

from functools import partial

import torch
import torch.nn as nn

# pyre-fixme[21]: Could not find name `fbnet_building_blocks` in
#  `mobile_cv.arch.fbnet_v2`.
from . import fbnet_building_blocks as bb


class MergeUpsample(nn.Module):
    def __init__(self, mode="nearest"):
        super().__init__()
        self.upsample = bb.Upsample(
            scale_factor=2.0,
            mode=mode,
            align_corners=None if mode == "nearest" else False,
        )

    def forward(self, x1, x2):
        y1 = self.upsample(x1)
        y2 = self.upsample(x2)
        y = y1 + y2
        return y


class MergeShuffle(nn.Module):
    def __init__(self, mode="nearest"):
        super().__init__()
        self.shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x1, x2):
        y = torch.cat((x1, x2), dim=1)
        y = self.shuffle(y)
        return y


class MergeDeconv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.deconv1 = bb.ConvBNRelu(
            input_depth=channels,
            output_depth=channels,
            kernel=4,
            stride=2,
            pad=1,
            group=channels,
            conv_builder=nn.ConvTranspose2d,
            use_relu=None,
        )
        self.deconv2 = bb.ConvBNRelu(
            input_depth=channels,
            output_depth=channels,
            kernel=4,
            stride=2,
            pad=1,
            group=channels,
            conv_builder=nn.ConvTranspose2d,
            use_relu=None,
        )

    def forward(self, x1, x2):
        y1 = self.deconv1(x1)
        y2 = self.deconv2(x2)
        y = y1 + y2
        return y


class PoolConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=True,
        groups=1,
        dilation=1,
        weight_init="kaiming_normal",
        up_method="nearest",
    ):
        super().__init__()
        assert kernel_size == 1, kernel_size
        assert stride == 1, stride
        # 1x1 down sample
        # self.path1 = nn.Conv2d(
        #     in_channels=in_channels,
        #     out_channels=out_channels,
        #     kernel_size=1,
        #     stride=2,
        #     bias=bias,
        # )
        # if weight_init == "kaiming_normal":
        #     nn.init.kaiming_normal_(
        #         self.path1.weight, mode="fan_out", nonlinearity="relu"
        #     )
        # if self.path1.bias is not None:
        #     nn.init.constant_(self.path1.bias, 0.0)
        # self.path1 = bb.ConvBNRelu(
        #     input_depth=in_channels,
        #     output_depth=out_channels,
        #     kernel=1,
        #     stride=2,
        #     pad=0,
        #     no_bias=not bias,
        # )
        self.path1_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.path1_conv = bb.ConvBNRelu(
            input_depth=in_channels,
            output_depth=out_channels,
            kernel=1,
            stride=1,
            pad=0,
            no_bias=not bias,
        )

        # max pool to downsample
        self.path2_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # self.path2_conv = nn.Conv2d(
        #     in_channels=in_channels,
        #     out_channels=out_channels,
        #     kernel_size=1,
        #     stride=1,
        #     bias=bias,
        # )
        # if weight_init == "kaiming_normal":
        #     nn.init.kaiming_normal_(
        #         self.path2_conv.weight, mode="fan_out", nonlinearity="relu"
        #     )
        # if self.path2_conv.bias is not None:
        #     nn.init.constant_(self.path2_conv.bias, 0.0)
        self.path2_conv = bb.ConvBNRelu(
            input_depth=in_channels,
            output_depth=out_channels,
            kernel=1,
            stride=1,
            pad=0,
            no_bias=not bias,
        )

        if up_method == "conv_transpose":
            self.merge = MergeDeconv(out_channels)
        elif up_method == "shuffle":
            self.merge = MergeShuffle()
        else:
            self.merge = MergeUpsample(up_method)

    def forward(self, x):
        # x1 = self.path1(x)
        x1p = self.path1_pool(x)
        x1 = self.path1_conv(x1p)
        x2p = self.path2_pool(x)
        x2 = self.path2_conv(x2p)
        y = self.merge(x1, x2)
        return y


# pyre-fixme[11]: Annotation `ConvBNRelu` is not defined as a type.
class PoolConvBNRelu(bb.ConvBNRelu):
    def __init__(
        self,
        input_depth,
        output_depth,
        kernel=3,
        stride=1,
        pad=1,
        no_bias=True,
        use_relu="relu",
        bn_type="bn",
        group=1,
        dilation=1,
        weight_init="kaiming_normal",
        up_method="nearest",
        *args,
        **kwargs
    ):
        super().__init__(
            input_depth,
            output_depth,
            kernel=kernel,
            stride=stride,
            pad=pad,
            no_bias=no_bias,
            use_relu=use_relu,
            bn_type=bn_type,
            group=group,
            dilation=dilation,
            conv_builder=partial(
                PoolConv, up_method=up_method, weight_init=weight_init
            ),
            weight_init=None,
            *args,
            **kwargs
        )


# pyre-fixme[11]: Annotation `IRFBlock` is not defined as a type.
class PoolConvIRF(bb.IRFBlock):
    def __init__(
        self,
        input_depth,
        output_depth,
        expansion,
        stride,
        bn_type="bn",
        kernel=3,
        width_divisor=1,
        shuffle_type=None,
        pw_group=1,
        se=False,
        dw_conv=None,
        dw_skip_bn=False,
        dw_skip_relu=False,
        dw_dilation=1,
        dw_builder=None,
        init_add=False,
        relu_type="relu",
        use_shortcut_conv=False,
        always_use_pw=True,
        up_method="nearest",
    ):
        def conv_func(
            input_depth,
            output_depth,
            kernel,
            stride,
            pad,
            no_bias,
            use_relu,
            bn_type,
            group,
        ):
            return PoolConvBNRelu(
                input_depth,
                output_depth,
                kernel,
                stride,
                pad,
                no_bias,
                use_relu,
                bn_type,
                group,
                up_method=up_method,
            )

        super().__init__(
            input_depth,
            output_depth,
            expansion,
            stride,
            bn_type,
            kernel,
            width_divisor,
            shuffle_type,
            pw_group,
            se,
            dw_conv,
            dw_skip_bn,
            dw_skip_relu,
            dw_dilation,
            dw_builder,
            init_add,
            relu_type,
            conv=conv_func,
            use_shortcut_conv=use_shortcut_conv,
            always_use_pw=always_use_pw,
        )
