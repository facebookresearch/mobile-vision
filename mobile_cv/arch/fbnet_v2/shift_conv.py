#!/usr/bin/env python3

from functools import partial

import torch.nn as nn

# pyre-fixme[21]: Could not find name `fbnet_building_blocks` in
#  `mobile_cv.arch.fbnet_v2`.
from . import fbnet_building_blocks as bb


class ShiftConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        shift_count=0,
        weight_init="kaiming_normal",
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        if weight_init == "kaiming_normal":
            nn.init.kaiming_normal_(
                self.conv.weight, mode="fan_out", nonlinearity="relu"
            )
        self.shift_count = shift_count
        assert shift_count >= 0

    def forward(self, x):
        x_shift = x
        if self.shift_count > 0:
            channel_size = x.size()[1]
            assert channel_size > self.shift_count, channel_size
            shift_idx = channel_size - self.shift_count
            new_channel_indices = list(range(shift_idx, channel_size)) + list(
                range(shift_idx)
            )
            x_shift = x[:, new_channel_indices, :, :]
        y = self.conv(x_shift)
        return y


class ShiftMixConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        shift_count=1,
        weight_init="kaiming_normal",
    ):
        super().__init__()
        self.conv1 = ShiftConv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            shift_count=0,
            weight_init=weight_init,
        )
        self.conv2 = ShiftConv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            shift_count,
            weight_init=weight_init,
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        y = x1 + x2
        return y


# pyre-fixme[11]: Annotation `ConvBNRelu` is not defined as a type.
class ShiftConvBNRelu(bb.ConvBNRelu):
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
        shift_count=1,
        weight_init="kaiming_normal",
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
                ShiftMixConv, weight_init=weight_init, shift_count=shift_count
            ),
            weight_init=None,
            *args,
            **kwargs
        )


# pyre-fixme[11]: Annotation `IRFBlock` is not defined as a type.
class ShiftConvIRF(bb.IRFBlock):
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
        dw_skip_bn=False,
        dw_skip_relu=False,
        dw_dilation=1,
        init_add=False,
        relu_type="relu",
        conv=None,
        conv_pwl=None,
        use_shortcut_conv=False,
        always_use_pw=True,
        shift_count=1,
    ):
        def conv_func(
            input_depth,
            output_depth,
            kernel,
            stride,
            pad,
            group,
            no_bias,
            use_relu,
            bn_type,
            dilation,
        ):
            return ShiftConvBNRelu(
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
                shift_count=shift_count,
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
            dw_conv=conv_func,
            dw_skip_bn=dw_skip_bn,
            dw_skip_relu=dw_skip_relu,
            dw_dilation=dw_dilation,
            init_add=init_add,
            relu_type=relu_type,
            conv=conv,
            conv_pwl=conv_pwl,
            use_shortcut_conv=use_shortcut_conv,
            always_use_pw=always_use_pw,
        )
