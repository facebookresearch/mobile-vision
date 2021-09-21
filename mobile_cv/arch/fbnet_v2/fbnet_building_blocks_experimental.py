#!/usr/bin/env python3

import math

import torch  # noqa
import torch.nn as nn
import torch.nn.functional as F
from mobile_cv.arch.utils.resize import space2depth

# pyre-fixme[21]: Could not find name `fbnet_building_blocks` in
#  `mobile_cv.arch.fbnet_v2`.
from . import fbnet_building_blocks as bb


class SelectiveKernelConvBNRelu(nn.Module):
    def __init__(
        self,
        input_depth,
        output_depth,
        stride,
        dilation=1,
        kernel=3,
        group=1,
        bias=True,
        width_divisor=1,
        relu_type="relu",
        reduction_ratio=1,
        L=32,
        **kwargs
    ):
        super().__init__()
        self.conv_a = nn.Conv2d(
            in_channels=input_depth,
            out_channels=output_depth,
            kernel_size=kernel,
            stride=stride,
            padding=(kernel // 2),
            dilation=dilation,
            groups=group,
            bias=bias,
        )
        self.conv_b = nn.Conv2d(
            in_channels=input_depth,
            out_channels=output_depth,
            kernel_size=5,
            stride=stride,
            padding=(5 // 2),
            dilation=dilation,
            groups=group,
            bias=bias,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.d = int(max(float(output_depth) / reduction_ratio, L))
        self.A = nn.Linear(self.d, output_depth)

        self.fc = nn.Linear(output_depth, self.d)
        self.bn = nn.BatchNorm2d(self.d)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        U_a = self.conv_a(x)
        U_b = self.conv_b(x)
        U = U_a + U_b

        s = self.pool(U)[:, :, 0, 0]
        s = self.fc(s)[:, :, None, None]

        z = self.relu(self.bn(s))

        a = self.A(z[:, :, 0, 0])
        a = self.softmax(a[:, :, None, None])
        b = 1 - a
        V = U_a * a + U_b * b
        return V


class ConvBNReluResizeMaxPool2d(nn.Module):

    ALLOWED_RESIZE_MODES = ("bilinear", "nearest")

    def __init__(
        self,
        input_depth,
        output_depth,
        *args,
        stride=1,
        group=1,
        mode="nearest",
        **kwargs
    ):
        super().__init__()

        # conv
        self.conv = bb.ConvBNRelu(
            input_depth, output_depth, *args, stride=1, group=group, **kwargs
        )

        # resize
        self.scale_factor = math.ceil(stride) / float(stride)
        self.mode = mode

        # pool
        self.stride = int(math.ceil(stride))
        if self.stride != 1:
            self.pool = nn.MaxPool2d(2, stride=self.stride)

        # When adding more modes, update align_corners in `forward` accordingly
        assert (
            mode in self.ALLOWED_RESIZE_MODES
        ), "Resize mode '{}' must be one of: {}".format(mode, self.ALLOWED_RESIZE_MODES)

    def forward(self, x):
        x = self.conv(x)

        align_corners = True if self.mode == "bilinear" else None
        x = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=align_corners,
        )
        if self.stride != 1:
            x = self.pool(x)
        return x


# pyre-fixme[11]: Annotation `ConvBNRelu` is not defined as a type.
class ConvBNReluResize(bb.ConvBNRelu):

    ALLOWED_RESIZE_MODES = ("bilinear", "nearest", "space2depth")

    def __init__(self, *args, stride=1, mode="nearest", **kwargs):
        super().__init__(*args, stride=1, **kwargs)

        self.scale_factor = 1.0 / float(stride)
        self.mode = mode

        # When adding more modes, update align_corners in `forward` accordingly
        assert (
            mode in self.ALLOWED_RESIZE_MODES
        ), "Resize mode '{}' must be one of: {}".format(mode, self.ALLOWED_RESIZE_MODES)

    def forward(self, x):
        x = super().forward(x)

        if self.mode == "space2depth":
            x = space2depth(x, scale_factor=self.scale_factor)
        else:
            align_corners = True if self.mode == "bilinear" else None
            if self.mode == "nearest" and self.scale_factor == 0.5:
                # nearest biases towards top-left, need to center sampling.
                # note this offset is specifically for stride 2
                x = F.pad(x[:, :, 1:, 1:], (0, 1, 0, 1))
            x = F.interpolate(
                x,
                scale_factor=self.scale_factor,
                mode=self.mode,
                align_corners=align_corners,
            )
        return x
