#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# import typing
import math
from typing import Tuple  # , Dict, Optional, Union

import mobile_cv.arch.utils.helper as hp
import torch
import torch.nn as nn

from . import basic_blocks as bb


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + "_orig")
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * math.sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + "_orig", nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name="weight"):
    EqualLR.apply(module, name)

    return module


class ModulatedConv2d_V1(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_dim: int,
        modulate: bool = True,
        demodulate: bool = True,
        use_noise: bool = True,
        **conv_args,
    ):
        """Implement a variant of ModulatedConv2d for efficient model
        Based on paper: "Analyzing and Improving the Image Quality of StyleGAN"
        and https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py
        The difference is that
            rather than weights level the modulate and demodulate are performed at activation level
            in the order: channel-wise Mod std, conv, channel-wise demod(i.e., norm std), add noise
        """
        super().__init__()
        self.eps = 1e-8
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 1. channel-wise Mod std
        if modulate:
            self.modulation = bb.build_conv(
                "eqlr_linear",
                style_dim,
                in_channels,
                bias=True,
                bias_init=1.0,
            )
        else:
            self.modulation = None

        # 2. conv
        conv_args["bias"] = True
        self.conv = bb.build_conv("eqlr_conv2d", in_channels, out_channels, **conv_args)

        # 3. channel-wise demod
        self.demodulate = demodulate

        # 4. move bias from self.conv to this step does not work
        # self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

        # 5. add noise
        self.use_noise = use_noise
        if use_noise:
            self.noise_weight = nn.Parameter(torch.zeros(1))
        else:
            self.noise_weight = None

        self.mul = bb.TorchMultiply()
        self.add = bb.TorchAdd()
        self.add_scalar = bb.TorchAddScalar(self.eps)

    def forward(self, data: Tuple[torch.Tensor, torch.Tensor]):
        x, style = data

        # 1. channel-wise Mod std
        if self.modulation is not None:
            scalar = self.modulation(style).view(-1, self.in_channels, 1, 1)
            # batch, in_channels, height, width
            x = self.mul(x, scalar)

        # 2. conv
        # batch, out_channels, height, width
        out = self.conv(x)

        # 3. channel-wise demod
        if self.demodulate:
            # batch, out_channels
            demod = torch.rsqrt(self.add_scalar(out.pow(2).sum([2, 3])))
            # batch, out_channels, height, width
            out = self.mul(out, demod.view(-1, self.out_channels, 1, 1))

        # 4. add bias
        # out = self.add(out, self.bias)

        # 5. add moise
        if self.use_noise:
            batch, _, height, width = out.shape
            noise = out.new_empty(batch, 1, height, width).normal_()

            out = self.add(out, self.mul(self.noise_weight, noise))

        return (out, style)


@bb.CONV_REGISTRY.register()
def eqlr_conv2d(in_channels, out_channels, **conv_args):
    conv_args = hp.filter_kwargs(nn.Conv2d, conv_args)
    assert conv_args["bias"], "eqlr_conv2d only supports bias=True!"

    conv = nn.Conv2d(in_channels, out_channels, **conv_args)
    conv.weight.data.normal_()
    conv.bias.data.zero_()

    return equal_lr(conv)


@bb.CONV_REGISTRY.register()
def eqlr_linear(in_channels, out_channels, bias_init=0.0, bias=True):
    assert bias, "eqlr_linear only supports bias=True!"
    linear = nn.Linear(in_channels, out_channels, bias=bias)
    linear.weight.data.normal_()
    # if bias:
    linear.bias.data.fill_(bias_init)

    return equal_lr(linear)


@bb.CONV_REGISTRY.register()
def style_conv_v1(in_channels, out_channels, style_dim, demodulate=True, **conv_args):
    return ModulatedConv2d_V1(
        in_channels, out_channels, style_dim, demodulate, **conv_args
    )
