#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
An Improved One millisecond Mobile Backbone (https://arxiv.org/abs/2206.04040)
"""


import mobile_cv.arch.utils.helper as hp
import torch.nn as nn

from mobile_cv.arch.fbnet_v2.basic_blocks import build_bn, build_relu, ConvBNRelu


class MobileOneBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        bias=False,
        conv_args="conv",
        bn_args="bn",
        relu_args="relu",
        over_param_branches=1,  # number of over-parameterization branches
        deploy=False,
        # additional arguments for conv
        **kwargs,
    ):
        super().__init__()

        def _build_model_for_train():
            self.dw_1x1 = ConvBNRelu(
                in_channels=in_channels,
                out_channels=in_channels,
                conv_args={
                    "kernel_size": 1,
                    "stride": stride,
                    "padding": 0,
                    "groups": in_channels,
                    "bias": bias,
                    **conv_args,
                },
                bn_args=bn_args,
                relu_args=None,
            )

            self.dw_3x3_blocks = nn.ModuleList()
            for _ in range(over_param_branches):
                self.dw_3x3_blocks.append(
                    ConvBNRelu(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        conv_args={
                            "kernel_size": 3,
                            "stride": stride,
                            "padding": 1,
                            "groups": in_channels,
                            "bias": bias,
                            **conv_args,
                        },
                        bn_args=bn_args,
                        relu_args=None,
                    )
                )

            self.pw_1x1_blocks = nn.ModuleList()
            for _ in range(over_param_branches):
                self.pw_1x1_blocks.append(
                    ConvBNRelu(
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
                        relu_args=None,
                    )
                )

            self.dw_skip = (
                build_bn(num_channels=in_channels, **hp.unify_args(bn_args))
                if stride == 1
                else None
            )
            # In MobileNetOne (stride == 1) is euqivalent to (in_channels == out_channels) and vice versa.
            self.pw_skip = (
                build_bn(num_channels=out_channels, **hp.unify_args(bn_args))
                if in_channels == out_channels
                else None
            )

            self.relu = build_relu(
                num_channels=out_channels, **hp.unify_args(relu_args)
            )

        def _build_model_for_deploy():
            self.dw_3x3 = ConvBNRelu(
                in_channels=in_channels,
                out_channels=in_channels,
                conv_args={
                    "kernel_size": 3,
                    "stride": stride,
                    "padding": 1,
                    "groups": in_channels,
                    "bias": bias,
                    **conv_args,
                },
                bn_args=bn_args,
                relu_args=relu_args,
            )

            self.pw_1x1 = ConvBNRelu(
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
                relu_args=relu_args,
            )

        # TODO(xfw): add switch_to_deploy() to convert train-time models to deploy-time models
        conv_args = hp.unify_args(conv_args)
        bn_args = hp.unify_args(bn_args)
        relu_args = hp.unify_args(relu_args)
        self.deploy = deploy
        if deploy:
            _build_model_for_deploy()
        else:
            _build_model_for_train()

    def forward(self, x):
        if self.deploy:
            out = self.pw_1x1(self.dw_3x3(x))
        else:
            dw_out = self.dw_1x1(x) + sum([dw_3x3(x) for dw_3x3 in self.dw_3x3_blocks])
            dw_out += self.dw_skip(x) if self.dw_skip is not None else 0
            dw_out = self.relu(dw_out)

            out = sum([pw_1x1(dw_out) for pw_1x1 in self.pw_1x1_blocks])
            out += self.pw_skip(dw_out) if self.pw_skip is not None else 0
            out = self.relu(out)
        return out
