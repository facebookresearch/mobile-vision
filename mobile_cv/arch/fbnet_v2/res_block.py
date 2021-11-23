#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
from typing import Optional

import mobile_cv.arch.utils.helper as hp
import torch.nn as nn

from .basic_blocks import ConvBNRelu, build_relu, build_residual_connect


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv_args="conv",
        bn_args="bn",
        relu_args="relu",
        upsample_args="default",
        res_conn_args="default",
        drop_connect_rate=None,
        downsample_in_conv2=True,
        bn_in_skip=False,
        bias_in_skip=True,
        # method for qconfig_dict for fx quantization
        qmethod: Optional[str] = None,
        # additional arguments for conv
        **kwargs,
    ):
        super().__init__()

        self.qmethod = qmethod
        conv_full_args = hp.merge_unify_args(conv_args, kwargs)
        if conv_full_args.get("stride", 1) == 1 and in_channels == out_channels:
            self.conv1 = ConvBNRelu(
                in_channels,
                out_channels,
                conv_args,
                bn_args,
                relu_args,
                upsample_args,
                **kwargs,
            )
            self.conv2 = ConvBNRelu(
                out_channels,
                out_channels,
                conv_args,
                bn_args,
                None,
                upsample_args,
                **kwargs,
            )
            self.skip = None
        else:  # otherwise, use conv
            conv_args1 = copy.deepcopy(conv_args)
            mid_channels = in_channels if downsample_in_conv2 else out_channels

            if downsample_in_conv2:
                conv_args1["stride"] = 1
            self.conv1 = ConvBNRelu(
                in_channels,
                mid_channels,
                conv_args1,
                bn_args,
                relu_args,
                upsample_args,
                **kwargs,
            )

            conv_args2 = copy.deepcopy(conv_args)
            if not downsample_in_conv2:
                conv_args2["stride"] = 1
            self.conv2 = ConvBNRelu(
                mid_channels,
                out_channels,
                conv_args2,
                bn_args,
                None,
                upsample_args,
                **kwargs,
            )

            skip_args = copy.deepcopy(conv_args)
            skip_args["kernel_size"] = 1
            skip_args["padding"] = 0
            skip_args["bias"] = bias_in_skip
            self.skip = ConvBNRelu(
                in_channels,
                out_channels,
                conv_args=skip_args,
                bn_args=bn_args if bn_in_skip else None,
                relu_args=None,
                **kwargs,
            )

        self.add = build_residual_connect(
            in_channels=out_channels,
            out_channels=out_channels,
            stride=1,
            drop_connect_rate=drop_connect_rate,
            **hp.unify_args(res_conn_args),
        )
        self.relu = build_relu(num_channels=out_channels, **hp.unify_args(relu_args))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        skip = self.skip(x) if self.skip is not None else x
        out = self.add(out, skip)
        out = self.relu(out)
        return out

    def get_qconfig_dict(self, qconfig):
        qconfig_dict = {"": qconfig}
        if self.qmethod == "fp32_skip":
            qconfig_dict["module_name"] = [
                ("skip", None),
                ("add", None),
            ]
        elif self.qmethod == "fp32_skip_relu":
            qconfig_dict["module_name"] = [
                ("skip", None),
                ("add", None),
                ("relu", None),
            ]
        return qconfig_dict


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv_args="conv",
        bn_args="bn",
        relu_args="relu",
        upsample_args="default",
        res_conn_args="default",
        drop_connect_rate=None,
        bn_in_skip=False,
        expand_ratio=0.25,
        width=None,
        # additional arguments for conv
        **kwargs,
    ) -> None:
        super(Bottleneck, self).__init__()
        if width is None:
            width = int(out_channels * expand_ratio)
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        conv_args = hp.merge_unify_args(conv_args, kwargs)
        conv_args1 = copy.deepcopy(conv_args)
        conv_args1["kernel_size"] = 1
        conv_args1["stride"] = 1
        conv_args1["padding"] = 0
        self.conv1 = ConvBNRelu(
            in_channels, width, conv_args1, bn_args, relu_args, upsample_args
        )
        self.conv2 = ConvBNRelu(
            width, width, conv_args, bn_args, relu_args, upsample_args
        )
        self.conv3 = ConvBNRelu(
            width, out_channels, conv_args1, bn_args, None, upsample_args
        )
        if conv_args["stride"] == 1 and in_channels == out_channels:
            self.downsample = None
        else:
            skip_args = copy.deepcopy(conv_args)
            skip_args["kernel_size"] = 1
            skip_args["padding"] = 0
            self.downsample = ConvBNRelu(
                in_channels,
                out_channels,
                conv_args=skip_args,
                bn_args=bn_args if bn_in_skip else None,
                relu_args=None,
            )
        self.relu = build_relu(num_channels=out_channels, **hp.unify_args(relu_args))
        self.add = build_residual_connect(
            in_channels=out_channels,
            out_channels=out_channels,
            stride=1,
            drop_connect_rate=drop_connect_rate,
            **hp.unify_args(res_conn_args),
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add(out, identity)

        out = self.relu(out)

        return out
