#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import torch
from caffe2.python import dyndep


dyndep.InitOpsLibrary("@/mobile-vision/mobile_cv/mobile_cv/torch/cpp:caffe2_ops")


class MetaConv(torch.nn.Module):
    def __init__(self, weight_cl=(0.0, 0.0), bias_cl=(0.0, 0.0), output_cl=(0.0, 0.0)):
        super().__init__()
        self.weight_cl = weight_cl
        self.bias_cl = bias_cl
        self.output_cl = output_cl

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        return torch.ops._caffe2.MetaConv(
            x,
            self.weight_cl[0],
            self.weight_cl[1],
            self.bias_cl[0],
            self.bias_cl[1],
            self.output_cl[0],
            self.output_cl[1],
        )


class MetaOutput(torch.nn.Module):
    def __init__(self, output_cl=(0.0, 0.0)):
        super().__init__()
        self.output_cl = output_cl

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        return torch.ops._caffe2.MetaOutput(x, self.output_cl[0], self.output_cl[1])
