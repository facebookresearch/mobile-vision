#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import random
import unittest

import mobile_cv.arch.utils.fuse_utils as fuse_utils
import torch
from mobile_cv.arch.fbnet_v2.fbnet_builder import FBNetBuilder
from mobile_cv.arch.fbnet_v2.fbnet_hr import FBNetHRBuilder


def _build_model(input_channels):
    arch_def = {
        "stages": [[("conv", random.randint(2, 6), 1, 1)]] * random.randrange(3, 16, 3)
    }
    builder = FBNetBuilder()
    builder.last_depth = input_channels
    builder_hr = FBNetHRBuilder(builder)
    return builder_hr.build_model(arch_def)


class TestFBNetHRQuantize(unittest.TestCase):
    def test_post_quant(self):
        """Check that model can be quantized and traced"""
        input_channels = random.randint(1, 5)
        model = _build_model(input_channels)
        model.eval()
        gt_shape = model(torch.randn(1, input_channels, 32, 32)).shape

        # prepare quantization
        model = torch.quantization.QuantWrapper(model)
        model.eval()
        fuse_utils.fuse_model(model, inplace=True)
        model.qconfig = torch.quantization.default_qconfig
        torch.quantization.prepare(model, inplace=True)

        # calibration
        model(torch.rand(1, input_channels, 32, 32))

        # Convert to quantized model
        torch.quantization.convert(model, inplace=True)

        # Make sure model can be traced
        traced_model = torch.jit.trace(model, torch.randn(1, input_channels, 32, 32))
        traced_model.eval()
        output = traced_model(torch.randn(1, input_channels, 32, 32))
        self.assertEqual(output.shape, gt_shape)

    def test_qat(self):
        """Check that model can be quantized and traced with qat parameters

        The difference here is the use of a different quant config, prepare_qat
        """
        input_channels = random.randint(1, 5)
        model = _build_model(input_channels)
        model.eval()
        gt_shape = model(torch.randn(1, input_channels, 32, 32)).shape

        # prepare quantization
        model = torch.quantization.QuantWrapper(model)
        model.eval()
        fuse_utils.fuse_model(model, inplace=True)
        model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
        torch.quantization.prepare_qat(model, inplace=True)

        # simulated training
        model.train()
        model(torch.rand(1, input_channels, 32, 32))
        model.eval()

        # Convert to quantized model
        torch.quantization.convert(model, inplace=True)

        # Make sure model can be traced
        traced_model = torch.jit.trace(model, torch.randn(1, input_channels, 32, 32))
        traced_model.eval()
        output = traced_model(torch.randn(1, input_channels, 32, 32))
        self.assertEqual(output.shape, gt_shape)
