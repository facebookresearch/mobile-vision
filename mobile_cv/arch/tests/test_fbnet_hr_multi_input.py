#!/usr/bin/env python3
# buck test mobile-vision/mobile_cv/mobile_cv/arch/tests:test_fbnet_hr

import random
import unittest
from typing import NamedTuple

import torch
import torch.nn as nn
from mobile_cv.arch.fbnet_v2.fbnet_builder import FBNetBuilder
from mobile_cv.arch.fbnet_v2.fbnet_hr_add_input import FBNetHRMultiInputBuilder


def _rand_input(n=None, c=None, h=None, w=None):
    if n is None:
        n = random.randint(1, 5)
    if c is None:
        c = random.randint(1, 5)
    if h is None:
        h = random.randrange(32, 129, 32)
    if w is None:
        w = random.randrange(32, 129, 32)
    return torch.randn(n, c, h, w)


def _additional_input(input):
    n, c, h, w = input.shape
    return {"5": torch.randn(n, 1, h, w)}


class TestModel(torch.nn.Module):
    def __init__(self, in_c, out_c, k):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, k, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_c + 1, out_c, k, padding=1, bias=False)

    def forward(self, x, additional_input):
        additional_value = list(additional_input.values())[0]
        y = self.conv1(x)
        concat_y = torch.cat([y, additional_value], dim=1)
        return y + self.conv2(concat_y)


class DummyStage(NamedTuple):
    out_channels: int


def compare_model_outputs(model_A, model_B, data_shape):
    """Checks whether the same input results in the same output"""
    input_data = torch.rand(*data_shape)
    additional_input = _additional_input(input_data)
    model_A.eval()
    model_B.eval()
    with torch.no_grad():
        output_A = model_A(input_data, additional_input)
        output_B = model_B(input_data, additional_input)
    torch.testing.assert_allclose(output_A, output_B)


class TestFBNetHRMultiInput(unittest.TestCase):
    def test_build_and_run_model(self):
        """Check that model can run on random input"""
        input_channels = random.randint(1, 5)
        arch_def = {
            "stages": [[("conv", random.randint(2, 6), 1, 1)]]
            * random.randrange(3, 16, 3),
            "additional_inputs": {
                "5": 1,  # stage 5 has additional input
            },
        }
        builder = FBNetBuilder()
        builder.last_depth = input_channels
        builder_hr = FBNetHRMultiInputBuilder(builder)
        model = builder_hr.build_model(arch_def, input_channels)
        input = _rand_input(c=input_channels)
        additional_input = _additional_input(input)
        output = model(input, additional_input)
        gt_output_shape = list(input.shape)
        gt_output_shape[1] = arch_def["stages"][2][0][1]
        self.assertEqual(output.shape, torch.Size(gt_output_shape))

    def test_vs_manual(self):
        """Compare the output of the model with a hand created model

        The manual model is:
            output = conv1(input) + conv2(conv1(input))
        """
        in_c = random.randint(1, 5)
        out_c = random.randint(1, 5)

        # build manual
        model_A = TestModel(in_c, out_c, 3)

        # build fbnet_hr
        builder = FBNetBuilder()
        builder.add_basic_args(bn_args=None, relu_args=None)
        builder.last_depth = in_c
        builder_hr = FBNetHRMultiInputBuilder(builder)
        arch_def = {
            "stages": [
                [("conv_k3", out_c, 1, 1)],
                [("skip", out_c, 1, 1)],
                [("skip", out_c, 1, 1)],
                [("skip", out_c, 1, 1)],
                [("skip", out_c, 1, 1)],
                [("conv_k3", out_c, 1, 1)],
            ],
            "additional_inputs": {
                "5": 1,  # stage 5 has additional input
            },
        }
        model_B = builder_hr.build_model(arch_def, in_c)

        # sync models
        model_B.stages[0][0].xif0_0.conv.weight.data = model_A.conv1.weight.clone()
        model_B.stages[2][1].xif5_0.conv.weight.data = model_A.conv2.weight.clone()
        compare_model_outputs(model_A, model_B, _rand_input(c=in_c).shape)
