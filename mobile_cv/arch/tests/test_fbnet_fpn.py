#!/usr/bin/env python3
# buck test mobile-vision/mobile_cv/mobile_cv/arch/tests:test_fbnet_fpn

import unittest

import torch
from mobile_cv.arch.fbnet_v2.fbnet_builder import FBNetBuilder
from mobile_cv.arch.fbnet_v2.fbnet_fpn import FBNetFPNBuilder


class TestFBNetFPN(unittest.TestCase):
    def test_single_res_fpn(self):
        """
        Build FPN with a single path (resolution)
        """
        input_channels = [8, 16]
        arch_def = {
            "stages": [
                # [op, c, s, n]
                # stage 0
                [("conv_k3", 32, 1, 2)],
                # stage 1
                [("conv_k3", 32, 1, 2)],
                # stage 2
                [("conv_k3", 64, 1, 2)],
                # stage 3
                [("conv_k3", 32, 1, 2)],
            ]
        }
        builder = FBNetBuilder()
        builder_fpn = FBNetFPNBuilder(builder)
        model = builder_fpn.build_model(input_channels, arch_def)
        input = [
            torch.randn(1, input_channels[0], 32, 32),
            torch.randn(1, input_channels[1], 32, 32),
        ]
        output = model(input)
        assert isinstance(output, list)
        self.assertEqual(output[0].shape[1], arch_def["stages"][-1][0][1])

    def test_multi_res_fpn(self):
        input_channels = [16, 16, 32, 32]
        arch_def = {
            "stages": [
                # [op, c, s, n]
                # stage 0
                [("conv_k3", 16, 1, 1)],
                # stage 1
                [("skip", 16, 1, 1)],
                # stage 2
                [("conv_k3", 32, 1, 1)],
                # stage 3
                [("conv_k3", 32, 1, 1)],
                # stage 4
                [("conv_k3", 32, 2, 1)],
                # stage 5
                [("conv_k3", 32, 1, 1)],
                # stage 6
                [("skip", 32, 1, 1)],
                # stage 7
                [("conv_k3", 64, 1, 1)],
                # stage 8
                [("conv_k3", 64, 1, 1)],
            ]
        }
        builder = FBNetBuilder()
        builder_fpn = FBNetFPNBuilder(builder)
        model = builder_fpn.build_model(input_channels, arch_def)
        input = [
            torch.randn(1, input_channels[0], 32, 32),
            torch.randn(1, input_channels[1], 32, 32),
            torch.randn(1, input_channels[2], 16, 16),
            torch.randn(1, input_channels[3], 16, 16),
        ]
        output = model(input)
        assert isinstance(output, list)
        self.assertEqual(len(output), 2)
        self.assertEqual(output[0].shape[1], arch_def["stages"][3][0][1])
        self.assertEqual(output[1].shape[1], arch_def["stages"][-1][0][1])

    def test_multi_res_single_input_fpn(self):
        input_channels = [16, None, 32, 32]
        arch_def = {
            "stages": [
                # [op, c, s, n]
                # stage 0
                [("conv_k3", 16, 1, 1)],
                # stage 1
                [("noop", 16, 1, 1)],
                # stage 2
                [("conv_k3", 32, 1, 1)],
                # stage 3
                [("conv_k3", 32, 1, 1)],
                # stage 4
                [("conv_k3", 32, 2, 1)],
                # stage 5
                [("conv_k3", 32, 1, 1)],
                # stage 6
                [("conv_k3", 32, 1, 1)],
                # stage 7
                [("conv_k3", 64, 1, 1)],
                # stage 8
                [("conv_k3", 64, 1, 1)],
            ]
        }
        builder = FBNetBuilder()
        builder_fpn = FBNetFPNBuilder(builder)
        model = builder_fpn.build_model(input_channels, arch_def)
        input = [
            torch.randn(1, input_channels[0], 32, 32),
            None,  # represent no input for the skip connection
            torch.randn(1, input_channels[2], 16, 16),
            torch.randn(1, input_channels[3], 16, 16),
        ]
        output = model(input)
        assert isinstance(output, list)
        self.assertEqual(len(output), 2)
        self.assertEqual(output[0].shape[1], arch_def["stages"][3][0][1])
        self.assertEqual(output[1].shape[1], arch_def["stages"][-1][0][1])

    def test_repeat_fpn(self):
        builder = FBNetBuilder()
        builder_fpn = FBNetFPNBuilder(builder)
        # arch def
        arch_def = {
            "high_to_low": {
                "stages": [
                    # [op, c, s, n]
                    # stage 0
                    [("conv_k3", 16, 1, 1)],
                    # stage 1
                    [("noop", 16, 1, 1)],
                    # stage 2
                    [("conv_k3", 32, 1, 1)],
                    # stage 3
                    [("conv_k3", 32, 1, 1)],
                    # stage 4
                    [("conv_k3", 32, 2, 1)],
                    # stage 5
                    [("conv_k3", 32, 1, 1)],
                    # stage 6
                    [("noop", 32, 1, 1)],
                    # stage 7
                    [("conv_k3", 64, 1, 1)],
                    # stage 8
                    [("conv_k3", 64, 1, 1)],
                ]
            },
            "low_to_high": {
                "stages": [
                    # [op, c, s, n]
                    # stage 0
                    [("conv_k3", 64, 1, 1)],
                    # stage 1
                    [("conv_k3", 64, 1, 1)],
                    # stage 2
                    [("conv_k3", 64, 1, 1)],
                    # stage 3
                    [("conv_k3", 32, 1, 1)],
                    # stage 4
                    [("upsample", 64, 2, 1), ("conv_k3", 32, 1, 1)],
                    # stage 5
                    [("conv_k3", 32, 1, 1)],
                    # stage 6
                    [("conv_k3", 32, 1, 1)],
                    # stage 7
                    [("conv_k3", 32, 1, 1)],
                    # stage 8
                    [("conv_k3", 32, 1, 1)],
                ]
            },
        }
        # high to low resolution
        input_channels = [16, None, 32, None]
        downsample_arch_def = arch_def["high_to_low"]
        model = builder_fpn.build_model(input_channels, downsample_arch_def)
        input = [
            torch.randn(1, input_channels[0], 32, 32),
            None,
            torch.randn(1, input_channels[2], 16, 16),
            None,
        ]
        output = model(input)
        assert isinstance(output, list)
        self.assertEqual(len(output), 2)
        self.assertEqual(output[0].shape[1], downsample_arch_def["stages"][3][0][1])
        self.assertEqual(output[1].shape[1], downsample_arch_def["stages"][-1][0][1])

        # low to high resolution
        input, output = input[::-1], output[::-1]
        input = [output[0], input[1], output[1], input[3]]
        input_channels = [t.shape[1] for t in input]
        upsample_arch_def = arch_def["low_to_high"]
        model = builder_fpn.build_model(input_channels, upsample_arch_def)
        output = model(input)

        assert isinstance(output, list)
        self.assertEqual(len(output), 2)
        self.assertEqual(output[0].shape[1], upsample_arch_def["stages"][3][0][1])
        self.assertEqual(output[1].shape[1], upsample_arch_def["stages"][-1][0][1])
