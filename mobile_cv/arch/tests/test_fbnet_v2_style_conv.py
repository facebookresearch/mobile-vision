# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import mobile_cv.arch.fbnet_v2.basic_blocks as bb
import mobile_cv.arch.fbnet_v2.style_conv as sc
import mobile_cv.lut.lib.pt.flops_utils as flops_utils
import torch
import torch.nn as nn


def _has_module(model, module_type):
    for x in model.modules():
        if isinstance(x, module_type):
            return True
    return False


class TestStyleConv(unittest.TestCase):
    def test_style_conv(self):
        conv_args = {"kernel_size": 3, "stride": 1, "padding": 1}
        conv = sc.ModulatedConv2d_V1(
            in_channels=8,
            out_channels=16,
            style_dim=24,
            demodulate=True,
            **conv_args,
        )

        data = torch.ones((2, 8, 4, 4))
        style = torch.ones((2, 24))
        out = conv((data, style))

        self.assertEqual(out[0].shape, (2, 16, 4, 4))
        self.assertEqual(out[1].sum(), style.sum())

        nparams, nflops = flops_utils.get_model_flops(conv, [(data, style)])
        self.assertEqual(nparams, 0.001344)
        self.assertEqual(nflops, 0.037248)

        conv_build = bb.build_conv(
            "style_conv_v1",
            in_channels=8,
            out_channels=16,
            style_dim=24,
            demodulate=True,
            **conv_args,
        ).eval()
        self.assertIsInstance(conv_build, sc.ModulatedConv2d_V1)

        nparams1, nflops1 = flops_utils.get_model_flops(conv_build, [(data, style)])
        self.assertEqual(nparams, nparams1)
        self.assertEqual(nflops, nflops1)

    def test_style_conv_relu(self):
        scr = bb.ConvBNRelu(
            in_channels=8,
            out_channels=16,
            style_dim=24,
            demodulate=True,
            conv_args={
                "name": "style_conv_v1",
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            bn_args=None,
            relu_args={"name": "relu_tuple_left", "relu_name": "leakyrelu"},
        ).eval()

        self.assertTrue(_has_module(scr, sc.ModulatedConv2d_V1))
        self.assertTrue(_has_module(scr, nn.LeakyReLU))

        data = torch.ones((2, 8, 4, 4))
        style = torch.ones((2, 24))

        out = scr((data, style))
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, (2, 16, 4, 4))
        self.assertEqual(out[1].sum(), style.sum())

    def test_style_conv_relu_upsample(self):
        scr = bb.ConvBNRelu(
            in_channels=8,
            out_channels=16,
            style_dim=24,
            demodulate=True,
            conv_args={
                "name": "style_conv_v1",
                "kernel_size": 3,
                "stride": -2,
                "padding": 1,
            },
            bn_args=None,
            relu_args={"name": "relu_tuple_left", "relu_name": "leakyrelu"},
            upsample_args="upsample_tuple_left",
        ).eval()

        self.assertTrue(_has_module(scr, sc.ModulatedConv2d_V1))
        self.assertTrue(_has_module(scr, nn.LeakyReLU))

        data = torch.ones((2, 8, 4, 4))
        style = torch.ones((2, 24))

        out = scr((data, style))
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, torch.Size([2, 16, 8, 8]))
        self.assertEqual(out[1].sum(), style.sum())

    def test_style_conv_relu_downsample(self):
        scr = bb.ConvBNRelu(
            in_channels=8,
            out_channels=16,
            style_dim=24,
            demodulate=True,
            conv_args={
                "name": "style_conv_v1",
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
            },
            bn_args=None,
            relu_args={"name": "relu_tuple_left", "relu_name": "leakyrelu"},
            upsample_args="upsample_tuple_left",
        ).eval()

        self.assertTrue(_has_module(scr, sc.ModulatedConv2d_V1))
        self.assertTrue(_has_module(scr, nn.LeakyReLU))

        data = torch.ones((2, 8, 4, 4))
        style = torch.ones((2, 24))

        out = scr((data, style))
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, torch.Size([2, 16, 2, 2]))
        self.assertEqual(out[1].sum(), style.sum())
