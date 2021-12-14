# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import mobile_cv.arch.fbnet_v2.basic_blocks as bb  # noqa
import mobile_cv.arch.fbnet_v2.res_block as rb
import torch
import torch.nn as nn


# import mobile_cv.lut.lib.pt.flops_utils as flops_utils

# from mobile_cv.arch.fbnet_v2.fbnet_builder import FBNetBuilder
# from mobile_cv.arch.fbnet_v2.fbnet_hr import FBNetHRBuilder


def _has_module(model, module_type):
    for x in model.modules():
        if isinstance(x, module_type):
            return True
    return False


class TestResBlock(unittest.TestCase):
    def test_res_block_equal_in_out_channels(self):
        conv_args = {"kernel_size": 3, "stride": 1, "padding": 1}
        conv = rb.BasicBlock(
            in_channels=3,
            out_channels=3,
            conv_args=conv_args,
        )

        data = torch.ones((2, 3, 4, 4))
        out = conv(data)

        self.assertEqual(out.shape, (2, 3, 4, 4))

    def test_res_block_different_in_out_channels(self):
        conv_args = {"kernel_size": 3, "stride": 1, "padding": 1}
        conv = rb.BasicBlock(
            in_channels=3,
            out_channels=8,
            conv_args=conv_args,
        )

        data = torch.ones((2, 3, 4, 4))
        out = conv(data)
        self.assertEqual(out.shape, (2, 8, 4, 4))

    def test_res_block_downsample(self):
        conv_args = {"kernel_size": 3, "stride": 2, "padding": 1}
        conv = rb.BasicBlock(
            in_channels=3,
            out_channels=8,
            conv_args=conv_args,
        )

        data = torch.ones((2, 3, 4, 4))
        out = conv(data)
        self.assertEqual(out.shape, (2, 8, 2, 2))

    def test_res_block_upsample(self):
        conv_args = {"kernel_size": 3, "stride": -2, "padding": 1}
        conv = rb.BasicBlock(
            in_channels=8,
            out_channels=8,
            conv_args=conv_args,
        )

        data = torch.ones((2, 8, 4, 4))
        out = conv(data)
        self.assertEqual(out.shape, (2, 8, 8, 8))

    def test_res_block_quantize_partial(self):
        import mobile_cv.arch.utils.quantize_utils as qu
        from torch.ao.quantization import get_default_qconfig
        from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

        qconfig = get_default_qconfig("qnnpack")

        model = nn.Sequential(
            rb.BasicBlock(
                8, 8, downsample_in_conv2=False, bn_in_skip=True, qmethod="fp32_skip"
            ),
            rb.BasicBlock(
                8,
                8,
                downsample_in_conv2=False,
                bn_in_skip=True,
                qmethod="fp32_skip_relu",
            ),
        )
        model.eval()
        data = torch.zeros(1, 8, 4, 4)

        qconfig_dict = qu.get_qconfig_dict(model, qconfig)
        model = prepare_fx(model, qconfig_dict)
        model = convert_fx(model)
        print(model)

        dequants = [x.name for x in model.graph.nodes if x.op == "call_method"]
        self.assertEqual(dequants, ["dequantize_1", "dequantize_3", "dequantize_4"])

        funcs = [x.name for x in model.graph.nodes if x.op == "call_function"]
        self.assertEqual(
            funcs, ["quantize_per_tensor", "add", "quantize_per_tensor_2", "add_1"]
        )

        out = model(data)
        self.assertEqual(out.shape, torch.Size([1, 8, 4, 4]))


class TestResBottleNeckBlock(unittest.TestCase):
    def test_res_block_equal_in_out_channels(self):
        conv_args = {"kernel_size": 3, "stride": 1, "padding": 1}
        conv = rb.Bottleneck(
            in_channels=3,
            out_channels=8,
            conv_args=conv_args,
        )

        data = torch.ones((2, 3, 4, 4))
        out = conv(data)

        self.assertEqual(out.shape, (2, 8, 4, 4))

    def test_res_block_different_in_out_channels(self):
        conv_args = {"kernel_size": 3, "stride": 1, "padding": 1}
        conv = rb.Bottleneck(
            in_channels=3,
            out_channels=8,
            conv_args=conv_args,
        )

        data = torch.ones((2, 3, 4, 4))
        out = conv(data)
        self.assertEqual(out.shape, (2, 8, 4, 4))

    def test_res_block_downsample(self):
        conv_args = {"kernel_size": 3, "stride": 2, "padding": 1}
        conv = rb.Bottleneck(
            in_channels=3,
            out_channels=8,
            conv_args=conv_args,
        )

        data = torch.ones((2, 3, 4, 4))
        out = conv(data)
        self.assertEqual(out.shape, (2, 8, 2, 2))

    def test_res_block_upsample(self):
        conv_args = {"kernel_size": 3, "stride": -2, "padding": 1}
        conv = rb.Bottleneck(
            in_channels=8,
            out_channels=8,
            conv_args=conv_args,
        )

        data = torch.ones((2, 8, 4, 4))
        out = conv(data)
        self.assertEqual(out.shape, (2, 8, 8, 8))
