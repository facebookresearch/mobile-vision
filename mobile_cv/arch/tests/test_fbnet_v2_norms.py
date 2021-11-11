#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import mobile_cv.arch.fbnet_v2.basic_blocks as bb
import mobile_cv.arch.fbnet_v2.norms as norms
import torch


def _has_module(model, module_type):
    for x in model.modules():
        if isinstance(x, module_type):
            return True
    return False


class TestNorms(unittest.TestCase):
    def test_instance_layer_norm(self):
        iln = norms.ILN(32).eval()

        data = torch.ones((2, 32, 8, 8))

        out = iln(data)
        self.assertEqual(out.shape, data.shape)
        self.assertEqual(out.sum(), 0)

        iln_build = bb.build_bn("instance_layer", num_channels=32).eval()
        self.assertIsInstance(iln_build, norms.ILN)

        out2 = iln_build(data)
        self.assertEqual(out2.shape, data.shape)
        self.assertEqual(out2.sum(), 0)

    def test_ada_instance_layer_norm(self):
        ada_iln = norms.adaILN(32, 24).eval()

        data = torch.ones((2, 32, 8, 8))
        style = torch.ones((2, 24))

        out = ada_iln((data, style))
        self.assertEqual(out[0].shape, data.shape)
        self.assertEqual(out[1].sum(), style.sum())

        ada_iln_build = bb.build_bn(
            "ada_instance_layer", num_channels=32, style_dim=24
        ).eval()
        self.assertIsInstance(ada_iln_build, norms.adaILN)

        out2 = ada_iln_build((data, style))
        self.assertEqual(out2[0].shape, data.shape)
        self.assertEqual(out2[1].sum(), style.sum())

    def test_norm_conv_bn_relu_iln(self):
        cbr = bb.ConvBNRelu(
            3,
            32,
            conv_args="conv",
            bn_args={
                "name": "instance_layer",
            },
        ).eval()
        self.assertTrue(_has_module(cbr, norms.ILN))

        data = torch.ones((2, 3, 8, 8))
        out = cbr(data)
        self.assertEqual(out.shape, torch.Size([2, 32, 8, 8]))

    def test_norm_conv_bn_relu_ada_iln(self):
        cbr = bb.ConvBNRelu(
            3,
            32,
            conv_args="conv_tuple_left",
            bn_args={
                "name": "ada_instance_layer",
                "style_dim": 24,
            },
            relu_args={"name": "relu_tuple_left", "relu_name": "leakyrelu"},
        ).eval()
        self.assertTrue(_has_module(cbr, norms.adaILN))

        data = torch.ones((2, 3, 8, 8))
        style = torch.ones((2, 24))
        out = cbr((data, style))
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, torch.Size([2, 32, 8, 8]))
        self.assertEqual(out[1].sum(), style.sum())
