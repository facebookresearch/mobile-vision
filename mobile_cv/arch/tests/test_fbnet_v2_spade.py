#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import mobile_cv.arch.fbnet_v2.basic_blocks as bb
import mobile_cv.arch.fbnet_v2.spade as sp
import mobile_cv.arch.utils.fuse_utils as fuse_utils
import mobile_cv.lut.lib.pt.flops_utils as flops_utils
import torch

from .helper import find_modules, run_and_compare


class TestSpade(unittest.TestCase):
    def test_spade_norm(self):
        sn = sp.SpadeNorm(
            32, bn_args="bn", kernel_size=1, seg_channels=1, seg_mid_channels=64
        ).eval()

        data = torch.ones((2, 32, 8, 8))
        mask = torch.ones((2, 1, 6, 6))

        sn_out = sn((data, mask))
        self.assertEqual(sn_out.shape, data.shape)

        nparams, nflops = flops_utils.get_model_flops(sn, [(data, mask)])

        sn_build = bb.build_bn(
            "spade_norm",
            num_channels=32,
            bn_args="bn",
            kernel_size=1,
            seg_channels=1,
            seg_mid_channels=64,
        ).eval()
        self.assertIsInstance(sn_build, sp.SpadeNorm)

        nparams1, nflops1 = flops_utils.get_model_flops(sn_build, [(data, mask)])
        self.assertEqual(nparams, nparams1)
        self.assertEqual(nflops, nflops1)

    def test_spade_norm_return_seg_map(self):
        sn = sp.SpadeNorm(
            32,
            bn_args="bn",
            kernel_size=1,
            seg_channels=1,
            seg_mid_channels=64,
            seg_return_type="input",
        ).eval()

        data = torch.ones((2, 32, 8, 8))
        mask = torch.ones((2, 1, 6, 6))

        sn_out = sn((data, mask))
        self.assertIsInstance(sn_out, tuple)
        self.assertEqual(sn_out[0].shape, data.shape)
        self.assertEqual(sn_out[1].shape, mask.shape)

        sn1 = sp.SpadeNorm(
            32,
            bn_args="bn",
            kernel_size=1,
            seg_channels=1,
            seg_mid_channels=64,
            seg_return_type="resized",
        ).eval()

        sn_out1 = sn1((data, mask))
        self.assertIsInstance(sn_out1, tuple)
        self.assertEqual(sn_out1[0].shape, data.shape)
        self.assertEqual(sn_out1[1].shape, torch.Size([2, 1, 8, 8]))

    def test_spade_norm_conv_bn_relu_spade(self):
        cbr = bb.ConvBNRelu(
            3,
            32,
            conv_args="conv_tuple_left",
            bn_args={
                "name": "spade_norm",
                "bn_args": "bn",
                "kernel_size": 1,
                "seg_channels": 1,
                "seg_mid_channels": 16,
                "seg_return_type": "input",
            },
            relu_args={"name": "relu_tuple_left", "relu_name": "leakyrelu"},
        ).eval()

        self.assertTrue(find_modules(cbr, sp.SpadeNorm))

        data = torch.ones((2, 3, 8, 8))
        mask = torch.ones((2, 1, 6, 6))

        out = cbr((data, mask))
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, torch.Size([2, 32, 8, 8]))
        self.assertEqual((out[1] - mask).norm(), 0.0)

    def test_spade_norm_conv_bn_relu_spade_upsample(self):
        cbr = bb.ConvBNRelu(
            3,
            32,
            stride=-2,
            conv_args="conv_tuple_left",
            bn_args={
                "name": "spade_norm",
                "bn_args": "bn",
                "kernel_size": 1,
                "seg_channels": 1,
                "seg_mid_channels": 16,
                "seg_return_type": "resized",
            },
            relu_args={"name": "relu_tuple_left", "relu_name": "leakyrelu"},
            upsample_args="upsample_tuple_left",
        ).eval()

        self.assertTrue(find_modules(cbr, sp.SpadeNorm))

        data = torch.ones((2, 3, 4, 4))
        mask = torch.ones((2, 1, 6, 6))

        out = cbr((data, mask))
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, torch.Size([2, 32, 8, 8]))
        self.assertEqual(out[1].shape, torch.Size([2, 1, 4, 4]))

    def test_spade_norm_conv_bn_relu_spade_fusebn(self):
        cbr = bb.ConvBNRelu(
            3,
            32,
            stride=-2,
            conv_args="conv_tuple_left",
            bn_args={
                "name": "spade_norm",
                "bn_args": "bn",
                "kernel_size": 1,
                "seg_channels": 1,
                "seg_mid_channels": 16,
                "seg_return_type": "resized",
            },
            relu_args={"name": "relu_tuple_left", "relu_name": "relu"},
            upsample_args="upsample_tuple_left",
        ).eval()

        self.assertTrue(find_modules(cbr, sp.SpadeNorm))

        data = torch.ones((2, 3, 4, 4))
        mask = torch.ones((2, 1, 6, 6))

        for use_fx in [True, False]:
            _test_fuse_cbr(self, cbr, (data, mask), torch.nn.BatchNorm2d, use_fx=use_fx)

    def test_spade_norm_conv_bn_relu_spade_fuse_syncbn(self):
        cbr = bb.ConvBNRelu(
            3,
            32,
            stride=-2,
            conv_args="conv_tuple_left",
            bn_args={
                "name": "spade_norm",
                "bn_args": "sync_bn",
                "kernel_size": 1,
                "seg_channels": 1,
                "seg_mid_channels": 16,
                "seg_return_type": "resized",
            },
            relu_args={"name": "relu_tuple_left", "relu_name": "relu"},
            upsample_args="upsample_tuple_left",
        ).eval()

        self.assertTrue(find_modules(cbr, sp.SpadeNorm))

        data = torch.ones((2, 3, 4, 4))
        mask = torch.ones((2, 1, 6, 6))

        for use_fx in [True, False]:
            _test_fuse_cbr(
                self, cbr, (data, mask), bb.NaiveSyncBatchNorm, use_fx=use_fx
            )


def _test_fuse_cbr(self, cbr, inputs, bn_class, use_fx):
    cbr.train()
    for _ in range(3):
        cbr(inputs)

    cbr.eval()

    print(f"Model before fuse:\n{cbr}")
    fused = fuse_utils.fuse_model(cbr, inplace=False, use_fx=use_fx)
    print(f"Model after fuse:\n{fused}")

    self.assertTrue(find_modules(cbr, bn_class))
    self.assertFalse(find_modules(fused, bn_class))

    run_and_compare(cbr, fused, inputs)
