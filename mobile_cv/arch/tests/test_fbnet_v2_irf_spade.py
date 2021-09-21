#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import mobile_cv.arch.fbnet_v2.fbnet_builder as fbnet_builder
import mobile_cv.arch.fbnet_v2.irf_spade as isp
import mobile_cv.arch.fbnet_v2.spade as sp
import mobile_cv.arch.utils.fuse_utils as fuse_utils
import torch

from .helper import find_modules, run_and_compare


def _build_model(arch_def, dim_in):
    arch_def = fbnet_builder.unify_arch_def(arch_def, ["blocks"])
    torch.manual_seed(0)
    builder = fbnet_builder.FBNetBuilder(1.0)
    model = builder.build_blocks(arch_def["blocks"], dim_in=dim_in)
    model.eval()
    return model


class TestIRFSpade(unittest.TestCase):
    def test_spade_norm_irf_spade(self):
        irf_spade = isp.irf_spade(
            3,
            32,
            spade_args={
                "seg_channels": 1,
                "seg_mid_channels": 16,
                "seg_return_type": "input",
            },
        ).eval()

        self.assertTrue(find_modules(irf_spade.dw, sp.SpadeNorm))

        data = torch.ones((2, 3, 8, 8))
        mask = torch.ones((2, 1, 6, 6))

        out = irf_spade((data, mask))
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, torch.Size([2, 32, 8, 8]))
        self.assertEqual((out[1] - mask).norm(), 0.0)

    def test_spade_norm_irf_spade_pwl(self):
        irf_spade = isp.irf_spade_pwl(
            3,
            32,
            spade_args={
                "seg_channels": 1,
                "seg_mid_channels": 16,
                "seg_return_type": "resized",
            },
        ).eval()

        self.assertTrue(find_modules(irf_spade.pwl, sp.SpadeNorm))

        data = torch.ones((2, 3, 8, 8))
        mask = torch.ones((2, 1, 6, 6))

        out = irf_spade((data, mask))
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, torch.Size([2, 32, 8, 8]))
        self.assertEqual(out[1].shape, torch.Size([2, 1, 8, 8]))

    def test_spade_norm_irf_spade_arch(self):
        arch_def = {
            "blocks": [
                [
                    ("conv_k3_tuple_left", 32, 1, 1),
                    ("irf_spade", 32, 1, 1, {"spade_args": {"seg_mid_channels": 32}}),
                    (
                        "irf_spade_pwl",
                        32,
                        1,
                        1,
                        {"spade_args": {"seg_mid_channels": 32}},
                    ),
                    (
                        "irf_spade_pwl",
                        16,
                        2,
                        1,
                        {"spade_args": {"seg_mid_channels": 32}},
                    ),
                    (
                        "irf_spade",
                        16,
                        -2,
                        1,
                        {"spade_args": {"seg_mid_channels": 32}},
                    ),
                ]
            ]
        }

        model = _build_model(arch_def, dim_in=3).eval()
        data = torch.ones((2, 3, 8, 8))
        mask = torch.ones((2, 1, 6, 6))

        out = model((data, mask))
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, torch.Size([2, 16, 8, 8]))
        self.assertEqual(out[1].shape, torch.Size([2, 1, 6, 6]))

    def test_spade_norm_irf_spade_arch_fuse(self):
        arch_def = {
            "blocks": [
                [
                    ("conv_k3_tuple_left", 32, 1, 1),
                    ("irf_spade", 32, 1, 1, {"spade_args": {"seg_mid_channels": 32}}),
                    (
                        "irf_spade_pwl",
                        32,
                        1,
                        1,
                        {"spade_args": {"seg_mid_channels": 32}},
                    ),
                    (
                        "irf_spade_pwl",
                        16,
                        2,
                        1,
                        {"spade_args": {"seg_mid_channels": 32}},
                    ),
                    (
                        "irf_spade",
                        16,
                        -2,
                        1,
                        {"spade_args": {"seg_mid_channels": 32}},
                    ),
                ]
            ]
        }

        model = _build_model(arch_def, dim_in=3).eval()
        data = torch.ones((2, 3, 8, 8))
        mask = torch.ones((2, 1, 6, 6))

        self.assertTrue(find_modules(model, sp.SpadeNorm))
        self.assertTrue(find_modules(model, torch.nn.BatchNorm2d))

        _test_fuse_cbr(self, model, (data, mask), torch.nn.BatchNorm2d, use_fx=True)
        _test_fuse_cbr(self, model, (data, mask), torch.nn.BatchNorm2d, use_fx=False)

    def test_spade_norm_irf_spade_arch_fuse_syncbn(self):
        SYNC_BN = {"bn_args": "sync_bn"}
        arch_def = {
            "blocks": [
                [
                    ("conv_k3_tuple_left", 32, 1, 1, SYNC_BN),
                    (
                        "irf_spade",
                        32,
                        1,
                        1,
                        {"spade_args": {"seg_mid_channels": 32, **SYNC_BN}},
                        SYNC_BN,
                    ),
                    (
                        "irf_spade_pwl",
                        32,
                        1,
                        1,
                        {"spade_args": {"seg_mid_channels": 32, **SYNC_BN}},
                        SYNC_BN,
                    ),
                    (
                        "irf_spade_pwl",
                        16,
                        2,
                        1,
                        {"spade_args": {"seg_mid_channels": 32, **SYNC_BN}},
                        SYNC_BN,
                    ),
                    (
                        "irf_spade",
                        16,
                        -2,
                        1,
                        {"spade_args": {"seg_mid_channels": 32, **SYNC_BN}},
                        SYNC_BN,
                    ),
                ]
            ]
        }

        model = _build_model(arch_def, dim_in=3).eval()
        data = torch.ones((2, 3, 8, 8))
        mask = torch.ones((2, 1, 6, 6))

        self.assertTrue(find_modules(model, sp.SpadeNorm))
        self.assertFalse(find_modules(model, torch.nn.BatchNorm2d, exact_match=True))
        self.assertTrue(find_modules(model, sp.bb.NaiveSyncBatchNorm))

        _test_fuse_cbr(
            self,
            model,
            (data, mask),
            (sp.bb.NaiveSyncBatchNorm, torch.nn.BatchNorm2d),
            use_fx=True,
        )
        _test_fuse_cbr(
            self,
            model,
            (data, mask),
            (sp.bb.NaiveSyncBatchNorm, torch.nn.BatchNorm2d),
            use_fx=False,
        )


def _test_fuse_cbr(self, model, inputs, bn_class, use_fx):
    model.train()
    for _ in range(3):
        model(inputs)

    model.eval()

    print(f"Model before fuse:\n{model}")
    fused = fuse_utils.fuse_model(model, inplace=False, use_fx=use_fx)
    print(f"Model after fuse:\n{fused}")

    self.assertTrue(find_modules(model, bn_class))
    self.assertFalse(find_modules(fused, bn_class))

    run_and_compare(model, fused, inputs)
