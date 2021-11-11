#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import tempfile
import unittest

import mobile_cv.arch.fbnet_v2.fbnet_builder as fbnet_builder
import mobile_cv.arch.utils.fuse_utils as fuse_utils
import torch


def _build_model(arch_def, dim_in):
    arch_def = fbnet_builder.unify_arch_def(arch_def, ["blocks"])
    torch.manual_seed(0)
    builder = fbnet_builder.FBNetBuilder(1.0)
    model = builder.build_blocks(arch_def["blocks"], dim_in=dim_in)
    model.eval()
    return model


class TestFBNetV2Scriptable(unittest.TestCase):
    def test_fbnet_v2_scriptable(self):
        e6 = {"expansion": 6}
        bn_args = {"bn_args": {"name": "bn", "momentum": 0.003}}
        arch_def = {
            "blocks": [
                # [op, c, s, n, ...]
                # stage 0
                [("conv_k3", 4, 2, 1, bn_args)],
                # stage 1
                [
                    ("ir_k3", 8, 2, 2, e6, bn_args),
                    ("ir_k5", 8, 1, 1, e6, bn_args),
                ],
            ]
        }

        model = _build_model(arch_def, dim_in=3)
        model.eval()
        model = fuse_utils.fuse_model(model, inplace=False)

        print(f"Fused model {model}")

        # Make sure model can be traced
        script_model = torch.jit.script(model)

        data = torch.zeros(1, 3, 32, 32)
        model_output = model(data)
        script_output = script_model(data)

        self.assertEqual(model_output.norm(), script_output.norm())

        with tempfile.TemporaryDirectory() as tmp_dir:
            fn = os.path.join(tmp_dir, "model_script.jit")
            torch.jit.save(script_model, fn)
            self.assertTrue(os.path.isfile(fn))

    def test_fbnet_v2_scriptable_empty_batch(self):
        e6 = {"expansion": 6}
        bn_args = {"bn_args": {"name": "bn", "momentum": 0.003}}
        arch_def = {
            "blocks": [
                # [op, c, s, n, ...]
                # stage 0
                [("conv_k3", 4, 2, 1, bn_args)],
                # stage 1
                [
                    ("ir_k3", 8, 2, 2, e6, bn_args),
                    ("ir_k5", 8, 1, 1, e6, bn_args),
                ],
            ]
        }

        model = _build_model(arch_def, dim_in=3)
        model.eval()
        model = fuse_utils.fuse_model(model, inplace=False)

        # Make sure model can be traced
        script_model = torch.jit.script(model)

        # empty batch
        data = torch.zeros(0, 3, 32, 32)
        script_output = script_model(data)

        self.assertEqual(script_output.shape, torch.Size([0, 8, 8, 8]))
