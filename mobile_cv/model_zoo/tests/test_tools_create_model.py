#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
import os
import tempfile
import unittest

import mobile_cv.arch.fbnet_v2.blocks_factory as blocks_factory
import torch
from mobile_cv.model_zoo.tools import create_model


def _test_model_zoo_tools_create_model(self, arch_name, builder=None):
    with tempfile.TemporaryDirectory() as temp_dir:
        input_args = [
            "--builder",
            builder or "fbnet_v2",
            "--arch_name",
            arch_name,
            "--arch_args",
            '{"pretrained": false}',
            "--data_shape",
            "[[1, 3, 224, 224]]",
            "--output_dir",
            temp_dir,
            "--int8_backend",
            "fbgemm",
            "--self_contained",
            "1",
        ]
        create_model.main(input_args)

        jit_file = os.path.join(temp_dir, "jit", "model.jit")
        self.assertTrue(
            os.path.isfile(jit_file), f"Converted file not existed {jit_file}"
        )

        int8_jit_file = os.path.join(temp_dir, "int8_jit", "model.jit")
        self.assertTrue(
            os.path.isfile(int8_jit_file), f"Converted file not existed {int8_jit_file}"
        )

        int8_jit_sc_file = os.path.join(temp_dir, "int8_jit_sc", "model.jit")
        self.assertTrue(
            os.path.isfile(int8_jit_sc_file),
            f"Converted file not existed {int8_jit_sc_file}",
        )


class TestModelZooToolsCreateModel(unittest.TestCase):
    def test_model_zoo_tools_create_model(self):
        _test_model_zoo_tools_create_model(self, arch_name="fbnet_c")

    def test_model_zoo_tools_create_model_custom_arch(self):
        arch = {
            "blocks": [
                # [op, c, s, n, ...]
                # stage 0
                [("conv_k3", 32, 2, 1)],
                # stage 1
                [("ir_k3", 16, 1, 1)],
            ]
        }
        _test_model_zoo_tools_create_model(self, json.dumps(arch))

    def test_model_zoo_tools_create_model_cached_output(self):
        # cache output of the module, similar to ClassyBlock
        # the cahced value may contain grad information which is not copyable
        # add torch.no_grad() when running the model to make sure the cached value
        # does not contain grad information, or set the `weights` to not require
        # grads
        class CachedPlus(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weights = torch.nn.Parameter(torch.zeros([1]))
                self.output = torch.zeros(0)

            def forward(self, x):
                output = self.weights * x
                self.output = output
                return output

        blocks_factory.PRIMITIVES.register_dict(
            {
                "cached_plus": lambda in_channels, out_channels, stride, **kwargs: CachedPlus()
            }
        )
        arch = {
            "blocks": [
                # [op, c, s, n, ...]
                # stage 0
                [("cached_plus", 1, 1, 1)]
            ]
        }
        _test_model_zoo_tools_create_model(
            self, json.dumps(arch), builder="fbnet_v2_backbone"
        )


if __name__ == "__main__":
    unittest.main()
