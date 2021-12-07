#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import mobile_cv.arch.fbnet_v2.fbnet_builder as fbnet_builder
import mobile_cv.arch.fbnet_v2.fbnet_modeldef_cls as fbnet_modeldef_cls
import mobile_cv.lut.lib.pt.flops_utils as flops_utils
import torch


def _create_and_run(self, arch_name, model_arch):
    arch = fbnet_builder.unify_arch_def(model_arch, ["blocks"])
    builder = fbnet_builder.FBNetBuilder(basic_args=arch.get("basic_args", None))
    model = builder.build_blocks(arch["blocks"], dim_in=3)
    model.eval()
    res = model_arch.get("input_size", 224)
    inputs = (torch.zeros([1, 3, res, res]),)
    output = flops_utils.print_model_flops(model, inputs)
    self.assertEqual(output.shape[0], 1)


class TestFBNetV2Archs(unittest.TestCase):
    def test_unify_all_predefinied_archs(self):
        """Initial check for arch definitions"""
        arch_factory = fbnet_modeldef_cls.MODEL_ARCH

        self.assertGreater(len(arch_factory), 0)
        for name, arch in arch_factory.items():
            with self.subTest(arch=name):
                print("Unifiying {}".format(name))
                fbnet_builder.unify_arch_def(arch, ["blocks"])

    def test_selected_arches(self):
        arch_factory = fbnet_modeldef_cls.MODEL_ARCH
        selected_archs = [
            "default",
            "mnv3",
            "mnv3_small",
            "fbnet_a",
            "fbnet_b",
            "fbnet_c",
            "ResNet50",
            "ResNet18",
            "ResNet34",
            "ResNet101",
            "ResNet152",
            "fbnet_vit_tiny_v3",
        ]

        for name in selected_archs:
            with self.subTest(arch=name):
                print("Testing {}".format(name))
                model_arch = arch_factory.get(name)
                _create_and_run(self, name, model_arch)


if __name__ == "__main__":
    unittest.main()
