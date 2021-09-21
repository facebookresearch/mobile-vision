#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import mobile_cv.arch.fbnet_v2.fbnet_builder as fbnet_builder
import mobile_cv.arch.fbnet_v2.fbnet_modeldef_cls as fbnet_modeldef_cls
import mobile_cv.arch.utils.fuse_utils as fuse_utils
import mobile_cv.lut.lib.pt.flops_utils as flops_utils
import torch


def _create_and_run(self, arch_name, model_arch):
    arch = fbnet_builder.unify_arch_def(model_arch, ["blocks"])
    builder = fbnet_builder.FBNetBuilder(basic_args=arch.get("basic_args", None))
    model = builder.build_blocks(arch["blocks"], dim_in=3)
    model.eval()
    res = model_arch.get("input_size", 112)
    inputs = (torch.zeros([1, 3, 8, res, res]),)
    output = flops_utils.print_model_flops(model, inputs)
    self.assertEqual(output.shape[0], 1)


def _create_and_trace(self, arch_name, model_arch):
    arch = fbnet_builder.unify_arch_def(model_arch, ["blocks"])
    builder = fbnet_builder.FBNetBuilder(basic_args=arch.get("basic_args", None))
    model = builder.build_blocks(arch["blocks"], dim_in=3)
    model.eval()
    res = model_arch.get("input_size", 112)
    inputs = (torch.zeros([1, 3, 8, res, res]),)

    # it will raise if tracing failed
    model = fuse_utils.fuse_model(model, inplace=True)
    traced = torch.jit.trace(model, inputs)

    print(traced)
    traced_output = traced(*inputs)
    self.assertEqual(traced_output.shape[0], 1)


class TestFBNetV2VideoArchs(unittest.TestCase):
    def test_selected_arches(self):
        """
        buck run @mode/dev-nosan mobile-vision/mobile_cv/mobile_cv/arch/tests:test_fbnet_v2_archs_video \
        -- mobile_cv.arch.test_fbnet_v2_archs_video.TestFBNetV2VideoArchs.test_selected_arches
        """
        arch_factory = fbnet_modeldef_cls.MODEL_ARCH
        selected_archs = [
            # "vt_3d",
            # "vt_3d_2",
            # "eff_2_2dp1",
            # "eff_2_3d",
            "FBNetV3_3D_E4VT",
            "FBNetV3_3D_E5VT",
            # "FBNetV3_3D_F4_VT",
            # "FBNetV3_3D_G4_VT",
        ]

        for name in selected_archs:
            with self.subTest(arch=name):
                print("Testing {}".format(name))
                model_arch = arch_factory.get(name)
                _create_and_run(self, name, model_arch)

    def test_selected_arches_trace(self):
        arch_factory = fbnet_modeldef_cls.MODEL_ARCH
        selected_archs = [
            "vt_3d",
            "eff_2_2dp1",
        ]

        for name in selected_archs:
            with self.subTest(arch=name):
                print("Testing {}".format(name))
                model_arch = arch_factory.get(name)
                _create_and_trace(self, name, model_arch)


if __name__ == "__main__":
    unittest.main()
