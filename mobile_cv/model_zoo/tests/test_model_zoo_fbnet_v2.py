#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import mobile_cv.lut.lib.pt.flops_utils as flops_utils
import torch
from mobile_cv.model_zoo.models.fbnet_v2 import fbnet, fbnet_backbone
from parameterized import parameterized


class TestModelZooFBNetV2(unittest.TestCase):
    @parameterized.expand([[True], [False]])
    def test_fbnet_v2(self, load_pretrained):
        for name in [
            "fbnet_a",
            "fbnet_b",
            "fbnet_c",
            "FBNetV2_F1",
            "FBNetV2_F2",
            "FBNetV2_F3",
            "FBNetV2_F4",
            "FBNetV2_L1",
            "FBNetV2_L2",
            "FBNetV3_A",
            "FBNetV3_B",
            "FBNetV3_C",
            "FBNetV3_D",
            "FBNetV3_E",
            "FBNetV3_F",
            "FBNetV3_G",
        ]:
            with self.subTest(arch=name):
                print(f"Testing {name}...")
                model = fbnet(name, pretrained=load_pretrained)
                res = model.arch_def.get("input_size", 224)
                print(f"Test res: {res}")
                data = torch.zeros([1, 3, res, res])
                out = model(data)
                self.assertEqual(out.size(), torch.Size([1, 1000]))

    @parameterized.expand([[True], [False]])
    def test_fbnet_v2_backbone(self, load_pretrained):
        for name in [
            "fbnet_c",
            # "FBNetV2_F2",
        ]:
            print(f"Testing {name} backbone...")
            model = fbnet_backbone(name, pretrained=load_pretrained)
            res = model.arch_def.get("input_size", 224)
            out_chan = model.arch_def["blocks"][-1]["block_cfg"]["out_channels"]
            print(f"Test res: {res}")
            data = torch.zeros([1, 3, res, res])
            out = model(data)
            self.assertEqual(out.size()[:2], torch.Size([1, out_chan]))

    @parameterized.expand([[True], [False]])
    def test_fbnet_v2_backbone_stage_indices(self, load_pretrained):
        name = "fbnet_c"
        print(f"Testing {name} backbone selected stage indices...")
        stage_indices = [0, 1, 2]
        model = fbnet_backbone(
            name, pretrained=load_pretrained, stage_indices=stage_indices
        )
        res = model.arch_def.get("input_size", 224)
        out_chan = 24
        print(f"Test res: {res}")
        data = torch.zeros([1, 3, res, res])
        out = model(data)
        self.assertEqual(out.size(), torch.Size([1, out_chan, 224 // 4, 224 // 4]))

    def test_fbnet_arch_def(self):
        model_arch = {
            "blocks": [
                # [c, s, n]
                # stage 0
                [["conv_k3_hs", 16, 2, 1]],
                # stage 1
                [["ir_k3", 16, 2, 1]],
                # stage 2
                [["ir_k3", 24, 2, 1]],
                # stage 3
                [["ir_pool_hs", 24, 1, 1]],
            ]
        }

        model = fbnet(model_arch, pretrained=False, num_classes=8)
        data = torch.zeros([1, 3, 32, 32])
        out = model(data)
        self.assertEqual(out.size(), torch.Size([1, 8]))

    def test_fbnet_flops(self):
        """
        buck run @mode/dev-nosan //mobile-vision/projects/model_zoo/tests:test_model_zoo_fbnet_v2 \
        -- test_model_zoo_fbnet_v2.TestModelZooFBNetV2.test_fbnet_flops
        """
        for x in [
            "fbnet_a",
            "fbnet_b",
            "fbnet_c",
            "FBNetV2_F1",
            "FBNetV2_F2",
            "FBNetV2_F3",
            "FBNetV2_F4",
            "FBNetV2_F5",
        ]:
            print(f"model name: {x}")
            model = fbnet(x, pretrained=False)
            res = model.arch_def.get("input_size", 224)
            inputs = (torch.zeros([1, 3, res, res]),)
            flops_utils.print_model_flops(model, inputs)


if __name__ == "__main__":
    unittest.main()
