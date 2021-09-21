#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import mobile_cv.lut.lib.pt.flops_utils as flops_utils
import torch
from mobile_cv.model_zoo.models.vits import vit
from utils import is_devserver


ON_DEVSERVER = is_devserver()


class TestModelZooFBNetV2(unittest.TestCase):
    def test_fbnet_v2(self):
        for name in [
            "LeViT_128",
            "LeViT_256",
            "LeViT_384",
            "DeiT-Tiny",
            "DeiT-Tiny_noDistill",
            "DeiT-Small",
            "DeiT-Small_noDistill",
            "DeiT-Base",
            "DeiT-Base_noDistill",
        ]:
            with self.subTest(arch=name):
                for isTest in [False, True]:
                    for load_pretrained in [False, ON_DEVSERVER]:
                        print(f"Testing {name}...")
                        model = vit(name, pretrained=load_pretrained)
                        # print(model)
                        res = model.arch_def.get("input_size", 224)
                        print("Is test?", isTest)
                        print(f"Test res: {res}")
                        print("load_pretrained", load_pretrained)
                        num_params = sum(
                            (parameter.nelement() for parameter in model.parameters())
                        )
                        print("Number of parameters", num_params)
                        if isTest:
                            bs = 1
                            model.eval()
                        else:
                            bs = 2  # have to be larger than 1 bc of BatchNorm
                            model.train()
                        data = torch.zeros([bs, 3, res, res])
                        out = model(data)
                        if isinstance(out, tuple):  # distillation
                            self.assertEqual(out[0].size(), torch.Size([bs, 1000]))
                            self.assertEqual(out[1].size(), torch.Size([bs, 1000]))
                        else:
                            self.assertEqual(out.size(), torch.Size([bs, 1000]))

    def test_fbnet_flops(self):
        """
        buck run @mode/dev-nosan //mobile-vision/projects/model_zoo/tests:test_model_zoo_fbnet_v2 -- test_model_zoo_fbnet_v2.TestModelZooFBNetV2.test_fbnet_flops
        """
        for name in [
            "LeViT_128",
            "LeViT_256",
            "LeViT_384",
            "DeiT-Tiny",
            "DeiT-Tiny_noDistill",
            "DeiT-Small",
            "DeiT-Small_noDistill",
            "DeiT-Base",
            "DeiT-Base_noDistill",
        ]:
            print(f"model name: {name}")
            model = vit(name, pretrained=False)
            model.eval()
            res = model.arch_def.get("input_size", 224)
            inputs = (torch.zeros([1, 3, res, res]),)
            flops_utils.print_model_flops(model, inputs)


if __name__ == "__main__":
    unittest.main()
