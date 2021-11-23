#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import mobile_cv.lut.lib.pt.flops_utils as flops_utils
import torch
from mobile_cv.model_zoo.models import model_zoo_factory
from mobile_cv.model_zoo.models.fbnet_v2 import fbnet


def _set_weights(model, value):
    for param in model.parameters():
        torch.nn.init.constant_(param, value)


class TestModelZooFBNetV2ResNet(unittest.TestCase):
    def test_resnet_flops(self):
        """
        buck run @mode/dev-nosan //mobile-vision/mobile_cv/mobile_cv/model_zoo/tests:test_model_zoo_fbnet_v2_resnet -- -r test_resnet_flops
        """
        for arch, arch_tv in [
            ("ResNet18", "resnet18"),
            ("ResNet50", "resnet50"),
        ]:
            print(f"model name: {arch}")
            model = fbnet(arch, pretrained=False).eval()
            _set_weights(model, 0.01)
            res = model.arch_def.get("input_size", 224)
            inputs = (torch.ones([1, 3, res, res]),)
            out = flops_utils.print_model_flops(model, inputs)
            self.assertEqual(out.size(), torch.Size([1, 1000]))

            model_tv = model_zoo_factory.get_model(
                arch_tv, num_classes=1000, pretrained=False
            ).eval()
            _set_weights(model_tv, 0.01)
            out_tv = flops_utils.print_model_flops(model_tv, inputs)
            self.assertEqual(out_tv.size(), torch.Size([1, 1000]))

            self.assertEqual((out_tv - out).norm(), 0)


if __name__ == "__main__":
    unittest.main()
