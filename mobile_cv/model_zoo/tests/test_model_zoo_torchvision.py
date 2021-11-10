#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import mobile_cv.lut.lib.pt.flops_utils as flops_utils
import torch
from mobile_cv.model_zoo.models import model_zoo_factory
from parameterized import parameterized


class TestModelZooTorchVision(unittest.TestCase):
    @parameterized.expand([[True], [False]])
    def test_model_zoo_factory_resnet(self, load_pretrained):
        model = model_zoo_factory.get_model(
            "resnet50", num_classes=1000, pretrained=load_pretrained
        )
        model.eval()
        with torch.no_grad():
            data = torch.zeros([1, 3, 32, 32])
            out = flops_utils.print_model_flops(model, [data])
            self.assertEqual(out.size(), torch.Size([1, 1000]))


if __name__ == "__main__":
    unittest.main()
