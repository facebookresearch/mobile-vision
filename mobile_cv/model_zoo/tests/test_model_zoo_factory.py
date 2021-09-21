#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import torch
from mobile_cv.model_zoo.models import model_zoo_factory


class TestModelZooFactory(unittest.TestCase):
    def test_model_zoo_factory_register(self):
        self.assertGreater(len(model_zoo_factory.MODEL_ZOO_FACTORY), 0)

    def test_model_zoo_factory_fbnet_v2(self):
        model = model_zoo_factory.get_model(
            "fbnet_v2", arch_name="fbnet_a", num_classes=8
        )
        model.eval()
        with torch.no_grad():
            data = torch.zeros([1, 3, 32, 32])
            out = model(data)
            self.assertEqual(out.size(), torch.Size([1, 8]))

    def test_model_zoo_factory_resnet(self):
        model = model_zoo_factory.get_model("resnet50", num_classes=8)
        model.eval()
        with torch.no_grad():
            data = torch.zeros([1, 3, 32, 32])
            out = model(data)
            self.assertEqual(out.size(), torch.Size([1, 8]))


if __name__ == "__main__":
    unittest.main()
