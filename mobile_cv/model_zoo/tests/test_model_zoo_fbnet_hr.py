#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import mobile_cv.lut.lib.pt.flops_utils as flops_utils
import torch
from mobile_cv.model_zoo.models.fbnet_hr import fbnet_hr


class TestModelZooFBNetHR(unittest.TestCase):
    def test_fbnet_hr(self):
        for name in [
            "TestModel",
        ]:
            print(f"Testing {name}...")
            model = fbnet_hr(name)
            data = torch.zeros([1, 3, 32, 32])
            out = model(data)
            self.assertEqual(out.size(), torch.Size([1, 16, 32, 32]))

    def test_fbnet_hr_flops(self):
        """
        buck run @mode/dev-nosan //mobile-vision/projects/model_zoo/tests:test_model_zoo_fbnet_hr \
        -- test_model_zoo_fbnet_hr.TestModelZooFBNetHR.test_fbnet_hr_flops
        """
        for x in [
            "TestModel",
        ]:
            print(f"model name: {x}")
            model = fbnet_hr(x)
            inputs = (torch.zeros([1, 3, 32, 32]),)
            flops_utils.print_model_flops(model, inputs)


if __name__ == "__main__":
    unittest.main()
