#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import mobile_cv.arch.utils.model_utils as mu
import torch


class TestUtilsModelUtils(unittest.TestCase):
    def test_copy_attributes(self):
        model = torch.nn.Sequential(torch.nn.Identity(), torch.nn.ReLU())
        model[0].aa = 1
        model[0].bb = 2
        model[1].cc = 3

        model1 = torch.nn.Sequential(
            torch.nn.Identity(), torch.nn.ReLU(), torch.nn.Identity()
        )
        mu.copy_model_attributes(model, model1, ["aa", "bb", "cc"])

        self.assertEqual(model1[0].aa, 1)
        self.assertEqual(model1[0].bb, 2)
        self.assertEqual(model1[1].cc, 3)
