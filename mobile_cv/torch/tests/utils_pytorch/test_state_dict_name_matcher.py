#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import unittest

import mobile_cv.torch.utils_pytorch.state_dict_name_matcher as sdnm
import torch


class TestUtilsPytorchStateDictNameMatcher(unittest.TestCase):
    def test_matcher(self):
        sd1 = {
            "model.m1.aa.weight": torch.zeros(1, 3, 4, 4),
            "model.m1.ab.bias": torch.zeros(1, 3, 4, 4),
            "model.m2.conv.weight": torch.zeros(1, 3, 4, 4),
            "model.m2.bn.weight": torch.zeros(1, 3, 4, 4),
            "model.m2.relu.weight": torch.zeros(1, 3, 4, 4),
        }
        sd2 = {
            "model.m2_conv.weight": torch.zeros(1, 3, 4, 4),
            "model.m1.aa_post.weight": torch.zeros(1, 3, 4, 4),
            "model.m2_bn.weight": torch.zeros(1, 3, 4, 4),
            "model.m1.ab_post.bias": torch.zeros(1, 3, 4, 4),
            "model.m2_relu.weight": torch.zeros(1, 3, 4, 4),
        }

        ret = sdnm.get_state_dict_name_mapping(sd1, sd2)
        self.assertEqual(
            ret,
            {
                "model.m1.aa.weight": "model.m1.aa_post.weight",
                "model.m1.ab.bias": "model.m1.ab_post.bias",
                "model.m2.conv.weight": "model.m2_conv.weight",
                "model.m2.bn.weight": "model.m2_bn.weight",
                "model.m2.relu.weight": "model.m2_relu.weight",
            },
        )

    def test_matcher_shape(self):
        sd1 = {
            "model.m1.aa.weight": torch.zeros(1, 3, 4, 4),
            "model.m1.bb.weight": torch.zeros(1, 3, 2, 2),
        }
        sd2 = {
            "model.m1.ccb.weight": torch.zeros(1, 3, 4, 4),
            "model.m1.dda.weight": torch.zeros(1, 3, 2, 2),
        }

        ret = sdnm.get_state_dict_name_mapping(sd1, sd2)
        self.assertEqual(
            ret,
            {
                "model.m1.aa.weight": "model.m1.ccb.weight",
                "model.m1.bb.weight": "model.m1.dda.weight",
            },
        )
