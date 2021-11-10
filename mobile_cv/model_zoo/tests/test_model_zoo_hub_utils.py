#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import unittest

import mobile_cv.model_zoo.models.hub_utils as hu
import torch
from mobile_cv.model_zoo.models.hub_utils import pretrained_download
from mobile_cv.model_zoo.models.utils import is_devserver


ON_DEVSERVER = is_devserver()


class TestModelZooHubUtils(unittest.TestCase):
    def test_model_zoo_dict_modifier(self):
        dict_to_test = {
            "ab": 1234,
            "cd": 2345,
            "ef": None,
        }
        new_dict = {
            "ab": 4321,
            "gh": 2342,
            "NoneVal": None,
        }

        gt_dict = copy.deepcopy(dict_to_test)
        gt_modifed = {
            "ab": 4321,
            "cd": 2345,
            "ef": None,
            "gh": 2342,
            "NoneVal": None,
        }
        with hu.DictModifier(dict_to_test, new_dict):
            self.assertEqual(dict_to_test, gt_modifed)

        self.assertEqual(dict_to_test, gt_dict)

    @unittest.skipIf(not ON_DEVSERVER, "Test only on devserver")
    def test_pretrain_downloader(self):
        data = pretrained_download(torch.hub.load_state_dict_from_url)(
            "https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth"
        )
        self.assertIsInstance(data, dict)


if __name__ == "__main__":
    unittest.main()
