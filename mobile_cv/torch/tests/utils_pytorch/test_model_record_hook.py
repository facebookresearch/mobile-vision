#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import copy
import unittest

import torch
from mobile_cv.torch.utils_pytorch.model_record_hook import (
    add_model_record_hook,
    compare_hook_items,
    print_hook_items_difference,
)


class Model(torch.nn.Module):
    def __init__(self, count=5):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 8, kernel_size=1, stride=1)
        self.conv1 = torch.nn.Conv2d(8, 6, kernel_size=1, stride=2)
        self.count = count

    def forward(self, x, y):
        for _ in range(self.count):
            x = self.conv(x)
        return self.conv1(x) + y


class TestUtilsPytorchMOdelRecordHook(unittest.TestCase):
    def test_model_record_hook(self):
        """
        buck2 run @mode/dev-nosan //mobile-vision/mobile_cv/mobile_cv/torch/tests:utils_pytorch_test_model_record_hook -- -r test_model_record_hook$
        """

        input_x = torch.ones(2, 8, 4, 4) * 2
        input_y = torch.ones(2, 6, 2, 2)

        model1 = Model(1).eval()
        model2 = Model(1).eval()

        hook1 = add_model_record_hook(model1)
        hook2 = add_model_record_hook(model2)

        out1 = model1(input_x, input_y)
        out2 = model2(input_x, input_y)

        self.assertNotEqual((out1 - out2).norm(), 0.0)

        diff_items = list(compare_hook_items(hook1, hook2))
        self.assertEqual(len(diff_items), 3)
        print_hook_items_difference(hook1, hook2)

    def test_model_record_hook_same_model(self):
        """
        buck2 run @mode/dev-nosan //mobile-vision/mobile_cv/mobile_cv/torch/tests:utils_pytorch_test_model_record_hook -- -r test_model_record_hook_same_model$
        """

        input_x = torch.ones(2, 8, 4, 4) * 2
        input_y = torch.ones(2, 6, 2, 2)

        model1 = Model(1).eval()
        model2 = copy.deepcopy(model1).eval()

        hook1 = add_model_record_hook(model1)
        hook2 = add_model_record_hook(model2)

        out1 = model1(input_x, input_y)
        out2 = model2(input_x, input_y)

        self.assertEqual((out1 - out2).norm(), 0.0)

        diff_items = list(compare_hook_items(hook1, hook2))
        self.assertEqual(len(diff_items), 0)
        print_hook_items_difference(hook1, hook2)

    def test_model_record_hook_repeat(self):
        """
        buck2 run @mode/dev-nosan //mobile-vision/mobile_cv/mobile_cv/torch/tests:utils_pytorch_test_model_record_hook -- -r test_model_record_hook_repeat$
        """

        input_x = torch.ones(2, 8, 4, 4) * 2
        input_y = torch.ones(2, 6, 2, 2)

        model1 = Model(5).eval()
        model2 = Model(5).eval()

        hook1 = add_model_record_hook(model1)
        hook2 = add_model_record_hook(model2)

        out1 = model1(input_x, input_y)
        out2 = model2(input_x, input_y)

        self.assertNotEqual((out1 - out2).norm(), 0.0)

        diff_items = list(compare_hook_items(hook1, hook2))
        self.assertEqual(len(diff_items), 7)
        print_hook_items_difference(hook1, hook2)
