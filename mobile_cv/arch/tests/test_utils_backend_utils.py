#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
from unittest import mock

import mobile_cv.arch.utils.backend_utils as bu
import torch


class TestUtilsBackendUtils(unittest.TestCase):
    @mock.patch("torch.Tensor.to")
    def test_move_to_device(self, mock_tensor_to):
        data = {
            "a": torch.zeros(1),
            "b": [torch.zeros(2), torch.zeros(3), (torch.zeros(4), torch.zeros(5))],
        }

        mock_tensor_to.side_effect = lambda x: x

        data_out = bu.move_to_device(data, "cuda")
        self.assertEqual(
            data_out, {"a": "cuda", "b": ["cuda", "cuda", ("cuda", "cuda")]}
        )

    @mock.patch.object(torch.Tensor, "cuda", lambda x: x)
    @mock.patch.object(torch.nn.Module, "cuda", lambda x: x)
    def test_seq_module_list_to_gpu(self):
        class Add(torch.nn.Module):
            def __init__(self, num):
                super().__init__()
                self.num = num

            def forward(self, x):
                return x + self.num

        module_list = torch.nn.ModuleList([Add(2), Add(3)])

        inputs = torch.tensor([1.0])
        y = inputs
        for m in module_list:
            y = m(y)
        self.assertEqual(y, 6.0)

        new_model = bu.seq_module_list_to_gpu(module_list)
        y = inputs
        for m in new_model:
            y = m(y)
        self.assertEqual(y, 6.0)
