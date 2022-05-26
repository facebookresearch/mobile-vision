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
