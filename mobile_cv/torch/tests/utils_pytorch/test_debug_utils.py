#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import tempfile
import unittest

import torch

from mobile_cv.torch.utils_pytorch import debug_utils as du


class TestUtilsPytorchDebugUtils(unittest.TestCase):
    def test_debug_utils_save_load(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            du._DEFAULT_PATH = tmp_dir
            uid = "test_du"

            du.save_data(uid, "d1", torch.ones(1, 3, 4, 4))
            du.save_data(uid, "d2", {"a": 1, "b": 2})

            self.assertEqual(du.list_all_names(uid), ["d1", "d2"])

            loaded_data = du.load_all_data(uid)
            matched, mismatched = du.compare_data(
                loaded_data, {"d1": torch.ones(1, 3, 4, 4), "d2": {"a": 1, "b": 2}}
            )
            self.assertEqual(mismatched, [])

    def test_debug_utils_compare_data(self):
        d1 = {
            "d1": torch.ones(1, 3, 4, 4),
            "d2": {"a": 1, "b": 2},
        }
        d2 = {
            "d1": torch.ones(1, 3, 4, 4),
            "d2": {"a": 1, "b": 2},
            "d3": {"a1": 1, "b1": 2},
        }
        d3 = {
            "d1": torch.ones(1, 3, 4, 4),
            "d2": {"a": 1, "b": 2, "c": torch.ones(1, 3, 4, 4)},
            "d3": {"a1": 1, "b1": 2},
        }

        matched_keys, mismatched_keys = du.compare_data(d1, d1)
        self.assertEqual(sorted(matched_keys), sorted(["d1", "d2.a", "d2.b"]))
        self.assertEqual(mismatched_keys, [])

        matched_keys, mismatched_keys = du.compare_data(d1, d2)
        self.assertEqual(sorted(matched_keys), sorted(["d1", "d2.b", "d2.a"]))
        self.assertEqual(sorted(mismatched_keys), sorted(["d3"]))

        matched_keys, mismatched_keys = du.compare_data(d1, d3)
        self.assertEqual(sorted(matched_keys), sorted(["d1", "d2.b", "d2.a"]))
        self.assertEqual(sorted(mismatched_keys), sorted(["d3", "d2.c"]))
