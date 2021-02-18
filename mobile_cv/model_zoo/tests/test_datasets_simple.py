#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import torch
from mobile_cv.model_zoo.datasets import dataset_factory


class TestDatasetsSimple(unittest.TestCase):
    def test_dataset_from_shape(self):
        input_shapes = [[1, 3, 16, 16], [1, 3, 6, 6]]

        dl = dataset_factory.get("tensor_shape", input_shapes=input_shapes)
        first_batch = next(iter(dl))
        self.assertEqual(len(first_batch), len(input_shapes))
        for data, gt_size in zip(first_batch, input_shapes):
            self.assertEqual(data.shape, torch.Size(gt_size))


if __name__ == "__main__":
    unittest.main()
