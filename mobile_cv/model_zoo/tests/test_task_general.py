#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import torch
from mobile_cv.model_zoo.tasks import task_factory


class TestTaskGeneral(unittest.TestCase):
    def test_task_general(self):
        fbnet_args = {"builder": "fbnet_v2", "arch_name": "fbnet_c"}
        dataset_args = {
            "builder": "tensor_shape",
            "input_shapes": [[1, 3, 224, 224]],
        }

        task = task_factory.get("general", fbnet_args, dataset_args)
        model = task.get_model()
        data_loader = task.get_dataloader()
        count = 0
        for data in data_loader:
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0].shape, torch.Size(dataset_args["input_shapes"][0]))
            output = model(*data)
            self.assertEqual(output.shape, torch.Size([1, 1000]))
            count += 1
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
