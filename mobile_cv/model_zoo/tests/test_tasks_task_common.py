# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


import unittest

import torch
from mobile_cv.model_zoo.tasks.task_common import TaskCommon


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.version = 1
        self.input_shapes = [[1, 3]]
        self.output_shapes = [[1, 3]]

    def forward(self, inputs):
        return inputs * 2


class TestTask(TaskCommon):
    def get_model(self):
        return SimpleModel().eval()

    def get_dataloader(self):
        return [[torch.ones(1, 3)]]


class TestTasksTaskCommon(unittest.TestCase):
    def test_model(self):
        task = TestTask()
        model = task.get_model()
        self.assertIsInstance(model, SimpleModel)
