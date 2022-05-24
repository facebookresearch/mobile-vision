#!/usr/bin/env python3

import unittest

import torch
from mobile_cv.torch.utils_pytorch.vis import vis_model


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 8, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = torch.nn.functional.relu(y)
        y += x

        return y


class TestUtilsPytorchVis(unittest.TestCase):
    def test_utils_pytorch_vis_vis_model(self):
        model = Model()
        inputs = torch.zeros((1, 8, 4, 4))
        log_dir, model_url = vis_model("test_model", model, [inputs])
