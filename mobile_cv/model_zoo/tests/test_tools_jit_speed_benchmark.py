#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import tempfile
import unittest

import torch
import torch.nn as nn
from mobile_cv.model_zoo.models import model_utils
from mobile_cv.model_zoo.tools import jit_speed_benchmark


class Model(nn.Module):
    def forward(self, x):
        return x + 1.0


class ModelDict(nn.Module):
    def forward(self, x, y):
        return {"out1": x + 1.0, "out2": y + 2.0}


def _save_model(model, data, output_dir, save_type):
    if save_type == "trace":
        traced = torch.jit.trace(model, data)
    else:
        traced = torch.jit.script(model)

    model_utils.save_model(output_dir, traced, data)

    return output_dir


def _create_data(num_counts):
    ret = []
    for idx in range(num_counts):
        cur = torch.zeros(1, 3, 4, 4) + idx
        ret.append(cur)
    return ret


class TestJitSpeedBenchmark(unittest.TestCase):
    def test_jit_speed_benchmark(self):
        model = Model()
        data = _create_data(1)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = _save_model(model, data, temp_dir, "trace")
            jit_speed_benchmark.main(
                [
                    "--model",
                    os.path.join(model_dir, "model.jit"),
                    "--input_dims",
                    "1,3,4,4",
                ]
            )
