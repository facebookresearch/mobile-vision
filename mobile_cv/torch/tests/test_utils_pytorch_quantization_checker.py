#!/usr/bin/env python3

import tempfile
import unittest

import mobile_cv.torch.utils_pytorch.quantization_checker as qc
import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 8, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.relu(y)
        y += x

        return y


class TestUtilsPytorchQuantizationChecker(unittest.TestCase):
    def test_quantization_checker(self):
        write_manifold = False
        with tempfile.TemporaryDirectory() as log_dir:
            model = Model().eval()

            # add observers
            qc.add_stats_observers(model)

            # run models
            with torch.no_grad():
                for _ in range(5):
                    inputs = torch.rand((1, 8, 4, 4))
                    model(inputs)

            # get stats results
            results = qc.get_stats_results(model)

            # visualize the results
            cur_log_dir = log_dir if not write_manifold else None
            qc.visualize_stats_results(model, results, log_dir=cur_log_dir)

    def test_quantization_checker_compare(self):
        write_manifold = False
        with tempfile.TemporaryDirectory() as log_dir:

            def _collect(model):
                # add observers
                qc.add_stats_observers(model)

                # run models
                with torch.no_grad():
                    for _ in range(5):
                        inputs = torch.rand((1, 8, 4, 4))
                        model(inputs)

                # get stats results
                results = qc.get_stats_results(model)
                return results

            model1 = Model().eval()
            res1 = _collect(model1)

            model2 = Model().eval()
            res2 = _collect(model2)

            # visualize the results
            cur_log_dir = log_dir if not write_manifold else None
            with qc.vis.tensorboard(log_dir=cur_log_dir) as logger:
                qc.add_visualize_stats_results(logger, 0, model1, res1)
                qc.add_visualize_stats_results(logger, 1, model2, res2)
