#!/usr/bin/env python3

import copy
import json
import os
import tempfile
import unittest

import mobile_cv.arch.fbnet_v2.blocks_factory as blocks_factory
import torch
from mobile_cv.model_zoo.tools import model_exporter


class TestToolsModelExporter(unittest.TestCase):
    def test_tools_model_exporter(self):
        fbnet_args = {"builder": "fbnet_v2", "arch_name": "fbnet_c"}
        dataset_args = {
            "builder": "tensor_shape",
            "input_shapes": [[1, 3, 64, 64]],
        }

        with tempfile.TemporaryDirectory() as output_dir:
            export_args = [
                "--task",
                "general",
                "--task_args",
                json.dumps({"model_args": fbnet_args, "dataset_args": dataset_args}),
                "--output_dir",
                output_dir,
                "--export_types",
                "torchscript",
                "torchscript_int8",
                "--post_quant_backend",
                "default",
                # currently int8 will fail due to copy issue in quantized op
                # "--use_get_traceable",
                # "1",
            ]
            out_paths = model_exporter.run_with_cmdline_args_list(export_args)
            self.assertEqual(len(out_paths), 2)
            self.assertSetEqual(
                set(out_paths.keys()), {"torchscript", "torchscript_int8"}
            )
            for _, path in out_paths.items():
                self.assertTrue(os.path.exists(path))

    def test_tools_model_exporter_use_get_traceable(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.is_traceable = False

            def forward(self, x):
                # the tracing will fail if get_tracable_model() was not called
                if not self.is_traceable and torch.jit.is_tracing():
                    raise Exception()
                return x

            def to_traceable(self):
                self.is_traceable = True

        blocks_factory.PRIMITIVES.register_dict(
            {"trace_test": lambda in_channels, out_channels, stride, **kwargs: Model()}
        )
        arch = {"blocks": [[("trace_test", 1, 1, 1)]]}

        fbnet_args = {
            "builder": "fbnet_v2_backbone",
            "arch_name": json.dumps(arch),
        }
        dataset_args = {
            "builder": "tensor_shape",
            "input_shapes": [[1, 3, 4, 4]],
        }

        with tempfile.TemporaryDirectory() as output_dir:
            export_args = [
                "--task",
                "general",
                "--task_args",
                json.dumps({"model_args": fbnet_args, "dataset_args": dataset_args}),
                "--output_dir",
                output_dir,
                "--export_types",
                "torchscript",
                "--use_get_traceable",
                "1",
            ]
            out_paths = model_exporter.run_with_cmdline_args_list(export_args)
            self.assertEqual(len(out_paths), 1)
            self.assertSetEqual(set(out_paths.keys()), {"torchscript"})
            for _, path in out_paths.items():
                self.assertTrue(os.path.exists(path))
