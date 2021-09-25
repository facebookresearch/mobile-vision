#!/usr/bin/env python3

import json
import os
import tempfile
import unittest

import mobile_cv.arch.fbnet_v2.blocks_factory as blocks_factory
import torch
from mobile_cv.model_zoo.tasks import task_factory
from mobile_cv.model_zoo.tasks.task_base import TaskBase
from mobile_cv.model_zoo.tools import model_exporter


@task_factory.TASK_FACTORY.register("test_task_unittest")
def ext_task(**kwargs):
    return UnitTestTask()


class UnitTestTask(TaskBase):
    def get_model(self):
        return torch.nn.Identity()

    def get_quantized_model(self, model, data_loader):
        ret = torch.nn.Identity()
        return ret

    def get_backend1_model(self, quantized_model):
        ret = torch.nn.Identity()
        return ret

    def get_dataloader(self):
        return [[torch.Tensor(1)]]


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

    def test_tools_model_exporter_ext_task(self):
        with tempfile.TemporaryDirectory() as output_dir:
            export_args = [
                "--task",
                "test_task@ext.test.lib.external_task_for_test",
                "--output_dir",
                output_dir,
                "--export_types",
                "torchscript",
                "--opt_for_mobile",
                "1",
            ]
            out_paths = model_exporter.run_with_cmdline_args_list(export_args)
            self.assertEqual(len(out_paths), 1)
            self.assertSetEqual(set(out_paths.keys()), {"torchscript"})
            for _, path in out_paths.items():
                self.assertTrue(os.path.exists(path))

    def test_tools_model_exporter_test_dynamic(self):
        with tempfile.TemporaryDirectory() as output_dir:
            export_args = [
                "--task",
                "test_task_unittest",
                "--output_dir",
                output_dir,
                "--export_types",
                "torchscript",
                "torchscript_int8",
                "torchscript_backend1",
            ]
            out_paths = model_exporter.run_with_cmdline_args_list(export_args)
            self.assertEqual(len(out_paths), 3)
            self.assertSetEqual(
                set(out_paths.keys()),
                {"torchscript", "torchscript_int8", "torchscript_backend1"},
            )
            for _, path in out_paths.items():
                self.assertTrue(os.path.exists(path))

    def test_tools_model_exporter_with_annotations(self):
        from mobile_cv.model_zoo.tasks import task_base, task_factory

        class Model(torch.nn.Module):
            def forward(self, x):
                return x

        @task_factory.TASK_FACTORY.register("task_ann")
        class TaskWithAnn(task_base.TaskBase):
            def get_model(self):
                annos = {"attr1": "attr1", "attr2": 2}
                return Model(), annos

            def get_dataloader(self):
                return [[torch.Tensor(1)], [torch.Tensor(1)]]

        with tempfile.TemporaryDirectory() as output_dir:
            export_args = [
                "--task",
                "task_ann",
                "--task_args",
                json.dumps({}),
                "--output_dir",
                output_dir,
                "--export_types",
                "torchscript",
                "torchscript_int8",
                "--post_quant_backend",
                "default",
                "--trace_type",
                "script",
            ]
            out_paths = model_exporter.run_with_cmdline_args_list(export_args)
            self.assertEqual(len(out_paths), 2)
            self.assertSetEqual(
                set(out_paths.keys()), {"torchscript", "torchscript_int8"}
            )
            for _, path in out_paths.items():
                self.assertTrue(os.path.exists(path))
                loaded_model = torch.load(path)
                self.assertEqual(loaded_model.attr1, "attr1")
                self.assertEqual(loaded_model.attr2, 2)

                ann_file = os.path.join(os.path.dirname(path), "annotations.pth")
                self.assertTrue(os.path.exists(ann_file))
                loaded_ann = torch.load(ann_file)
                self.assertEqual(loaded_ann, {"attr1": "attr1", "attr2": 2})
