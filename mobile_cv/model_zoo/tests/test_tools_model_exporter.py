#!/usr/bin/env python3

import copy
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


class TestModel(torch.nn.Module):
    def __init__(self, num=0):
        super().__init__()
        self.num = num

    def forward(self, x):
        return x + self.num


class UnitTestTask(TaskBase):
    def __init__(self, num=0):
        super().__init__()
        self.num = num

    def get_model(self):
        return TestModel(self.num)

    def get_quantized_model(self, model, data_loader):
        return TestModel(self.num)

    def get_backend1_model(self, quantized_model):
        return TestModel(self.num)

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

    def test_tools_model_exporter_test_dynamic(self):
        with tempfile.TemporaryDirectory() as output_dir:
            export_args = [
                "--task",
                "test_task_unittest",
                "--output_dir",
                output_dir,
                "--raise_if_failed",
                "1",
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

    def test_tools_model_exporter_bundle_input(self):
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
                "--save_for_lite_interpreter",
                "--save_bundle_input",
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
                model = torch.jit.load(path).eval()
                self.assertEqual(model.get_num_bundled_inputs(), 1)
                self.assertEqual(len(model.get_all_bundled_inputs()[0]), 1)
                with torch.no_grad():
                    out = model(*(model.get_all_bundled_inputs()[0]))
                self.assertEqual(out.shape, torch.Size([1, 1000]))

    def test_tools_model_exporter_with_annotations(self):
        gt_anns = {"attr1": "attr1", "attr2": 2}
        gt_anns_quantized = {"attr1": "attr1", "attr2": 2, "attr_quant": 3}

        @task_factory.TASK_FACTORY.register("task_ann")
        class TaskWithAnn(TaskBase):
            def get_model(self):
                ret = TestModel()
                ret.attrs = gt_anns
                return ret

            def get_quantized_model(self, model, data_loader):
                model = copy.deepcopy(model)
                model.attrs = gt_anns_quantized
                return model

            def get_dataloader(self):
                return [[torch.Tensor(1)], [torch.Tensor(1)]]

        for trace_type in [
            "trace",
            "script",
        ]:
            with tempfile.TemporaryDirectory() as output_dir:
                export_args = [
                    "--task",
                    "task_ann",
                    "--task_args",
                    json.dumps({}),
                    "--output_dir",
                    output_dir,
                    "--raise_if_failed",
                    "1",
                    "--export_types",
                    "torchscript",
                    "torchscript_int8",
                    "--post_quant_backend",
                    "default",
                    "--trace_type",
                    trace_type,
                ]
                out_paths = model_exporter.run_with_cmdline_args_list(export_args)
                self.assertEqual(len(out_paths), 2)
                self.assertSetEqual(
                    set(out_paths.keys()), {"torchscript", "torchscript_int8"}
                )
                for key, path in out_paths.items():
                    self.assertTrue(os.path.exists(path))
                    loaded_model = torch.load(path)
                    self.assertEqual(loaded_model.attr1, "attr1")
                    self.assertEqual(loaded_model.attr2, 2)
                    if key != "torchscript_int8":
                        self.assertFalse(hasattr(loaded_model, "attr_quant"))
                    else:
                        self.assertEqual(loaded_model.attr_quant, 3)

                    ann_file = os.path.join(os.path.dirname(path), "annotations.pth")
                    self.assertTrue(os.path.exists(ann_file))
                    loaded_ann = torch.load(ann_file)
                    cur_gt = gt_anns if key != "torchscript_int8" else gt_anns_quantized
                    self.assertEqual(loaded_ann, cur_gt)

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

    def test_tools_model_exporter_quantization_fx(self):
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
                # fx quantization
                "--use_graph_mode_quant",
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

    def test_tools_model_exporter_fx_quant(self):
        blocks_factory.PRIMITIVES.register_dict(
            {
                "exporter_fx_quant_test": lambda in_channels, out_channels, stride, **kwargs: TestModel(
                    num=2
                )
            }
        )
        arch = {"blocks": [[("exporter_fx_quant_test", 1, 1, 1)]]}

        fbnet_args = {
            "builder": "fbnet_v2_backbone",
            "arch_name": json.dumps(arch),
        }
        dataset_args = {
            "builder": "tensor_shape",
            "input_shapes": [[2, 2]],
        }

        with tempfile.TemporaryDirectory() as output_dir:
            export_args = [
                "--task",
                "general",
                "--task_args",
                json.dumps({"model_args": fbnet_args, "dataset_args": dataset_args}),
                "--output_dir",
                output_dir,
                "--use_graph_mode_quant",
                "--raise_if_failed",
                "1",
                "--export_types",
                "torchscript",
                "torchscript_int8",
            ]
            out_paths = model_exporter.run_with_cmdline_args_list(export_args)
            self.assertEqual(len(out_paths), 2)
            self.assertSetEqual(
                set(out_paths.keys()), {"torchscript", "torchscript_int8"}
            )
            for _, path in out_paths.items():
                self.assertTrue(os.path.exists(path))

    def test_tools_model_exporter_quantization_fx_swap_module(self):
        arch_def = {
            "input_size": 64,
            "basic_args": {
                "bn_args": "naiveSyncBN",
            },
            "blocks": [
                [["conv_k3", 16, 2, 1]],
                [["ir_k3", 16, 1, 1]],
            ],
        }

        fbnet_args = {"builder": "fbnet_v2", "arch_name": arch_def}
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
                # fx quantization
                "--use_graph_mode_quant",
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

    def test_tools_model_exporter_data_loader_generator(self):
        @task_factory.TASK_FACTORY.register("task_dl_gen")
        class TaskWithAnn(TaskBase):
            def get_model(self):
                ret = TestModel()
                return ret

            def get_dataloader(self):
                return (x for x in [[torch.Tensor(1)], [torch.Tensor(1)]])

        with tempfile.TemporaryDirectory() as output_dir:
            export_args = [
                "--task",
                "task_dl_gen",
                "--task_args",
                json.dumps({}),
                "--output_dir",
                output_dir,
                "--raise_if_failed",
                "1",
                "--export_types",
                "torchscript",
                "--trace_type",
                "trace",
            ]
            out_paths = model_exporter.run_with_cmdline_args_list(export_args)
            self.assertEqual(len(out_paths), 1)
            self.assertSetEqual(set(out_paths.keys()), {"torchscript"})

    def test_tools_model_exporter_batch_mode(self):
        @task_factory.TASK_FACTORY.register("batch_task_test")
        def batch_task_test():
            return {
                "task1": {"num": 1},
                "task2": {"num": 2},
            }

        with tempfile.TemporaryDirectory() as output_dir:
            export_args = [
                "--task",
                "test_task_unittest",
                "--output_dir",
                output_dir,
                "--export_types",
                "torchscript",
                "--batch_mode",
                "batch_task_test",
            ]
            out_paths = model_exporter.run_with_cmdline_args_list(export_args)
            self.assertIsInstance(out_paths, dict)
            self.assertEqual(len(out_paths), 2)
            self.assertSetEqual(
                set(out_paths.keys()),
                {"task1", "task2"},
            )

            for _task_name, paths in out_paths.items():
                for _export_name, path in paths.items():
                    self.assertTrue(os.path.exists(path))
