#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
import os
import tempfile
import unittest
from typing import NamedTuple

import torch
import torch.nn as nn
from mobile_cv.model_zoo.models import model_utils
from mobile_cv.model_zoo.tools import create_self_contained_model


class Model(nn.Module):
    def forward(self, x):
        return x + 1.0


class ModelDict(nn.Module):
    def forward(self, x, y):
        return {"out1": x + 1.0, "out2": y + 2.0}


class ModelDictCheckFormat(nn.Module):
    def forward(self, x, y):
        return (
            torch.ones(1) * int(x.is_contiguous()),
            torch.ones(1) * int(y.is_contiguous()),
        )


class ModelDictOutput(NamedTuple):
    out1: torch.Tensor
    out2: torch.Tensor


class ModelDictIO(nn.Module):
    def forward(self, data):
        return ModelDictOutput(out1=data["video"] + 1.0, out2=data["audio"] + 2.0)


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


def _test_create_self_contained(self, model, data, model_save_type, sc_model_save_type):
    print(
        f"Create {model_save_type} model and {sc_model_save_type} self-contained model..."
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = _save_model(model, data, temp_dir, model_save_type)
        sc_model_dir = os.path.join(temp_dir, "sc")
        input_args = [
            "--model",
            model_dir,
            "--output_dir",
            sc_model_dir,
            "--trace_type",
            sc_model_save_type,
            "--self_container_type",
            "wrapper",
        ]
        sc_model_dir = create_self_contained_model.main(input_args)
        self.assertTrue(os.path.isfile(os.path.join(sc_model_dir, "model.jit")))


def _test_create_self_contained_from_shape(
    self,
    model,
    input_shape,
    model_save_type,
    sc_model_save_type,
    input_memory_format="contiguous",
    self_container_type="wrapper",
):
    print(
        f"Create {model_save_type} model and {sc_model_save_type} self-contained model..."
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = _save_model(model, [], temp_dir, model_save_type)
        sc_model_dir = os.path.join(temp_dir, "sc")
        input_args = [
            "--model",
            model_dir + "/model.jit",
            "--input_type",
            "shape",
            "--input_shape",
            json.dumps(input_shape),
            "--output_dir",
            sc_model_dir,
            "--trace_type",
            sc_model_save_type,
            "--input_memory_format",
            input_memory_format,
            "--self_container_type",
            self_container_type,
        ]
        print(input_args)
        sc_model_dir = create_self_contained_model.main(input_args)
        out_file = os.path.join(sc_model_dir, "model.jit")
        self.assertTrue(os.path.isfile(out_file))
        loaded_model = torch.jit.load(out_file)
        return loaded_model


def _test_create_self_contained_from_data(
    self,
    model,
    input_data,
    model_save_type,
    sc_model_save_type,
    input_memory_format=None,
    self_container_type="wrapper",
):
    print(
        f"Create {model_save_type} model and {sc_model_save_type} self-contained model..."
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = _save_model(model, input_data, temp_dir, model_save_type)
        sc_model_dir = os.path.join(temp_dir, "sc")

        input_data_path = os.path.join(temp_dir, "data.pth")
        torch.save(input_data, input_data_path)

        input_args = [
            "--model",
            model_dir + "/model.jit",
            "--input_type",
            "file",
            "--input_file",
            input_data_path,
            "--output_dir",
            sc_model_dir,
            "--trace_type",
            sc_model_save_type,
            "--input_memory_format",
            str(input_memory_format),
            "--self_container_type",
            self_container_type,
            "--optimize_for_mobile",
            "1",
        ]
        print(input_args)
        sc_model_dir = create_self_contained_model.main(input_args)
        out_file = os.path.join(sc_model_dir, "model.jit")
        self.assertTrue(os.path.isfile(out_file))
        loaded_model = torch.jit.load(out_file)
        return loaded_model


class TestModelZooToolsCreateSelfContainedModel(unittest.TestCase):
    def test_create_self_contained_model_simple(self):
        model = Model()
        data = _create_data(1)
        _test_create_self_contained(self, model, data, "trace", "trace")
        _test_create_self_contained(self, model, data, "script", "trace")
        _test_create_self_contained(self, model, data, "trace", "script")
        _test_create_self_contained(self, model, data, "script", "script")

    def test_create_self_contained_model_dict(self):
        model = ModelDict()
        data = _create_data(2)
        # the model could only be scripted as dict return type is not supported by tracing
        _test_create_self_contained(self, model, data, "script", "script")

    def test_create_self_contained_model_from_shape(self):
        model = Model()
        data_shape = [[1, 3, 2, 4]]
        _test_create_self_contained_from_shape(
            self, model, data_shape, "script", "script"
        )

    def test_create_self_contained_model_from_shape_dict(self):
        model = ModelDictIO()
        data_shape = [{"video": (1, 3, 2, 4), "audio": (1, 1, 1, 1)}]

        data = [{x: torch.zeros(y) for x, y in data_shape[0].items()}]
        traced_model = torch.jit.trace(model, data)

        _test_create_self_contained_from_shape(
            self,
            traced_model,
            data_shape,
            "script",
            "script",
            self_container_type="wrapper",
        )

    def test_create_self_contained_model_channels_last(self):
        model = ModelDictCheckFormat()
        data_shape = [(1, 3, 2), (1, 3, 4, 2)]

        traced_model = torch.jit.script(model)

        run_out = _test_create_self_contained_from_shape(
            self,
            traced_model,
            data_shape,
            "trace",
            "trace",
            input_memory_format="channels_last",
            self_container_type="wrapper",
        )()
        # 1: contiguous, not channels last, 0: not contiguous, channels last
        # run_out[0] is `contiguous` (1) because channels_last is only applied
        #  when the input is a 4d tensor
        self.assertEqual(run_out, (torch.Tensor([1]), torch.Tensor([0])))

        run_out = _test_create_self_contained_from_shape(
            self,
            traced_model,
            data_shape,
            "trace",
            "trace",
            input_memory_format="contiguous",
            self_container_type="wrapper",
        )()
        # 1: contiguous, not channels last, 0: not contiguous, channels last
        self.assertEqual(run_out, (torch.Tensor([1]), torch.Tensor([1])))

    def test_create_self_contained_model_memory_format_in_file(self):
        model = ModelDictCheckFormat()
        data_shape = [
            ((1, 3, 2, 2), torch.channels_last),
            ((1, 3, 4, 2), torch.contiguous_format),
        ]
        data = [torch.zeros(x[0]).contiguous(memory_format=x[1]) for x in data_shape]

        traced_model = torch.jit.script(model)

        run_out = _test_create_self_contained_from_data(
            self,
            traced_model,
            data,
            "trace",
            "trace",
            input_memory_format="None",
            self_container_type="wrapper",
        )()
        # 1: contiguous, not channels last, 0: not contiguous, channels last
        self.assertEqual(run_out, (torch.Tensor([0]), torch.Tensor([1])))

        run_out = _test_create_self_contained_from_data(
            self,
            traced_model,
            data,
            "trace",
            "trace",
            input_memory_format="contiguous",
            self_container_type="wrapper",
        )()
        # 1: contiguous, not channels last, 0: not contiguous, channels last
        self.assertEqual(run_out, (torch.Tensor([1]), torch.Tensor([1])))

    def test_create_self_contained_model_bundle_input(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 4, 1, stride=2)

            def forward(self, x):
                return self.conv(x)

        model = Model()
        data_shape = [(1, 3, 4, 4)]
        data = [torch.ones(data_shape[0])]
        out_shape = (1, 4, 2, 2)

        out_model = _test_create_self_contained_from_data(
            self,
            model,
            data,
            "trace",
            "script",
            self_container_type="bundle_input",
        )
        bundle_inputs = out_model.get_all_bundled_inputs()
        self.assertEqual(len(bundle_inputs), 1)
        self.assertEqual(bundle_inputs[0][0].shape, torch.Size(data_shape[0]))

        model_out = out_model(*bundle_inputs[0])
        self.assertEqual(model_out.shape, torch.Size(out_shape))

    def test_create_self_contained_model_wrapper(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 4, 1, stride=2)

            def forward(self, x):
                return self.conv(x)

        model = Model()
        data_shape = [(1, 3, 4, 4)]
        data = [torch.ones(data_shape[0])]
        out_shape = (1, 4, 2, 2)

        out_model = _test_create_self_contained_from_data(
            self,
            model,
            data,
            "trace",
            "trace",
            self_container_type="wrapper",
        )
        # make sure conv is actually called
        self.assertIn("conv2d_clamp_run", out_model.model.code)
        self.assertEqual(out_model().shape, torch.Size(out_shape))

    def test_invalid_self_contained_model_wrapper(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 4, 1, stride=2)

            def forward(self, x):
                return self.conv(x)

        model = Model()
        data_shape = [(1, 1, 4, 4)]
        data = [torch.ones(data_shape[0])]
        with self.assertRaises(RuntimeError):
            model(*data)

        with self.assertRaises(RuntimeError):
            _test_create_self_contained_from_data(
                self,
                model,
                data,
                "trace",
                "trace",
                self_container_type="wrapper",
            )


if __name__ == "__main__":
    unittest.main()
