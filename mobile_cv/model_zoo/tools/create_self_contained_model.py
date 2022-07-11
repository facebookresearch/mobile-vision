#!/usr/bin/env python3

"""
Create a self-contained model from given model and data
"""

import argparse
import json
import os
import tempfile
import typing

import mobile_cv.common.misc.iter_utils as iu
import torch
import torch.utils.bundled_inputs
import torchvision  # noqa

# @manual=//mobile-vision/mobile_cv/mobile_cv/common:utils_io
from mobile_cv.common.utils_io import get_path_manager

# @manual=//mobile-vision/mobile_cv/mobile_cv/model_zoo/tools:tools_common_libs
from mobile_cv.model_zoo.tools.common_libs import load_libraries

path_manager = get_path_manager()

load_libraries()


def parse_arguments(input_args):
    parser = argparse.ArgumentParser(description="Create a self-contained model")
    parser.add_argument("--model", type=str, required=True, help="Model file")
    parser.add_argument(
        "--input_type",
        type=str,
        default="file",
        choices=["file", "shape"],
        help="Input type",
    )
    parser.add_argument(
        "--input_file", type=str, default=None, help="Input data file (.pth)"
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default=None,
        help="Shape of input data as a list of list",
    )
    parser.add_argument(
        "--input_memory_format",
        type=str,
        default="contiguous",
        choices=["contiguous", "channels_last", "None"],
        help="Specify the data format for input data",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output dir")
    parser.add_argument(
        "--trace_type",
        type=str,
        default="trace",
        choices=["trace", "script"],
        help="Trace or script the output model",
    )
    parser.add_argument(
        "--self_contained",
        type=int,
        default=1,
        help="Create self-contained model if 1, otherwise save model and data separately",
    )
    parser.add_argument(
        "--self_container_type",
        type=str,
        default="wrapper",
        choices=["bundle_input", "wrapper"],
        help="Use bundle input inside the model to store data",
    )
    parser.add_argument(
        "--bundle_format",
        type=str,
        default="none",
        choices=["none", "pytorch", "caffe2"],
        help="Use bundle input inside the model to store data",
    )
    parser.add_argument(
        "--optimize_for_mobile",
        type=int,
        default=0,
        help="Run optimize_for_mobile if 1, mainly useful for fp32 mobile model",
    )
    parser.add_argument(
        "--int8_backend",
        type=str,
        default=None,
        help="int8 quantization backend: qnnpack or fbgemm",
    )
    parser.add_argument(
        "--lite_format",
        type=int,
        default=1,
        help="Save model with lite format",
    )

    ret = parser.parse_args(input_args)
    ret.output_dir = os.path.abspath(ret.output_dir)
    if ret.input_shape is not None:
        ret.input_shape = json.loads(ret.input_shape)
    return ret


def load_model(args):
    if path_manager.isdir(args.model):
        model_path = os.path.join(args.model, "model.jit")
    else:
        model_path = args.model
    print(f"Loading model from {model_path}...")
    if args.int8_backend is not None:
        torch.backends.quantized.engine = args.int8_backend
    h_in = path_manager.get_local_path(model_path)
    model = torch.jit.load(h_in)
    model.eval()
    return model


def load_inputs(args):
    if args.input_type == "file":
        if args.input_file is not None:
            input_path = args.input_file
        elif path_manager.isdir(args.model):
            input_path = os.path.join(args.model, "data.pth")
        else:
            base_dir = os.path.dirname(args.model)
            input_path = os.path.join(base_dir, "data.pth")
        assert path_manager.isfile(input_path), f"Data file {input_path} not existed."
        with path_manager.open(input_path, "rb") as h_in:
            ret = torch.load(h_in)
    elif args.input_type == "shape":
        ret = get_input_data(args)
    else:
        raise f"Invalid input type {args.input_type}"

    ret = _make_data_contiguous(ret, memory_format=args.input_memory_format)

    return ret


def get_input_data(args):
    data_shape = args.input_shape
    assert isinstance(data_shape, (list, dict)), f"Invalid data shape type {data_shape}"

    def _is_value_list(obj):
        if not isinstance(obj, list):
            return False
        return all(isinstance(val, int) for val in obj)

    iters = iu.recursive_iterate(
        data_shape, seq_check_func=lambda x: iu.is_seq(x) and not _is_value_list(x)
    )
    for x in iters:
        iters.send(torch.zeros(x))
    ret = iters.value

    return ret


def trace_and_save_torchscript(
    model: torch.nn.Module,
    inputs: typing.Tuple[typing.Any, ...],
    output_path: str,
    trace_type: str = "trace",
    lite_format: bool = True,
):
    print("Tracing and saving TorchScript to {} ...".format(output_path))
    with torch.no_grad():
        if trace_type == "trace":
            script_model = torch.jit.trace(model, inputs)
        else:
            script_model = torch.jit.script(model)

    # Sanity check on the forward pass of the scripted model
    script_model(*inputs)

    path_manager.mkdirs(output_path)

    try:
        op_list = torch.jit.export_opnames(script_model)
        for x in op_list:
            print(f'"{x}",')
    except Exception:
        print("Could not create opnames, skipped")

    model_file = os.path.join(output_path, "model.jit")
    with tempfile.NamedTemporaryFile() as tmp_fn:
        if lite_format:
            script_model._save_for_lite_interpreter(tmp_fn.name)
        else:
            script_model.save(tmp_fn.name)
        path_manager.copy(tmp_fn.name, model_file, overwrite=True)

    data_file = os.path.join(output_path, "data.pth")
    with path_manager.open(data_file, "wb") as h_out:
        torch.save(inputs, h_out)

    return model_file


def _make_data_contiguous(data, memory_format):
    MAP = {
        "contiguous": torch.contiguous_format,
        "channels_last": torch.channels_last,
        "None": None,
        None: None,
    }
    memory_format = MAP[memory_format]
    citer = iu.recursive_iterate(data, iter_types=torch.Tensor)
    for cur in citer:
        if memory_format is not None:
            if memory_format == torch.channels_last and len(cur.shape) != 4:
                print("channels_last could only be used with 4d tensor")
            else:
                cur = cur.contiguous(memory_format=memory_format).clone()
        citer.send(cur)
    return citer.value


class SelfContainedModelBundleInput(torch.nn.Module):
    """Use bundle input to create self contained model, support any inputs, but
    has size limitation on the data
    """

    def __init__(self, model, data):
        super().__init__()
        self.model = model
        torch.utils.bundled_inputs.augment_model_with_bundled_inputs(model, [data])
        self.train(getattr(model, "training", False))

    def forward(self):
        return self.model(*self.model.get_all_bundled_inputs()[0])


class SelfContainedModelP1(torch.nn.Module):
    def __init__(self, model, data):
        super().__init__()
        self.model = model
        self.data = data[0]
        self.train(getattr(model, "training", False))

    def forward(self):
        return self.model(self.data)


class SelfContainedModelP2(torch.nn.Module):
    def __init__(self, model, data):
        super().__init__()
        self.model = model
        self.data = data
        self.train(getattr(model, "training", False))

    def forward(self):
        return self.model(self.data[0], self.data[1])


class SelfContainedModelP3(torch.nn.Module):
    def __init__(self, model, data):
        super().__init__()
        self.model = model
        self.data = data
        self.train(getattr(model, "training", False))

    def forward(self):
        return self.model(self.data[0], self.data[1], self.data[2])


class SelfContainedModelP5(torch.nn.Module):
    def __init__(self, model, data):
        super().__init__()
        self.model = model
        self.data = data
        self.train(getattr(model, "training", False))

    def forward(self):
        return self.model(
            self.data[0], self.data[1], self.data[2], self.data[3], self.data[4]
        )


def SelfContainedModel(model, data):
    # torch scripting could not handle self.model(*data) so we create different
    # classes for different number of inputs
    if len(data) == 1:
        return SelfContainedModelP1(model, data)
    if len(data) == 2:
        return SelfContainedModelP2(model, data)
    if len(data) == 3:
        return SelfContainedModelP3(model, data)
    if len(data) == 5:
        return SelfContainedModelP5(model, data)
    raise ValueError(f"Not supported number of inputs {len(data)}")


def _run_optimize_for_mobile(model):
    from torch.utils.mobile_optimizer import optimize_for_mobile

    preserved_methods = [
        "get_all_bundled_inputs",
        "get_num_bundled_inputs",
    ]
    preserved_methods = [x for x in preserved_methods if hasattr(model, x)]
    model = optimize_for_mobile(
        model,
        preserved_methods=preserved_methods,
    )
    return model


def run(args):
    output_dir = args.output_dir

    model = load_model(args)
    inputs = load_inputs(args)

    """Run optimize_for_mobile on the original model instead of the SelfContainedModel
      as it could cause side effects (e.g., "Runtime Error: module contains
      attributes values that overlaps" for the "test_create_self_contained_model_channels_last"
      test )
    """
    """Run optimize_for_mobile on the original model instead of the SelfContainedModel
      as it could cause side effects (e.g., "Runtime Error: module contains
      attributes values that overlaps" for the "test_create_self_contained_model_channels_last"
      test )
    """
    if args.optimize_for_mobile:
        model = _run_optimize_for_mobile(model)

    if args.self_contained:
        if args.self_container_type == "bundle_input":
            if args.bundle_format == "pytorch":
                inputs = [(inputs,)]
            elif args.bundle_format == "caffe2":
                height = inputs.size()[2]
                width = inputs.size()[3]
                model_info = torch.tensor([[float(height), float(width), 1.0]])
                inputs = [((inputs, model_info),)]
            if args.bundle_format == "none":
                inputs = [inputs]
            torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
                model, inputs, skip_size_check=True
            )
            inputs = inputs[0]
        elif args.self_container_type == "wrapper":
            model = SelfContainedModel(model, inputs)
            print(model)
            inputs = []
        else:
            raise Exception(f"Invalid container type {args.self_container_type}")

    # Run sanity check before scripting to make sure forward pass is working with the provided inputs
    # incase of self contained model the inputs are empty
    model(*inputs)

    # export model
    trace_and_save_torchscript(
        model,
        inputs,
        output_path=output_dir,
        trace_type=args.trace_type,
        lite_format=args.lite_format,
    )

    print(f"Models saved to {output_dir}")

    return output_dir


def main(args_list=None):
    args = parse_arguments(args_list)
    return run(args)


if __name__ == "__main__":
    main()
