#!/usr/bin/env python3

"""
Create model from model zoo
"""

import argparse
import json
import os

import mobile_cv.arch.utils.quantize_utils as qu
import mobile_cv.common.misc.iter_utils as iu
import torch
from mobile_cv.lut.lib.pt import flops_utils
from mobile_cv.model_zoo.models import model_utils, model_zoo_factory


def parse_arguments(input_args):
    parser = argparse.ArgumentParser(description="Create a model from model zoo")
    parser.add_argument(
        "--builder", type=str, default="fbnet_v2", help="model builder name"
    )
    parser.add_argument(
        "--arch_name",
        type=str,
        default=None,
        help="model builder argument arch_name, empty to not use",
    )
    parser.add_argument(
        "--arch_args",
        type=str,
        default=None,
        help="Additional arguments as a str of dict",
    )
    parser.add_argument(
        "--data_shape",
        type=str,
        default="[[1, 3, 224, 224]]",
        help="Shape of input data as a list of list",
    )
    parser.add_argument("--output_dir", type=str, help="Output dir", default="output")
    parser.add_argument(
        "--fuse_bn", type=int, default=1, help="Fuse bn in the model if 1"
    )
    parser.add_argument(
        "--use_get_traceable",
        type=int,
        default=0,
        help="Use get_traceable() to convert the model before trace if 1",
    )
    parser.add_argument("--int8_backend", type=str, default=None, help="int8 backend")
    parser.add_argument(
        "--self_contained",
        type=int,
        default=0,
        help="Create self-contained model as well if 1",
    )
    parser.add_argument(
        "--convert_int8",
        type=int,
        default=1,
        help="convert to int8 if 1",
    )
    parser.add_argument(
        "--print_per_layer_flops",
        type=int,
        default=1,
        help="Print per layer flops if 1",
    )
    ret = parser.parse_args(input_args)
    ret.output_dir = os.path.abspath(ret.output_dir)
    if ret.arch_args is not None:
        ret.arch_args = json.loads(ret.arch_args)
    if ret.data_shape is not None:
        ret.data_shape = json.loads(ret.data_shape)
    return ret


def create_model(args):
    model_args = {"builder": args.builder}
    if args.arch_name is not None:
        model_args["arch_name"] = args.arch_name
    if args.arch_args is not None:
        model_args = {**model_args, **args.arch_args}
    print(f"Building model: {model_args}")
    model = model_zoo_factory.get_model(**model_args)
    model.eval()

    return model


def get_input_data(args):
    data_shape = args.data_shape
    assert isinstance(data_shape, (list, dict)), f"Invalid data shape type {data_shape}"

    def _is_value_list(obj):
        if not isinstance(obj, list):
            return False
        return all(isinstance(val, int) for val in obj)

    iters = iu.recursive_iterate(
        data_shape, seq_check_func=lambda x: not _is_value_list(x)
    )
    for x in iters:
        iters.send(torch.zeros(x))
    ret = iters.value

    return ret


def convert_jit(args, model, data, folder_name="jit"):
    print("Converting to jit...")
    # trace model
    traced_model, traced_output = model_utils.convert_torch_script(
        model,
        data,
        fuse_bn=args.fuse_bn,
        use_get_traceable=bool(args.use_get_traceable),
    )
    output_dir = os.path.join(args.output_dir, folder_name)
    model_utils.save_model(output_dir, traced_model, data)

    return traced_model, traced_output, output_dir


USE_GRAPH_MODE_QUANT = True


def convert_int8_jit(args, model, data, folder_name="int8_jit"):
    if not args.convert_int8:
        return None, None, None
    try:
        print("Converting to int8 jit...")
        if args.int8_backend is not None:
            torch.backends.quantized.engine = args.int8_backend
        if not USE_GRAPH_MODE_QUANT:
            # trace model
            traced_model, traced_output = model_utils.convert_int8_jit(
                model, data, int8_backend=args.int8_backend, add_quant_stub=False
            )
        else:
            quant = qu.PostQuantizationFX(model)
            quant_model = (
                quant.set_quant_backend("default")
                .prepare()
                .calibrate_model([data], 1)
                .convert_model()
            )
            traced_model, traced_output = model_utils.convert_torch_script(
                quant_model,
                data,
                fuse_bn=args.fuse_bn,
                use_get_traceable=bool(args.use_get_traceable),
            )

        print(traced_model)

        output_dir = os.path.join(args.output_dir, folder_name)
        model_utils.save_model(output_dir, traced_model, data)
        return traced_model, traced_output, output_dir
    except Exception as e:
        print(f"Converting to int8 jit failed. {e}")
        return None, None, None


class SelfContainedModel(torch.nn.Module):
    def __init__(self, model, data):
        super().__init__()
        self.model = model
        self.data = data
        self.train(model.training)

    def forward(self):
        return self.model(*self.data)


def run(args):
    base_target_dir = args.output_dir
    os.makedirs(base_target_dir, exist_ok=True)

    model = create_model(args)
    model_quant = (
        torch.ao.quantization.QuantWrapper(model) if not USE_GRAPH_MODE_QUANT else model
    )
    data = get_input_data(args)

    with torch.no_grad():
        flops_utils.print_model_flops(
            model, data, print_per_layer_flops=args.print_per_layer_flops
        )

    convert_jit(args, model, data)
    convert_int8_jit(args, model_quant, data)

    if args.self_contained:
        model_with_data = SelfContainedModel(model, data)
        convert_jit(args, model_with_data, [], folder_name="jit_sc")
        model_with_data_quant = SelfContainedModel(model_quant, data)
        convert_int8_jit(args, model_with_data_quant, [], folder_name="int8_jit_sc")

    print(f"Models saved to {args.output_dir}")


def main(args_list=None):
    args = parse_arguments(args_list)
    run(args)


if __name__ == "__main__":
    main()
