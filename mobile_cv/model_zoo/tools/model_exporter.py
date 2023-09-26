#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

"""
General model exporter, support torchscript and torchscript int8
"""

import argparse
import copy
import importlib
import itertools
import json
import logging
import multiprocessing as mp
import os
import typing
from typing import Dict

import mobile_cv.arch.utils.fuse_utils as fuse_utils
import mobile_cv.arch.utils.jit_utils as ju

import mobile_cv.common.misc.registry as registry
import mobile_cv.lut.lib.pt.flops_utils as flops_utils
import mobile_cv.model_zoo.tasks.task_factory as task_factory
import torch

from mobile_cv.common import utils_io
from mobile_cv.common.misc.file_utils import recompress_export
from mobile_cv.model_zoo.tools.utils import get_model_attributes, get_ptq_model
from torch.utils.bundled_inputs import augment_model_with_bundled_inputs
from torch.utils.mobile_optimizer import optimize_for_mobile


path_manager = utils_io.get_path_manager()
logger = logging.getLogger("model_zoo_tools.export")


ExportFactory = registry.Registry("ExportFactory")
DEFAULT_EXPORT_FORMATS = ["torchscript", "torchscript_int8"]


def parse_args(args_list=None):
    parser = argparse.ArgumentParser(description="Model zoo model exporter")
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task name, if @ is inside the name, use the str after it as the "
        "path to import",
    )
    parser.add_argument("--task_args", type=json.loads, default={}, help="Task args")
    parser.add_argument("--output_dir", type=str, required=True, help="Output base dir")
    parser.add_argument(
        "--export_types",
        type=str,
        nargs="+",
        default=DEFAULT_EXPORT_FORMATS,
        help=f"Export format, supported formats: {ExportFactory.keys()}",
    )
    parser.add_argument(
        "--raise_if_failed",
        type=int,
        default=0,
        help="Throw an exception if conversion failed, elsewise skipped",
    )
    parser.add_argument(
        "--post_quant_backend",
        type=str,
        choices=["qnnpack", "qnnpack_symmetric", "fbgemm", "default"],
        default="fbgemm",
        help="Post quantization: backend",
    )
    parser.add_argument(
        "--post_quant_calibration_batches",
        type=int,
        default=1,
        help="Post quantization: Num of batches of images for calibration",
    )
    parser.add_argument(
        "--use_graph_mode_quant",
        action="store_true",
        help="Use fx quantization for int8 models",
    )
    parser.add_argument(
        "--use_get_traceable",
        type=int,
        default=0,
        help="Use get_traceable_model to convert the model before tracing if 1",
    )
    parser.add_argument(
        "--trace_type",
        type=str,
        default="trace",
        choices=["trace", "script"],
        help="Use trace or script to get the torchscript model."
        + "Ignored when trace_type_per_export is provided",
    )
    parser.add_argument(
        "--trace_type_per_export",
        type=str,
        nargs="+",
        default=[],
        help="choice of trace or script for each export_type",
    )
    parser.add_argument(
        "--opt_for_mobile",
        type=int,
        default=0,
        help="Run optimize for mobile if 1",
    )
    parser.add_argument(
        "--save_for_lite_interpreter",
        action="store_true",
        help="Also export lite interpreter model",
    )
    parser.add_argument(
        "--save_bundle_input",
        action="store_true",
        help="Bundle inputs to the models when exporting",
    )
    parser.add_argument(
        "--batch_mode",
        type=str,
        default=None,
        help="Specify the registed name to run export in batch",
    )

    assert len(ExportFactory.keys()) > 0

    ret = parser.parse_args(args_list)
    if len(ret.trace_type_per_export) > 0:
        assert len(ret.trace_type_per_export) == len(
            ret.export_types
        ), "trace_type_per_export should be provided for each export_type"

    return ret


def _set_attrs_to_model(model, attrs):
    assert isinstance(attrs, dict)
    for k, w in attrs.items():
        if hasattr(model, k) and getattr(model, k) != w:
            logger.warning(
                f"{k} has already existed inside the model, the value {getattr(model, k)} is override with {w}"
            )
        setattr(model, k, w)


def _get_model_with_attrs(model, attrs):
    if not attrs:
        return model
    _set_attrs_to_model(model, attrs)
    return model


class TraceWrapperP0(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.training = getattr(model, "training", False)

    def forward(self):
        return self.model()


class TraceWrapperP1(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.training = getattr(model, "training", False)

    def forward(self, x):
        return self.model(x)


class TraceWrapperP2(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.training = getattr(model, "training", False)

    def forward(self, x, y):
        return self.model(x, y)


class TraceWrapperP3(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, y, z):
        return self.model(x, y, z)


def _get_model_extra_files(model):
    if hasattr(model, "extra_files"):
        return model.extra_files
    else:
        return None


def _get_traced_model_with_attrs(traced_model, num_inputs, model_attrs):
    """For a module that has already been traced, adding additional attributes on
    top of it and script it again will not work. This workaround add a wrapper to
    the traced model with attributes and script it.
    Currently only with up to two
    parameters could be supported, and the parameter names may not match the exact
    names of the traced model.
    """
    assert isinstance(traced_model, torch.jit.ScriptModule)
    if num_inputs == 0:
        traced_model = TraceWrapperP0(traced_model)
    elif num_inputs == 1:
        traced_model = TraceWrapperP1(traced_model)
    elif num_inputs == 2:
        traced_model = TraceWrapperP2(traced_model)
    elif num_inputs == 3:
        traced_model = TraceWrapperP3(traced_model)
    else:
        raise Exception("Traced models with at most two parameters are supported.")
    _set_attrs_to_model(traced_model, model_attrs)
    return traced_model


def get_script_model_with_attrs(model, trace_type, model_inputs=None, model_attrs=None):
    """Trace or script the model and store model_attrs as model attributes."""
    assert trace_type in ("trace", "script"), f"Invalid trace_type {trace_type}"

    info = f'{"Tracing" if trace_type == "trace" else "Scripting"} model'
    if model_attrs is not None:
        info += f" with attributes: {model_attrs}"
    logger.info(info)

    if trace_type == "trace":
        script_model = torch.jit.trace(model, model_inputs, strict=False)
        if model_attrs is not None:
            script_model = _get_traced_model_with_attrs(
                script_model, len(model_inputs), model_attrs
            )
            script_model = torch.jit.script(script_model)
    else:
        if isinstance(model, torch.jit.ScriptModule) and model_attrs is not None:
            logger.warning(
                f"Model has been scripted, could not add attributes {model_attrs}"
            )
        else:
            model = _get_model_with_attrs(model, model_attrs)
        script_model = torch.jit.script(model)

    return script_model


def trace_and_save_torchscript(
    model: torch.nn.Module,
    inputs: typing.Tuple[typing.Any, ...],
    output_path: str,
    use_get_traceable: bool = False,
    trace_type: str = "trace",
    opt_for_mobile: bool = False,
    model_attrs: Dict = None,
    save_for_lite_interpreter: bool = True,
    save_bundle_input: bool = False,
    model_extra_files: Dict = None,
):
    logger.info("Tracing and saving TorchScript to {} ...".format(output_path))

    with torch.no_grad():
        if use_get_traceable:
            model = ju.get_traceable_model(model)

    script_model = get_script_model_with_attrs(
        model, trace_type, model_inputs=inputs, model_attrs=model_attrs
    )

    if opt_for_mobile:
        logger.info("Running optimize_for_mobile...")
        script_model = optimize_for_mobile(script_model)

    if not path_manager.isdir(output_path):
        path_manager.mkdirs(output_path)

    if save_bundle_input:
        augment_model_with_bundled_inputs(script_model, [inputs], skip_size_check=True)

    model_file = os.path.join(output_path, "model.jit")
    with path_manager.open(model_file, "wb") as fp:
        torch.jit.save(script_model, fp, _extra_files=model_extra_files)

    link_model_file = os.path.join(output_path, "model.pt")
    path_manager.symlink(model_file, link_model_file)

    data_file = os.path.join(output_path, "data.pth")
    with path_manager.open(data_file, "wb") as fp:
        torch.save(inputs, fp)

    if model_attrs is not None:
        attrs_file = os.path.join(output_path, "annotations.pth")
        with path_manager.open(attrs_file, "wb") as fp:
            torch.save(model_attrs, fp)

    if save_for_lite_interpreter:
        lite_model_file = os.path.join(output_path, "model.ptl")
        with path_manager.open(lite_model_file, "wb") as fp:
            if model_extra_files:
                fp.write(
                    script_model._save_to_buffer_for_lite_interpreter(
                        _extra_files=model_extra_files
                    )
                )
            else:
                fp.write(script_model._save_to_buffer_for_lite_interpreter())
        lite_compressed_file = os.path.join(output_path, "model_compressed.ptl")
        recompress_export(lite_model_file, lite_compressed_file)

    return model_file


@ExportFactory.register("torchscript")
def export_to_torchscript(
    args, task, model, inputs, output_base_dir, trace_type, **kwargs
):
    model_attrs = get_model_attributes(model)

    output_dir = os.path.join(output_base_dir, "torchscript")

    with torch.no_grad():
        fused_model = fuse_utils.fuse_model(model, inplace=False)

    # get model_extra_files from model if it exists
    model_extra_files = _get_model_extra_files(model)

    print("fused model {}".format(fused_model))
    torch_script_path = trace_and_save_torchscript(
        fused_model,
        inputs,
        output_dir,
        use_get_traceable=bool(args.use_get_traceable),
        trace_type=trace_type,
        opt_for_mobile=args.opt_for_mobile,
        model_attrs=model_attrs,
        save_for_lite_interpreter=args.save_for_lite_interpreter,
        save_bundle_input=args.save_bundle_input,
        model_extra_files=model_extra_files,
    )
    return torch_script_path


@ExportFactory.register("torchscript_int8")
def export_to_torchscript_int8(
    args, task, model, inputs, output_base_dir, trace_type, *, data_iter, **kwargs
):
    ptq_model, model_attrs = get_ptq_model(args, task, model, inputs, data_iter)

    logger.info(ptq_model)

    # get model_extra_files from ptq_model if it exists
    model_extra_files = _get_model_extra_files(ptq_model)

    ptq_folder = os.path.join(output_base_dir, "torchscript_int8")
    ptq_torchscript_path = trace_and_save_torchscript(
        ptq_model,
        inputs,
        ptq_folder,
        use_get_traceable=bool(args.use_get_traceable),
        trace_type=trace_type,
        opt_for_mobile=args.opt_for_mobile,
        model_attrs=model_attrs,
        save_for_lite_interpreter=args.save_for_lite_interpreter,
        save_bundle_input=args.save_bundle_input,
        model_extra_files=model_extra_files,
    )

    return ptq_torchscript_path


@ExportFactory.register("_dynamic_")
def export_to_torchscript_dynamic(
    args,
    task,
    model,
    inputs,
    output_base_dir,
    trace_type,
    *,
    export_format=None,
    **kwargs,
):
    """Task returns the model based on the given export_format
    The code will try to access `task.get_{model_name}_model()` to get the model
    for export, `model_name` is extracted from `export_format` without `torchscript_`
    prefix.
    """

    assert hasattr(task, "get_model_by_name")
    assert export_format.startswith("torchscript_")
    model_name = export_format[len("torchscript_") :]
    model = task.get_model_by_name(model_name, model)
    model_attrs = get_model_attributes(model)

    if model_attrs is not None and "data" in model_attrs:
        inputs = model_attrs["data"]

    # get model_extra_files from model if it exists
    model_extra_files = _get_model_extra_files(model)

    print(f"Converting to {model_name}...")
    output_dir = os.path.join(output_base_dir, export_format)
    torch_script_path = trace_and_save_torchscript(
        model,
        inputs,
        output_dir,
        use_get_traceable=bool(args.use_get_traceable),
        trace_type=trace_type,
        opt_for_mobile=args.opt_for_mobile,
        model_attrs=model_attrs,
        save_for_lite_interpreter=args.save_for_lite_interpreter,
        save_bundle_input=args.save_bundle_input,
        model_extra_files=model_extra_files,
    )
    return torch_script_path


@ExportFactory.register("_executorch_")
def export_to_executorch_dynamic(
    args,
    task,
    model,
    inputs,
    output_base_dir,
    *,
    data_iter,
    export_format=None,
    **kwargs,
):
    """Task returns the model based on the given export_format
    The code will try to access `task.get_{model_name}_model()` to get the model
    for export, `model_name` starts with `executorch_` prefix.
    """
    assert hasattr(task, "get_model_by_name")

    if export_format in ["executorch_int8", "executorch_boltnn"]:
        cur_loader = itertools.chain([inputs], data_iter)
        if hasattr(task, "get_quantized_model_for_et"):
            logger.info("calling get quantized model for executorch")
            ptq_model = task.get_quantized_model_for_et(model, cur_loader)
            ptq_model(*inputs)
        else:
            logger.error("no quantization method found.")

    exec_prog = task.get_model_by_name(export_format, model)
    flatbuffer = exec_prog.buffer
    output_dir = os.path.join(output_base_dir, export_format)

    logger.info("Saving Executorch flatbuffer to {} ...".format(output_dir))
    if not path_manager.isdir(output_dir):
        path_manager.mkdirs(output_dir)

    executorch_path = os.path.join(output_dir, "model.ff")
    with path_manager.open(executorch_path, "wb") as fp:
        fp.write(flatbuffer)

    return executorch_path


def _import_tasks(task_name):
    """if "@" inside the task_name, use the path after it as the path to import"""
    if "@" not in task_name:
        return

    _, module_name = task_name.split("@", 1)
    importlib.import_module(module_name)
    return


def main(
    args,
    output_dir,
    export_formats: typing.List[str] = DEFAULT_EXPORT_FORMATS,
    raise_if_failed: bool = False,
) -> typing.Dict[str, str]:
    _import_tasks(args.task)
    task = task_factory.get(args.task, **args.task_args)
    model = task.get_model()
    if isinstance(model, tuple):
        model, model_attrs = model
        model.attrs = model_attrs
    model.eval()
    data_loader = task.get_dataloader()
    data_iter = iter(data_loader)

    try:
        first_batch = next(data_iter)
    except Exception:
        first_batch = []
        logger.warning("No data available from the data loader")

    try:
        with torch.no_grad():
            flops_utils.print_model_flops(model, first_batch)
    except Exception:
        logger.warning("Failed to print model flops")

    ret = {}
    trace_type_per_export = args.trace_type_per_export
    if len(trace_type_per_export) == 0:
        trace_type_per_export = [args.trace_type] * len(export_formats)
    for ef, trace_type in zip(export_formats, trace_type_per_export):
        assert ef not in ret, f"Export format {ef} has already existed."
        try:
            if ef in ExportFactory:
                export_func = ExportFactory.get(ef)
            elif ef.startswith("executorch"):
                export_func = ExportFactory.get("_executorch_")
            else:
                export_func = ExportFactory.get("_dynamic_")
            out_path = export_func(
                args,
                task,
                model,
                first_batch,
                output_dir,
                trace_type=trace_type,
                # NOTE: output model maybe difference if data_loader is used multiple times
                data_iter=data_iter,
                export_format=ef,
            )
            ret[ef] = out_path
        except Exception as e:
            logger.warning(f"Export format {ef} failed: {e}")
            if raise_if_failed:
                raise e

    return ret


def _run_main_single(args_tuple):
    args, (name, task_args) = args_tuple
    cur_args = copy.deepcopy(args)
    cur_args.task_args.update(task_args)
    cur_output_dir = os.path.join(args.output_dir, name)
    ret = main(
        cur_args,
        cur_output_dir,
        export_formats=cur_args.export_types,
        raise_if_failed=cur_args.raise_if_failed,
    )
    return name, ret


def main_batch_mode(args):
    """Run model exporter in a batch mode.
    To use batch mode, the user needs to create a function, register it as a
    task and pass the name to `args.batch_mode`. The function returns a dict of
    task arguments with their names that could be used to create the task
    specified in `args.task`
      func() -> Dict[str, TaskArgs]
    All the exported models will be stored in the sub folders of `args.output_dir`
    indicated by their names.
    """
    assert args.batch_mode is not None
    _import_tasks(args.batch_mode)
    all_tasks_args = task_factory.get(args.batch_mode)
    ret = {}
    total_count = len(all_tasks_args)

    with mp.Pool(16) as pool:
        all_items = ((args, sub_item) for sub_item in all_tasks_args.items())
        for idx, (name, cur) in enumerate(
            pool.imap_unordered(_run_main_single, all_items)
        ):
            print(f"Exported {idx}/{total_count}: {name}")
            ret[name] = cur

    print(f"{total_count} models exported to {args.output_dir}")

    return ret


def _get_task_info(args):
    ret = {"task": args.task, "task_args": args.task_args}
    return ret


def run_with_cmdline_args(args):
    if args.batch_mode is not None:
        return main_batch_mode(args)

    return main(
        args,
        args.output_dir,
        export_formats=args.export_types,
        raise_if_failed=args.raise_if_failed,
    )


def run_with_cmdline_args_list(
    args_list: typing.Optional[typing.List[str]] = None,
) -> typing.Dict[str, typing.Any]:
    args = parse_args(args_list)
    return run_with_cmdline_args(args)


if __name__ == "__main__":
    run_with_cmdline_args_list()
