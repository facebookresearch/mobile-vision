#!/usr/bin/env python3
"""
General model exporter, support torchscript and torchscript int8
"""

import argparse
import importlib
import itertools
import json
import logging
import os
import typing

import mobile_cv.arch.utils.fuse_utils as fuse_utils
import mobile_cv.arch.utils.jit_utils as ju
import mobile_cv.arch.utils.quantize_utils as quantize_utils
import mobile_cv.common.misc.registry as registry
import mobile_cv.lut.lib.pt.flops_utils as flops_utils
import mobile_cv.model_zoo.tasks.task_factory as task_factory
import torch
from mobile_cv.common import utils_io
from mobile_cv.model_zoo.models import model_utils
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
        choices=["qnnpack", "fbgemm", "default"],
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
        help="Use graph mode quantization for int8 models",
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
        help="Use trace or script to get the torchscript model",
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

    assert len(ExportFactory.keys()) > 0

    ret = parser.parse_args(args_list)
    return ret


def _set_attrs_to_model(model, attrs):
    assert isinstance(attrs, dict)
    for k, w in attrs.items():
        assert not hasattr(model, k), f"{k} has already existed inside the model"
        setattr(model, k, w)


def _get_model_with_attrs(model, attrs):
    if not attrs:
        return model
    _set_attrs_to_model(model, attrs)
    return model


def trace_and_save_torchscript(
    model: torch.nn.Module,
    inputs: typing.Tuple[typing.Any, ...],
    output_path: str,
    use_get_traceable=False,
    trace_type="trace",
    opt_for_mobile=False,
    model_attrs=None,
    save_for_lite_interpreter=False,
):
    logger.info("Tracing and saving TorchScript to {} ...".format(output_path))

    with torch.no_grad():
        if use_get_traceable:
            model = ju.get_traceable_model(model)
        if trace_type == "trace":
            script_model = torch.jit.trace(model, inputs, strict=False)
            if model_attrs is not None:
                script_model = _get_model_with_attrs(script_model, model_attrs)
                script_model = torch.jit.script(script_model)
        else:
            logger.info(f"Saving model with attributes: {model_attrs}")
            model = _get_model_with_attrs(model, model_attrs)
            script_model = torch.jit.script(model)

    if opt_for_mobile:
        logger.info("Running optimize_for_mobile...")
        script_model = optimize_for_mobile(script_model)

    if not path_manager.isdir(output_path):
        path_manager.mkdirs(output_path)

    model_file = os.path.join(output_path, "model.jit")
    with path_manager.open(model_file, "wb") as fp:
        torch.jit.save(script_model, fp)

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
            fp.write(script_model._save_to_buffer_for_lite_interpreter())

    return model_file


@ExportFactory.register("torchscript")
def export_to_torchscript(
    args, task, model, inputs, output_base_dir, *, model_attrs=None, **kwargs
):
    output_dir = os.path.join(output_base_dir, "torchscript")

    with torch.no_grad():
        fused_model = fuse_utils.fuse_model(model, inplace=False)
    print("fused model {}".format(fused_model))
    torch_script_path = trace_and_save_torchscript(
        fused_model,
        inputs,
        output_dir,
        use_get_traceable=bool(args.use_get_traceable),
        trace_type=args.trace_type,
        opt_for_mobile=args.opt_for_mobile,
        model_attrs=model_attrs,
        save_for_lite_interpreter=args.save_for_lite_interpreter,
    )
    return torch_script_path


@ExportFactory.register("torchscript_int8")
def export_to_torchscript_int8(
    args, task, model, inputs, output_base_dir, *, data_iter, model_attrs=None, **kwargs
):
    if args.use_graph_mode_quant:
        print(f"Post quantization using {args.post_quant_backend} backend...")
        print("Converting to int8 jit...")
        quant = quantize_utils.PostQuantizationGraph(model)
        cur_loader = itertools.chain([inputs], data_iter)

        traced_model = (
            quant.set_quant_backend(args.post_quant_backend)
            .set_calibrate(cur_loader, 1)
            .trace(inputs, strict=False)
            .convert_model()
        )
        traced_model(*inputs)

        print(traced_model)
        output_dir = os.path.join(output_base_dir, "torchscript_int8")
        model_utils.save_model(output_dir, traced_model, inputs)
        return output_dir

    cur_loader = itertools.chain([inputs], data_iter)
    if hasattr(task, "get_quantized_model"):
        ptq_model = task.get_quantized_model(model, cur_loader)
    else:
        print(f"Post quantization using {args.post_quant_backend} backend...")
        qa_model = task.get_quantizable_model(model)
        post_quant = quantize_utils.PostQuantization(qa_model)
        post_quant.fuse_bn().set_quant_backend(args.post_quant_backend)
        ptq_model = post_quant.prepare().calibrate_model(cur_loader, 1).convert_model()

    print(ptq_model)
    ptq_model(*inputs)

    ptq_folder = os.path.join(output_base_dir, "torchscript_int8")
    ptq_torchscript_path = trace_and_save_torchscript(
        ptq_model,
        inputs,
        ptq_folder,
        use_get_traceable=bool(args.use_get_traceable),
        trace_type=args.trace_type,
        opt_for_mobile=args.opt_for_mobile,
        model_attrs=model_attrs,
        save_for_lite_interpreter=args.save_for_lite_interpreter,
    )

    return ptq_torchscript_path


@ExportFactory.register("_dynamic_")
def export_to_torchscript_dynamic(
    args,
    task,
    model,
    inputs,
    output_base_dir,
    *,
    model_attrs=None,
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
    model_name = export_format[len("torchscript_"):]
    model = task.get_model_by_name(model_name, model)

    print(f"Converting to {model_name}...")
    output_dir = os.path.join(output_base_dir, export_format)
    torch_script_path = trace_and_save_torchscript(
        model,
        inputs,
        output_dir,
        use_get_traceable=bool(args.use_get_traceable),
        trace_type=args.trace_type,
        opt_for_mobile=args.opt_for_mobile,
        model_attrs=model_attrs,
        save_for_lite_interpreter=args.save_for_lite_interpreter,
    )
    return torch_script_path


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
):
    _import_tasks(args.task)
    task = task_factory.get(args.task, **args.task_args)
    model = task.get_model()
    if isinstance(model, tuple):
        model, model_attrs = model
    else:
        model_attrs = None
    model.eval()
    data_loader = task.get_dataloader()
    data_iter = iter(data_loader)

    first_batch = next(data_iter) if len(data_loader) > 0 else []
    with torch.no_grad():
        flops_utils.print_model_flops(model, first_batch)

    ret = {}
    for ef in export_formats:
        assert ef not in ret, f"Export format {ef} has already existed."
        try:
            export_func = (
                ExportFactory.get(ef)
                if ef in ExportFactory
                else ExportFactory.get("_dynamic_")
            )
            out_path = export_func(
                args,
                task,
                model,
                first_batch,
                output_dir,
                # NOTE: output model maybe difference if data_loader is used multiple times
                data_iter=data_iter,
                model_attrs=model_attrs,
                export_format=ef,
            )
            ret[ef] = out_path
        except Exception as e:
            logger.warning(f"Export format {ef} failed: {e}")
            if raise_if_failed:
                raise e

    return ret


def _get_task_info(args):
    ret = {"task": args.task, "task_args": args.task_args}
    return ret


def run_with_cmdline_args(args):
    return main(
        args,
        args.output_dir,
        export_formats=args.export_types,
        raise_if_failed=args.raise_if_failed,
    )


def run_with_cmdline_args_list(args_list=None):
    args = parse_args(args_list)
    return run_with_cmdline_args(args)


if __name__ == "__main__":
    run_with_cmdline_args_list()
