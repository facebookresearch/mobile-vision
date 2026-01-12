#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import argparse
import itertools
import json
import logging
import os
import shutil
from typing import Any, Dict, Iterable, Optional, Tuple

import mobile_cv.arch.utils.fuse_utils as fuse_utils
import torch
from mobile_cv.arch.utils import quantize_utils
from mobile_cv.model_zoo.tasks.task_base import TaskBase

logger = logging.getLogger(__name__)


def copy_file(src, dst, skip_exists=True):
    if os.path.exists(src):
        if skip_exists and os.path.exists(dst):
            return
        shutil.copy2(src, dst)
    else:
        print(f"Warning: {src} does not exist!")


def save_json(file, data):
    with open(file, "w") as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True)


def get_model_attributes(model: torch.nn.Module) -> Optional[Dict[str, Any]]:
    model_attrs = None
    if hasattr(model, "attrs"):
        model_attrs = model.attrs
        if model_attrs is not None:
            assert isinstance(
                model_attrs, dict
            ), f"Invalid model attributes type: {model_attrs}"
    return model_attrs


def get_ptq_model(
    args: argparse.Namespace,
    task: TaskBase,
    model: torch.nn.Module,
    inputs: Iterable[Any],
    data_iter: Iterable[Any],
) -> Tuple[torch.nn.Module, Optional[Dict[str, Any]]]:
    cur_loader = itertools.chain([inputs], data_iter)

    example_inputs = tuple(inputs)
    if hasattr(task, "get_quantized_model"):
        logger.info("calling get quantized model")
        ptq_model = task.get_quantized_model(model, cur_loader)
        model_attrs = get_model_attributes(ptq_model)
        logger.info("after calling get quantized model")
    elif args.use_graph_mode_quant:
        logger.info(
            f"Post quantization using {args.post_quant_backend} backend fx mode..."
        )
        model_attrs = get_model_attributes(model)
        # swap models that fx could not support
        model = fuse_utils.swap_modules(model)
        quant = quantize_utils.PostQuantizationFX(model)
        ptq_model = (
            quant.set_quant_backend(args.post_quant_backend)
            .prepare(example_inputs=example_inputs)
            .calibrate_model(cur_loader, 1)
            .convert_model()
        )
        logger.info("after calling callback")
    else:
        logger.info(f"Post quantization using {args.post_quant_backend} backend...")
        qa_model = task.get_quantizable_model(model)
        model_attrs = get_model_attributes(qa_model)
        post_quant = quantize_utils.PostQuantization(qa_model)
        post_quant.fuse_bn().set_quant_backend(args.post_quant_backend)
        ptq_model = post_quant.prepare().calibrate_model(cur_loader, 1).convert_model()

    return ptq_model, model_attrs
