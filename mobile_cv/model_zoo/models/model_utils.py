#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import os

import mobile_cv.arch.utils.jit_utils as ju
import mobile_cv.common.misc.iter_utils as iu
import numpy as np
import torch
from mobile_cv.arch.utils import fuse_utils, quantize_utils as qu
from mobile_cv.common import utils_io
from torch.utils.mobile_optimizer import generate_mobile_module_lints


path_manager = utils_io.get_path_manager()


def convert_torch_script(
    model, inputs, fuse_bn=True, verify_output=True, use_get_traceable=False
):
    assert isinstance(inputs, (tuple, list)), f"Invalid input types {inputs}"
    if verify_output:
        print("Run pytorch model")
        with torch.no_grad():
            output_before = model(*inputs)

    if fuse_bn:
        print("Fusing bn...")
        fused_model = fuse_utils.fuse_model(model)
        if fuse_utils.check_bn_exist(fused_model):
            print(f"WARNING: BN existed after fusing, {fused_model}")
    else:
        fused_model = copy.deepcopy(model)

    for x in fused_model.parameters():
        x.requires_grad = False

    if use_get_traceable:
        print("Get traceable model...")
        fused_model = ju.get_traceable_model(fused_model)

    print("Start tracing...")
    with torch.no_grad():
        traced_model = torch.jit.trace(fused_model, inputs, strict=False)
    # print(f"Traced model {traced_model}")
    # print(f"Traced model {traced_model.code}")

    # print("Optimizing traced model...")
    # traced_model = optimize_for_mobile(traced_model)
    print("Generating traced model lints...")
    print(generate_mobile_module_lints(traced_model))

    print("Run traced model")
    with torch.no_grad():
        outputs = traced_model(*inputs)

    if verify_output:
        paired_outputs = iu.create_pair(output_before, outputs)
        for x in iu.recursive_iterate(paired_outputs, iter_types=torch.Tensor):
            np.testing.assert_allclose(
                x.lhs.detach(), x.rhs.detach(), rtol=0, atol=1e-4
            )

    return traced_model, outputs


def save_model(output_dir, traced_model, data=None):
    if not path_manager.isdir(output_dir):
        path_manager.mkdirs(output_dir)

    out_file = os.path.join(output_dir, "model.jit")
    print(f"Saving traced model {out_file}")

    with path_manager.open(out_file, "wb") as fp:
        torch.jit.save(traced_model, fp)
    print(f"Model saved to {out_file}")

    if data is not None:
        input_file = os.path.join(output_dir, "data.pth")

        with path_manager.open(input_file, "wb") as fp:
            torch.save(data, fp)
        print(f"Data saved to {input_file}")

    return out_file


def convert_int8_jit(
    model,
    inputs,
    add_quant_stub=True,
    int8_backend=None,
    use_get_traceable=True,
):
    assert isinstance(inputs, list), f"Invalid inputs {inputs}"
    # torch.set_num_threads(1)
    old_engine = torch.backends.quantized.engine
    if int8_backend is not None:
        torch.backends.quantized.engine = int8_backend

    def _build():
        return copy.deepcopy(model)

    # qconfig = (
    #     torch.ao.quantization.get_default_qconfig(int8_backend)
    #     if int8_backend is not None
    #     else torch.ao.quantization.get_default_qconfig()
    # )
    qconfig = torch.ao.quantization.QConfig(
        activation=torch.ao.quantization.MinMaxObserver.with_args(reduce_range=False),
        weight=torch.ao.quantization.default_weight_observer,
    )
    quant_model = qu.quantize_model(
        _build, inputs, add_quant_stub=add_quant_stub, quant_config=qconfig
    )

    if use_get_traceable:
        quant_model = ju.get_traceable_model(quant_model)

    jit_model = torch.jit.trace(quant_model, inputs)
    jit_output = jit_model(*inputs)

    torch.backends.quantized.engine = old_engine

    return jit_model, jit_output
