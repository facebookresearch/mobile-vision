#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import contextlib
import typing
import unittest.mock as mock

import torch
from torch import nn
from torch.quantization import DeQuantStub, QuantStub

from . import fuse_utils


class QuantModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()

    def forward(self, x):
        y = self.quant(x)
        y = self.model(y)
        y = self.dequant(y)
        return y


class TorchAddPact(nn.Module):
    def __init__(self):
        super().__init__()
        self.float_functional = nn.quantized.FloatFunctional()

    def forward(self, x, y):
        return self.float_functional.add(x, y)


class TorchAddScalarPact(nn.Module):
    def __init__(self):
        super().__init__()
        self.float_functional = nn.quantized.FloatFunctional()

    def forward(self, x, y):
        return self.float_functional.add_scalar(x, y)


class TorchMulPact(nn.Module):
    def __init__(self):
        super().__init__()
        self.float_functional = nn.quantized.FloatFunctional()

    def forward(self, x, y):
        return self.float_functional.mul(x, y)


class TorchMulScalarPact(nn.Module):
    def __init__(self):
        super().__init__()
        self.float_functional = nn.quantized.FloatFunctional()

    def forward(self, x, y):
        return self.float_functional.mul_scalar(x, y)


class TorchCatPact(nn.Module):
    def __init__(self):
        super().__init__()
        self.float_functional = nn.quantized.FloatFunctional()

    def forward(self, tensors, dim):
        return self.float_functional.cat(tensors, dim)


@contextlib.contextmanager
def mock_quant_ops(
    quant_op=None, calling_module="mobile_cv.arch.fbnet_v2.basic_blocks"
):
    """Context manager to swap operators

    The swap depends on the input variable quant_op. Calling_module should be
    considered when patching, default calling_module is fbb because this
    is often where the operators are created. However, there are cases where
    another module creates the operators (e.g., unit_tests)
    Input: (str) key indicating what ops to replace
           (str) where operators to be replaced were created
    Return: None
    """
    MAPPINGS = {
        "quant_add": (f"{calling_module}.TorchAdd", TorchAddPact),
        "quant_add_scalar": (
            f"{calling_module}.TorchAddScalar",
            TorchAddScalarPact,
        ),
        "quant_mul": (f"{calling_module}.TorchMultiply", TorchMulPact),
        "quant_mul_scalar": (
            f"{calling_module}.TorchMulScalar",
            TorchMulScalarPact,
        ),
        "quant_cat": (f"{calling_module}.TorchCat", TorchCatPact),
    }

    if quant_op in MAPPINGS:
        with mock.patch(
            MAPPINGS[quant_op][0], side_effect=MAPPINGS[quant_op][1]
        ):
            yield
    else:
        yield


@contextlib.contextmanager
def build_model_context():
    mock_ctx_managers = [
        mock_quant_ops(quant_op="quant_add"),
        mock_quant_ops(quant_op="quant_add_scalar"),
        mock_quant_ops(quant_op="quant_mul"),
        mock_quant_ops(quant_op="quant_mul_scalar"),
        mock_quant_ops(quant_op="quant_cat"),
    ]

    with contextlib.ExitStack() as stack:
        for mgr in mock_ctx_managers:
            stack.enter_context(mgr)
        yield


def quantize_model(
    model_builder: typing.Callable,
    inputs,
    add_quant_stub=True,
    quant_config=None,
):
    print("Building quantization compatiable model...")
    with build_model_context():
        model = model_builder()
    if add_quant_stub:
        model = QuantModel(model)

    print("Fusing bn...")
    model = fuse_utils.fuse_model(model)
    assert not fuse_utils.check_bn_exist(model), model

    model.qconfig = quant_config or torch.quantization.default_qconfig
    print(f"Quant config: {model.qconfig}")

    torch.quantization.prepare(model, inplace=True)
    print("Collecting stats...")
    model(inputs)
    quant_model = torch.quantization.convert(model, inplace=False)

    return quant_model
