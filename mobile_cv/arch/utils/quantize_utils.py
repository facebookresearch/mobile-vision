#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import typing
from contextlib import contextmanager

import mobile_cv.common.misc.iter_utils as iu
import torch
import torch.nn as nn
from mobile_cv.arch.layers import NaiveSyncBatchNorm
from torch.quantization.stubs import DeQuantStub, QuantStub

from . import fuse_utils


def get_backend_qconfig(backend):
    if backend == "default":
        qconfig = torch.quantization.default_qconfig
    elif backend == "qnnpack_per_channel":
        qconfig = torch.quantization.QConfig(
            activation=torch.quantization.HistogramObserver.with_args(
                reduce_range=False
            ),
            weight=torch.quantization.default_per_channel_weight_observer,
        )
    else:
        qconfig = torch.quantization.get_default_qconfig(backend)

    return qconfig


def map_backend_name(backend_config_name):
    """Map backend config name used in `get_backend_qconfig` to the backend name
    in pytoch
    """
    if backend_config_name in ["qnnpack", "qnnpack_per_channel"]:
        return "qnnpack"
    if backend_config_name in ["default", "fbgemm"]:
        return "fbgemm"
    raise Exception(f"Invalid backend config name {backend_config_name}")


@contextmanager
def use_backends_quantized_engine(backend_config_name):
    backend_name = map_backend_name(backend_config_name)
    old_backend = torch.backends.quantized.engine
    torch.backends.quantized.engine = backend_name
    yield
    torch.backends.quantized.engine = old_backend


def calibrate_model(model, data_loader, num_batches=1):
    assert not model.training
    with torch.no_grad():
        for idx, data in enumerate(data_loader):
            print(f"Collecting stats {idx}/{num_batches}...")
            model(*data)
            if idx + 1 == num_batches:
                break
        else:
            print(
                f"Only ran {idx} bathces data for calibration, expected {num_batches}"
            )


class PostQuantization(object):
    """Post quantization"""

    def __init__(self, model, copy_model=True):
        self.model = copy.deepcopy(model) if copy_model else model
        if not hasattr(self.model, "qconfig"):
            self.model.qconfig = torch.quantization.default_qconfig

    def fuse_bn(self):
        print("Fusing bn...")
        self.model = fuse_utils.fuse_model(self.model, inplace=True)
        assert not fuse_utils.check_bn_exist(self.model), self.model
        return self

    def add_quant_stub(self):
        self.model = torch.quantization.QuantWrapper(self.model)
        return self

    def set_quant_backend(self, backend="fbgemm"):
        self.model.qconfig = get_backend_qconfig(backend)
        return self

    def set_quant_config(self, quant_cfg):
        self.model.qconfig = quant_cfg
        return self

    def prepare(self):
        torch.quantization.prepare(self.model, inplace=True)
        return self

    def calibrate_model(self, data_loader, num_batches=1):
        for idx, data in enumerate(data_loader):
            print(f"Collecting stats {idx}/{num_batches}...")
            self.model(*data)
            if idx + 1 == num_batches:
                break
        else:
            print(
                f"Only ran {idx} bathces data for calibration, expected {num_batches}"
            )
        return self

    def convert_model(self):
        quant_model = torch.quantization.convert(self.model, inplace=True)
        return quant_model


def quantize_model(
    model_builder: typing.Callable,
    inputs: typing.List[typing.Any],
    add_quant_stub=True,
    quant_config=None,
):
    assert isinstance(inputs, (list, tuple)), f"Invalid inputs type {inputs}"
    print("Building quantization compatiable model...")

    model = model_builder()
    model.eval()
    if add_quant_stub:
        model = torch.quantization.QuantWrapper(model)

    print("Fusing bn...")
    model = fuse_utils.fuse_model(model)
    assert not fuse_utils.check_bn_exist(model), model

    model.qconfig = quant_config or torch.quantization.default_qconfig
    print(f"Quant config: {model.qconfig}")

    torch.quantization.prepare(model, inplace=True)
    print("Collecting stats...")
    model(*inputs)
    quant_model = torch.quantization.convert(model, inplace=False)

    return quant_model


class QuantStubNested(nn.Module):
    """Extension of QuantStub/DeQuantStub that supports any number of predefined
    nested input as a dict or list.
    The number of stubs needs to match the number of inputs after expanding
    the nested input as a list, and the order of the nested inputs could not
    be changed between forward() calls.
    """

    def __init__(self, stub_list):
        super().__init__()
        self.stubs = nn.ModuleList(x or torch.Identity() for x in stub_list)

    @classmethod
    def FromCount(cls, num_stubs, stub_func=QuantStub, stub_func_args=None):
        stubs = [stub_func(*(stub_func_args or ())) for _ in range(num_stubs)]
        return cls(stubs)

    def forward(self, x):
        assert len(list(iu.recursive_iterate(x))) == len(self.stubs)
        data_iter = iu.recursive_iterate(x, wait_on_send=True)
        for data, stub in zip(data_iter, self.stubs):
            data_iter.send(stub(data))
        return data_iter.value


def wrap_quant_subclass(module, n_inputs, n_outputs, wrapped_method_name="forward"):
    """Create a subclass of type(module) but quantize the input and dequant
    the output
    Similar to `QuantWrapper` but the return object is a subclass of
    type(module) so other functions in it could be called directly.
    """
    ModuleType = type(module)

    def wrapped_method(self, *args, **kwargs):
        (q_args, q_kwargs) = self.quant_stubs((args, kwargs))
        q_outputs = getattr(ModuleType, wrapped_method_name)(self, *q_args, **q_kwargs)
        outputs = self.dequant_stubs(q_outputs)
        return outputs

    class QuantWrapSubClass(ModuleType):
        def __init__(self, module):
            assert isinstance(module, ModuleType)
            # initialize the parent by copying the dict directly
            self.__dict__ = module.__dict__.copy()
            self.quant_stubs = QuantStubNested.FromCount(
                n_inputs, stub_func=QuantStub, stub_func_args=[None]
            )
            self.dequant_stubs = QuantStubNested.FromCount(
                n_outputs, stub_func=DeQuantStub
            )

    setattr(QuantWrapSubClass, wrapped_method_name, wrapped_method)

    ret = QuantWrapSubClass(module)
    return ret


def _swap_bn(module, bn_source_cls, bn_target_cls):
    for name, child in module.named_children():
        if isinstance(child, bn_source_cls):
            new_bn = bn_target_cls(
                child.num_features,
                child.eps,
                child.momentum,
                child.affine,
                child.track_running_stats,
            )
            new_bn.weight = child.weight
            if child.affine:
                new_bn.bias = child.bias
            new_bn.running_mean = child.running_mean
            new_bn.running_var = child.running_var
            new_bn.training = child.training
            module._modules[name] = new_bn
            del child
        else:
            _swap_bn(child, bn_source_cls, bn_target_cls)


def swap_syncbn_to_bn(module):
    """
    Replaces all instances of NaiveSyncBatchNorm in module with BatchNorm2d.
    This is needed for FX graph mode quantization, as the swaps and fusions
    assume the OSS BatchNorm2d.
    Note: this function is recursive, and it modifies module inplace.
    """
    _swap_bn(module, NaiveSyncBatchNorm, torch.nn.BatchNorm2d)


def swap_bn_to_syncbn(module):
    """
    Replaces all instances of BatchNorm2d  in module with NaiveSyncBatchNorm.
    Note: this function is recursive, and it modifies module inplace.
    """
    _swap_bn(module, torch.nn.BatchNorm2d, NaiveSyncBatchNorm)
