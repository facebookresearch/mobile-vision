#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import typing
from contextlib import contextmanager
from typing import List

import mobile_cv.arch.utils.helper as hp
import mobile_cv.arch.utils.jit_utils as ju
import mobile_cv.common.misc.iter_utils as iu
import torch
import torch.nn as nn
from mobile_cv.arch.layers import NaiveSyncBatchNorm
from torch.ao.quantization.stubs import DeQuantStub, QuantStub

from . import fuse_utils


def get_backend_qconfig(backend):
    if backend == "default":
        qconfig = torch.ao.quantization.default_qconfig
    elif backend == "qnnpack_per_channel":
        qconfig = torch.ao.quantization.QConfig(
            activation=torch.ao.quantization.HistogramObserver.with_args(
                reduce_range=False
            ),
            weight=torch.ao.quantization.default_per_channel_weight_observer,
        )
    else:
        qconfig = torch.ao.quantization.get_default_qconfig(backend)

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
            self.model.qconfig = torch.ao.quantization.default_qconfig

    def fuse_bn(self):
        print("Fusing bn...")
        self.model = fuse_utils.fuse_model(self.model, inplace=True)
        bn_count = fuse_utils.count_bn_exist(self.model)
        if bn_count > 0:
            print(f"Warning: Found {bn_count} BatchNorms after fusing: {self.model}")
        return self

    def add_quant_stub(self):
        self.model = torch.ao.quantization.QuantWrapper(self.model)
        return self

    def set_quant_backend(self, backend="fbgemm"):
        self.model.qconfig = get_backend_qconfig(backend)
        return self

    def set_quant_config(self, quant_cfg):
        self.model.qconfig = quant_cfg
        return self

    def prepare(self):
        torch.ao.quantization.prepare(self.model, inplace=True)
        return self

    def calibrate_model(self, data_loader, num_batches=1):
        calibrate_model(self.model, data_loader, num_batches)
        return self

    def convert_model(self):
        quant_model = torch.ao.quantization.convert(self.model, inplace=True)
        return quant_model


class PostQuantizationGraph(object):
    """Graph Mode post quantization"""

    def __init__(self, model, copy_model=True):
        self.model = copy.deepcopy(model) if copy_model else model
        self.processed_model = None
        self.calibrate_func = None
        self.qconfig = torch.ao.quantization.default_qconfig
        if hasattr(self.model, "qconfig"):
            self.qconfig = self.model.qconfig

    def set_quant_backend(self, backend="fbgemm"):
        self.qconfig = get_backend_qconfig(backend)
        return self

    def set_quant_config(self, quant_cfg):
        self.qconfig = quant_cfg
        return self

    def trace(self, inputs, check_inputs=None, strict=True, use_get_traceable=False):
        if use_get_traceable:
            model = ju.get_traceable_model(self.model)
        else:
            model = self.model
        self.processed_model = torch.jit.trace(
            model, inputs, check_inputs=check_inputs, strict=strict
        )
        return self

    def script(self):
        self.processed_model = torch.jit.script(self.model)
        return self

    def set_calibrate(self, data_loader, num_batches=1):
        def _calibrate_func(model, _not_used):
            model.eval()
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

        self.calibrate_func = _calibrate_func
        return self

    def convert_model(self):
        assert self.processed_model is not None
        assert self.calibrate_func is not None
        ret = torch.ao.quantization.quantize_jit(
            self.processed_model,
            {"": self.qconfig},
            self.calibrate_func,
            ["_not_needed"],
        )
        return ret


class PostQuantizationFX(object):
    """Post quantization using FX"""

    def __init__(self, model, copy_model=True, qconfig=None):
        self.model = copy.deepcopy(model) if copy_model else model
        if qconfig is None:
            self.qconfig = torch.ao.quantization.default_qconfig

    def set_quant_backend(self, backend="fbgemm"):
        self.qconfig = get_backend_qconfig(backend)
        return self

    def set_quant_config(self, quant_cfg):
        self.qconfig = quant_cfg
        return self

    def prepare(self, example_inputs, qconfig_dict=None):
        if qconfig_dict is None:
            qconfig_dict = get_qconfig_dict(self.model, self.qconfig)
        if qconfig_dict is None:
            qconfig_dict = {"": self.qconfig}
        self._prepared_model = torch.ao.quantization.quantize_fx.prepare_fx(
            self.model,
            qconfig_dict=qconfig_dict,
            example_inputs=example_inputs,
        )
        return self

    def calibrate_model(self, data_loader, num_batches=1):
        assert hasattr(self, "_prepared_model"), "Call prepare() first"
        calibrate_model(self._prepared_model, data_loader, num_batches)
        return self

    def convert_model(self):
        assert hasattr(self, "_prepared_model"), "Call prepare() first"
        quant_model = torch.ao.quantization.quantize_fx.convert_fx(self._prepared_model)
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
        model = torch.ao.quantization.QuantWrapper(model)

    print("Fusing bn...")
    model = fuse_utils.fuse_model(model)
    assert not fuse_utils.check_bn_exist(model), model

    model.qconfig = quant_config or torch.ao.quantization.default_qconfig
    print(f"Quant config: {model.qconfig}")

    torch.ao.quantization.prepare(model, inplace=True)
    print("Collecting stats...")
    model(*inputs)
    quant_model = torch.ao.quantization.convert(model, inplace=False)

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


class QuantizableModule(nn.Module):
    """
    This class helps create quantize/dequantize stubs that are needed for eager
    mode quantization.
    """

    def __init__(self, eager_mode, qconfig=None, n_inputs=1, n_outputs=1, **kwargs):
        super(QuantizableModule, self).__init__(**kwargs)
        self.eager_mode = eager_mode
        if self.eager_mode:
            self.quant_stubs = QuantStubNested.FromCount(
                n_inputs, stub_func=QuantStub, stub_func_args=[qconfig]
            )
            self.dequant_stubs = QuantStubNested.FromCount(
                n_outputs, stub_func=DeQuantStub
            )

    @staticmethod
    def quant_dequant(bypass_kwargs: typing.Optional[typing.List] = None):
        """wrap the forward function, run quant/dequant stubs before/after it"""
        bypass_kwargs = bypass_kwargs or []

        def decorator(forward_func):
            def quant_forward_dequant(self, *args, **kwargs):
                assert isinstance(self, QuantizableModule)
                if self.eager_mode:
                    bypassed_kwargs = {x: kwargs.pop(x) for x in bypass_kwargs}
                    (q_args, q_kwargs) = self.quant_stubs((args, kwargs))
                    q_kwargs.update(bypassed_kwargs)
                    q_outputs = forward_func(self, *q_args, **q_kwargs)
                    return self.dequant_stubs(q_outputs)
                else:
                    return forward_func(self, *args, **kwargs)

            return quant_forward_dequant

        return decorator

    @staticmethod
    def dequant_quant(bypass_kwargs: typing.Optional[typing.List] = None):
        """wrap the forward function, run dequant/quant stubs before/after it"""
        bypass_kwargs = bypass_kwargs or []

        def decorator(forward_func):
            def dequant_forward_quant(self, *args, **kwargs):
                assert isinstance(self, QuantizableModule)
                if self.eager_mode:
                    bypassed_kwargs = {x: kwargs.pop(x) for x in bypass_kwargs}
                    (q_args, q_kwargs) = self.dequant_stubs((args, kwargs))
                    q_kwargs.update(bypassed_kwargs)
                    q_outputs = forward_func(self, *q_args, **q_kwargs)
                    return self.quant_stubs(q_outputs)
                else:
                    return forward_func(self, *args, **kwargs)

            return dequant_forward_quant

        return decorator


class QuantWrapper(QuantizableModule):
    def __init__(self, module, **kwargs):
        qconfig = module.qconfig if hasattr(module, "qconfig") else None
        super().__init__(eager_mode=True, qconfig=qconfig, **kwargs)
        self.module = module

    @QuantizableModule.quant_dequant()
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


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


class NonQuantWrapper(QuantizableModule):
    def __init__(self, module):
        super().__init__(eager_mode=True)
        self.module = module

    @QuantizableModule.dequant_quant()
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def wrap_non_quant_group_norm(module):
    if isinstance(module, nn.GroupNorm):
        return NonQuantWrapper(module)
    for name, child in module.named_children():
        module._modules[name] = wrap_non_quant_group_norm(child)
    return module


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


def _add_prefix_qconfig_dict(qconfig_dict, prefix):
    ret = {"module_name": []}

    if "" not in qconfig_dict:
        ret["module_name"].append((prefix, None))
    else:
        ret["module_name"].append((prefix, qconfig_dict[""]))

    for name, item in qconfig_dict.items():
        if name == "":
            pass
        elif name == "module_name":
            for sub_items in item:
                new_item = (prefix + "." + sub_items[0], sub_items[1])
                ret["module_name"].append(new_item)
        else:
            print(f"Unsupported type for qconfig_dict {name}: {item}")

    return ret


def _merge_qconfig_dict(base_dict, dict_to_merge, prefix):
    ret = copy.deepcopy(base_dict)
    dict_to_merge = _add_prefix_qconfig_dict(dict_to_merge, prefix)
    hp.update_dict_merge_list(ret, dict_to_merge)
    return ret


def get_qconfig_dict_sub_modules(
    module: torch.nn.Module, sub_module_names: List[str], qconfig
):
    """
    Get a qconfig_dict that collects all qconfig_dicts from specified sub modules
    """
    ret = {}
    for name in sub_module_names:
        sub_module = getattr(module, name)
        sub_qd = get_qconfig_dict(sub_module, qconfig)
        if sub_qd is not None:
            ret = _merge_qconfig_dict(ret, sub_qd, prefix=name)
    if not ret:
        ret = None

    return ret


def get_qconfig_dict_default(module: torch.nn.Module, qconfig):
    """
    Get a qconfig_dict that collects all qconfig_dicts from all sub modules,
    and set the global default to qconfig.
    """
    ret = get_qconfig_dict_sub_modules(
        module, [x[0] for x in module.named_children()], qconfig
    )
    if ret:
        ret[""] = qconfig

    return ret


def get_qconfig_dict(module: torch.nn.Module, qconfig):
    """Get qconfig_dict recursively from the module, used for fx quantization
    (prepare_fx/prepare_qat_fx)
    """
    if hasattr(module, "get_qconfig_dict"):
        qd = module.get_qconfig_dict(qconfig)
    else:
        qd = get_qconfig_dict_default(module, qconfig)

    return qd
