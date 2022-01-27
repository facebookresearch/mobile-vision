#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import typing
from typing import Callable, Dict, Type, Tuple

import mobile_cv.arch.fbnet_v2.basic_blocks as bb
import mobile_cv.arch.layers
import mobile_cv.common.misc.registry as registry
import torch
import torch.nn as nn
from mobile_cv.arch.fbnet_v2.spade import _get_fuser_name_convbnrelu_with_tuple_left
from mobile_cv.arch.layers.batch_norm import (
    NaiveSyncBatchNorm,
    NaiveSyncBatchNorm1d,
    NaiveSyncBatchNorm3d,
)


TORCH_VERSION: Tuple[int, ...] = tuple(int(x) for x in torch.__version__.split(".")[:2])
if TORCH_VERSION > (1, 10):
    from torch.ao.quantization import fuse_modules
    from torch.ao.quantization.fuse_modules import (
        fuse_conv_bn,
        fuse_conv_bn_relu,
        fuse_known_modules,
    )
    from torch.ao.quantization.quantize_fx import _fuse_fx
else:
    from torch.quantization import fuse_modules
    from torch.quantization.fuse_modules import (
        fuse_conv_bn,
        fuse_conv_bn_relu,
        fuse_known_modules,
    )
    from torch.quantization.quantize_fx import _fuse_fx


# Registry to get the names for fusing the supported module
# returns the list of list for the sub module to fuse
# func(
#    module: torch.nn.Module,
#    supported_fusing_types: Dict[str, List[torch.nn.Module]]
# ) -> List[List[str]]
FUSE_LIST_GETTER = registry.Registry("fuse_list_getter")


CONV_BN_RELU_SUPPORTED_FUSING_TYPES = {
    "conv": [nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear],
    "bn": [
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        NaiveSyncBatchNorm,
        NaiveSyncBatchNorm1d,
        NaiveSyncBatchNorm3d,
    ],
    "relu": [nn.ReLU],
}


def _cast_func(model):
    return model.cast(model)


SWAPPING_MODULES = {
    mobile_cv.arch.layers.Conv2d: _cast_func,
    NaiveSyncBatchNorm: _cast_func,
    NaiveSyncBatchNorm1d: _cast_func,
    NaiveSyncBatchNorm3d: _cast_func,
}


def swap_modules_inplace(
    model: nn.Module,
    # pyre-fixme[9]: module_mapping has type `Dict[Type[nn.Module],
    #  typing.Callable[[nn.Module], nn.Module]]`; used as
    #  `Dict[Type[typing.Union[NaiveSyncBatchNorm, NaiveSyncBatchNorm1d,
    #  NaiveSyncBatchNorm3d, mobile_cv.arch.layers.misc.Conv2d]], typing.Any]`.
    module_mapping: Dict[
        Type[nn.Module], Callable[[nn.Module], nn.Module]
    ] = SWAPPING_MODULES,
):
    is_training = model.training
    if type(model) in module_mapping:
        model = module_mapping[type(model)](model)

    # pyre-ignore Undefined attribute [16]: `torch.nn.Module` has no attribute `named_children`.
    for name, module in model.named_children():
        module = swap_modules_inplace(module, module_mapping)
        setattr(model, name, module)

    model.train(is_training)

    return model


def swap_modules(
    model: nn.Module,
    # pyre-fixme[9]: module_mapping has type `Dict[Type[nn.Module],
    #  typing.Callable[[nn.Module], nn.Module]]`; used as
    #  `Dict[Type[typing.Union[NaiveSyncBatchNorm, NaiveSyncBatchNorm1d,
    #  NaiveSyncBatchNorm3d, mobile_cv.arch.layers.misc.Conv2d]], typing.Any]`.
    module_mapping: Dict[
        Type[nn.Module], Callable[[nn.Module], nn.Module]
    ] = SWAPPING_MODULES,
):
    model = copy.deepcopy(model)
    return swap_modules_inplace(model, module_mapping)


# TODO: Is this the same as fuse_known_modules? should we just refactor this using `additioanl_fuser_method_mapping`?
def fuse_more_modules(
    mod_list: typing.List[nn.Module], additional_fuser_method_mapping=None
):
    r"""Returns a list of modules that fuses the operations specified
     in the input module list.

    Supports NaiveSyncBatchNorm, will call `fuse_known_modules` to fuse the rest
    Only support fusing in inference time

    For these sequences, the first element in the output module list performs
    the fused operation. The rest of the elements are set to nn.Identity()
    """

    assert len(mod_list) > 0

    OP_LIST_TO_FUSER_METHOD = {
        (torch.nn.Conv1d, NaiveSyncBatchNorm1d): fuse_conv_bn,
        (
            torch.nn.Conv1d,
            NaiveSyncBatchNorm1d,
            torch.nn.ReLU,
        ): fuse_conv_bn_relu,
        (torch.nn.Conv1d, nn.SyncBatchNorm): fuse_conv_bn,
        (torch.nn.Conv1d, nn.SyncBatchNorm, torch.nn.ReLU): fuse_conv_bn_relu,
        (torch.nn.Conv2d, NaiveSyncBatchNorm): fuse_conv_bn,
        (torch.nn.Conv2d, NaiveSyncBatchNorm, torch.nn.ReLU): fuse_conv_bn_relu,
        (torch.nn.Conv2d, nn.SyncBatchNorm): fuse_conv_bn,
        (torch.nn.Conv2d, nn.SyncBatchNorm, torch.nn.ReLU): fuse_conv_bn_relu,
        (torch.nn.Conv3d, NaiveSyncBatchNorm3d): fuse_conv_bn,
        (
            torch.nn.Conv3d,
            NaiveSyncBatchNorm3d,
            torch.nn.ReLU,
        ): fuse_conv_bn_relu,
        (torch.nn.Conv3d, nn.SyncBatchNorm): fuse_conv_bn,
        (torch.nn.Conv3d, nn.SyncBatchNorm, torch.nn.ReLU): fuse_conv_bn_relu,
        (NaiveSyncBatchNorm, torch.nn.ReLU): torch.nn.intrinsic.BNReLU2d,
        (NaiveSyncBatchNorm3d, torch.nn.ReLU): torch.nn.intrinsic.BNReLU3d,
    }

    types = tuple(type(m) for m in mod_list)
    # pyre-fixme[6]: Expected `Union[typing.Tuple[], typing.Tuple[]]` for 1st param
    #  but got `Tuple[Type[nn.Module], ...]`.
    fuser_method = OP_LIST_TO_FUSER_METHOD.get(types, None)
    # Currently only support fusing for evaluation mode
    if fuser_method is None or mod_list[0].training:
        return fuse_known_modules(mod_list)

    new_mod = [None] * len(mod_list)
    new_mod[0] = fuser_method(*mod_list)

    for i in range(1, len(mod_list)):
        new_mod[i] = torch.nn.Identity()
        new_mod[i].training = mod_list[0].training

    return new_mod


@FUSE_LIST_GETTER.register(bb.ConvBNRelu)
@FUSE_LIST_GETTER.register(bb.ConvNormAct)
def _get_fuser_name_cbr(
    module: torch.nn.Module,
    supported_types: typing.Dict[str, typing.List[torch.nn.Module]],
):
    assert isinstance(module, (bb.ConvBNRelu, bb.ConvNormAct))

    # handle convbnrelu with tuple_left/spadenorm differently
    ret = _get_fuser_name_convbnrelu_with_tuple_left(module, supported_types)
    if ret is not None:
        return ret

    MODULE_NAME_MAPPING = {
        bb.ConvBNRelu: {"conv": "conv", "bn": "bn", "relu": "relu"},
        bb.ConvNormAct: {"conv": "conv", "bn": "norm", "relu": "act"},
    }

    op_names_map = MODULE_NAME_MAPPING[type(module)]
    ret = []
    for op_name in ["conv", "bn", "relu"]:
        op_name_real = op_names_map[op_name]
        op = getattr(module, op_name_real, None)
        if type(op) in supported_types[op_name]:
            ret.append(op_name_real)

    # do not need to fuse if there is only one op
    if len(ret) <= 1:
        return []
    return [ret]


def fuse_convbnrelu(module, inplace=False):
    if not isinstance(module, tuple(FUSE_LIST_GETTER.keys())):
        return module

    ret = module if inplace else copy.deepcopy(module)

    fuse_names = FUSE_LIST_GETTER[type(module)](
        module, CONV_BN_RELU_SUPPORTED_FUSING_TYPES
    )

    if len(fuse_names) > 0:
        ret = fuse_modules(ret, fuse_names, inplace=True, fuser_func=fuse_more_modules)
    return ret


def fuse_model_inplace(model: nn.Module):
    model = fuse_convbnrelu(model, inplace=True)
    children = {}
    for name, child in model.named_children():
        children[name] = fuse_model_inplace(child)
    return model


def fuse_model(model: nn.Module, inplace=False, use_fx=False):
    if use_fx:
        assert inplace is False
        return fuse_model_fx(model)
    if not inplace:
        model = copy.deepcopy(model)
    return fuse_model_inplace(model)


def check_bn_exist(model):
    for x in model.modules():
        if isinstance(x, nn.BatchNorm2d.__base__):
            return True
    return False


def count_bn_exist(model):
    ret = 0
    for x in model.modules():
        if isinstance(x, nn.BatchNorm2d.__base__):
            ret += 1
    return ret


def _find_input_nodes(cur_node, nodes_to_remove):
    """Find all nodes that produces outputs as the input for cur_node"""
    all_arg_users_in_to_remove = True
    for arg in cur_node.all_input_nodes:
        # Check uses, if all uses are to be removed, then
        # this node can also be removed.
        for arg_user_node in arg.users.keys():
            if arg_user_node not in nodes_to_remove:
                all_arg_users_in_to_remove = False
    if all_arg_users_in_to_remove:
        for arg in cur_node.all_input_nodes:
            nodes_to_remove.add(arg)
        for arg in cur_node.all_input_nodes:
            _find_input_nodes(arg, nodes_to_remove)
    else:
        return


def _remove_asserts(mod: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # remove all assert nodes and the nodes that used by them
    graph = mod.graph
    nodes_to_remove = set()
    for node in graph.nodes:
        if node.op == "call_function" and node.target == torch._assert:
            nodes_to_remove.add(node)
            _find_input_nodes(node, nodes_to_remove)
    # node: need to delete in reverse order to prevent dangling nodes
    for node in reversed(graph.nodes):
        if node in nodes_to_remove:
            graph.erase_node(node)
    return torch.fx.GraphModule(mod, graph)


def _fuse_model_fx_single(model: torch.nn.Module) -> torch.nn.Module:
    model = torch.fx.symbolic_trace(model)
    model = _remove_asserts(model)
    model = _fuse_fx(model)
    return model


def _fuse_model_fx_recursive(model: torch.nn.Module):
    traceable = True
    try:
        model = _fuse_model_fx_single(model)
    except Exception as e:  # noqa
        # print(f"Error in tracing {model}: {e}")
        traceable = False

    if not traceable:
        # pyre-ignore Undefined attribute [16]: `torch.nn.Module` has no attribute `named_children`.
        for name, module in model.named_children():
            module = _fuse_model_fx_recursive(module)
            setattr(model, name, module)

    return model


def fuse_model_fx(model: torch.nn.Module):
    """
    Fusing model using fx, the fusing will run recursively if some of the modules
    are not symbolic traceable
    Note the returned model is a graph module
    """
    model = swap_modules(model)
    return _fuse_model_fx_recursive(model)
