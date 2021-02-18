#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import typing

import mobile_cv.arch.fbnet_v2.basic_blocks as bb
import mobile_cv.common.misc.registry as registry
import torch
import torch.nn as nn
from mobile_cv.arch.layers.batch_norm import (
    NaiveSyncBatchNorm,
    NaiveSyncBatchNorm1d,
    NaiveSyncBatchNorm3d,
)
from torch.quantization.fuse_modules import (
    fuse_conv_bn,
    fuse_conv_bn_relu,
    fuse_known_modules,
)


# Registry to get the names for fusing the supported module
# returns the list of list for the sub module to fuse
# func(
#    module: torch.nn.Module,
#    supported_fusing_types: Dict[str, List[torch.nn.Module]]
# ) -> List[List[str]]
FUSE_LIST_GETTER = registry.Registry("fuse_list_getter")


CONV_BN_RELU_SUPPORTED_FUSING_TYPES = {
    "conv": [nn.Conv1d, nn.Conv2d, nn.Conv3d],
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


# TODO: Is this the same as fuse_known_modules? should we just refactor this
# using `additioanl_fuser_method_mapping`?
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
        ret = torch.quantization.fuse_modules(
            ret, fuse_names, inplace=True, fuser_func=fuse_more_modules
        )
    return ret


def fuse_model_inplace(model: nn.Module):
    model = fuse_convbnrelu(model, inplace=True)
    children = {}
    for name, child in model.named_children():
        children[name] = fuse_model_inplace(child)
    return model


def fuse_model(model: nn.Module, inplace=False):
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
