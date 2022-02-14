#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging

import torch
import torch.nn as nn
from mobile_cv.arch.builder import meta_builder as mbuilder
from mobile_cv.arch.fbnet_v2 import (
    fbnet_modeldef_cls as modeldef,
)
from mobile_cv.arch.utils.helper import format_dict_expanding_list_values

logger = logging.getLogger(__name__)


class MultiIONetBackbone(nn.Module):
    # TODO: clean up the __init__ if we want to reuse MultiIONetBackbone without calling
    # via build_multi_io_net_backbone
    def __init__(
        self,
        num_paths,
        num_stages,
        paths,
        fusions,
        vertical_connect_stage0=False,
        header=None,
    ):
        super().__init__()
        self.num_paths = num_paths
        self.num_stages = num_stages
        self.paths = paths
        self.fusions = fusions
        self.vertical_connect_stage0 = vertical_connect_stage0
        self.header = header  # header for classification

    def forward(self, inputs):
        # first stage
        if self.vertical_connect_stage0:
            outputs = []
            for path_idx in range(self.num_paths):
                outputs.append(self.paths[f"path{path_idx}_s{0}"](inputs))
                inputs = torch.clone(outputs[path_idx])
        else:
            outputs = [
                self.paths[f"path{path_idx}_s{0}"](inputs)
                for path_idx in range(self.num_paths)
            ]

        # feature fusion -> block
        for stage_idx in range(1, self.num_stages):
            outputs = self.fusions[f"fusions_s{stage_idx}"](outputs)
            outputs = [
                self.paths[f"path{path_idx}_s{stage_idx}"](outputs[path_idx])
                for path_idx in range(self.num_paths)
            ]

        # header
        if self.header is not None:
            output = self.header(outputs)
            return output  # single tensor
        else:
            return outputs  # list of tensors


def get_arch_def(arch_name, arch_factory):
    assert arch_name in arch_factory
    arch_def = arch_factory[arch_name]
    return arch_def


def create_builder(arch_name):
    arch_def = get_arch_def(arch_name, modeldef.MODEL_ARCH)

    logger.info(
        'Using arch_def for ARCH "{}" (without scaling):\n{}'.format(
            arch_name, format_dict_expanding_list_values(arch_def)
        )
    )

    builder = mbuilder.FBNetBuilder()
    builder.add_basic_args(**arch_def.get("basic_args", {}))

    return builder, arch_def


def get_num_paths_stages(arch_def):
    num_paths = 0
    num_stages = 0
    for k in arch_def:
        if "path" in k:  # path defination
            num_paths = max(int(k.split("_")[0][4:]) + 1, num_paths)
            num_stages = max(int(k.split("_")[1][1:]) + 1, num_stages)
    return num_paths, num_stages


def build_multi_io_net_backbone(arch_name, build_cls_header=True):
    builder, arch_def = create_builder(arch_name)
    num_paths, num_stages = get_num_paths_stages(arch_def)
    logger.info("{}: {} paths, {} stages".format(arch_name, num_paths, num_stages))

    # FIXME: this check is not general
    vertical_connect_stage0 = "multitask_nas_v2" in arch_name

    # unify architecture defination
    unify_names = [f"path{path_idx}_s{0}" for path_idx in range(num_paths)]
    for stage_idx in range(1, num_stages):
        unify_names.append(f"fusions_s{stage_idx}")
        for path_idx in range(num_paths):
            unify_names.append(f"path{path_idx}_s{stage_idx}")
    unify_names.append("header")

    arch_def = mbuilder.unify_arch_def(arch_def, unify_names)
    unified_arch_def = arch_def

    # build `nn.Module` from `blocks`
    paths = torch.nn.ModuleDict({})
    fusions = torch.nn.ModuleDict({})

    in_channels = [3] * num_paths
    for path_idx in range(num_paths):
        block_name = f"path{path_idx}_s{0}"
        dim_in = in_channels[path_idx]
        paths[block_name] = builder.build_blocks(
            unified_arch_def[block_name], dim_in=dim_in
        )
        in_channels[path_idx] = mbuilder.get_stages_dim_out(
            unified_arch_def[block_name]
        )[-1]
        if vertical_connect_stage0:
            if path_idx + 1 < num_paths:
                in_channels[path_idx + 1] = in_channels[path_idx]

    for stage_idx in range(1, num_stages):
        block_name = f"fusions_s{stage_idx}"
        dim_in = in_channels
        fusions[block_name] = builder.build_blocks(
            unified_arch_def[block_name], dim_in=dim_in
        )
        in_channels = mbuilder.get_stages_dim_out(unified_arch_def[block_name])[-1]

        for path_idx in range(num_paths):
            block_name = f"path{path_idx}_s{stage_idx}"
            dim_in = in_channels[path_idx]
            paths[block_name] = builder.build_blocks(
                unified_arch_def[block_name], dim_in=dim_in
            )
            in_channels[path_idx] = mbuilder.get_stages_dim_out(
                unified_arch_def[block_name]
            )[-1]

    if build_cls_header:
        block_name = "header"
        dim_in = in_channels
        header = builder.build_blocks(unified_arch_def[block_name], dim_in=dim_in)
    else:
        header = None

    return MultiIONetBackbone(
        num_paths,
        num_stages,
        paths,
        fusions,
        vertical_connect_stage0=vertical_connect_stage0,
        header=header,
    )
