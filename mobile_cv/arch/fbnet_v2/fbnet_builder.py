#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This file is for backward compatiblity.
Use mobile_cv.arch.builder.meta_builder instead.
"""

from mobile_cv.arch.builder.meta_builder import (
    add_block_kwargs,
    BLOCK_KWARGS_NAME,
    count_stride_each_block,
    count_strides,
    expand_repeats,
    FBNetBuilder,
    flatten_stages,
    get_block_kwargs,
    get_num_blocks_in_stage,
    get_num_stages,
    get_stages_dim_out,
    MetaBuilder,
    parse_block_cfg,
    parse_block_cfgs,
    PRIMITIVES,
    unify_arch_def,
    unify_arch_def_blocks,
    update_with_block_kwargs,
)


__all__ = [
    "add_block_kwargs",
    "BLOCK_KWARGS_NAME",
    "count_stride_each_block",
    "count_strides",
    "expand_repeats",
    "FBNetBuilder",
    "flatten_stages",
    "get_block_kwargs",
    "get_num_blocks_in_stage",
    "get_num_stages",
    "get_stages_dim_out",
    "MetaBuilder",
    "parse_block_cfg",
    "parse_block_cfgs",
    "PRIMITIVES",
    "unify_arch_def",
    "unify_arch_def_blocks",
    "update_with_block_kwargs",
]
