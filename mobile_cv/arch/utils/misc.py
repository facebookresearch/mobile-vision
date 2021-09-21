#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch


def drop_connect_batch(inputs, drop_prob, training):
    """Randomly drop batch during training"""
    assert drop_prob < 1.0, f"Invalid drop_prob {drop_prob}"
    if not training or drop_prob == 0.0:
        return inputs
    keep_prob = 1 - drop_prob
    shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
    random_tensor = (
        torch.rand(shape, dtype=inputs.dtype, device=inputs.device) + keep_prob
    )
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    # output = inputs * binary_tensor
    return output


def add_dropout(dropout_ratio):
    if dropout_ratio > 0:
        return torch.nn.Dropout(dropout_ratio)
    return None


def add_drop_connect_args(mbuilder, block_cfgs, drop_rate, start_idx=0, total_idx=None):
    if drop_rate is None:
        return
    assert isinstance(block_cfgs, list)
    if total_idx is None:
        total_idx = len(block_cfgs)
    for idx, block in enumerate(block_cfgs):
        cur_drop_rate = drop_rate * (idx + start_idx) / total_idx
        args = {"drop_connect_rate": cur_drop_rate}
        mbuilder.add_block_kwargs(block, args)
