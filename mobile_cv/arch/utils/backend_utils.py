#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from mobile_cv.common.misc import iter_utils as iu


def move_to_device(data, device: str):
    """Move data to the given device, data could be a nested dict/list"""
    diter = iu.recursive_iterate(data, iter_types=torch.Tensor)
    for cur in diter:
        diter.send(cur.to(device))

    return diter.value
