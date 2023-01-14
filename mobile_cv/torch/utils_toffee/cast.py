#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import torch


def to_device(t, device_str):
    """
    This function is a replacement of .to(another_device) such that it allows the
    casting to be traced properly by explicitly calling the underlying copy ops.
    It also avoids introducing unncessary op when casting to the same device.
    """
    src = t.device
    dst = torch.device(device_str)

    if src == dst:
        return t
    elif src.type == "cuda" and dst.type == "cpu":
        return torch.ops._caffe2.CopyGPUToCPU(t)
    elif src.type == "cpu" and dst.type == "cuda":
        return torch.ops._caffe2.CopyCPUToGPU(t)
    else:
        raise RuntimeError(
            "Can't cast tensor from device {} to device {}".format(src, dst)
        )
