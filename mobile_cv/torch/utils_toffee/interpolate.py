#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import functools
import logging
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
from mobile_cv.torch.utils_toffee.cast import to_device
from torch.nn.functional import interpolate as interp


logger = logging.getLogger(__name__)


# Note: borrowed from vision/detection/fair/detectron/detectron/modeling/detector.py
def BilinearInterpolation(tensor_in, up_scale):
    assert up_scale % 2 == 0, "Scale should be even"

    def upsample_filt(size):
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5

        og = np.ogrid[:size, :size]
        return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

    kernel_size = int(up_scale) * 2
    bil_filt = upsample_filt(kernel_size)

    dim = int(tensor_in.shape[1])
    kernel = np.zeros((dim, dim, kernel_size, kernel_size), dtype=np.float32)
    kernel[range(dim), range(dim), :, :] = bil_filt

    tensor_out = F.conv_transpose2d(
        tensor_in,
        weight=to_device(torch.Tensor(kernel), tensor_in.device),
        bias=None,
        stride=int(up_scale),
        padding=int(up_scale / 2),
    )

    return tensor_out


# NOTE: ONNX is incompatible with traced torch.nn.functional.interpolate if
# using dynamic `scale_factor` rather than static `size`. (T43166860)
# NOTE: Caffe2 Int8 conversion might not be able to quantize `size` properly.
def onnx_compatibale_interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    # NOTE: The input dimensions are interpreted in the form:
    # `mini-batch x channels x [optional depth] x [optional height] x width`.
    if size is None and scale_factor is not None:
        if input.dim() == 4:
            if isinstance(scale_factor, (int, float)):
                height_scale, width_scale = (scale_factor, scale_factor)
            else:
                assert isinstance(scale_factor, (tuple, list))
                assert len(scale_factor) == 2
                height_scale, width_scale = scale_factor

            assert not align_corners, "No matching C2 op for align_corners == True"
            if mode == "nearest":
                return torch.ops._caffe2.ResizeNearest(
                    input,
                    order="NCHW",
                    width_scale=width_scale,
                    height_scale=height_scale,
                )
            elif mode == "bilinear":
                logger.warning(
                    "Use F.conv_transpose2d for bilinear interpolate"
                    " because there's no such C2 op, this may cause significant"
                    " slowdown and the boundary pixels won't be as same as"
                    " using F.interpolate due to padding."
                )
                assert height_scale == width_scale
                return BilinearInterpolation(input, up_scale=height_scale)
        logger.warning(
            "Output size is not static, it might cause ONNX conversion issue"
        )

    return interp(input, size, scale_factor, mode, align_corners)


def mock_torch_nn_functional_interpolate():
    def decorator(func):
        @functools.wraps(func)
        def _mock_torch_nn_functional_interpolate(*args, **kwargs):
            if torch.onnx.is_in_onnx_export():
                with mock.patch(
                    "torch.nn.functional.interpolate",
                    side_effect=onnx_compatibale_interpolate,
                ):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return _mock_torch_nn_functional_interpolate

    return decorator
