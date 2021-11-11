#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Taken from detectron2

"""
helper class that supports empty tensors on some nn functions.

Ideally, add support directly in PyTorch to empty tensors in
those functions.

This can be removed once https://github.com/pytorch/pytorch/issues/12013
is implemented
"""

import copy
import math
from typing import List, NamedTuple, Tuple

import torch
from torch.nn.modules.utils import _pair


# for backward compatibility
BatchNorm2d = torch.nn.BatchNorm2d
interpolate = torch.nn.functional.interpolate
GroupNorm = torch.nn.GroupNorm


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single
    element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class Conv2dArgs(NamedTuple):
    kernel_size: Tuple[int, int]
    padding: Tuple[int, int]
    stride: Tuple[int, int]
    dilation: Tuple[int, int]
    out_channels: int

    @classmethod
    # pyre-fixme[47]: `Conv2dArgs` cannot be the type of `cls`.
    def FromConv2d(cls: "Conv2dArgs", conv: torch.nn.Conv2d) -> "Conv2dArgs":
        # pyre-fixme[29]: `Conv2dArgs` is not a function.
        return cls(
            conv.kernel_size,
            conv.padding,
            conv.stride,
            conv.dilation,
            conv.out_channels,
        )


def _get_conv_2d_output_shape(conv_args: Conv2dArgs, x: torch.Tensor) -> List[int]:
    # When input is empty, we want to return a empty tensor with "correct" shape,
    # So that the following operations will not panic
    # if they check for the shape of the tensor.
    # This computes the height and width of the output tensor
    assert len(x.shape) == 4
    x_res: Tuple[int, int] = (x.shape[2], x.shape[3])
    output_shape = [
        (i + 2 * p - (di * (k - 1) + 1)) // s + 1
        for i, p, di, k, s in zip(
            x_res,
            conv_args.padding,
            conv_args.dilation,
            conv_args.kernel_size,
            conv_args.stride,
        )
    ]
    output_shape = [x.shape[0], conv_args.out_channels] + output_shape
    return output_shape


class Conv2dEmptyOutput(torch.nn.Module):
    def __init__(self, conv_op: torch.nn.Conv2d):
        super().__init__()
        assert isinstance(conv_op, torch.nn.Conv2d)
        self.conv_args: Conv2dArgs = Conv2dArgs.FromConv2d(conv_op)

    # NOTE: `torch.autograd.Function` (used for empty batch) is not supported
    # for scripting now, so we skip it in scripting mode
    # We should remove empty batch function after empty batch is fully supported
    # by pytorch
    # See https://github.com/pytorch/pytorch/issues/22329
    @torch.jit.unused
    def forward(self, x):
        assert x.numel() == 0, "Only handle empty batch"
        output_shape = _get_conv_2d_output_shape(Conv2dArgs(*self.conv_args), x)
        return _NewEmptyTensorOp.apply(x, output_shape)


class Conv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        norm (nn.Module, optional): a normalization layer
        activation (callable(Tensor) -> Tensor): a callable activation function
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)
        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    @classmethod
    def cast(cls, module: "Conv2d"):
        """Cast mobile_cv.arch.layers.Conv2d to a normal conv2d"""
        assert type(module) == cls
        module = copy.deepcopy(module)
        assert module.norm is None
        assert module.activation is None
        module.__class__ = torch.nn.Conv2d
        return module


class ConvTranspose2d(torch.nn.ConvTranspose2d):
    def forward(self, x):
        if x.numel() > 0:
            return super(ConvTranspose2d, self).forward(x)
        # get output shape

        output_shape = [
            (i - 1) * d - 2 * p + (di * (k - 1) + 1) + op
            for i, p, di, k, d, op in zip(
                x.shape[-2:],
                self.padding,
                self.dilation,
                self.kernel_size,
                self.stride,
                self.output_padding,
            )
        ]
        output_shape = [x.shape[0], self.bias.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


# pyre-fixme[11]: Annotation `AvgPool2d` is not defined as a type.
class AvgPool2d(torch.nn.AvgPool2d):
    def forward(self, x):
        if x.numel() > 0:
            return super(AvgPool2d, self).forward(x)
        # get output shape
        floor_func = math.floor if not self.ceil_mode else math.ceil
        output_shape = [
            int(floor_func((i + 2 * p - k) / s + 1))
            for i, p, k, s in zip(
                x.shape[-2:],
                _pair(self.padding),
                _pair(self.kernel_size),
                _pair(self.stride),
            )
        ]
        output_shape = [x.shape[0], x.shape[1]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)
