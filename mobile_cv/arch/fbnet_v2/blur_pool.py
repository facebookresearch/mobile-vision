# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn


class BlurPool2d(nn.Module):
    def __init__(self, num_channels, kernel_size=4, stride=2, padding_mode="zeros"):
        super().__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_mode = padding_mode
        assert num_channels is not None
        assert (
            self.stride >= 1
        ), f"Only positive stride >= 1 is allowed, got: stride={self.stride}"

        kernel = _make_2d_blur_kernel(kernel_size)
        kernel = kernel.repeat((num_channels, 1, 1, 1))
        if kernel_size > 1 and kernel_size % 2 == 0:
            # Add extra one pixel padding if kernel_size % 2 == 0
            self.uneven_pad = _make_uneven_pad(padding_mode, (kernel_size - 1) // 2)
            left_pad = 0
        else:
            self.uneven_pad = None
            left_pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            num_channels,
            num_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=left_pad,
            padding_mode=padding_mode,
            groups=num_channels,
            bias=False,
        )
        assert (
            self.conv.weight.shape == kernel.shape
        ), f"{self.conv.weight.shape} == {kernel.shape}"
        del self.conv.weight
        self.conv.weight = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x):
        if self.uneven_pad is not None:
            x = self.uneven_pad(x)
        return self.conv(x)


def _make_2d_blur_kernel(kernel_size):
    """
    Create 2D blur kernel of size [kernel_size, kernel_size]
    """
    kernels_1d = [
        None,
        [1.0],
        [1.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 3.0, 3.0, 1.0],
        [1.0, 4.0, 6.0, 4.0, 1.0],
        [1.0, 5.0, 10.0, 10.0, 5.0, 1.0],
        [1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0],  # 7
    ]
    if kernel_size == 0 or kernel_size >= len(kernels_1d):
        raise ValueError("kernel_size must be in range [1, 7]")

    k = torch.tensor(kernels_1d[kernel_size])
    k = k[:, None] * k[None, :]
    k /= k.sum()
    return k


def _make_uneven_pad(padding_mode, size):
    padding_types = {
        "zeros": nn.ZeroPad2d,
        "replicate": nn.ReplicationPad2d,
        "reflect": nn.ReflectionPad2d,
    }
    if padding_mode not in padding_types:
        raise ValueError("Unknown Padding mode: [%s]" % padding_mode)
    return padding_types[padding_mode]([size, size + 1, size, size + 1])
