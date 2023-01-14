#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from mobile_cv.arch.fbnet_v2.blocks_factory import PRIMITIVES

torch.fx.wrap("len")


class ReshapeToBatch(torch.nn.Module):
    """Reshape the channel dimension of the tensor to batch dimension
    [n, c, h, w] -> [n * (c // out_channels), out_channels, h, w]
    """

    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels

    @staticmethod
    @PRIMITIVES.register("reshape_to_batch")
    def from_block_factory(in_channels, out_channels, stride, **kwargs):
        return ReshapeToBatch(out_channels=out_channels)

    def forward(self, x):
        torch._assert(len(x.shape) == 4, "must have 4 dims")
        torch._assert(
            x.shape[1] % self.out_channels == 0,
            f"{x.shape[1]} is not divisible by {self.out_channels}",
        )
        num_views = x.shape[1] // self.out_channels
        return x.reshape(
            x.shape[0] * num_views, self.out_channels, x.shape[2], x.shape[3]
        )


class ReshapeToChannel(torch.nn.Module):
    """Reshape the batch dimension of the tensor to channel dimension
    [n, c, h, w] -> [n / (out_channels // c), out_channels, h, w]
    """

    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels

    @staticmethod
    @PRIMITIVES.register("reshape_to_channel")
    def from_block_factory(in_channels, out_channels, stride, **kwargs):
        return ReshapeToChannel(out_channels=out_channels)

    def forward(self, x):
        torch._assert(len(x.shape) == 4, "must has 4 dims")
        torch._assert(
            self.out_channels % x.shape[1] == 0,
            f"out_channels = {self.out_channels}, x = {x.shape}",
        )
        num_views = self.out_channels // x.shape[1]
        torch._assert(
            x.shape[0] % num_views == 0, f"x = {x.shape}, num_views = {num_views}"
        )
        return x.reshape(
            x.shape[0] // num_views,
            self.out_channels,
            x.shape[2],
            x.shape[3],
        )
