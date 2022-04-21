#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Taken from detectron2

import copy

import torch
import torch.distributed as dist
from torch import nn
from torch.autograd.function import Function


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    It contains non-trainable buffers called "weight" and "bias".
    The two buffers are computed from the original four parameters of BN:
    mean, variance, scale (gamma), offset (beta).
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - mean) / std * scale + offset`, but will be slightly cheaper.
    The pre-trained backbone models from Caffe2 are already in such a frozen format.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))

    def forward(self, x):
        scale = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        return x * scale + bias


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


class _AllReduce(Function):
    @staticmethod
    def forward(ctx, input):
        input_list = [torch.zeros_like(input) for k in range(dist.get_world_size())]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output


def differentiable_all_reduce(input: torch.Tensor) -> torch.Tensor:
    """
    Code forked from vision/fair/fvcore/fvcore/nn/distributed.py
    Differentiable counterpart of `dist.all_reduce`.
    """
    if (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size() == 1
    ):
        return input
    return _AllReduce.apply(input)


# pyre-fixme[11]: Annotation `BatchNorm2d` is not defined as a type.
class NaiveSyncBatchNorm(nn.BatchNorm2d):
    """
    Code forked from detectron2/layers/batch_norm.py

    In PyTorch<=1.5, ``nn.SyncBatchNorm`` has incorrect gradient
    when the batch size on each worker is different.
    (e.g., when scale augmentation is used, or when it is applied to mask head).

    This is a slower but correct alternative to `nn.SyncBatchNorm`.

    Note:
        There isn't a single definition of Sync BatchNorm.

        When ``stats_mode==""``, this module computes overall statistics by using
        statistics of each worker with equal weight.  The result is true statistics
        of all samples (as if they are all on one worker) only when all workers
        have the same (N, H, W). This mode does not support inputs with zero batch size.

        When ``stats_mode=="N"``, this module computes overall statistics by weighting
        the statistics of each worker by their ``N``. The result is true statistics
        of all samples (as if they are all on one worker) only when all workers
        have the same (H, W). It is slower than ``stats_mode==""``.

        Even though the result of this module may not be the true statistics of all samples,
        it may still be reasonable because it might be preferrable to assign equal weights
        to all workers, regardless of their (H, W) dimension, instead of putting larger weight
        on larger images. From preliminary experiments, little difference is found between such
        a simplified implementation and an accurate computation of overall mean & variance.
    """

    def __init__(self, *args, stats_mode="", **kwargs):
        super().__init__(*args, **kwargs)
        assert stats_mode in ["", "N"]
        self._stats_mode = stats_mode

    def forward(self, input):
        if get_world_size() == 1 or not self.training:
            return super().forward(input)

        B, C = input.shape[0], input.shape[1]

        half_input = input.dtype == torch.float16
        if half_input:
            # fp16 does not have good enough numerics for the reduction here
            input = input.float()
        mean = torch.mean(input, dim=[0, 2, 3])
        meansqr = torch.mean(input * input, dim=[0, 2, 3])

        if self._stats_mode == "":
            assert (
                B > 0
            ), 'SyncBatchNorm(stats_mode="") does not support zero batch size.'
            vec = torch.cat([mean, meansqr], dim=0)
            vec = differentiable_all_reduce(vec) * (1.0 / dist.get_world_size())
            mean, meansqr = torch.split(vec, C)
            momentum = self.momentum
        else:
            if B == 0:
                vec = torch.zeros([2 * C + 1], device=mean.device, dtype=mean.dtype)
                vec = vec + input.sum()  # make sure there is gradient w.r.t input
            else:
                vec = torch.cat(
                    [
                        mean,
                        meansqr,
                        torch.ones([1], device=mean.device, dtype=mean.dtype),
                    ],
                    dim=0,
                )
            vec = differentiable_all_reduce(vec * B)

            total_batch = vec[-1].detach()
            momentum = (
                total_batch.clamp(max=1) * self.momentum
            )  # no update if total_batch is 0
            mean, meansqr, _ = torch.split(
                vec / total_batch.clamp(min=1), C
            )  # avoid div-by-zero

        var = meansqr - mean * mean
        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)

        self.running_mean += momentum * (mean.detach() - self.running_mean)
        self.running_var += momentum * (var.detach() - self.running_var)
        ret = input * scale + bias
        if half_input:
            ret = ret.half()
        return ret

    @classmethod
    def cast(cls, module: "NaiveSyncBatchNorm"):
        assert type(module) == cls
        module = copy.deepcopy(module)
        module.__class__ = nn.BatchNorm2d
        return module


class NaiveSyncBatchNorm1d(nn.BatchNorm1d):
    """
    torch.nn.SyncBatchNorm has bugs. Use this before it is fixed.
    only supports input shape of (N, C, L) or (N, C)
    """

    def forward(self, input):
        if get_world_size() == 1 or not self.training:
            return super().forward(input)

        assert input.shape[0] > 0, "SyncBatchNorm does not support empty inputs"
        assert len(input.shape) in (
            2,
            3,
        ), "SyncBatchNorm1d supports (N, C, L) or (N, C) inputs"
        C = input.shape[1]

        reduce_dims = [*range(len(input.shape))]
        reduce_dims.remove(1)

        mean = torch.mean(input, dim=reduce_dims)
        meansqr = torch.mean(input * input, dim=reduce_dims)

        vec = torch.cat([mean, meansqr], dim=0)
        vec = differentiable_all_reduce(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape([1, -1] + [1] * (len(input.shape) - 2))
        bias = bias.reshape([1, -1] + [1] * (len(input.shape) - 2))
        return input * scale + bias

    @classmethod
    def cast(cls, module: "NaiveSyncBatchNorm1d"):
        assert type(module) == cls
        module = copy.deepcopy(module)
        module.__class__ = nn.BatchNorm1d
        return module


# pyre-fixme[11]: Annotation `BatchNorm3d` is not defined as a type.
class NaiveSyncBatchNorm3d(nn.BatchNorm3d):
    """
    torch.nn.SyncBatchNorm has bugs. Use this before it is fixed.
    only supports input shape of (N, C, T, H, W)
    """

    def forward(self, input):
        if get_world_size() == 1 or not self.training:
            return super().forward(input)

        assert input.shape[0] > 0, "SyncBatchNorm does not support empty inputs"
        C = input.shape[1]
        mean = torch.mean(input, dim=[0, 2, 3, 4])
        meansqr = torch.mean(input * input, dim=[0, 2, 3, 4])

        vec = torch.cat([mean, meansqr], dim=0)
        vec = differentiable_all_reduce(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1, 1)
        return input * scale + bias

    @classmethod
    def cast(cls, module: "NaiveSyncBatchNorm3d"):
        assert type(module) == cls
        module = copy.deepcopy(module)
        module.__class__ = nn.BatchNorm3d
        return module


class SyncBatchNormWrapper(nn.SyncBatchNorm, NaiveSyncBatchNorm):
    """
    A wrapper to use torch.nn.SyncBatchNorm when the input is on GPU,
    or NaiveSyncBatchNorm when the input is on CPU
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(num_channels, **kwargs)

    def forward(self, input):
        # nn.SyncBatchNorm only supports GPUs, so use NaiveSyncBatchNorm
        # when the input is on CPU and the communication backend is gloo
        if input.is_cuda:
            return nn.SyncBatchNorm.forward(self, input)
        else:
            return NaiveSyncBatchNorm.forward(self, input)
