#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from . import basic_blocks as bb


@bb.BN_REGISTRY.register("ada_instance_layer")
class adaILN(nn.Module):
    def __init__(
        self,
        num_channels: int,
        style_dim: int = 0,
        zero_gamma: Optional[bool] = None,
        eps: float = 1e-8,
    ):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.empty(1, num_channels, 1, 1))
        # pyre-fixme[16]: `Parameter` has no attribute `fill_`.
        self.rho.data.fill_(0.9)

        if style_dim <= 0:
            style_dim = num_channels
        self.gamma = nn.Linear(style_dim, num_channels, bias=False)
        self.beta = nn.Linear(style_dim, num_channels, bias=False)

        self.add_eps = bb.TorchAddScalar(eps)
        self.add_one = bb.TorchAddScalar(1.0)
        self.mul_neg = bb.TorchMulScalar(-1.0)
        self.add = bb.TorchAdd()
        self.mul = bb.TorchMultiply()
        # todo: wrapper for torch.rsqrt ?

    def forward(self, data: Tuple[torch.Tensor, torch.Tensor]):
        x, style = data

        in_mean, in_var = torch.mean(x, dim=[2, 3], keepdim=True), torch.var(
            x, dim=[2, 3], keepdim=True
        )
        # out_in = (x - in_mean) / torch.sqrt(in_var + self.eps)
        out_in = self.mul(
            self.add(x, self.mul_neg(in_mean)), torch.rsqrt(self.add_eps(in_var))
        )

        ln_mean, ln_var = torch.mean(x, dim=[1, 2, 3], keepdim=True), torch.var(
            x, dim=[1, 2, 3], keepdim=True
        )
        # out_ln = (x - ln_mean) / torch.sqrt(ln_var + self.eps)
        out_ln = self.mul(
            self.add(x, self.mul_neg(ln_mean)), torch.rsqrt(self.add_eps(ln_var))
        )

        # out = self.rho.expand(x.shape[0], -1, -1, -1) * out_in + (1.0 - self.rho.expand(x.shape[0], -1, -1, -1)) * out_ln
        rho = self.rho.expand(x.shape[0], -1, -1, -1)
        out = self.mul(rho, out_in) + self.mul(self.add_one(self.mul_neg(rho)), out_ln)

        gamma = self.gamma(style).unsqueeze(2).unsqueeze(3)
        beta = self.beta(style).unsqueeze(2).unsqueeze(3)
        out = self.add(self.mul(out, gamma), beta)
        return (out, style)


@bb.BN_REGISTRY.register("instance_layer")
class ILN(nn.Module):
    def __init__(
        self,
        num_channels: int,
        zero_gamma: Optional[bool] = None,
        eps: float = 1e-8,
    ):
        super(ILN, self).__init__()
        self.rho = Parameter(torch.empty(1, num_channels, 1, 1))
        self.gamma = Parameter(torch.empty(1, num_channels, 1, 1))
        self.beta = Parameter(torch.empty(1, num_channels, 1, 1))
        # pyre-fixme[16]: `Parameter` has no attribute `fill_`.
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

        self.add_eps = bb.TorchAddScalar(eps)
        self.add_one = bb.TorchAddScalar(1.0)
        self.mul_neg = bb.TorchMulScalar(-1.0)
        self.add = bb.TorchAdd()
        self.mul = bb.TorchMultiply()
        # todo: wrapper for torch.rsqrt ?

    def forward(self, x):
        in_mean, in_var = torch.mean(x, dim=[2, 3], keepdim=True), torch.var(
            x, dim=[2, 3], keepdim=True
        )
        # out_in = (x - in_mean) * torch.rsqrt(self.add_eps(in_var))
        out_in = self.mul(
            self.add(x, self.mul_neg(in_mean)), torch.rsqrt(self.add_eps(in_var))
        )

        ln_mean, ln_var = torch.mean(x, dim=[1, 2, 3], keepdim=True), torch.var(
            x, dim=[1, 2, 3], keepdim=True
        )
        # out_ln = (x - ln_mean) / torch.sqrt(ln_var + self.eps)
        out_ln = self.mul(
            self.add(x, self.mul_neg(ln_mean)), torch.rsqrt(self.add_eps(ln_var))
        )

        # out = self.rho.expand(x.shape[0], -1, -1, -1) * out_in + (1.0 - self.rho.expand(x.shape[0], -1, -1, -1)) * out_ln

        rho = self.rho.expand(x.shape[0], -1, -1, -1)
        out = self.mul(rho, out_in) + self.mul(self.add_one(self.mul_neg(rho)), out_ln)

        gamma = self.gamma.expand(x.shape[0], -1, -1, -1)
        beta = self.beta.expand(x.shape[0], -1, -1, -1)
        out = self.add(self.mul(out, gamma), beta)

        return out
