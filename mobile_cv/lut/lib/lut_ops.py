#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

""" Represents ops in LUT, following pytorch's interface """

import functools
import json
import operator

from mobile_cv.lut.lib.lut_schema import OpBase


def to_tuple1(x):
    if isinstance(x, (tuple, list)):
        assert len(x) == 1
        return tuple(x)
    return (x,)


def to_tuple2(x):
    if isinstance(x, (tuple, list)):
        assert len(x) == 2
        return tuple(x)
    return (x, x)


def to_tuple3(x):
    if isinstance(x, (tuple, list)):
        assert len(x) == 3
        return tuple(x)
    return (x, x, x)


def compute_prod(alist):
    assert isinstance(alist, list)
    return functools.reduce(operator.mul, alist)


def compute_sum_of_prod(list_of_list):
    assert isinstance(list_of_list, list)
    ret = sum(compute_prod(x) for x in list_of_list)
    return ret


class OpProperty(OpBase):
    """Convenience class to store arguments in a dict"""

    def __init__(self, info):
        super().__init__()
        assert isinstance(info, dict)
        self.info = info

    def __eq__(self, rhs):
        return self.info == rhs.info

    def __hash__(self):
        # only support dict without nesting
        return hash(frozenset(self.info.items()))

    def __repr__(self):
        info = {"type": self.name(), "info": self.info}
        return json.dumps(info)

    def __getattr__(self, key):
        if "info" in vars(self) and key in self.info:
            return self.info[key]
        return super().__getattr__(key)

    def __setattr__(self, key, value):
        if "info" in vars(self) and key in self.info:
            self.info[key] = value
        super().__setattr__(key, value)


class Conv2d(OpProperty):
    def __init__(
        self,
        in_channels=-1,
        out_channels=-1,
        kernel_size=-1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        allow_frac=False,
    ):
        info = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
            "bias": bias,
        }
        super().__init__(info)
        self._unify()
        self.allow_frac = allow_frac

        if not self.allow_frac:
            in_channels = self.in_channels
            assert in_channels % self.groups == 0, (in_channels, self.groups)

    def get_output_shape(self, input_shape):
        N, C, H, W = input_shape[0][:]
        oH = int(
            (H + self.padding[0] * 2 - self.dilation[0] * (self.kernel_size[0] - 1) - 1)
            // self.stride[0]
            + 1
        )
        oW = int(
            (W + self.padding[1] * 2 - self.dilation[1] * (self.kernel_size[1] - 1) - 1)
            // self.stride[1]
            + 1
        )
        return [[N, self.out_channels, oH, oW]]

    def get_params_shape(self):
        ret = []
        in_channels = self.in_channels // self.groups
        if self.allow_frac:
            in_channels = self.in_channels / self.groups
        ret.append(
            [
                self.out_channels,
                in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
            ]
        )
        return ret

    def get_flops(self, input_shape):
        nparams = self.get_nparams()
        out_shape = self.get_output_shape(input_shape)
        oN, _, oH, oW = out_shape[0]
        flops = nparams * oN * oH * oW
        return flops

    def get_nparams(self):
        params_shape = self.get_params_shape()
        nparams = compute_sum_of_prod(params_shape)
        return nparams

    def _unify(self):
        """Unify the representation"""
        for name in ["kernel_size", "stride", "padding", "dilation"]:
            self.info[name] = to_tuple2(self.info[name])


class Conv1d(OpProperty):
    def __init__(
        self,
        in_channels=-1,
        out_channels=-1,
        kernel_size=-1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        allow_frac=False,
    ):
        info = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
            "bias": bias,
        }
        super().__init__(info)
        self._unify()
        self.allow_frac = allow_frac

        if not self.allow_frac:
            in_channels = self.in_channels
            assert in_channels % self.groups == 0, (in_channels, self.groups)

    def get_output_shape(self, input_shape):
        N, _, L = input_shape[0][:]
        oL = int(
            (L + self.padding[0] * 2 - self.dilation[0] * (self.kernel_size[0] - 1) - 1)
            // self.stride[0]
            + 1
        )
        return [[N, self.out_channels, oL]]

    def get_params_shape(self):
        ret = []
        in_channels = self.in_channels // self.groups
        if self.allow_frac:
            in_channels = self.in_channels / self.groups
        ret.append([self.out_channels, in_channels, self.kernel_size[0]])
        return ret

    def get_flops(self, input_shape):
        nparams = self.get_nparams()
        out_shape = self.get_output_shape(input_shape)
        oN, _, oL = out_shape[0]
        flops = nparams * oN * oL
        return flops

    def get_nparams(self):
        params_shape = self.get_params_shape()
        nparams = compute_sum_of_prod(params_shape)
        return nparams

    def _unify(self):
        """Unify the representation"""
        for name in ["kernel_size", "stride", "padding", "dilation"]:
            self.info[name] = to_tuple1(self.info[name])


class Conv3d(OpProperty):
    def __init__(
        self,
        in_channels=-1,
        out_channels=-1,
        kernel_size=-1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        allow_frac=False,
    ):
        info = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
            "bias": bias,
        }
        super().__init__(info)
        self._unify()
        self.allow_frac = allow_frac

        if not self.allow_frac:
            in_channels = self.in_channels
            assert in_channels % self.groups == 0, (in_channels, self.groups)

    def get_output_shape(self, input_shape):
        N, C, D, H, W = input_shape[0][:]
        oD = int(
            (D + self.padding[0] * 2 - self.dilation[0] * (self.kernel_size[0] - 1) - 1)
            // self.stride[0]
            + 1
        )
        oH = int(
            (H + self.padding[1] * 2 - self.dilation[1] * (self.kernel_size[1] - 1) - 1)
            // self.stride[1]
            + 1
        )
        oW = int(
            (W + self.padding[2] * 2 - self.dilation[2] * (self.kernel_size[2] - 1) - 1)
            // self.stride[2]
            + 1
        )
        return [[N, self.out_channels, oD, oH, oW]]

    def get_params_shape(self):
        ret = []
        in_channels = self.in_channels // self.groups
        if self.allow_frac:
            in_channels = self.in_channels / self.groups
        ret.append(
            [
                self.out_channels,
                in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
                self.kernel_size[2],
            ]
        )
        return ret

    def get_flops(self, input_shape):
        nparams = self.get_nparams()
        out_shape = self.get_output_shape(input_shape)
        oN, _, oD, oH, oW = out_shape[0]
        flops = nparams * oN * oD * oH * oW
        return flops

    def get_nparams(self):
        params_shape = self.get_params_shape()
        nparams = compute_sum_of_prod(params_shape)
        return nparams

    def _unify(self):
        """Unify the representation"""
        for name in ["kernel_size", "stride", "padding", "dilation"]:
            self.info[name] = to_tuple3(self.info[name])


class ConvTranspose2d(OpProperty):
    def __init__(
        self,
        in_channels=-1,
        out_channels=-1,
        kernel_size=-1,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
    ):
        info = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "output_padding": output_padding,
            "groups": groups,
            "bias": bias,
            "dilation": dilation,
        }
        super().__init__(info)
        self._unify()
        assert (
            self.out_channels % self.groups == 0
        ), f"output_channels: {self.out_channels}, groups: {self.groups}"

    def get_output_shape(self, input_shape):
        N, C, H, W = input_shape[0][:]
        oH = int(
            (H - 1) * self.stride[0]
            - 2 * self.padding[0]
            + self.dilation[0] * (self.kernel_size[0] - 1)
            + self.output_padding[0]
            + 1
        )
        oW = int(
            (W - 1) * self.stride[1]
            - 2 * self.padding[1]
            + self.dilation[1] * (self.kernel_size[1] - 1)
            + self.output_padding[1]
            + 1
        )
        return [[N, self.out_channels, oH, oW]]

    def get_params_shape(self):
        ret = []
        ret.append(
            [
                self.in_channels,
                self.out_channels // self.groups,
                self.kernel_size[0],
                self.kernel_size[1],
            ]
        )
        return ret

    def get_flops(self, input_shape):
        nparams = self.get_nparams()
        out_shape = self.get_output_shape(input_shape)
        oN, _, oH, oW = out_shape[0]
        flops = nparams * oN * oH * oW
        return flops

    def get_nparams(self):
        params_shape = self.get_params_shape()
        nparams = compute_sum_of_prod(params_shape)
        return nparams

    def _unify(self):
        """Unify the representation"""
        for name in [
            "kernel_size",
            "stride",
            "padding",
            "output_padding",
            "dilation",
        ]:
            self.info[name] = to_tuple2(self.info[name])


class Linear(OpProperty):
    def __init__(self, in_features=-1, out_features=-1, bias=True):
        info = {
            "in_features": in_features,
            "out_features": out_features,
            "bias": bias,
        }
        super().__init__(info)

    def get_output_shape(self, input_shape):
        assert len(input_shape) == 1 and len(input_shape[0]) >= 2
        ret = [list(input_shape[0][:-1]) + [self.out_features]]
        return ret

    def get_params_shape(self):
        ret = []
        ret.append([self.out_features, self.in_features])
        return ret

    def get_flops(self, input_shape):
        nparams = self.get_nparams()
        out_shape = self.get_output_shape(input_shape)
        flops = nparams * compute_prod(out_shape[0][:-1])
        return flops

    def get_nparams(self):
        params_shape = self.get_params_shape()
        nparams = compute_sum_of_prod(params_shape)
        return nparams


class AdaptiveAvgPool2d(OpProperty):
    def __init__(self, output_size=1):
        info = {"output_size": output_size}
        super().__init__(info)

    def get_output_shape(self, input_shape):
        assert len(input_shape) == 1
        N, C, H, W = input_shape[0][:]
        ret = [(N, C) + self.output_size]
        return ret

    def get_params_shape(self):
        ret = []
        ret.append([0])
        return ret

    def get_flops(self, input_shape):
        if not isinstance(input_shape[0], (int, float)):
            input_shape = input_shape[0]
        return functools.reduce(operator.mul, input_shape)

    def get_nparams(self):
        params_shape = self.get_params_shape()
        nparams = compute_sum_of_prod(params_shape)
        return nparams


class MatMul(OpProperty):
    def __init__(self):
        info = {}
        super().__init__(info)

    def get_output_shape(self, input_shape):
        assert len(input_shape) == 2
        ins1, ins2 = input_shape[0], input_shape[1]
        assert len(ins1) == len(ins2)
        assert len(ins1) >= 2
        assert ins1[-1] == ins2[-2]

        out_shape = []
        for _idx in range(len(ins1) - 2):
            i1, i2 = ins1[_idx], ins2[_idx]
            assert isinstance(i1, int)
            assert isinstance(i2, int)
            assert i1 > 0
            assert i2 > 0

            if i1 == 1:
                out_shape.append(i2)
            elif i2 == 1:
                out_shape.append(i1)
            else:
                assert i1 == i2
                out_shape.append(i1)

        out_shape.extend([ins1[-2], ins2[-1]])
        return [out_shape]

    def get_params_shape(self):
        return [[0]]

    def get_flops(self, input_shape):
        out_shape = self.get_output_shape(input_shape)
        flops = 1
        for os in out_shape[0]:
            flops *= os
        flops *= input_shape[0][-1]
        return flops

    def get_nparams(self):
        return 0


class MultiheadAttention(OpProperty):
    def __init__(
        self,
        embed_dim=-1,
        num_heads=-1,
        kdim=None,
        vdim=None,
    ):
        kdim = embed_dim if kdim is None else kdim
        vdim = embed_dim if vdim is None else vdim
        info = {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "kdim": kdim,
            "vdim": vdim,
            "_qkv_same_embed_dim": (embed_dim == kdim and kdim == vdim),
        }
        super().__init__(info)

    def get_output_shape(self, input_shape):
        # FIXME: According to the pytorch document:
        # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html,
        # MultiheadAttention has 2 outputs if need_weights is set to True during
        # calling the forward function. Else, it has 1 output. Here, we assumes
        # it only has 1 output since we don't have access to the value of
        # need_weights.
        return [input_shape[0]]

    def get_params_shape(self):
        if self._qkv_same_embed_dim:
            return [
                [self.embed_dim, self.embed_dim],  # q
                [self.embed_dim, self.embed_dim],  # k
                [self.embed_dim, self.embed_dim],  # v
                [self.embed_dim, self.embed_dim],  # out_proj
            ]
        else:
            return [
                [self.embed_dim, self.embed_dim],  # q
                [self.embed_dim, self.kdim],  # k
                [self.embed_dim, self.vdim],  # v
                [self.embed_dim, self.embed_dim],  # out_proj
            ]

    def get_nparams(self):
        output_shapes = self.get_params_shape()
        return compute_sum_of_prod(output_shapes)

    def get_flops(self, input_shape):
        qs, ks, vs = input_shape
        L, N, E = qs
        S, K = ks[0], ks[2]
        V = vs[2]
        assert N == ks[1] and N == vs[1]
        assert S == vs[0]

        flops = N * (
            L * E * E
            + S * K * E
            + S * V * E
            + L * E * S
            + L * S * E
            + L * E * E
            # -------    ---------   ---------   ---------   ---------   ---------
            # Q_proj    K_proj      V_proj      Q * K       (QK) * V    out_proj
        )

        return flops
