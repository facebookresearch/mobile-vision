#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import typing
from typing import Dict, Optional, Tuple, Union

import mobile_cv.arch.utils.helper as hp
import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F

from . import basic_blocks as bb, blocks_factory


# needed for SpadeNorm with fx fusing
torch.fx.wrap("len")


@bb.BN_REGISTRY.register("spade_norm")
class SpadeNorm(nn.Module):
    """Spade implementation
    Based on Paper: "Semantic Image Synthesis with SPADE"
    and https://github.com/NVlabs/SPADE.
    seg_return_type: which seg map to return as the second output
        "None": do not return the seg map,
        "input": return the input seg map,
        "resized": return the resized seg map
    """

    def __init__(
        self,
        num_channels: int,
        bn_args: Union[str, Dict[str, str]],
        kernel_size: int = 1,
        seg_channels: int = 1,
        seg_mid_channels: int = 32,
        zero_gamma: Optional[bool] = None,
        seg_return_type: str = "None",
    ):
        assert seg_return_type in ["None", "input", "resized"]

        super().__init__()
        self.out_channels = num_channels
        self.seg_return_type = seg_return_type

        self.bn = bb.build_bn(
            num_channels=num_channels,
            **hp.merge_unify_args(bn_args, {"zero_gamma": zero_gamma}),
        )

        padding = kernel_size // 2
        self.conv_seg = nn.Sequential(
            nn.Conv2d(
                seg_channels, seg_mid_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.ReLU(),
        )
        self.conv_gamma = nn.Conv2d(
            seg_mid_channels, num_channels, kernel_size=kernel_size, padding=padding
        )
        self.conv_beta = nn.Conv2d(
            seg_mid_channels, num_channels, kernel_size=kernel_size, padding=padding
        )
        self.mul = bb.TorchMultiply()
        self.add = bb.TorchAdd()
        self.add_scalar = bb.TorchAddScalar(1.0)

    def forward(self, data: Tuple[torch.Tensor, torch.Tensor]):
        x, seg_map = data

        # need to reuse the same x_len in message in torch._assert() otherwise
        # it may not be removed correctly when removing assert and cause failure
        # to fuse the bn, see unit test `test_spade_norm_conv_bn_relu_spade_fusebn`
        x_len = len(x.shape)
        torch._assert(x_len == 4, f"{x_len}")
        seg_map_len = len(seg_map.shape)
        torch._assert(seg_map_len == 4, f"{seg_map_len}")

        x = self.bn(x)

        # scaling and bias conditioned on semantic map
        resized_seg_map = F.interpolate(seg_map, size=x.size()[2:], mode="nearest")
        feature_seg = self.conv_seg(resized_seg_map)
        gamma = self.conv_gamma(feature_seg)
        beta = self.conv_beta(feature_seg)

        # apply scale and bias
        # ret = x * (1 + gamma) + beta
        ret = self.add(self.mul(x, self.add_scalar(gamma)), beta)

        if self.seg_return_type == "input":
            return (ret, seg_map)

        if self.seg_return_type == "resized":
            return (ret, resized_seg_map)

        return ret


class TupleLeft(nn.Module):
    """Apply moudle on the first element of the input, keep the second element
    unchanged, return both elements
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        if not isinstance(x, torch.fx.proxy.Proxy):
            # need to reuse the same x_len in message in torch._assert() otherwise
            # it may not be removed correctly when removing assert and cause failure
            # to fuse the bn, see unit test `test_spade_norm_conv_bn_relu_spade_fusebn`
            len_x = len(x)
            torch._assert(
                isinstance(x, tuple) and len_x == 2, f"type: {type(x)}, len={len_x}"
            )
        r1 = self.module(x[0])
        return (r1, x[1])


@bb.CONV_REGISTRY.register()
def conv_tuple_left(in_channels, out_channels, conv_name="conv2d", **conv_args):
    conv = bb.build_conv(
        conv_name, in_channels=in_channels, out_channels=out_channels, **conv_args
    )
    if conv is None:
        return None
    return TupleLeft(conv)


@bb.BN_REGISTRY.register()
def bn_tuple_left(num_channels, bn_name="bn", **bn_args):
    bn = bb.build_bn(bn_name, num_channels=num_channels, **bn_args)
    if bn is None:
        return None
    return TupleLeft(bn)


@bb.RELU_REGISTRY.register()
def relu_tuple_left(relu_name="relu", num_channels=None, **kwargs):
    relu = bb.build_relu(relu_name, num_channels=num_channels, **kwargs)
    if relu is None:
        return None
    return TupleLeft(relu)


@bb.UPSAMPLE_REGISTRY.register()
def upsample_tuple_left(scales=None, upsample_name="default", **kwargs):
    upsample = bb.build_upsample(upsample_name, scales=scales, **kwargs)
    if upsample is None:
        return None
    return TupleLeft(upsample)


class TupleLeft2(nn.Module):
    """Apply moudle on the first element of both inputs, keep the second element
    unchanged, return both elements
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, y):
        if not isinstance(x, torch.fx.proxy.Proxy):
            # need to reuse the same x_len in message in torch._assert() otherwise
            # it may not be removed correctly when removing assert and cause failure
            # to fuse the bn, see unit test `test_spade_norm_conv_bn_relu_spade_fusebn`
            len_x = len(x)
            torch._assert(
                isinstance(x, tuple) and len_x == 2, f"type: {type(x)}, len={len_x}"
            )
        r1 = self.module(x[0], y[0])
        return (r1, x[1])


@bb.RESIDUAL_REGISTRY.register()
def residual_connect_tuple_left(
    in_channels,
    out_channels,
    stride,
    residual_connect_name,
    drop_connect_rate=None,
    **kwargs,
):
    res_conn = bb.build_residual_connect(
        name=residual_connect_name,
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
        drop_connect_rate=drop_connect_rate,
        **kwargs,
    )
    if res_conn is None:
        return None
    return TupleLeft2(res_conn)


_TL_PRIMITIVES = {
    f"{name}_tuple_left": lambda in_channels, out_channels, stride, **kwargs: TupleLeft(
        blocks_factory.PRIMITIVES[name](
            in_channels=in_channels, out_channels=out_channels, stride=stride, **kwargs
        )
    )
    for name in ["conv", "conv_k3", "conv_k5"]
}
blocks_factory.PRIMITIVES.register_dict(_TL_PRIMITIVES)


def _get_fuser_name_convbnrelu_with_tuple_left(
    module: torch.nn.Module,
    supported_types: typing.Dict[str, typing.List[torch.nn.Module]],
):
    """Get the fuser name for ConvBNRelu with TupleLeft(conv) and SpadeNorm
    Return None if not the supported types
    """

    if not isinstance(module, (bb.ConvBNRelu, bb.ConvNormAct)):
        return None

    MODULE_NAME_MAPPING = {
        bb.ConvBNRelu: {"conv": "conv", "bn": "bn", "relu": "relu"},
        bb.ConvNormAct: {"conv": "conv", "bn": "norm", "relu": "act"},
    }

    def _get_op(name):
        op_name_real = MODULE_NAME_MAPPING[type(module)][name]
        return op_name_real, getattr(module, op_name_real, None)

    class NotMatch(torch.nn.Module):
        pass

    not_match_op = NotMatch()

    def _get_tuple_left(op_name):
        op_name, op = _get_op(op_name)
        if op is None:
            return None, None
        if not isinstance(op, TupleLeft):
            return not_match_op, None
        return op.module, op_name + ".module"

    def _get_bn_spade():
        op_name, op = _get_op("bn")
        if op is None:
            return None, None
        if not isinstance(op, SpadeNorm):
            return not_match_op, None
        return op.bn, op_name + ".bn"

    SUPPORTED_FUSING_TYPES = [
        {
            "conv": lambda: _get_tuple_left("conv"),
            "bn": lambda: _get_tuple_left("bn"),
            "relu": lambda: _get_tuple_left("relu"),
        },
        {
            # could not fuse relu for spadebn
            "conv": lambda: _get_tuple_left("conv"),
            "bn": _get_bn_spade,
        },
    ]

    for fuse_type in SUPPORTED_FUSING_TYPES:
        cur_ret = []
        for op_name, op_func in fuse_type.items():
            op_real, op_real_name = op_func()
            # if the op type not match, we will skip (so will not fuse
            #   conv + relu if bn in the middle is not supported)
            if isinstance(op_real, NotMatch):
                break
            if op_real is None:
                continue
            if type(op_real) in supported_types[op_name]:
                cur_ret.append(op_real_name)
        if len(cur_ret) > 1:
            return [cur_ret]
    return None
