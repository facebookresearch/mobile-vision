#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy

import mobile_cv.arch.utils.helper as hp

from . import blocks_factory, irf_block


IRF_SPADE_DEFAULT_ARGS = {"seg_return_type": "input"}


def _get_tuple_left(args, op_type):
    ret = copy.deepcopy(args)
    assert "name" in args
    op_name = op_type + "_name"
    assert op_name not in args
    ret[op_name] = args["name"]
    ret["name"] = op_type + "_tuple_left"
    return ret


def irf_spade(
    in_channels,
    out_channels,
    expansion=6,
    kernel_size=3,
    stride=1,
    conv_args="conv",
    bn_args="bn",
    relu_args="relu",
    res_conn_args="default",
    upsample_args="default",
    spade_args=None,
    **kwargs,
):
    conv_args = hp.unify_args(conv_args)
    conv_args = _get_tuple_left(conv_args, "conv")

    relu_args = hp.unify_args(relu_args)
    relu_args = _get_tuple_left(relu_args, "relu")

    res_conn_args = hp.unify_args(res_conn_args)
    res_conn_args = _get_tuple_left(res_conn_args, "residual_connect")

    upsample_args = hp.unify_args(upsample_args)
    upsample_args = _get_tuple_left(upsample_args, "upsample")

    bn_args = hp.unify_args(bn_args)
    bn_args_normal = _get_tuple_left(bn_args, "bn")

    # apply spade after dw
    bn_args_dw = {
        "name": "spade_norm",
        "bn_args": bn_args,
        **hp.merge_unify_args(IRF_SPADE_DEFAULT_ARGS, spade_args),
    }

    ret = irf_block.IRFBlock(
        in_channels,
        out_channels,
        expansion=expansion,
        kernel_size=kernel_size,
        stride=stride,
        conv_args=conv_args,
        bn_args={},
        relu_args=relu_args,
        res_conn_args=res_conn_args,
        upsample_args=upsample_args,
        pw_bn_args=bn_args_normal,
        dw_bn_args=bn_args_dw,
        pwl_bn_args=bn_args_normal,
        **kwargs,
    )
    return ret


def irf_spade_pwl(
    in_channels,
    out_channels,
    expansion=6,
    kernel_size=3,
    stride=1,
    conv_args="conv",
    bn_args="bn",
    relu_args="relu",
    res_conn_args="default",
    upsample_args="default",
    spade_args=None,
    **kwargs,
):
    conv_args = hp.unify_args(conv_args)
    conv_args = _get_tuple_left(conv_args, "conv")

    relu_args = hp.unify_args(relu_args)
    relu_args = _get_tuple_left(relu_args, "relu")

    res_conn_args = hp.unify_args(res_conn_args)
    res_conn_args = _get_tuple_left(res_conn_args, "residual_connect")

    upsample_args = hp.unify_args(upsample_args)
    upsample_args = _get_tuple_left(upsample_args, "upsample")

    bn_args = hp.unify_args(bn_args)
    bn_args_normal = _get_tuple_left(bn_args, "bn")

    # apply spade after pwl
    bn_args_pwl = {
        "name": "spade_norm",
        "bn_args": bn_args,
        **hp.merge_unify_args(IRF_SPADE_DEFAULT_ARGS, spade_args),
    }

    ret = irf_block.IRFBlock(
        in_channels,
        out_channels,
        expansion=expansion,
        kernel_size=kernel_size,
        stride=stride,
        conv_args=conv_args,
        bn_args={},
        relu_args=relu_args,
        res_conn_args=res_conn_args,
        upsample_args=upsample_args,
        pw_bn_args=bn_args_normal,
        dw_bn_args=bn_args_normal,
        pwl_bn_args=bn_args_pwl,
        **kwargs,
    )
    return ret


_PRIMITIVES = {
    "irf_spade": lambda in_channels, out_channels, stride, **kwargs: irf_spade(
        in_channels=in_channels, out_channels=out_channels, stride=stride, **kwargs
    ),
    "irf_spade_pwl": lambda in_channels, out_channels, stride, **kwargs: irf_spade_pwl(
        in_channels=in_channels, out_channels=out_channels, stride=stride, **kwargs
    ),
}
blocks_factory.PRIMITIVES.register_dict(_PRIMITIVES)
