#!/usr/bin/env python3
"""
This file is for analyze and/or optimize state transition (eg. NHWC/NCHW,
fp32/int8, CPU/GPU).
"""

from typing import Callable, Dict, Tuple

from caffe2.proto import caffe2_pb2
from caffe2.python import core

from .protobuf import get_pb_arg_vals


def _get_order_arg_from_op(op, default):
    order_bytes = get_pb_arg_vals(op, "order", b"").decode()
    return order_bytes or default


# modified from mobile-vision/common/utils/switch_order.py
def _update_order_status(op, inputs_order, outputs_order):
    """
    Given op and currently known order for input & output blobs,
    return an updated order of input & output blobs.
    Order can be one of "NCHW", "NHWC", None.

    Return:
        The returned inputs_order can depends on given outputs_order.
        The returned outputs_order can depends on given inputs_order.
    """

    STATIC_ORDERED_OPS = {
        # order switch ops
        "NCHW2NHWC": (["NCHW"], ["NHWC"]),
        "NHWC2NCHW": (["NHWC"], ["NCHW"]),
        # int8 quantize ops
        "Int8Quantize": (["NHWC"], ["NHWC"]),
        "Int8Dequantize": (["NHWC"], ["NHWC"]),
    }
    _inputs_order = [o for o in inputs_order]  # noqa
    _outputs_order = [o for o in outputs_order]  # noqa

    if op.type in STATIC_ORDERED_OPS:
        return STATIC_ORDERED_OPS[op.type]
    # == for ops whose all input/output follow the `order` arg
    elif op.type in [
        "Conv",
        "ConvTranspose",
        "ChannelShuffle",
        "Sum",
    ]:
        order = _get_order_arg_from_op(op, "NCHW")
        _inputs_order = [order for _ in _inputs_order]
        _outputs_order = [order for _ in _outputs_order]
    # == for ops whose first input/output follows the `order` arg
    elif op.type in ["RoIAlign", "RoIWarp"]:
        order = _get_order_arg_from_op(op, "NCHW")
        _inputs_order[0] = order
        _outputs_order[0] = order
    # == for ops whose first input/output follows its output/input
    elif op.type in ["Sigmoid", "Softmax", "Relu"]:
        if _outputs_order[0] is not None:
            _inputs_order[0] = outputs_order[0]
        if _inputs_order[0] is not None:
            _outputs_order[0] = inputs_order[0]
    # == for ops whose all input/output is NHWC
    elif op.type in ["Int8Conv", "Int8ConvRelu", "Int8ConvTranspose"]:
        _inputs_order = ["NHWC" for _ in _inputs_order]
        _outputs_order = ["NHWC" for _ in _outputs_order]

    return _inputs_order, _outputs_order


def _update_int8_status(op, inputs_type, outputs_type):
    if op.type == "Int8Quantize":
        _inputs_type = ["fp32" for _ in inputs_type]
        _outputs_type = ["int8" for _ in outputs_type]
    elif op.type == "Int8Dequantize":
        _inputs_type = ["int8" for _ in inputs_type]
        _outputs_type = ["fp32" for _ in outputs_type]
    elif op.type.startswith("Int8"):
        _inputs_type = ["int8" for _ in inputs_type]
        _outputs_type = ["int8" for _ in outputs_type]
    else:
        _inputs_type = ["fp32" for _ in inputs_type]
        _outputs_type = ["fp32" for _ in outputs_type]

    return _inputs_type, _outputs_type


def _has_conflict(order, updated_order):
    if order is not None and updated_order is not None:
        return order != updated_order
    return False


def static_static_analyzer(
    predict_net: caffe2_pb2.NetDef,
    status_updater: Callable,
) -> Dict[Tuple[str, int], str]:
    """
    Statically analyze the status of blob. status can be anything, eg. it can
        be one of ["NCHW", "NHWC"] when referring to order; it can be
        ["int8", "fp32"] when referring to precision; it can be ["cpu", "gpu"]
        when reffering device. It uses `None` for unknown status.

    Args:
        predict_net: the NetDef.
        status_updater:

    Return:
        dictioanry mapping versioned blob to its status
    """
    ssa, versions = core.get_ssa(predict_net)

    def _update_status_for_op(i, status_map):
        inputs = ssa[i][0]
        outputs = ssa[i][1]
        inputs_status = [status_map.get(inp, None) for inp in inputs]
        outputs_status = [status_map.get(outp, None) for outp in outputs]
        updated_inputs_status, updated_outputs_status = status_updater(
            predict_net.op[i], inputs_status, outputs_status
        )
        for name, status, updated_status in zip(
            inputs + outputs,
            inputs_status + outputs_status,
            updated_inputs_status + updated_outputs_status,
        ):
            if _has_conflict(status, updated_status):
                raise RuntimeError("{} has status conflict.".format(name))
            status_map[name] = updated_status

    cur_status_map = {}
    cur_cnt = 0
    while True:
        new_status_map = {k: v for k, v in cur_status_map.items()}
        for i in range(len(ssa)):
            _update_status_for_op(i, new_status_map)
        for i in range(len(ssa)):
            _update_status_for_op(len(ssa) - i - 1, new_status_map)
        if new_status_map == cur_status_map:
            break
        cur_status_map = new_status_map
        # NOTE: stop updating after iterating through the whole net N times,
        # where N is number of ops. Normally one forward and backward pass
        # is enough.
        cur_cnt += 1
        if cur_cnt == len(predict_net.op):
            break

    return cur_status_map


def analyze_order(predict_net):
    """Return dict mapping versioned blob to "NCHW" / "NHWC" / None"""
    return static_static_analyzer(predict_net, _update_order_status)


def analyze_type(predict_net):
    """Return dict mapping versioned blob to "fp32" / "int8" """
    return static_static_analyzer(predict_net, _update_int8_status)
