#!/usr/bin/env python3

import torch
from mobile_cv.common.misc.oss_utils import is_oss


if not is_oss():
    from caffe2.python import dyndep

    dyndep.InitOpsLibrary("@/mobile-vision/mobile_cv/mobile_cv/torch/cpp:caffe2_ops")


def alias(x, name, is_backward=False):
    if not torch.onnx.is_in_onnx_export():
        return x
    assert isinstance(x, torch.Tensor)
    return torch.ops._caffe2.AliasWithName(x, name, is_backward=is_backward)


def fuse_alias_placeholder(predict_net, init_net):
    """Remove AliasWithName placeholder and rename the input/output of it"""

    # Delay the import in case caffe2 is not available
    from mobile_cv.torch.utils_caffe2.graph_transform import (
        rename_op_input,
        rename_op_output,
    )
    from mobile_cv.torch.utils_caffe2.protobuf import get_pb_arg_vali, get_pb_arg_vals

    # First we finish all the re-naming
    for i, op in enumerate(predict_net.op):
        if op.type == "AliasWithName":
            assert len(op.input) == 1
            assert len(op.output) == 1
            name = get_pb_arg_vals(op, "name", None).decode()
            is_backward = bool(get_pb_arg_vali(op, "is_backward", 0))
            rename_op_input(
                predict_net, init_net, i, 0, name, from_producer=is_backward
            )
            rename_op_output(predict_net, i, 0, name)

    # Remove AliasWithName, should be very safe since it's a non-op
    new_ops = []
    for op in predict_net.op:
        if op.type != "AliasWithName":
            new_ops.append(op)
        else:
            # safety check
            assert op.input == op.output
            assert op.input[0] == op.arg[0].s.decode()
    del predict_net.op[:]
    predict_net.op.extend(new_ops)
