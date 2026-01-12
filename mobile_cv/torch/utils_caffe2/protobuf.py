#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

"""
Utilify functions for retrieving information from caffe2 net, i.e. protobuf.
This file also contains static analysis tools.
"""

import collections
import copy
import logging
from typing import Any, Callable, Dict, Optional, Tuple, Union

import caffe2.python.utils as putils
import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from mobile_cv.torch.utils_caffe2.ws_utils import fetch_any_blob, ScopedWS


logger = logging.getLogger(__name__)


def get_pb_arg(pb, arg_name):
    for x in pb.arg:
        if x.name == arg_name:
            return x
    return None


def get_pb_arg_valf(pb, arg_name, default_val):
    arg = get_pb_arg(pb, arg_name)
    return arg.f if arg is not None else default_val


def get_pb_arg_floats(pb, arg_name, default_val):
    arg = get_pb_arg(pb, arg_name)
    return list(map(float, arg.floats)) if arg is not None else default_val


def get_pb_arg_ints(pb, arg_name, default_val):
    arg = get_pb_arg(pb, arg_name)
    return list(map(int, arg.ints)) if arg is not None else default_val


def get_pb_arg_vali(pb, arg_name, default_val):
    arg = get_pb_arg(pb, arg_name)
    return arg.i if arg is not None else default_val


def get_pb_arg_vals(pb, arg_name, default_val):
    arg = get_pb_arg(pb, arg_name)
    return arg.s if arg is not None else default_val


def get_pb_arg_valstrings(pb, arg_name, default_val):
    arg = get_pb_arg(pb, arg_name)
    return list(arg.strings) if arg is not None else default_val


def check_set_pb_arg(pb, arg_name, arg_attr, arg_value, allow_override=False):
    arg = get_pb_arg(pb, arg_name)
    if arg is None:
        arg = putils.MakeArgument(arg_name, arg_value)
        assert hasattr(arg, arg_attr)
        pb.arg.extend([arg])
    if allow_override and getattr(arg, arg_attr) != arg_value:
        logger.warning(
            "Override argument {}: {} -> {}".format(
                arg_name, getattr(arg, arg_attr), arg_value
            )
        )
        setattr(arg, arg_attr, arg_value)
    else:
        assert arg is not None
        assert (
            getattr(arg, arg_attr) == arg_value
        ), "Existing value {}, new value {}".format(getattr(arg, arg_attr), arg_value)


def _create_const_fill_op_from_numpy(name, tensor, device_option=None):
    assert type(tensor) == np.ndarray
    kTypeNameMapper = {
        np.dtype("float32"): "GivenTensorFill",
        np.dtype("int32"): "GivenTensorIntFill",
        np.dtype("int64"): "GivenTensorInt64Fill",
        np.dtype("uint8"): "GivenTensorStringFill",
    }

    args_dict = {}
    if tensor.dtype == np.dtype("uint8"):
        args_dict.update(
            {
                "values": [str(tensor.data)],
                "shape": [1],
            }
        )
    else:
        args_dict.update({"values": tensor, "shape": tensor.shape})

    if device_option is not None:
        args_dict["device_option"] = device_option

    return core.CreateOperator(kTypeNameMapper[tensor.dtype], [], [name], **args_dict)


def _create_const_fill_op_from_c2_int8_tensor(name, int8_tensor):
    assert type(int8_tensor) == workspace.Int8Tensor
    kTypeNameMapper = {
        np.dtype("int32"): "Int8GivenIntTensorFill",
        np.dtype("uint8"): "Int8GivenTensorFill",
    }

    tensor = int8_tensor.data
    assert tensor.dtype in [np.dtype("uint8"), np.dtype("int32")]
    values = tensor.tobytes() if tensor.dtype == np.dtype("uint8") else tensor

    return core.CreateOperator(
        kTypeNameMapper[tensor.dtype],
        [],
        [name],
        values=values,
        shape=tensor.shape,
        Y_scale=int8_tensor.scale,
        Y_zero_point=int8_tensor.zero_point,
    )


def create_const_fill_op(
    name: str,
    blob: Union[np.ndarray, workspace.Int8Tensor],
    device_option: Optional[caffe2_pb2.DeviceOption] = None,
) -> caffe2_pb2.OperatorDef:
    """
    Given a blob object, return the Caffe2 operator that creates this blob
    as constant. Currently support NumPy tensor and Caffe2 Int8Tensor.
    """

    tensor_type = type(blob)
    assert tensor_type in [np.ndarray, workspace.Int8Tensor], (
        'Error when creating const fill op for "{}", unsupported blob type: {}'
    ).format(name, type(blob))

    if tensor_type == np.ndarray:
        return _create_const_fill_op_from_numpy(name, blob, device_option)
    elif tensor_type == workspace.Int8Tensor:
        assert device_option is None
        return _create_const_fill_op_from_c2_int8_tensor(name, blob)


def get_producer_map(ssa):
    """
    Return dict from versioned blob to (i, j),
        where i is index of producer op, j is the index of output of that op.
    """
    producer_map = {}
    for i in range(len(ssa)):
        outputs = ssa[i][1]
        for j, outp in enumerate(outputs):
            producer_map[outp] = (i, j)
    return producer_map


def get_consumer_map(ssa):
    """
    Return dict from versioned blob to list of (i, j),
        where i is index of consumer op, j is the index of input of that op.
    """
    consumer_map = collections.defaultdict(list)
    for i in range(len(ssa)):
        inputs = ssa[i][0]
        for j, inp in enumerate(inputs):
            consumer_map[inp].append((i, j))
    return consumer_map


def construct_init_net_from_params(
    params: Dict[str, Any],
    device_options: Optional[Dict[str, caffe2_pb2.DeviceOption]] = None,
) -> caffe2_pb2.NetDef:
    """
    Construct the init_net from params dictionary
    """
    init_net = caffe2_pb2.NetDef()
    device_options = device_options or {}
    for name, blob in params.items():
        if isinstance(blob, str):
            logger.warning(
                (
                    "Blob {} with type {} is not supported in generating init net,"
                    " skipped.".format(name, type(blob))
                )
            )
            continue
        init_net.op.extend(
            [
                create_const_fill_op(
                    name, blob, device_option=device_options.get(name, None)
                )
            ]
        )
        init_net.external_output.append(name)
    return init_net


def get_params_from_init_net(
    init_net: caffe2_pb2.NetDef,
) -> [Dict[str, Any], Dict[str, caffe2_pb2.DeviceOption]]:
    """
    Take the output blobs from init_net by running it.
    Outputs:
        params: dict from blob name to numpy array
        device_options: dict from blob name to the device option of its creating op
    """

    # NOTE: this assumes that the params is determined by producer op with the
    # only exception be CopyGPUToCPU which is CUDA op but returns CPU tensor.
    def _get_device_option(producer_op):
        if producer_op.type == "CopyGPUToCPU":
            return caffe2_pb2.DeviceOption()
        else:
            return producer_op.device_option

    with ScopedWS("__get_params_from_init_net__", is_reset=True, is_cleanup=True) as ws:
        ws.RunNetOnce(init_net)
        params = {b: fetch_any_blob(b) for b in init_net.external_output}
    ssa, versions = core.get_ssa(init_net)
    producer_map = get_producer_map(ssa)
    device_options = {
        b: _get_device_option(init_net.op[producer_map[(b, versions[b])][0]])
        for b in init_net.external_output
    }
    return params, device_options


def _updater_raise(op, input_types, output_types):
    raise RuntimeError(
        "Failed to apply updater for op {} given input_types {} and"
        " output_types {}".format(op, input_types, output_types)
    )


def _generic_status_identifier(
    predict_net: caffe2_pb2.NetDef,
    status_updater: Callable,
    known_status: Dict[Tuple[str, int], Any],
) -> Dict[Tuple[str, int], Any]:
    """
    Statically infer the status of each blob, the status can be such as device type
        (CPU/GPU), layout (NCHW/NHWC), data type (float32/int8), etc. "Blob" here
        is versioned blob (Tuple[str, int]) in the format compatible with ssa.
    Inputs:
        predict_net: the caffe2 network
        status_updater: a callable, given an op and the status of its input/output,
            it returns the updated status of input/output. `None` is used for
            representing unknown status.
        known_status: a dict containing known status, used as initialization.
    Outputs:
        A dict mapping from versioned blob to its status
    """
    ssa, versions = core.get_ssa(predict_net)
    versioned_ext_input = [(b, 0) for b in predict_net.external_input]
    versioned_ext_output = [(b, versions[b]) for b in predict_net.external_output]
    all_versioned_blobs = set().union(*[set(x[0] + x[1]) for x in ssa])

    allowed_vbs = all_versioned_blobs.union(versioned_ext_input).union(
        versioned_ext_output
    )
    assert all(k in allowed_vbs for k in known_status)
    assert all(v is not None for v in known_status.values())
    _known_status = copy.deepcopy(known_status)

    def _check_and_update(key, value):
        assert value is not None
        if key in _known_status:
            if not _known_status[key] == value:
                raise RuntimeError(
                    "Confilict status for {}, existing status {}, new status {}".format(
                        key, _known_status[key], value
                    )
                )
        _known_status[key] = value

    def _update_i(op, ssa_i):
        versioned_inputs = ssa_i[0]
        versioned_outputs = ssa_i[1]

        inputs_status = [_known_status.get(b, None) for b in versioned_inputs]
        outputs_status = [_known_status.get(b, None) for b in versioned_outputs]

        new_inputs_status, new_outputs_status = status_updater(
            op, inputs_status, outputs_status
        )

        for versioned_blob, status in zip(
            versioned_inputs + versioned_outputs, new_inputs_status + new_outputs_status
        ):
            if status is not None:
                _check_and_update(versioned_blob, status)

    for op, ssa_i in zip(predict_net.op, ssa):
        _update_i(op, ssa_i)
    for op, ssa_i in zip(reversed(predict_net.op), reversed(ssa)):
        _update_i(op, ssa_i)

    # NOTE: This strictly checks all the blob from predict_net must be assgined
    # a known status. However sometimes it's impossible (eg. having deadend op),
    # we may relax this constraint if
    for k in all_versioned_blobs:
        if k not in _known_status:
            raise NotImplementedError(
                "Can not infer the status for {}. Currently only support the case where"
                " a single forward and backward pass can identify status for all blobs.".format(
                    k
                )
            )

    return _known_status


def infer_device_type(
    predict_net: caffe2_pb2.NetDef,
    known_status: Dict[Tuple[str, int], Any],
    device_name_style: str = "caffe2",
) -> Dict[Tuple[str, int], str]:
    """Return the device type ("cpu" or "gpu"/"cuda") of each (versioned) blob"""

    assert device_name_style in ["caffe2", "pytorch"]
    _CPU_STR = "cpu"
    _GPU_STR = "gpu" if device_name_style == "caffe2" else "cuda"

    def _copy_cpu_to_gpu_updater(op, input_types, output_types):
        if input_types[0] == _GPU_STR or output_types[0] == _CPU_STR:
            _updater_raise(op, input_types, output_types)
        return ([_CPU_STR], [_GPU_STR])

    def _copy_gpu_to_cpu_updater(op, input_types, output_types):
        if input_types[0] == _CPU_STR or output_types[0] == _GPU_STR:
            _updater_raise(op, input_types, output_types)
        return ([_GPU_STR], [_CPU_STR])

    def _other_ops_updater(op, input_types, output_types):
        non_none_types = [x for x in input_types + output_types if x is not None]
        if len(non_none_types) > 0:
            the_type = non_none_types[0]
            if not all(x == the_type for x in non_none_types):
                _updater_raise(op, input_types, output_types)
        else:
            the_type = None
        return ([the_type for _ in op.input], [the_type for _ in op.output])

    def _device_updater(op, *args, **kwargs):
        return {
            "CopyCPUToGPU": _copy_cpu_to_gpu_updater,
            "CopyGPUToCPU": _copy_gpu_to_cpu_updater,
        }.get(op.type, _other_ops_updater)(op, *args, **kwargs)

    return _generic_status_identifier(predict_net, _device_updater, known_status)
