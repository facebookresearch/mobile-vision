#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import copy
import os

import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace


class ScopedWS(object):
    def __init__(self, ws_name, is_reset, is_cleanup=False):
        self.ws_name = ws_name
        self.is_reset = is_reset
        self.is_cleanup = is_cleanup
        self.org_ws = ""

    def __enter__(self):
        self.org_ws = workspace.CurrentWorkspace()
        if self.ws_name is not None:
            workspace.SwitchWorkspace(self.ws_name, True)
        if self.is_reset:
            workspace.ResetWorkspace()

    def __exit__(self, *args):
        if self.is_cleanup:
            workspace.ResetWorkspace()
        if self.ws_name is not None:
            workspace.SwitchWorkspace(self.org_ws)


def create_blobs_if_not_existed(blob_names):
    existd_names = set(workspace.Blobs())
    for xx in blob_names:
        if xx not in existd_names:
            workspace.CreateBlob(str(xx))


def load_model_pb(
    net_file, init_file=None, is_run_init=True, is_create_net=True
):
    net = core.Net("net")
    if net_file is not None:
        net.Proto().ParseFromString(open(net_file, "rb").read())

    if init_file is None:
        fn, ext = os.path.splitext(net_file)
        init_file = fn + "_init" + ext

    init_net = caffe2_pb2.NetDef()
    init_net.ParseFromString(open(init_file, "rb").read())

    if is_run_init:
        workspace.RunNetOnce(init_net)
        create_blobs_if_not_existed(net.external_inputs)
        if net.Proto().name == "":
            net.Proto().name = "net"
    if is_create_net:
        workspace.CreateNet(net)

    return (net, init_net)


def create_fill_op(name, blob, device_option=None):
    """ Create an operator to store the tensor 'blob',
        return the operator
    """
    kTypeNameMapper = {
        np.dtype("float32"): "GivenTensorFill",
        np.dtype("int32"): "GivenTensorIntFill",
        np.dtype("int64"): "GivenTensorInt64Fill",
        np.dtype("uint8"): "GivenTensorStringFill",
        workspace.Int8Tensor: {
            np.dtype("int32"): "Int8GivenIntTensorFill",
            np.dtype("uint8"): "Int8GivenTensorFill",
        },
    }

    try:
        blob_type = blob.dtype
    except AttributeError:
        blob_type = type(blob)
    except Exception as e:
        print("Error when geting blob type {}: {}\n{}".format(name, blob, e))
        raise

    op_type = kTypeNameMapper[blob_type]
    args_dict = {}

    if blob_type == np.dtype("uint8"):
        args_dict.update({"values": [str(blob.data)], "shape": [1]})
    elif blob_type == workspace.Int8Tensor:
        data_type = blob.data.dtype
        shape = blob.data.shape
        assert data_type in [np.dtype("uint8"), np.dtype("int32")]

        op_type = op_type[data_type]
        values = blob.data
        scale = blob.scale
        zero_point = blob.zero_point

        if data_type == np.dtype("uint8"):
            values = values.tobytes()
        args_dict.update(
            {
                "values": values,
                "shape": shape,
                "Y_scale": scale,
                "Y_zero_point": zero_point,
            }
        )
    else:
        args_dict.update({"values": blob, "shape": blob.shape})

    if device_option is not None:
        args_dict["device_option"] = device_option

    op = core.CreateOperator(op_type, [], [name], **args_dict)
    return op


def get_ws_blobs(ws=None, blob_names=None):
    """ Get blobs in 'blob_names' in workspace 'ws',
        get all blobs if blob_names is None """
    blobs = {}
    with ScopedWS(ws, False):
        if blob_names is None:
            blob_names = workspace.Blobs()
        blobs = {x: _get_blob(x) for x in blob_names}

    return blobs


def _get_blob(name):
    bb = None
    try:
        bb = workspace.FetchBlob(name)
    except TypeError:
        bb = workspace.FetchInt8Blob(name)
    except Exception as e:
        print("Get blob {} error: {}".format(name, e))

    return bb


def _get_blob_shape(blob):
    bb = None
    try:
        bb = blob.shape
    except AttributeError:
        try:
            bb = blob.data.shape
        except Exception as e:
            print("Get blob shape {} error: {}".format(blob, e))
    except Exception as e:
        print("Get blob shape {} error: {}".format(blob, e))

    return bb


def _get_blob_dtype(blob):
    bb = None
    try:
        bb = blob.dtype
    except AttributeError:
        try:
            bb = blob.data.dtype
        except Exception as e:
            print("Get blob dtype {} error: {}".format(blob, e))
    except Exception as e:
        print("Get blob dtype {} error: {}".format(blob, e))

    return bb


def _get_blobs(blob_names):
    ret = [_get_blob(x) for x in blob_names]
    return ret


def _shape_to_list(shape):
    if shape is None:
        return None
    return list(shape)


def get_blobs_as_dict(blob_names):
    ret = {x: _get_blob(x) for x in blob_names}
    return ret


def get_blobs_shapes(blobs):
    blob_dims = {
        x: _shape_to_list(_get_blob_shape(blobs[x]))
        for x in blobs
        if blobs[x] is not None and not isinstance(blobs[x], (str, bytes))
    }
    return blob_dims


def get_blobs_dtypes(blobs):
    blob_dtypes = {
        x: _get_blob_dtype(blobs[x])
        for x in blobs
        if blobs[x] is not None and not isinstance(blobs[x], (str, bytes))
    }
    return blob_dtypes


def infer_model_shape(net, param_init_net, extra_inputs):
    with ScopedWS("__ws_tmp__", True):
        if param_init_net:
            workspace.RunNetOnce(param_init_net)
        for x in extra_inputs:
            workspace.FeedBlob(x, extra_inputs[x])
        try:
            workspace.RunNetOnce(net)
        except Exception as e:
            print("Run net error {}".format(e))
        blobs = {x: _get_blob(x) for x in workspace.Blobs()}
        ret = get_blobs_shapes(blobs)
        workspace.ResetWorkspace()

    return ret


def feed_blobs(name, blob, device_option=None):
    # feed data to workspace blobs, input can be both np.ndarray or Int8Tensor
    if isinstance(blob, caffe2_pb2.OperatorDef):
        op = blob
    else:
        op = create_fill_op(name, blob, device_option)
    try:
        workspace.RunOperatorOnce(op)
    except Exception as err:
        print("Feed blob {} error:\n{}".format(name, err))


def run_op(idx, op, extra_inputs):
    if extra_inputs:
        for x in op.input:
            if x in extra_inputs and type(extra_inputs[x]) != bytes:
                feed_blobs(x, extra_inputs[x])
    for x in op.input:
        bb = _get_blob(x)
        assert bb is not None, "Blob {} is None".format(x)
    workspace.RunOperatorOnce(op)
    for x in op.output:
        bb = _get_blob(x)
        assert bb is not None, "Blob {} is None".format(x)


def rand_blob(blob, scale=0.01, zero_point=0):
    assert isinstance(blob, workspace.Int8Tensor)

    data_type = blob.data.dtype
    shape = blob.data.shape

    values = np.array(np.random.randint(0, 255, size=shape), dtype=data_type)
    scale = scale
    zero_point = zero_point

    rand_blob = workspace.Int8Tensor(values, scale, zero_point)
    return rand_blob


def infer_model_shape_by_ops_raw(net, extra_inputs=None, skip_op_idx=None):
    def _run_op(idx, op):
        if skip_op_idx is not None and idx in skip_op_idx:
            print("Op #{} {}({}) skipped.".format(idx, op.type, op.name))
            return

        if op.type == "Int8Softmax":
            bb = _get_blob(op.input[0])
            blob = rand_blob(bb)
            feed_blobs(op.input[0], blob)

        run_op(idx, op, extra_inputs)

    for ix, x in enumerate(net.Proto().op):
        _run_op(ix, x)
    blobs = {x: _get_blob(x) for x in workspace.Blobs()}
    ret = get_blobs_shapes(blobs)
    dtype = get_blobs_dtypes(blobs)

    return ret, dtype


def infer_model_shape_by_ops(
    net,
    param_init_net=None,
    extra_inputs=None,
    is_run_gpu=False,
    skip_op_idx=None,
    get_dtype=False,
):
    cdo = get_device_option_cpu()
    if is_run_gpu:
        net.RunAllOnGPU()
        if param_init_net:
            param_init_net.RunAllOnGPU()
    with core.DeviceScope(cdo):
        if param_init_net:
            workspace.RunNetOnce(param_init_net)
        ret, dtype = infer_model_shape_by_ops_raw(
            net, extra_inputs=extra_inputs, skip_op_idx=skip_op_idx
        )
        workspace.ResetWorkspace()

    if not get_dtype:
        return ret
    else:
        return ret, dtype


def infer_model_shape_by_ops_shape(
    net,
    param_init_net=None,
    extra_input_shapes=None,
    is_run_gpu=False,
    extra_inputs=None,
):
    _extra_inputs = copy.deepcopy(extra_inputs) if extra_inputs else {}
    if extra_input_shapes is not None:
        eis = {}
        if isinstance(extra_input_shapes, list):
            ext_inputs = net.Proto().external_input
            assert len(extra_input_shapes) <= len(ext_inputs)
            for idx, x in enumerate(extra_input_shapes):
                assert isinstance(
                    x, (tuple, list)
                ), "{} in extra_input_shapes must be a list".format(x)
                eis[ext_inputs[idx]] = x
        else:
            eis = extra_input_shapes
        for x in eis:
            _extra_inputs[x] = np.zeros(eis[x], dtype=np.float32)
        print("Extra inputs:")
        for x in _extra_inputs:
            print("  {}: {}".format(x, _extra_inputs[x].shape))
    return infer_model_shape_by_ops(
        net, param_init_net, _extra_inputs, is_run_gpu
    )


def get_device_option_cpu():
    device_option = core.DeviceOption(caffe2_pb2.CPU)
    return device_option
