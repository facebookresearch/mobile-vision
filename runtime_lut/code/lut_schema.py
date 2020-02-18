#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json

import caffe2
from caffe2.python import core

# These two int8 operator args are not required to be matched.
ARGS_IGNORE = ["Y_scale", "Y_zero_point"]


def load_caffe2_op(
    op_record,
    caffe2_op,
    input_shapes,
    input_dtypes=core.DataType.FLOAT,
    device="SM-G930F-7.0-24",
    model_name="",
):
    assert isinstance(caffe2_op, caffe2.proto.caffe2_pb2.OperatorDef)
    input_shapes = list(input_shapes)

    if isinstance(input_dtypes, list):
        assert len(input_dtypes) == len(input_shapes)
    else:
        assert isinstance(input_dtypes, int)
        input_dtypes = [input_dtypes] * len(input_shapes)

    op_record.set_val("op_type", caffe2_op.type)
    op_args = [arg for arg in caffe2_op.arg if arg.name not in ARGS_IGNORE]
    op_args.sort(key=lambda x: x.name)
    op_record.set_val(
        "op_args", [str(a) for a in op_args if "ws_nbytes_limit" not in str(a)]
    )
    op_record.set_val("input_shapes", input_shapes)
    op_record.set_val("input_dtypes", input_dtypes)
    op_record.set_val("device", device)


class LUTSchema:
    """
    Immutable dictionary that defines the LUT schema.
    """

    def __init__(self):
        self.record = {}
        # operator type
        self.record["op_type"] = ""
        # a list of argument definitions, sorted by argument names. Each argument is
        # represented by its proto string,  containing argument name and values.
        self.record["op_args"] = []
        # a list of input shapes. E.g., a convolution op may have shapes like:
        #   [[1, 3, 224, 224], [16, 3, 3, 3], [16]]
        # which are input tensor shape, kernel shape, and bias shape
        self.record["input_shapes"] = []
        # a list of integers indicating input data types
        self.record["input_dtypes"] = []
        # runtime measurement in micro-second, percentile-0
        self.record["runtime_us_p0"] = 0.0
        # runtime measurement in micro-second, percentile-10
        self.record["runtime_us_p10"] = 0.0
        # runtime measurement in micro-second, percentile-50
        self.record["runtime_us_p50"] = 0.0
        # runtime measurement in micro-second, percentile-90
        self.record["runtime_us_p90"] = 0.0
        # runtime measurement in micro-second, percentile-100
        self.record["runtime_us_p100"] = 0.0
        # benchmarking device
        self.record["device"] = ""

    def get_val(self, key):
        assert key in self.record, "%s not in schema" % key
        return self.record[key]

    def set_val(self, key, val):
        assert key in self.record, "%s not in schema" % key
        self.record[key] = val

    def get_dict_record(self):
        return self.record

    def load_from_json(self, input_):
        assert isinstance(input_, str) or isinstance(input_, dict)
        if isinstance(input_, str):
            record = json.loads(input_)
        elif isinstance(input_, dict):
            record = input_
        for k, v in record.items():
            if k == "op_args":
                v = [str(a) for a in v if "ws_nbytes_limit" not in str(a)]
            self.set_val(k, v)
