#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
import os

import caffe2.proto.caffe2_pb2 as pb2
import numpy as np
from caffe2.python import core

import model_utils
from lut_schema import LUTSchema, load_caffe2_op

DB_DIR = "../"

TO_MATCH = ["op_type", "input_shapes", "input_dtypes", "device", "op_args"]

DTYPE_ENCODE = {
    "undefined": core.DataType.UNDEFINED,
    "float32": core.DataType.FLOAT,
    "int32": core.DataType.INT32,
    "byte": core.DataType.BYTE,
    "string": core.DataType.STRING,
    "bool": core.DataType.BOOL,
    "uint8": core.DataType.UINT8,
    "int8": core.DataType.INT8,
    "uint16": core.DataType.UINT16,
    "int16": core.DataType.INT16,
    "int64": core.DataType.INT64,
    "float16": core.DataType.FLOAT16,
    "double": core.DataType.DOUBLE,
}


def encode_dtype(np_dtype):
    res = DTYPE_ENCODE.get(np_dtype, None)
    if res is None:
        print("Unsupported encoding datatype!")
    return res


def decode_dtype(caffe2_dtype):
    for dtype in DTYPE_ENCODE:
        if DTYPE_ENCODE[dtype] == caffe2_dtype:
            return dtype
    print("Unsupported decoding datatype!")
    return None


def get_ops_from_net(model, blobs, input_dims):
    # Extract all operators and corresponding input shapes from a model
    blobs = {x: blobs[x] for x in blobs if not isinstance(blobs[x], str)}
    blobs.update(
        {x: np.ones(input_dims[x], dtype=np.float32) for x in input_dims}
    )

    blobs_dims, blobs_dtypes = model_utils.infer_model_shape_by_ops(
        model, extra_inputs=blobs, get_dtype=True
    )

    blobs = {
        x: np.zeros(blobs_dims[x], dtype=blobs_dtypes[x]) for x in blobs_dims
    }

    if type(model) == pb2.NetDef:
        proto = model
    else:
        proto = model.Proto()

    model_ops = proto.op

    ops, input_shapes, input_dtypes = [], [], []
    for op in model_ops:
        op_type, op_inputs = op.type, op.input

        if op_type in ["Conv", "FC"]:
            assert len(op_inputs) == 2 or len(op_inputs) == 3

        param_shape = [
            np.array(blobs[str(param_blob)]).shape for param_blob in op_inputs
        ]
        param_dtypes = [
            encode_dtype(str(blobs[str(param_blob)].dtype))
            for param_blob in op_inputs
        ]

        input_shapes.append(param_shape)
        input_dtypes.append(param_dtypes)
        ops.append(op)

    return ops, input_shapes, input_dtypes


class OpLut(object):
    def __init__(self, rand_digits=8):
        self.rand_digits = rand_digits
        self.ops = []

    def clear(self):
        self.ops = []

    def add_op(self, op_record):
        """
        Add a new record to the OpLut
        Args:
            op_record: a LUTSchema instance to be added
        """
        assert isinstance(op_record, LUTSchema)

        if self.find_op(op_record) != []:
            print("Operator already exists.")
            return
        self.ops.append(op_record)

    def _is_match(self, op_query, op):
        for key in TO_MATCH:
            if op_query.get_val(key) != op.get_val(key):
                return False
        return True

    def find_op(self, op_query):
        """
        Find an operator in the database
        Args:
            op_query: a LUTSchema instance, the operator to find in the database
        """
        records = []
        for op in self.ops:
            if self._is_match(op_query, op):
                records.append(op)

        if records == []:
            if op_query.get_val("op_type") == "Int8Conv":
                op_query.set_val("op_type", "Int8ConvRelu")
                return self.find_op(op_query)

            print("No match found in the database!")
        return records

    def load(self, dbfile):
        """
        Load a database file and convert all records to LUTSchema
        Args:
            dbfile: path to the database file
        """
        self.ops = []
        assert os.path.exists(dbfile)

        db_file = open(dbfile, "r")
        db = db_file.readlines()

        for record in db:
            op = LUTSchema()
            op.load_from_json(json.loads(record))
            self.ops.append(op)

        db_file.close()


class RunTimeAPI(object):
    def __init__(self, db_dir=DB_DIR, rand_digits=8):
        self.db_dir = db_dir
        self.rand_digits = rand_digits

        self.op_lut = OpLut()

        self.current_db = ""

    def get_op_runtime(self, op_query, mode="ave", verbose=True):
        """
        Extract the runtime of an operator from the database
        Args:
            op_query: a LUTSchema instance contains information about the op
            mode: return average/maximum/minmum/last runtime if multiple records
                found in the database, can only be 'ave'/'max'/'min'/'last'
        """
        records = self.op_lut.find_op(op_query)
        if records == []:
            return None

        assert mode in ["min", "max", "ave", "last"]
        run_times = [op.get_val("runtime_us_p50") for op in records]

        if verbose:
            print(
                "op type {} with input {}".format(
                    op_query.get_val("op_type"),
                    op_query.get_val("input_shapes"),
                )
            )

            print("run_time: {}".format(run_times))

        if mode == "min":
            return min(run_times)
        if mode == "max":
            return max(run_times)
        if mode == "ave":
            return sum(run_times) / float(len(run_times))
        if mode == "last":
            return run_times[-1]

    def get_net_runtime(
        self,
        model,
        blobs,
        input_dims,
        input_dtype=core.DataType.FLOAT,
        device="SM-G950U-7.0-24",
        mode="ave",
        verbose=True,
    ):
        """
        Calculated the run-time latency for a given model from the loop-up table
        Args:
            model: caffe2 pb
            input_dims: input dimensions
            device: device used for benchmarking
            wait: if not found, whether to wait for the result
        """
        db_file = os.path.join(
            self.db_dir, device + "-" + "op_lut_database.json"
        )

        if self.current_db != db_file:
            self.current_db = db_file
            self.op_lut.load(db_file)

        ops, input_shapes, input_dtypes = get_ops_from_net(
            model, blobs, input_dims
        )

        for i in range(len(input_shapes)):
            input_shapes[i] = map(lambda x: list(x), input_shapes[i])

        devices = [device for _ in range(len(input_shapes))]

        while True:
            op_time, net_time, all_found = [], 0, True
            for i, op in enumerate(ops):
                op_query = LUTSchema()
                load_caffe2_op(
                    op_query,
                    op,
                    input_shapes=input_shapes[i],
                    input_dtypes=input_dtypes[i],
                    device=devices[i],
                    model_name=str(model),
                )

                latency = self.get_op_runtime(
                    op_query, mode=mode, verbose=verbose
                )

                op_time.append({"op": op_query, "time": latency})

                if latency is not None:
                    net_time += latency
                else:
                    all_found = False

            return op_time, net_time, all_found
