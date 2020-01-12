#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import argparse
import json
import os
import sys

from caffe2.python import workspace

import model_utils
from api import RunTimeAPI

sys.path.append("runtime_lut/code/")


EXAMPLE_MODEL_1 = "ChamNet/models/ChamNet-E/int8"
EXAMPLE_MODEL_2 = "ChamNet/models/ChamNet-E/fp32"

DB_DIR = "runtime_lut/data"

EXAMPLE_MODELS = ",".join([EXAMPLE_MODEL_1, EXAMPLE_MODEL_2])

INPUT_DIMS = {"data": [1, 3, 160, 160]}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default=EXAMPLE_MODELS,
        help="Multiple models for benchmarking (separate with comma)",
    )
    parser.add_argument("--db_dir", type=str, default=DB_DIR)
    parser.add_argument(
        "--input_dims",
        type=lambda x: json.loads(x),
        default=json.dumps(INPUT_DIMS),
    )
    parser.add_argument("--input_dtype", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default="SM-G950U-7.0-24",
        help="SM-G950U-7.0-24: S8 US",
    )
    args = parser.parse_args()
    return args


def load_pb(model_path):
    # Load a caffe2 pb model and generate blobs
    model_path = os.path.join(model_path, "model.pb")
    model_init_path = (
        (model_path + "/model_init.pb")
        if os.path.exists(model_path + "/model_init.pb")
        else None
    )

    model, _ = model_utils.load_model_pb(
        model_path, model_init_path, is_run_init=True
    )

    return model


def main():
    args = parse_args()
    api = RunTimeAPI(db_dir=args.db_dir)
    model_paths = args.model_path.split(",")
    for model_path in model_paths:
        workspace.ResetWorkspace()
        workspace.GlobalInit(["caffe2", "--caffe2_log_level=0"])
        model = load_pb(model_path)
        blobs = model_utils.get_ws_blobs()
        # TODO: multiple inputs (e.g., for detection)
        op_time, net_time, all_found = api.get_net_runtime(
            model,
            blobs,
            args.input_dims,
            args.input_dtype,
            args.device,
            verbose=True,
        )

        print("All operators found in the table? {}".format(all_found))
        print("The network run-time latency is {}".format(net_time))


if __name__ == "__main__":
    main()
