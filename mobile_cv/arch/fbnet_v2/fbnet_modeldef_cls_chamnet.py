#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .fbnet_modeldef_cls import MODEL_ARCH
from .modeldef_utils import _ex, e1, e4, e6


use_bias = {"bias": True}

BASIC_ARGS = {
    "width_divisor": 8,
}


MODEL_ARCH_CHAM = {
    "cham_a": {
        "input_size": 224,
        "blocks": [
            # stage 0
            [("conv_k3", 32, 2, 1)],
            # stage 1
            [("ir_k3", 24, 1, 1, e1)],
            # stage 2
            [("ir_k3", 40, 2, 2, e4)],
            # stage 3
            [("ir_k3", 48, 2, 3, e6)],
            # stage 4
            [("ir_k3", 96, 2, 4, e4), ("ir_k3", 160, 1, 3, e4)],
            # stage 5
            [("ir_k3", 248, 2, 3, e4), ("ir_k3", 480, 1, 1, e6)],
            # stage 6
            [("conv_k1", 2016, 1, 1)],
        ],
    },
    "cham_b": {
        "input_size": 224,
        "blocks": [
            # stage 0
            [("conv_k3", 24, 2, 1)],
            # stage 1
            [("ir_k3", 16, 1, 1, e1)],
            # stage 2
            [("ir_k3", 40, 2, 2, _ex(5))],
            # stage 3
            [("ir_k3", 40, 2, 3, e6)],
            # stage 4
            [("ir_k3", 92, 2, 3, e6), ("ir_k3", 104, 1, 3, e6)],
            # stage 5
            [("ir_k3", 200, 2, 3, _ex(5)), ("ir_k3", 384, 1, 1, e6)],
            # stage 6
            [("conv_k1", 1984, 1, 1)],
        ],
    },
    #     "cham_c": {
    #         "block_op_type": {
    #             "first": "conv",
    #             "stages": [
    #                 # stage 0
    #                 ["ir_k3"],
    #                 # stage 1
    #                 ["ir_k3"],
    #                 # stage 2
    #                 ["ir_k3"] * 3,
    #                 # stage 3
    #                 ["ir_k3"] * 5,
    #                 # stage 4
    #                 ["ir_k3"] * 4,
    #             ],
    #         },
    #         "block_cfg": {
    #             "first": [16, 2],
    #             "stages": [
    #                 [[1, 16, 1, 1]],
    #                 [[4, 24, 1, 2]],
    #                 [[4, 32, 3, 2]],
    #                 [[5, 64, 2, 2], [6, 96, 3, 1]],
    #                 [[6, 160, 3, 2], [4, 256, 1, 1]],
    #             ],
    #             "last": [1520, 1.0],
    #         },
    #     },
    #     "cham_d": {
    #         "block_op_type": {
    #             "first": "conv",
    #             "stages": [
    #                 # stage 0
    #                 ["ir_k3"],
    #                 # stage 1
    #                 ["ir_k3"],
    #                 # stage 2
    #                 ["ir_k3"] * 2,
    #                 # stage 3
    #                 ["ir_k3"] * 7,
    #                 # stage 4
    #                 ["ir_k3"] * 4,
    #             ],
    #         },
    #         "block_cfg": {
    #             "first": [8, 2],
    #             "stages": [
    #                 [[1, 8, 1, 1]],
    #                 [[6, 24, 1, 2]],
    #                 [[4, 32, 2, 2]],
    #                 [[6, 40, 4, 2], [6, 80, 3, 1]],
    #                 [[5, 160, 3, 2], [6, 224, 1, 1]],
    #             ],
    #             "last": [1600, 1.0],
    #         },
    #     },
    #     "cham_e": {
    #         "block_op_type": {
    #             "first": "conv",
    #             "stages": [
    #                 # stage 0
    #                 ["ir_k3"],
    #                 # stage 1
    #                 ["ir_k3"],
    #                 # stage 2
    #                 ["ir_k3"] * 3,
    #                 # stage 3
    #                 ["ir_k3"] * 4,
    #                 # stage 4
    #                 ["ir_k3"] * 3,
    #             ],
    #         },
    #         "block_cfg": {
    #             "first": [8, 2],
    #             "stages": [
    #                 [[1, 8, 1, 1]],
    #                 [[4, 24, 1, 2]],
    #                 [[4, 32, 3, 2]],
    #                 [[6, 64, 1, 2], [6, 64, 3, 1]],
    #                 [[6, 96, 2, 2], [4, 128, 1, 1]],
    #             ],
    #             "last": [1600, 1.0],
    #         },
    #     },
    #     "cham_f": {
    #         "block_op_type": {
    #             "first": "conv",
    #             "stages": [
    #                 # stage 0
    #                 ["ir_k3"],
    #                 # stage 1
    #                 ["ir_k3"],
    #                 # stage 2
    #                 ["ir_k3"] * 3,
    #                 # stage 3
    #                 ["ir_k3"] * 4,
    #                 # stage 4
    #                 ["ir_k3"] * 3,
    #             ],
    #         },
    #         "block_cfg": {
    #             "first": [8, 2],
    #             "stages": [
    #                 [[1, 8, 1, 1]],
    #                 [[4, 24, 1, 2]],
    #                 [[4, 32, 3, 2]],
    #                 [[4, 40, 1, 2], [4, 64, 3, 1]],
    #                 [[4, 128, 2, 2], [4, 192, 1, 1]],
    #             ],
    #             "last": [1280, 1.0],
    #         },
    #     },
    #     "cham_b192": {
    #         "input_size": 192,
    #         "block_op_type": {
    #             "first": "conv",
    #             "stages": [
    #                 # stage 0
    #                 ["ir_k3"],
    #                 # stage 1
    #                 ["ir_k3"] * 2,
    #                 # stage 2
    #                 ["ir_k3"] * 3,
    #                 # stage 3
    #                 ["ir_k3"] * 6,
    #                 # stage 4
    #                 ["ir_k3"] * 4,
    #             ],
    #         },
    #         "block_cfg": {
    #             "first": [24, 2],
    #             "stages": [
    #                 [[1, 16, 1, 1]],
    #                 [[5, 40, 2, 2]],
    #                 [[6, 40, 3, 2]],
    #                 [[6, 92, 3, 2], [6, 104, 3, 1]],
    #                 [[5, 200, 3, 2], [6, 384, 1, 1]],
    #             ],
    #             "last": [1984, 1.0],
    #         },
    #     },
    #     "cham_d192": {
    #         "input_size": 192,
    #         "block_op_type": {
    #             "first": "conv",
    #             "stages": [
    #                 # stage 0
    #                 ["ir_k3"],
    #                 # stage 1
    #                 ["ir_k3"],
    #                 # stage 2
    #                 ["ir_k3"] * 2,
    #                 # stage 3
    #                 ["ir_k3"] * 7,
    #                 # stage 4
    #                 ["ir_k3"] * 4,
    #             ],
    #         },
    #         "block_cfg": {
    #             "first": [8, 2],
    #             "stages": [
    #                 [[1, 8, 1, 1]],
    #                 [[6, 24, 1, 2]],
    #                 [[4, 32, 2, 2]],
    #                 [[6, 40, 4, 2], [6, 80, 3, 1]],
    #                 [[5, 160, 3, 2], [6, 224, 1, 1]],
    #             ],
    #             "last": [1600, 1.0],
    #         },
    #     },
    #     "cham_e160": {
    #         "input_size": 160,
    #         "block_op_type": {
    #             "first": "conv",
    #             "stages": [
    #                 # stage 0
    #                 ["ir_k3"],
    #                 # stage 1
    #                 ["ir_k3"],
    #                 # stage 2
    #                 ["ir_k3"] * 3,
    #                 # stage 3
    #                 ["ir_k3"] * 4,
    #                 # stage 4
    #                 ["ir_k3"] * 3,
    #             ],
    #         },
    #         "block_cfg": {
    #             "first": [8, 2],
    #             "stages": [
    #                 [[1, 8, 1, 1]],
    #                 [[4, 24, 1, 2]],
    #                 [[4, 32, 3, 2]],
    #                 [[6, 64, 1, 2], [6, 64, 3, 1]],
    #                 [[6, 96, 2, 2], [4, 128, 1, 1]],
    #             ],
    #             "last": [1600, 1.0],
    #         },
    #     },
    #     "cham_f128": {
    #         "input_size": 128,
    #         "block_op_type": {
    #             "first": "conv",
    #             "stages": [
    #                 # stage 0
    #                 ["ir_k3"],
    #                 # stage 1
    #                 ["ir_k3"],
    #                 # stage 2
    #                 ["ir_k3"] * 3,
    #                 # stage 3
    #                 ["ir_k3"] * 4,
    #                 # stage 4
    #                 ["ir_k3"] * 3,
    #             ],
    #         },
    #         "block_cfg": {
    #             "first": [8, 2],
    #             "stages": [
    #                 [[1, 8, 1, 1]],
    #                 [[4, 24, 1, 2]],
    #                 [[4, 32, 3, 2]],
    #                 [[4, 40, 1, 2], [4, 64, 3, 1]],
    #                 [[4, 128, 2, 2], [4, 192, 1, 1]],
    #             ],
    #             "last": [1280, 1.0],
    #         },
    #     },
}
MODEL_ARCH.register_dict(MODEL_ARCH_CHAM)
