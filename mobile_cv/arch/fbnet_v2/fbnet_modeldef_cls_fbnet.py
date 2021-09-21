#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from . import modeldef_utils as mdu
from .fbnet_modeldef_cls import MODEL_ARCH
from .modeldef_utils import e1, e3, e6


use_bias = {"bias": True}

MODEL_ARCH_FBNET = {
    "fbnet_a": {
        "input_size": 224,
        "blocks": [
            # [op, c, s, n, ...]
            # stage 0
            [["conv_k3", 16, 2, 1]],
            # stage 1
            [["skip", 16, 1, 1]],
            # stage 2
            [
                ["ir_k3", 24, 2, 1, e3, use_bias],
                ["ir_k3", 24, 1, 1, e1, use_bias],
                ["skip", 24, 1, 1, use_bias],
                ["skip", 24, 1, 1, use_bias],
            ],
            # stage 3
            [
                ["ir_k5", 32, 2, 1, e6, use_bias],
                ["ir_k3", 32, 1, 1, e3, use_bias],
                ["ir_k5", 32, 1, 1, e1, use_bias],
                ["ir_k3", 32, 1, 1, e3, use_bias],
            ],
            # stage 4
            [
                ["ir_k5", 64, 2, 1, e6, use_bias],
                ["ir_k5", 64, 1, 1, e3, use_bias],
                ["ir_k5_g2", 64, 1, 1, e1, use_bias, {"pwl_groups": 2}],
                ["ir_k5", 64, 1, 1, e6, use_bias],
                ["ir_k3", 112, 1, 1, e6, use_bias],
                ["ir_k5_g2", 112, 1, 1, e1, use_bias, {"pwl_groups": 2}],
                ["ir_k5", 112, 1, 1, e3, use_bias],
                ["ir_k3_g2", 112, 1, 1, e1, use_bias, {"pwl_groups": 2}],
            ],
            # stage 5
            [
                ["ir_k5", 184, 2, 1, e6, use_bias],
                ["ir_k5", 184, 1, 1, e6, use_bias],
                ["ir_k5", 184, 1, 1, e3, use_bias],
                ["ir_k5", 184, 1, 1, e6, use_bias],
                ["ir_k5", 352, 1, 1, e6, use_bias],
            ],
            # stage 5
            [("conv_k1", 1504, 1, 1)],
        ],
    },
    "fbnet_b": {
        "input_size": 224,
        "blocks": [
            # [op, c, s, n, ...]
            # stage 0
            [["conv_k3", 16, 2, 1]],
            # stage 1
            [["ir_k3", 16, 1, 1, e1, use_bias]],
            # stage 2
            [
                ["ir_k3", 24, 2, 1, e6, use_bias],
                ["ir_k5", 24, 1, 1, e1, use_bias],
                ["ir_k3", 24, 1, 1, e1, use_bias],
                ["ir_k3", 24, 1, 1, e1, use_bias],
            ],
            # stage 3
            [
                ["ir_k5", 32, 2, 1, e6, use_bias],
                ["ir_k5", 32, 1, 1, e3, use_bias],
                ["ir_k3", 32, 1, 1, e6, use_bias],
                # ["ir_k3_sep", 32, 1, 1, e6, use_bias]],
                ["ir_k5", 32, 1, 1, e6, use_bias],
            ],
            # stage 4
            [
                ["ir_k5", 64, 2, 1, e6, use_bias],
                ["ir_k5", 64, 1, 1, e6, use_bias],
                ["skip", 64, 1, 1],
                ["ir_k5", 64, 1, 1, e3, use_bias],
                ["ir_k5", 112, 1, 1, e6, use_bias],
                ["ir_k3", 112, 1, 1, e1, use_bias],
                ["ir_k5", 112, 1, 1, e1, use_bias],
                ["ir_k5", 112, 1, 1, e3, use_bias],
            ],
            # stage 5
            [
                ["ir_k5", 184, 2, 1, e6, use_bias],
                ["ir_k5", 184, 1, 1, e1, use_bias],
                ["ir_k5", 184, 1, 1, e6, use_bias],
                ["ir_k5", 184, 1, 1, e6, use_bias],
                ["ir_k3", 352, 1, 1, e6, use_bias],
            ],
            # stage 6
            [("conv_k1", 1984, 1, 1)],
        ],
    },
    "fbnet_c": {
        "input_size": 224,
        "blocks": [
            # [op, c, s, n, ...]
            # stage 0
            [["conv_k3", 16, 2, 1]],
            # stage 1
            [["ir_k3", 16, 1, 1, e1, use_bias]],
            # stage 2
            [
                ["ir_k3", 24, 2, 1, e6, use_bias],
                ["skip", 24, 1, 1, e1, use_bias],
                ["ir_k3", 24, 1, 1, e1, use_bias],
                ["ir_k3", 24, 1, 1, e1, use_bias],
            ],
            # stage 3
            [
                ["ir_k5", 32, 2, 1, e6, use_bias],
                ["ir_k5", 32, 1, 1, e3, use_bias],
                # ["ir_k3_sep", 32, 1, 1, e6, use_bias],
                ["ir_k5", 32, 1, 1, e6, use_bias],
                ["ir_k3", 32, 1, 1, e6, use_bias],
            ],
            # stage 4
            [
                ["ir_k5", 64, 2, 1, e6, use_bias],
                ["ir_k5", 64, 1, 1, e3, use_bias],
                ["ir_k5", 64, 1, 1, e6, use_bias],
                ["ir_k5", 64, 1, 1, e6, use_bias],
                ["ir_k5", 112, 1, 1, e6, use_bias],
                #  ["ir_k3_sep", 112, 1, 1, e6, use_bias],
                ["ir_k5", 112, 1, 1, e6, use_bias],
                ["ir_k5", 112, 1, 1, e6, use_bias],
                ["ir_k5", 112, 1, 1, e3, use_bias],
            ],
            # stage 5
            [
                ["ir_k5", 184, 2, 1, e6, use_bias],
                ["ir_k5", 184, 1, 1, e6, use_bias],
                ["ir_k5", 184, 1, 1, e6, use_bias],
                ["ir_k5", 184, 1, 1, e6, use_bias],
                ["ir_k3", 352, 1, 1, e6, use_bias],
            ],
            # stage 6
            [("conv_k1", 1984, 1, 1)],
        ],
    },
    "fbnet_96": {
        "input_size": 96,
        "blocks": [
            # [op, c, s, n, ...]
            # stage 0
            [["conv_k3", 8, 2, 1]],
            # stage 1
            [["ir_k3", 8, 1, 1, e1]],
            # stage 2
            [["ir_k3", 16, 2, 1, e6], ["ir_k3", 16, 1, 1, e6]],
            # stage 3
            [["ir_k5", 16, 2, 1, e6], ["ir_k5", 16, 1, 1, e6]],
            # stage 4
            [["ir_k5", 24, 2, 1, e6], ["ir_k3", 40, 1, 1, e6]],
            # stage 5
            [
                ["ir_k5", 72, 2, 1, e6],
                ["ir_k5", 72, 1, 1, e6],
                ["ir_k5", 128, 1, 1, e6],
            ],
            # stage 6
            [("conv_k1", 1416, 1, 1)],
        ],
        "preprocessing": "resNet",
    },
}
MODEL_ARCH.register_dict(MODEL_ARCH_FBNET)
MODEL_ARCH.register_dict(mdu.get_i8f_models(MODEL_ARCH_FBNET))
