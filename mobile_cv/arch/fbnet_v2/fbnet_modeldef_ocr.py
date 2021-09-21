#!/usr/bin/env python3

import mobile_cv.common.misc.registry as registry

from . import modeldef_utils as mdu
from .modeldef_utils import _ex, e1, e3, e4, e6


MODEL_ARCH = registry.Registry("ocr_arch_factory")

MODEL_ARCH_FBNET = {
    "fbnet_b": {
        "input_size": 224,
        "blocks": [
            # [op, c, s, n, ...]
            # stage 0
            [["conv_k3", 16, 2, 1]],
            # stage 1
            [["ir_k3", 16, 1, 1, e1]],
            # stage 2
            [
                ["ir_k3", 24, 2, 1, e6],
                ["ir_k5", 24, 1, 1, e1],
                ["ir_k3", 24, 1, 1, e1],
                ["ir_k3", 24, 1, 1, e1],
            ],
            # stage 3
            [
                ["ir_k5", 32, 2, 1, e6],
                ["ir_k5", 32, 1, 1, e3],
                ["ir_k3", 32, 1, 1, e6],
                # ["ir_k3_sep", 32, 1, 1, e6]],
                ["ir_k5", 32, 1, 1, e6],
            ],
            # stage 4
            [
                ["ir_k5", 64, [2, 1], 1, e6],
                ["ir_k5", 64, 1, 1, e6],
                ["skip", 64, 1, 1],
                ["ir_k5", 64, 1, 1, e3],
                ["ir_k5", 112, 1, 1, e6],
                ["ir_k3", 112, 1, 1, e1],
                ["ir_k5", 112, 1, 1, e1],
                ["ir_k5", 112, 1, 1, e3],
            ],
            # stage 5
            [
                ["ir_k5", 184, 1, 1, e6],
                ["ir_k5", 184, 1, 1, e1],
                ["ir_k5", 184, 1, 1, e6],
                ["ir_k5", 184, 1, 1, e6],
                ["ir_k3", 352, 1, 1, e6],
            ],
            # stage 6
            [("conv_k1", 512, 1, 1)],
        ],
    },
    "fbnet_c": {
        "input_size": 224,
        "blocks": [
            # [op, c, s, n, ...]
            # stage 0
            [["conv_k3", 16, 2, 1]],
            # stage 1
            [["ir_k3", 16, 1, 1, e1]],
            # stage 2
            [
                ["ir_k3", 24, 2, 1, e6],
                ["skip", 24, 1, 1, e1],
                ["ir_k3", 24, 1, 1, e1],
                ["ir_k3", 24, 1, 1, e1],
            ],
            # stage 3
            [
                ["ir_k5", 32, 2, 1, e6],
                ["ir_k5", 32, 1, 1, e3],
                # ["ir_k3_sep", 32, 1, 1, e6],
                ["ir_k5", 32, 1, 1, e6],
                ["ir_k3", 32, 1, 1, e6],
            ],
            # stage 4
            [
                ["ir_k5", 64, [2, 1], 1, e6],
                ["ir_k5", 64, 1, 1, e3],
                ["ir_k5", 64, 1, 1, e6],
                ["ir_k5", 64, 1, 1, e6],
                ["ir_k5", 112, 1, 1, e6],
                #  ["ir_k3_sep", 112, 1, 1, e6],
                ["ir_k5", 112, 1, 1, e6],
                ["ir_k5", 112, 1, 1, e6],
                ["ir_k5", 112, 1, 1, e3],
            ],
            # stage 5
            [
                ["ir_k5", 184, 1, 1, e6],
                ["ir_k5", 184, 1, 1, e6],
                ["ir_k5", 184, 1, 1, e6],
                ["ir_k5", 184, 1, 1, e6],
                ["ir_k3", 352, 1, 1, e6],
            ],
            # stage 6
            [("conv_k1", 512, 1, 1)],
        ],
    },
}
MODEL_ARCH.register_dict(MODEL_ARCH_FBNET)
MODEL_ARCH.register_dict(mdu.get_i8f_models(MODEL_ARCH_FBNET))
