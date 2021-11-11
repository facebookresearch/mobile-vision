#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from . import modeldef_utils as mdu
from .fbnet_modeldef_cls import MODEL_ARCH
from .modeldef_utils import _ex, e1, e3, e4, e6


BASIC_ARGS = {
    "width_divisor": 8,
}

IRF_CFG = {
    "less_se_channels": False,
    "zero_last_bn_gamma": True,
}

SE_RELU = {"se_args": {"name": "se_hsig", "relu_args": "relu"}}


MODEL_ARCH_MOBILE_NET = {
    "mnv2": {
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [op, c, s, n, ...]
            # stage 0
            [("conv_k3", 32, 2, 1)],
            # stage 1
            [("ir_k3", 16, 1, 1, e1, IRF_CFG)],
            # stage 2
            [("ir_k3", 24, 2, 2, e6, IRF_CFG)],
            # stage 3
            [("ir_k3", 32, 2, 3, e6, IRF_CFG)],
            # stage 4
            [
                ("ir_k3", 64, 2, 4, e6, IRF_CFG),
                ("ir_k3", 96, 1, 3, e6, IRF_CFG),
            ],
            # stage 5
            [
                ("ir_k3", 160, 2, 3, e6, IRF_CFG),
                ("ir_k3", 320, 1, 1, e6, IRF_CFG),
            ],
            # stage 6
            [("conv_k1", 1280, 1, 1)],
        ],
    },
    "mnv3": {
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [("conv_k3_hs", 16, 2, 1)],
            # stage 1
            [["ir_k3", 16, 1, 1, e1, IRF_CFG]],
            # stage 2
            [
                ["ir_k3", 24, 2, 1, e4, IRF_CFG],
                ["ir_k3", 24, 1, 1, e3, IRF_CFG],
            ],
            # stage 3
            [["ir_k5_sehsig", 40, 2, 3, e3, IRF_CFG]],
            # stage 4
            [
                ["ir_k3_hs", 80, 2, 1, e6, IRF_CFG],
                ["ir_k3_hs", 80, 1, 1, _ex(2.5), IRF_CFG],
                ["ir_k3_hs", 80, 1, 2, _ex(2.3), IRF_CFG],
                ["ir_k3_sehsig_hs", 112, 1, 2, e6, IRF_CFG],
            ],
            # stage 5
            [["ir_k5_sehsig_hs", 160, 2, 3, e6, IRF_CFG]],
            # stage 6
            [["ir_pool_hs", 1280, 1, 1, e6, IRF_CFG]],
        ],
    },
    "mnv3_small": {
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [("conv_k3_hs", 16, 2, 1)],
            # stage 1
            [["ir_k3", 16, 2, 1, e1, SE_RELU, IRF_CFG]],
            # stage 2
            [
                ["ir_k3", 24, 2, 1, _ex(4.5), IRF_CFG],
                ["ir_k3", 24, 1, 1, _ex(3.67), IRF_CFG],
            ],
            # stage 3
            [
                ["ir_k5_hs", 40, 2, 1, e4, SE_RELU, IRF_CFG],
                ["ir_k5_hs", 40, 1, 2, e6, SE_RELU, IRF_CFG],
                ["ir_k5_hs", 48, 1, 2, e3, SE_RELU, IRF_CFG],
            ],
            # stage 4
            [
                ["ir_k5_hs", 96, 2, 3, e6, SE_RELU, IRF_CFG],
            ],
            # stage 5
            [
                [
                    "ir_pool_hs",
                    1024,
                    1,
                    1,
                    e6,
                    {"pw_se_args": {"name": "se_hsig", "relu_args": "relu"}},
                    IRF_CFG,
                ]
            ],
        ],
    },
}
MODEL_ARCH.register_dict(MODEL_ARCH_MOBILE_NET)
MODEL_ARCH.register_dict(mdu.get_i8f_models(MODEL_ARCH_MOBILE_NET))
