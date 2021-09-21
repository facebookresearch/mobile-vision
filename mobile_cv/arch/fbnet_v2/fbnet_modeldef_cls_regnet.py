#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .fbnet_modeldef_cls import MODEL_ARCH

BASIC_ARGS = {
    "relu_args": {"name": "swish"},
    "bias": False,
}

IRF_CFG = {
    "less_se_channels": True,
    "zero_last_bn_gamma": True,
    "round_se_channels": True,
}

MODEL_ARCH_REGNETZ = {
    "RegNet_IB_700M": {
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [["conv_k3", 32, 2, 1]],
            [
                ["ir_k3_se", 24, 2, 1, {"expansion": 3, "dw_group_ratio": 4}, IRF_CFG],
                ["ir_k3_se", 24, 1, 1, {"expansion": 4, "dw_group_ratio": 4}, IRF_CFG],
            ],
            [
                [
                    "ir_k3_se",
                    56,
                    2,
                    1,
                    {"expansion": 9.333, "dw_group_ratio": 4},
                    IRF_CFG,
                ],
                ["ir_k3_se", 56, 1, 3, {"expansion": 4, "dw_group_ratio": 4}, IRF_CFG],
            ],
            [
                [
                    "ir_k3_se",
                    136,
                    2,
                    1,
                    {"expansion": 9.7143, "dw_group_ratio": 4},
                    IRF_CFG,
                ],
                ["ir_k3_se", 136, 1, 9, {"expansion": 4, "dw_group_ratio": 4}, IRF_CFG],
            ],
            [
                [
                    "ir_k3_se",
                    336,
                    2,
                    1,
                    {"expansion": 9.8824, "dw_group_ratio": 4},
                    IRF_CFG,
                ],
                ["ir_k3_se", 336, 1, 2, {"expansion": 4, "dw_group_ratio": 4}, IRF_CFG],
            ],
        ],
    },
}

MODEL_ARCH.register_dict(MODEL_ARCH_REGNETZ)
