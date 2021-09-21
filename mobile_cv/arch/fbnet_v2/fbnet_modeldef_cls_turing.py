#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .fbnet_modeldef_cls import MODEL_ARCH

BASIC_ARGS = {
    "width_divisor": 8,
}

IRF_CFG = {
    "less_se_channels": True,
    "zero_last_bn_gamma": True,
}


MODEL_ARCH_FBNETV3_TURING = {
    "ARNet_b0_typeA": {
        # 78.3 % accuracy 801K cycles
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [
                ["conv_k3_hs", 40, 2, 1],
                ["conv_k3_hs", 32, 1, 1, {"groups": 4}],
                ["conv_k1", 16, 1, 1],
            ],
            [
                [
                    "ir_k3_hs",
                    40,
                    2,
                    1,
                    {"expansion": 6, "dw_group_ratio": 8, "pwl_groups": 4},
                ],
                [
                    "ir_k3_hs",
                    40,
                    1,
                    1,
                    {"expansion": 6, "dw_group_ratio": 8, "pwl_groups": 4},
                ],
            ],
            [
                ["ir_k3_hs", 80, 2, 1, {"expansion": 4, "dw_group_ratio": 8}],
                ["ir_k3_hs", 80, 1, 1, {"expansion": 4, "dw_group_ratio": 8}],
            ],
            [
                [
                    "ir_k3_hs",
                    96,
                    2,
                    1,
                    {"expansion": 8, "dw_group_ratio": 8, "pwl_groups": 2},
                ],
                [
                    "ir_k3_hs",
                    96,
                    1,
                    1,
                    {"expansion": 8, "dw_group_ratio": 8, "pwl_groups": 2},
                ],
                [
                    "ir_k3_hs",
                    96,
                    1,
                    1,
                    {"expansion": 8, "dw_group_ratio": 8, "pwl_groups": 2},
                ],
            ],
            [
                [
                    "ir_k5_hs",
                    160,
                    1,
                    1,
                    {"expansion": 6, "dw_group_ratio": 8, "pwl_groups": 2},
                ],
                [
                    "ir_k5_hs",
                    160,
                    1,
                    1,
                    {"expansion": 6, "dw_group_ratio": 8, "pwl_groups": 2},
                ],
                [
                    "ir_k5_hs",
                    160,
                    1,
                    1,
                    {"expansion": 6, "dw_group_ratio": 8, "pwl_groups": 2},
                ],
            ],
            [
                [
                    "ir_k3_hs",
                    184,
                    2,
                    1,
                    {"expansion": 8, "dw_group_ratio": 8, "pwl_groups": 2},
                ],
                [
                    "ir_k3_hs",
                    184,
                    1,
                    1,
                    {"expansion": 8, "dw_group_ratio": 8, "pwl_groups": 2},
                ],
                [
                    "ir_k3_hs",
                    184,
                    1,
                    1,
                    {"expansion": 8, "dw_group_ratio": 8, "pwl_groups": 2},
                ],
                [
                    "ir_k3_hs",
                    184,
                    1,
                    1,
                    {"expansion": 8, "dw_group_ratio": 8, "pwl_groups": 2},
                ],
            ],
            [
                [
                    "ir_k3_hs",
                    480,
                    1,
                    1,
                    {"expansion": 8, "dw_group_ratio": 8, "pwl_groups": 4},
                ],
            ],
            [("conv_k1", 1088, 1, 1)],
        ],
    },
    "FBNetV3_Turing_A": {
        "input_size": 256,  # 729K cycle
        "basic_args": BASIC_ARGS,
        "blocks": [
            [["conv_k3_hs", 32, 2, 1]],
            [["ir_k3_hs", 16, 1, 3, {"expansion": 1, "dw_group_ratio": 8}, IRF_CFG]],
            [
                ["ir_k5_hs", 32, 2, 1, {"expansion": 4, "dw_group_ratio": 8}, IRF_CFG],
                ["ir_k3_hs", 32, 1, 4, {"expansion": 2, "dw_group_ratio": 8}, IRF_CFG],
            ],
            [
                ["ir_k5_hs", 48, 2, 1, {"expansion": 4, "dw_group_ratio": 8}, IRF_CFG],
                ["ir_k3_hs", 48, 1, 4, {"expansion": 3, "dw_group_ratio": 8}, IRF_CFG],
            ],
            [
                ["ir_k5_hs", 80, 2, 1, {"expansion": 5, "dw_group_ratio": 8}, IRF_CFG],
                ["ir_k3_hs", 80, 1, 4, {"expansion": 3, "dw_group_ratio": 8}, IRF_CFG],
                ["ir_k3_hs", 128, 1, 1, {"expansion": 5, "dw_group_ratio": 8}, IRF_CFG],
                ["ir_k3_hs", 128, 1, 6, {"expansion": 3, "dw_group_ratio": 8}, IRF_CFG],
            ],
            [
                ["ir_k3_hs", 216, 2, 1, {"expansion": 6, "dw_group_ratio": 8}, IRF_CFG],
                ["ir_k3_hs", 216, 1, 5, {"expansion": 5, "dw_group_ratio": 8}, IRF_CFG],
                ["ir_k3_hs", 216, 1, 1, {"expansion": 6, "dw_group_ratio": 8}, IRF_CFG],
            ],
            [["ir_pool_hs", 1600, 1, 1, {"expansion": 6.0}]],
        ],
    },
    "FBNetV3_Turing_B": {
        "input_size": 256,  # 532912 cycle
        "basic_args": BASIC_ARGS,
        "blocks": [
            [["conv_k3_hs", 32, 2, 1]],
            [["ir_k3_hs", 16, 1, 3, {"expansion": 1, "dw_group_ratio": 8}, IRF_CFG]],
            [
                ["ir_k3_hs", 24, 2, 1, {"expansion": 4, "dw_group_ratio": 8}, IRF_CFG],
                ["ir_k3_hs", 24, 1, 3, {"expansion": 2, "dw_group_ratio": 8}, IRF_CFG],
            ],
            [
                ["ir_k5_hs", 40, 2, 1, {"expansion": 4, "dw_group_ratio": 8}, IRF_CFG],
                ["ir_k3_hs", 40, 1, 3, {"expansion": 2, "dw_group_ratio": 8}, IRF_CFG],
            ],
            [
                ["ir_k3_hs", 72, 2, 1, {"expansion": 5, "dw_group_ratio": 8}, IRF_CFG],
                ["ir_k3_hs", 72, 1, 4, {"expansion": 3, "dw_group_ratio": 8}, IRF_CFG],
                ["ir_k3_hs", 128, 1, 1, {"expansion": 5, "dw_group_ratio": 8}, IRF_CFG],
                ["ir_k3_hs", 128, 1, 4, {"expansion": 3, "dw_group_ratio": 8}, IRF_CFG],
            ],
            [
                ["ir_k3_hs", 192, 2, 1, {"expansion": 4, "dw_group_ratio": 8}, IRF_CFG],
                ["ir_k3_hs", 192, 1, 5, {"expansion": 4, "dw_group_ratio": 8}, IRF_CFG],
                ["ir_k3_hs", 192, 1, 1, {"expansion": 6, "dw_group_ratio": 8}, IRF_CFG],
            ],
            [["ir_pool_hs", 1440, 1, 1, {"expansion": 6.0}]],
        ],
    },
    "FBNetV3_Turing_C": {
        "input_size": 256,  # 916568 cycle
        "basic_args": BASIC_ARGS,
        "blocks": [
            [["conv_k3_hs", 32, 2, 1]],
            [["ir_k3_hs", 24, 1, 3, {"expansion": 1, "dw_group_ratio": 8}, IRF_CFG]],
            [
                ["ir_k5_hs", 32, 2, 1, {"expansion": 4, "dw_group_ratio": 8}, IRF_CFG],
                ["ir_k3_hs", 32, 1, 4, {"expansion": 2, "dw_group_ratio": 8}, IRF_CFG],
            ],
            [
                ["ir_k5_hs", 56, 2, 1, {"expansion": 4, "dw_group_ratio": 8}, IRF_CFG],
                ["ir_k3_hs", 56, 1, 4, {"expansion": 3, "dw_group_ratio": 8}, IRF_CFG],
            ],
            [
                ["ir_k5_hs", 96, 2, 1, {"expansion": 5, "dw_group_ratio": 8}, IRF_CFG],
                ["ir_k3_hs", 96, 1, 4, {"expansion": 3, "dw_group_ratio": 8}, IRF_CFG],
                ["ir_k3_hs", 160, 1, 1, {"expansion": 5, "dw_group_ratio": 8}, IRF_CFG],
                ["ir_k3_hs", 160, 1, 7, {"expansion": 3, "dw_group_ratio": 8}, IRF_CFG],
            ],
            [
                ["ir_k3_hs", 224, 2, 1, {"expansion": 6, "dw_group_ratio": 8}, IRF_CFG],
                ["ir_k3_hs", 224, 1, 6, {"expansion": 5, "dw_group_ratio": 8}, IRF_CFG],
                ["ir_k3_hs", 224, 1, 1, {"expansion": 6, "dw_group_ratio": 8}, IRF_CFG],
            ],
            [["ir_pool_hs", 1600, 1, 1, {"expansion": 6.0}]],
        ],
    },
}

MODEL_ARCH.register_dict(MODEL_ARCH_FBNETV3_TURING)
