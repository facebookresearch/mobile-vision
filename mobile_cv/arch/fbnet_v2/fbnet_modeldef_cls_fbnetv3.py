#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from mobile_cv.arch.fbnet_v2 import modeldef_utils as mdu
from mobile_cv.arch.fbnet_v2.fbnet_modeldef_cls import MODEL_ARCH


BASIC_ARGS = {
    "width_divisor": 8,
}

IRF_CFG = {
    "less_se_channels": True,
    "zero_last_bn_gamma": True,
}

FUSEDMB_CFG = {"skip_dw": True, "always_pw": True}
FUSEDMB_e1 = {"skip_pwl": True, "mid_expand_out": True, "expansion": 1}


MODEL_ARCH_FBNETV3 = {
    "FBNetV3_A0": {
        # a scaled version of FBNetV3_A, 217M FLOP, top-1 accuracy around 78.4%
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [["conv_k3_hs", 12, 2, 1]],
            [["ir_k3_hs", 12, 1, 1, {"expansion": 1}, IRF_CFG]],
            [
                ["ir_k3_hs", 24, 2, 1, {"expansion": 4}, IRF_CFG],
                ["ir_k3_hs", 24, 1, 2, {"expansion": 2}, IRF_CFG],
            ],
            [
                ["ir_k3_sehsig_hs", 40, 2, 1, {"expansion": 4}, IRF_CFG],
                ["ir_k3_sehsig_hs", 40, 1, 2, {"expansion": 2}, IRF_CFG],
            ],
            [
                ["ir_k5_hs", 72, 2, 1, {"expansion": 4}, IRF_CFG],
                ["ir_k3_hs", 72, 1, 2, {"expansion": 3}, IRF_CFG],
                ["ir_k3_sehsig_hs", 104, 1, 1, {"expansion": 4}, IRF_CFG],
                ["ir_k5_sehsig_hs", 104, 1, 3, {"expansion": 3}, IRF_CFG],
            ],
            [
                ["ir_k3_sehsig_hs", 176, 2, 1, {"expansion": 4}, IRF_CFG],
                ["ir_k3_sehsig_hs", 176, 1, 3, {"expansion": 4}, IRF_CFG],
                ["ir_k3_sehsig_hs", 208, 1, 1, {"expansion": 5}, IRF_CFG],
            ],
            [["ir_pool_hs", 1984, 1, 1, {"expansion": 5}]],
        ],
    },
    # applied a width multiplier of 8 for latency optimization
    "FBNetV3_A": {
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [["conv_k3_hs", 16, 2, 1]],
            [["ir_k3_hs", 16, 1, 2, {"expansion": 1}, IRF_CFG]],
            [
                ["ir_k5_hs", 24, 2, 1, {"expansion": 4}, IRF_CFG],
                ["ir_k5_hs", 24, 1, 3, {"expansion": 2}, IRF_CFG],
            ],
            [
                ["ir_k5_sehsig_hs", 40, 2, 1, {"expansion": 5}, IRF_CFG],
                ["ir_k5_sehsig_hs", 40, 1, 4, {"expansion": 3}, IRF_CFG],
            ],
            [
                [
                    "ir_k5_hs",
                    72,
                    2,
                    1,
                    {"expansion": 5},
                    IRF_CFG,
                    {"dw_args": {"padding": (1, 1)}},
                ],
                ["ir_k3_hs", 72, 1, 4, {"expansion": 3}, IRF_CFG],
                ["ir_k3_sehsig_hs", 120, 1, 1, {"expansion": 5}, IRF_CFG],
                ["ir_k5_sehsig_hs", 120, 1, 5, {"expansion": 3}, IRF_CFG],
            ],
            [
                [
                    "ir_k3_sehsig_hs",
                    184,
                    2,
                    1,
                    {"expansion": 6},
                    IRF_CFG,
                    {"dw_args": {"padding": (0, 0)}},
                ],
                ["ir_k5_sehsig_hs", 184, 1, 5, {"expansion": 4}, IRF_CFG],
                ["ir_k5_sehsig_hs", 224, 1, 1, {"expansion": 6}, IRF_CFG],
            ],
            [["ir_pool_hs", 1984, 1, 1, {"expansion": 6}]],
        ],
    },
    "FBNetV3_B": {
        "input_size": 248,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [["conv_k3_hs", 16, 2, 1]],
            [["ir_k3_hs", 16, 1, 2, {"expansion": 1}, IRF_CFG]],
            [
                ["ir_k5_hs", 24, 2, 1, {"expansion": 4}, IRF_CFG],
                ["ir_k5_hs", 24, 1, 3, {"expansion": 2}, IRF_CFG],
            ],
            [
                ["ir_k5_sehsig_hs", 40, 2, 1, {"expansion": 5}, IRF_CFG],
                ["ir_k5_sehsig_hs", 40, 1, 4, {"expansion": 3}, IRF_CFG],
            ],
            [
                [
                    "ir_k5_hs",
                    72,
                    2,
                    1,
                    {"expansion": 5},
                    IRF_CFG,
                    {"dw_args": {"padding": (1, 1)}},
                ],
                ["ir_k3_hs", 72, 1, 4, {"expansion": 3}, IRF_CFG],
                ["ir_k3_sehsig_hs", 120, 1, 1, {"expansion": 5}, IRF_CFG],
                ["ir_k5_sehsig_hs", 120, 1, 5, {"expansion": 3}, IRF_CFG],
            ],
            [
                [
                    "ir_k3_sehsig_hs",
                    184,
                    2,
                    1,
                    {"expansion": 6},
                    IRF_CFG,
                    {"dw_args": {"padding": (0, 0)}},
                ],
                ["ir_k5_sehsig_hs", 184, 1, 5, {"expansion": 4}, IRF_CFG],
                ["ir_k5_sehsig_hs", 224, 1, 1, {"expansion": 6}, IRF_CFG],
            ],
            [["ir_pool_hs", 1984, 1, 1, {"expansion": 6}]],
        ],
    },
    "FBNetV3_C": {
        "input_size": 248,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [["conv_k3_hs", 16, 2, 1]],
            [["ir_k3_hs", 16, 1, 2, {"expansion": 1}, IRF_CFG]],
            [
                ["ir_k5_hs", 24, 2, 1, {"expansion": 5}, IRF_CFG],
                ["ir_k3_hs", 24, 1, 4, {"expansion": 3}, IRF_CFG],
            ],
            [
                ["ir_k5_sehsig_hs", 48, 2, 1, {"expansion": 5}, IRF_CFG],
                ["ir_k5_sehsig_hs", 48, 1, 4, {"expansion": 2}, IRF_CFG],
            ],
            [
                [
                    "ir_k5_hs",
                    88,
                    2,
                    1,
                    {"expansion": 4},
                    IRF_CFG,
                    {"dw_args": {"padding": (1, 1)}},
                ],
                ["ir_k3_hs", 88, 1, 4, {"expansion": 3}, IRF_CFG],
                ["ir_k3_sehsig_hs", 120, 1, 1, {"expansion": 4}, IRF_CFG],
                ["ir_k5_sehsig_hs", 120, 1, 5, {"expansion": 3}, IRF_CFG],
            ],
            [
                [
                    "ir_k5_sehsig_hs",
                    216,
                    2,
                    1,
                    {"expansion": 5},
                    IRF_CFG,
                    {"dw_args": {"padding": (1, 1)}},
                ],
                ["ir_k5_sehsig_hs", 216, 1, 5, {"expansion": 5}, IRF_CFG],
                ["ir_k5_sehsig_hs", 216, 1, 1, {"expansion": 6}, IRF_CFG],
            ],
            [["ir_pool_hs", 1984, 1, 1, {"expansion": 6.0}]],
        ],
    },
    "FBNetV3_D": {
        "input_size": 248,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [["conv_k3_hs", 24, 2, 1]],
            [["ir_k3_hs", 16, 1, 2, {"expansion": 1}, IRF_CFG]],
            [
                ["ir_k3_hs", 24, 2, 1, {"expansion": 5}, IRF_CFG],
                ["ir_k3_hs", 24, 1, 5, {"expansion": 2}, IRF_CFG],
            ],
            [
                ["ir_k5_sehsig_hs", 40, 2, 1, {"expansion": 4}, IRF_CFG],
                ["ir_k3_sehsig_hs", 40, 1, 4, {"expansion": 3}, IRF_CFG],
            ],
            [
                ["ir_k3_hs", 72, 2, 1, {"expansion": 5}, IRF_CFG],
                ["ir_k3_hs", 72, 1, 4, {"expansion": 3}, IRF_CFG],
                ["ir_k3_sehsig_hs", 128, 1, 1, {"expansion": 5}, IRF_CFG],
                ["ir_k5_sehsig_hs", 128, 1, 6, {"expansion": 3}, IRF_CFG],
            ],
            [
                ["ir_k3_sehsig_hs", 208, 2, 1, {"expansion": 6}, IRF_CFG],
                ["ir_k5_sehsig_hs", 208, 1, 5, {"expansion": 5}, IRF_CFG],
                ["ir_k5_sehsig_hs", 240, 1, 1, {"expansion": 6}, IRF_CFG],
            ],
            [["ir_pool_hs", 1984, 1, 1, {"expansion": 6.0}]],
        ],
    },
    "FBNetV3_E": {
        "input_size": 264,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [["conv_k3_hs", 24, 2, 1]],
            [["ir_k3_hs", 16, 1, 3, {"expansion": 1}, IRF_CFG]],
            [
                ["ir_k5_hs", 24, 2, 1, {"expansion": 4}, IRF_CFG],
                ["ir_k5_hs", 24, 1, 4, {"expansion": 2}, IRF_CFG],
            ],
            [
                ["ir_k5_sehsig_hs", 48, 2, 1, {"expansion": 4}, IRF_CFG],
                ["ir_k5_sehsig_hs", 48, 1, 4, {"expansion": 3}, IRF_CFG],
            ],
            [
                [
                    "ir_k5_hs",
                    80,
                    2,
                    1,
                    {"expansion": 5},
                    IRF_CFG,
                    {"dw_args": {"padding": (1, 1)}},
                ],
                ["ir_k3_hs", 80, 1, 4, {"expansion": 3}, IRF_CFG],
                ["ir_k3_sehsig_hs", 128, 1, 1, {"expansion": 5}, IRF_CFG],
                ["ir_k5_sehsig_hs", 128, 1, 7, {"expansion": 3}, IRF_CFG],
            ],
            [
                ["ir_k3_sehsig_hs", 216, 2, 1, {"expansion": 6}, IRF_CFG],
                ["ir_k5_sehsig_hs", 216, 1, 5, {"expansion": 5}, IRF_CFG],
                ["ir_k5_sehsig_hs", 240, 1, 1, {"expansion": 6}, IRF_CFG],
            ],
            [["ir_pool_hs", 1984, 1, 1, {"expansion": 6.0}]],
        ],
    },
    "FBNetV3_F": {
        "input_size": 272,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [["conv_k3_hs", 24, 2, 1]],
            [["ir_k3_hs", 24, 1, 3, {"expansion": 1}, IRF_CFG]],
            [
                ["ir_k5_hs", 32, 2, 1, {"expansion": 4}, IRF_CFG],
                ["ir_k5_hs", 32, 1, 4, {"expansion": 2}, IRF_CFG],
            ],
            [
                ["ir_k5_sehsig_hs", 56, 2, 1, {"expansion": 4}, IRF_CFG],
                ["ir_k5_sehsig_hs", 56, 1, 4, {"expansion": 3}, IRF_CFG],
            ],
            [
                ["ir_k5_hs", 88, 2, 1, {"expansion": 5}, IRF_CFG],
                ["ir_k3_hs", 88, 1, 4, {"expansion": 3}, IRF_CFG],
                ["ir_k3_sehsig_hs", 144, 1, 1, {"expansion": 5}, IRF_CFG],
                ["ir_k5_sehsig_hs", 144, 1, 8, {"expansion": 3}, IRF_CFG],
            ],
            [
                [
                    "ir_k3_sehsig_hs",
                    248,
                    2,
                    1,
                    {"expansion": 6},
                    IRF_CFG,
                    {"dw_args": {"padding": (0, 0)}},
                ],
                ["ir_k5_sehsig_hs", 248, 1, 6, {"expansion": 5}, IRF_CFG],
                ["ir_k5_sehsig_hs", 272, 1, 1, {"expansion": 6}, IRF_CFG],
            ],
            [["ir_pool_hs", 1984, 1, 1, {"expansion": 6.0}]],
        ],
    },
    "FBNetV3_G": {
        "input_size": 320,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [["conv_k3_hs", 32, 2, 1]],
            [["ir_k3_hs", 24, 1, 3, {"expansion": 1}, IRF_CFG]],
            [
                ["ir_k5_hs", 40, 2, 1, {"expansion": 4}, IRF_CFG],
                ["ir_k5_hs", 40, 1, 4, {"expansion": 2}, IRF_CFG],
            ],
            [
                ["ir_k5_sehsig_hs", 56, 2, 1, {"expansion": 4}, IRF_CFG],
                ["ir_k5_sehsig_hs", 56, 1, 4, {"expansion": 3}, IRF_CFG],
            ],
            [
                ["ir_k5_hs", 104, 2, 1, {"expansion": 5}, IRF_CFG],
                ["ir_k3_hs", 104, 1, 4, {"expansion": 3}, IRF_CFG],
                ["ir_k3_sehsig_hs", 160, 1, 1, {"expansion": 5}, IRF_CFG],
                ["ir_k5_sehsig_hs", 160, 1, 8, {"expansion": 3}, IRF_CFG],
            ],
            [
                ["ir_k3_sehsig_hs", 264, 2, 1, {"expansion": 6}, IRF_CFG],
                ["ir_k5_sehsig_hs", 264, 1, 6, {"expansion": 5}, IRF_CFG],
                ["ir_k5_sehsig_hs", 288, 1, 2, {"expansion": 6}, IRF_CFG],
            ],
            [["ir_pool_hs", 1984, 1, 1, {"expansion": 6.0}]],
        ],
    },
    "FBNetV3_20G": {
        # A giant FBNetV3 model used for distillation.
        # 20.4G FLOPs, 101M Params
        # Top-1 accuracy 84.4% w/o extra data.
        "input_size": 256,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [["conv_k3_hs", 32, 2, 1]],
            [
                ["ir_k3_hs", 32, 1, 1, {"expansion": 1, "dw_group_ratio": 32}, IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, {"expansion": 1, "dw_group_ratio": 32}, IRF_CFG],
            ],
            [
                [
                    "ir_k5_hs",
                    108,
                    2,
                    1,
                    {"expansion": 4, "dw_group_ratio": 64},
                    IRF_CFG,
                ],
                [
                    "ir_k5_hs",
                    108,
                    1,
                    5,
                    {"expansion": 1, "dw_group_ratio": 54},
                    IRF_CFG,
                ],
            ],
            [
                [
                    "ir_k5_sehsig_hs",
                    160,
                    2,
                    1,
                    {"expansion": 2, "dw_group_ratio": 108},
                    IRF_CFG,
                ],
                [
                    "ir_k5_sehsig_hs",
                    160,
                    1,
                    5,
                    {"expansion": 2, "dw_group_ratio": 80},
                    IRF_CFG,
                ],
            ],
            [
                [
                    "ir_k5_hs",
                    256,
                    2,
                    1,
                    {"expansion": 2, "dw_group_ratio": 80},
                    IRF_CFG,
                ],
                [
                    "ir_k3_hs",
                    256,
                    1,
                    6,
                    {"expansion": 2, "dw_group_ratio": 128},
                    IRF_CFG,
                ],
                [
                    "ir_k3_sehsig_hs",
                    348,
                    1,
                    1,
                    {"expansion": 2, "dw_group_ratio": 116},
                    IRF_CFG,
                ],
                [
                    "ir_k5_sehsig_hs",
                    348,
                    1,
                    9,
                    {"expansion": 2, "dw_group_ratio": 116},
                    IRF_CFG,
                ],
            ],
            [
                [
                    "ir_k3_sehsig_hs",
                    580,
                    2,
                    1,
                    {"expansion": 2, "dw_group_ratio": 116},
                    IRF_CFG,
                ],
                [
                    "ir_k5_sehsig_hs",
                    580,
                    1,
                    7,
                    {"expansion": 2, "dw_group_ratio": 116},
                    IRF_CFG,
                ],
                [
                    "ir_k5_sehsig_hs",
                    580,
                    1,
                    3,
                    {"expansion": 2, "dw_group_ratio": 116},
                    IRF_CFG,
                ],
            ],
            [["ir_pool_hs", 3200, 1, 1, {"expansion": 6.0}]],
        ],
    },
}

# GPU-friendly FBNetV3
MODEL_ARCH_FBNETV3_GPU = {
    # applied a width multiplier of 8 for latency optimization
    "FBNetV3_A_GPU": {
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            MODEL_ARCH_FBNETV3["FBNetV3_A"]["blocks"][0],
            MODEL_ARCH_FBNETV3["FBNetV3_A"]["blocks"][1],
            [
                [
                    "ir_k5_hs",
                    24,
                    2,
                    1,
                    FUSEDMB_CFG,
                    {"expansion": 4},
                    {
                        "pw_args": {
                            "kernel_size": 5,
                            "padding": 2,
                            "stride": 2,
                        },
                    },
                    IRF_CFG,
                ],
                [
                    "ir_k5_hs",
                    24,
                    1,
                    3,
                    FUSEDMB_CFG,
                    {"expansion": 2},
                    {
                        "pw_args": {
                            "kernel_size": 5,
                            "padding": 2,
                            "stride": 1,
                        },
                    },
                    IRF_CFG,
                ],
            ],
            [
                [
                    "ir_k5_sehsig_hs",
                    40,
                    2,
                    1,
                    FUSEDMB_CFG,
                    {"expansion": 5},
                    {
                        "pw_args": {
                            "kernel_size": 5,
                            "padding": 2,
                            "stride": 2,
                        },
                    },
                    IRF_CFG,
                ],
                [
                    "ir_k5_sehsig_hs",
                    40,
                    1,
                    4,
                    FUSEDMB_CFG,
                    {"expansion": 3},
                    {
                        "pw_args": {
                            "kernel_size": 5,
                            "padding": 2,
                            "stride": 1,
                        },
                    },
                    IRF_CFG,
                ],
            ],
            MODEL_ARCH_FBNETV3["FBNetV3_A"]["blocks"][4],
            MODEL_ARCH_FBNETV3["FBNetV3_A"]["blocks"][5],
            MODEL_ARCH_FBNETV3["FBNetV3_A"]["blocks"][6],
        ],
    },
    "FBNetV3_B_GPU": {
        "input_size": 248,
        "basic_args": BASIC_ARGS,
        "blocks": [
            MODEL_ARCH_FBNETV3["FBNetV3_B"]["blocks"][0],
            MODEL_ARCH_FBNETV3["FBNetV3_B"]["blocks"][1],
            [
                [
                    "ir_k5_hs",
                    24,
                    2,
                    1,
                    FUSEDMB_CFG,
                    {"expansion": 4},
                    {
                        "pw_args": {
                            "kernel_size": 5,
                            "padding": 2,
                            "stride": 2,
                        },
                    },
                    IRF_CFG,
                ],
                [
                    "ir_k5_hs",
                    24,
                    1,
                    3,
                    FUSEDMB_CFG,
                    {"expansion": 2},
                    {
                        "pw_args": {
                            "kernel_size": 5,
                            "padding": 2,
                            "stride": 1,
                        },
                    },
                    IRF_CFG,
                ],
            ],
            [
                [
                    "ir_k5_sehsig_hs",
                    40,
                    2,
                    1,
                    FUSEDMB_CFG,
                    {"expansion": 5},
                    {
                        "pw_args": {
                            "kernel_size": 5,
                            "padding": 2,
                            "stride": 2,
                        },
                    },
                    IRF_CFG,
                ],
                [
                    "ir_k5_sehsig_hs",
                    40,
                    1,
                    4,
                    FUSEDMB_CFG,
                    {"expansion": 3},
                    {
                        "pw_args": {
                            "kernel_size": 5,
                            "padding": 2,
                            "stride": 1,
                        },
                    },
                    IRF_CFG,
                ],
            ],
            MODEL_ARCH_FBNETV3["FBNetV3_B"]["blocks"][4],
            MODEL_ARCH_FBNETV3["FBNetV3_B"]["blocks"][5],
            MODEL_ARCH_FBNETV3["FBNetV3_B"]["blocks"][6],
        ],
    },
    "FBNetV3_C_GPU": {
        "input_size": 248,
        "basic_args": BASIC_ARGS,
        "blocks": [
            MODEL_ARCH_FBNETV3["FBNetV3_C"]["blocks"][0],
            MODEL_ARCH_FBNETV3["FBNetV3_C"]["blocks"][1],
            [
                [
                    "ir_k5_hs",
                    24,
                    2,
                    1,
                    FUSEDMB_CFG,
                    {"expansion": 5},
                    {
                        "pw_args": {
                            "kernel_size": 5,
                            "padding": 2,
                            "stride": 2,
                        },
                    },
                    IRF_CFG,
                ],
                [
                    "ir_k3_hs",
                    24,
                    1,
                    4,
                    FUSEDMB_CFG,
                    {"expansion": 3},
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 1,
                        },
                    },
                    IRF_CFG,
                ],
            ],
            [
                [
                    "ir_k5_sehsig_hs",
                    48,
                    2,
                    1,
                    FUSEDMB_CFG,
                    {"expansion": 5},
                    {
                        "pw_args": {
                            "kernel_size": 5,
                            "padding": 2,
                            "stride": 2,
                        },
                    },
                    IRF_CFG,
                ],
                [
                    "ir_k5_sehsig_hs",
                    48,
                    1,
                    4,
                    FUSEDMB_CFG,
                    {"expansion": 2},
                    {
                        "pw_args": {
                            "kernel_size": 5,
                            "padding": 2,
                            "stride": 1,
                        },
                    },
                    IRF_CFG,
                ],
            ],
            MODEL_ARCH_FBNETV3["FBNetV3_C"]["blocks"][4],
            MODEL_ARCH_FBNETV3["FBNetV3_C"]["blocks"][5],
            MODEL_ARCH_FBNETV3["FBNetV3_C"]["blocks"][6],
        ],
    },
    "FBNetV3_D_GPU": {
        "input_size": 248,
        "basic_args": BASIC_ARGS,
        "blocks": [
            MODEL_ARCH_FBNETV3["FBNetV3_D"]["blocks"][0],
            MODEL_ARCH_FBNETV3["FBNetV3_D"]["blocks"][1],
            [
                [
                    "ir_k5_hs",
                    24,
                    2,
                    1,
                    FUSEDMB_CFG,
                    {"expansion": 5},
                    {
                        "pw_args": {
                            "kernel_size": 5,
                            "padding": 2,
                            "stride": 2,
                        },
                    },
                    IRF_CFG,
                ],
                [
                    "ir_k3_hs",
                    24,
                    1,
                    4,
                    FUSEDMB_CFG,
                    {"expansion": 3},
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 1,
                        },
                    },
                    IRF_CFG,
                ],
            ],
            [
                [
                    "ir_k5_sehsig_hs",
                    40,
                    2,
                    1,
                    FUSEDMB_CFG,
                    {"expansion": 4},
                    {
                        "pw_args": {
                            "kernel_size": 5,
                            "padding": 2,
                            "stride": 2,
                        },
                    },
                    IRF_CFG,
                ],
                [
                    "ir_k3_sehsig_hs",
                    40,
                    1,
                    4,
                    FUSEDMB_CFG,
                    {"expansion": 3},
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 1,
                        },
                    },
                    IRF_CFG,
                ],
            ],
            MODEL_ARCH_FBNETV3["FBNetV3_D"]["blocks"][4],
            MODEL_ARCH_FBNETV3["FBNetV3_D"]["blocks"][5],
            MODEL_ARCH_FBNETV3["FBNetV3_D"]["blocks"][6],
        ],
    },
    "FBNetV3_E_GPU": {
        "input_size": 264,
        "basic_args": BASIC_ARGS,
        "blocks": [
            MODEL_ARCH_FBNETV3["FBNetV3_E"]["blocks"][0],
            MODEL_ARCH_FBNETV3["FBNetV3_E"]["blocks"][1],
            [
                [
                    "ir_k5_hs",
                    24,
                    2,
                    1,
                    FUSEDMB_CFG,
                    {"expansion": 4},
                    {
                        "pw_args": {
                            "kernel_size": 5,
                            "padding": 2,
                            "stride": 2,
                        },
                    },
                    IRF_CFG,
                ],
                [
                    "ir_k5_hs",
                    24,
                    1,
                    4,
                    FUSEDMB_CFG,
                    {"expansion": 2},
                    {
                        "pw_args": {
                            "kernel_size": 5,
                            "padding": 2,
                            "stride": 1,
                        },
                    },
                    IRF_CFG,
                ],
            ],
            MODEL_ARCH_FBNETV3["FBNetV3_E"]["blocks"][3],
            MODEL_ARCH_FBNETV3["FBNetV3_E"]["blocks"][4],
            MODEL_ARCH_FBNETV3["FBNetV3_E"]["blocks"][5],
            MODEL_ARCH_FBNETV3["FBNetV3_E"]["blocks"][6],
        ],
    },
    "FBNetV3_F_GPU": {
        "input_size": 272,
        "basic_args": BASIC_ARGS,
        "blocks": [
            MODEL_ARCH_FBNETV3["FBNetV3_F"]["blocks"][0],
            MODEL_ARCH_FBNETV3["FBNetV3_F"]["blocks"][1],
            [
                [
                    "ir_k5_hs",
                    32,
                    2,
                    1,
                    FUSEDMB_CFG,
                    {"expansion": 4},
                    {
                        "pw_args": {
                            "kernel_size": 5,
                            "padding": 2,
                            "stride": 2,
                        },
                    },
                    IRF_CFG,
                ],
                [
                    "ir_k5_hs",
                    32,
                    1,
                    4,
                    FUSEDMB_CFG,
                    {"expansion": 2},
                    {
                        "pw_args": {
                            "kernel_size": 5,
                            "padding": 2,
                            "stride": 1,
                        },
                    },
                    IRF_CFG,
                ],
            ],
            [
                [
                    "ir_k5_sehsig_hs",
                    56,
                    2,
                    1,
                    FUSEDMB_CFG,
                    FUSEDMB_e1,
                    {
                        "pw_args": {
                            "kernel_size": 5,
                            "padding": 2,
                            "stride": 2,
                        },
                    },
                    IRF_CFG,
                ],
                [
                    "ir_k5_sehsig_hs",
                    56,
                    1,
                    4,
                    FUSEDMB_CFG,
                    FUSEDMB_e1,
                    {
                        "pw_args": {
                            "kernel_size": 5,
                            "padding": 2,
                            "stride": 1,
                        },
                    },
                    IRF_CFG,
                ],
            ],
            MODEL_ARCH_FBNETV3["FBNetV3_F"]["blocks"][4],
            MODEL_ARCH_FBNETV3["FBNetV3_F"]["blocks"][5],
            MODEL_ARCH_FBNETV3["FBNetV3_F"]["blocks"][6],
        ],
    },
    "FBNetV3_G_GPU": {
        "input_size": 320,
        "basic_args": BASIC_ARGS,
        "blocks": [
            MODEL_ARCH_FBNETV3["FBNetV3_G"]["blocks"][0],
            MODEL_ARCH_FBNETV3["FBNetV3_G"]["blocks"][1],
            [
                [
                    "ir_k5_hs",
                    40,
                    2,
                    1,
                    FUSEDMB_CFG,
                    {"expansion": 4},
                    {
                        "pw_args": {
                            "kernel_size": 5,
                            "padding": 2,
                            "stride": 2,
                        },
                    },
                    IRF_CFG,
                ],
                [
                    "ir_k5_hs",
                    40,
                    1,
                    4,
                    FUSEDMB_CFG,
                    {"expansion": 2},
                    {
                        "pw_args": {
                            "kernel_size": 5,
                            "padding": 2,
                            "stride": 1,
                        },
                    },
                    IRF_CFG,
                ],
            ],
            [
                [
                    "ir_k5_sehsig_hs",
                    56,
                    2,
                    1,
                    FUSEDMB_CFG,
                    FUSEDMB_e1,
                    {
                        "pw_args": {
                            "kernel_size": 5,
                            "padding": 2,
                            "stride": 2,
                        },
                    },
                    IRF_CFG,
                ],
                [
                    "ir_k5_sehsig_hs",
                    56,
                    1,
                    4,
                    FUSEDMB_CFG,
                    FUSEDMB_e1,
                    {
                        "pw_args": {
                            "kernel_size": 5,
                            "padding": 2,
                            "stride": 1,
                        },
                    },
                    IRF_CFG,
                ],
            ],
            MODEL_ARCH_FBNETV3["FBNetV3_G"]["blocks"][4],
            MODEL_ARCH_FBNETV3["FBNetV3_G"]["blocks"][5],
            MODEL_ARCH_FBNETV3["FBNetV3_G"]["blocks"][6],
        ],
    },
}

MODEL_ARCH.register_dict(MODEL_ARCH_FBNETV3)
MODEL_ARCH.register_dict(mdu.get_i8f_models(MODEL_ARCH_FBNETV3))

MODEL_ARCH.register_dict(MODEL_ARCH_FBNETV3_GPU)
MODEL_ARCH.register_dict(mdu.get_i8f_models(MODEL_ARCH_FBNETV3_GPU))
