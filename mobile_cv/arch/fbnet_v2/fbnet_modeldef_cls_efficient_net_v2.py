#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .fbnet_modeldef_cls import MODEL_ARCH
from .modeldef_utils import e4, e6


BASIC_ARGS = {
    "relu_args": "swish",
    "width_divisor": 8,
}

IRF_ARGS = {"less_se_channels": True}

FUSEDMB_CFG = {"skip_dw": True, "always_pw": True}
FUSEDMB_e1 = {"skip_pwl": True, "mid_expand_out": True, "expansion": 1}

MODEL_ARCH_EFFICIENT_NET_V2 = {
    "eff_v2_s": {
        "input_size": 384,
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3", 24, 2, 1]],
            # stage 1
            [
                [
                    "ir_k3",
                    24,
                    1,
                    2,
                    FUSEDMB_e1,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 1,
                        },
                    },
                    FUSEDMB_CFG,
                ]
            ],
            # stage 2
            [
                [
                    "ir_k3",
                    48,
                    2,
                    1,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 2,
                        },
                    },
                    FUSEDMB_CFG,
                ],
                [
                    "ir_k3",
                    48,
                    1,
                    3,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 1,
                        },
                    },
                    FUSEDMB_CFG,
                ],
            ],
            # stage 3
            [
                [
                    "ir_k3",
                    64,
                    2,
                    1,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 2,
                        },
                    },
                    FUSEDMB_CFG,
                ],
                [
                    "ir_k3",
                    64,
                    1,
                    3,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 1,
                        },
                    },
                    FUSEDMB_CFG,
                ],
            ],
            # stage 4
            [
                ["ir_k3_se", 128, 2, 1, e4, IRF_ARGS],
                ["ir_k3_se", 128, 1, 5, e4, IRF_ARGS],
            ],
            # stage 5
            [
                ["ir_k3_se", 160, 1, 9, e6, IRF_ARGS],
            ],
            # stage 6
            [
                ["ir_k3_se", 256, 2, 1, e6, IRF_ARGS],
                ["ir_k3_se", 256, 1, 14, e6, IRF_ARGS],
            ],
            # stage 7
            [["conv_k1", 1280, 1, 1]],
        ],
    },
    "eff_v2_m": {
        "input_size": 480,
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3", 24, 2, 1]],
            # stage 1
            [
                [
                    "ir_k3",
                    24,
                    1,
                    3,
                    FUSEDMB_e1,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 1,
                        },
                    },
                    FUSEDMB_CFG,
                ]
            ],
            # stage 2
            [
                [
                    "ir_k3",
                    48,
                    2,
                    1,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 2,
                        },
                    },
                    FUSEDMB_CFG,
                ],
                [
                    "ir_k3",
                    48,
                    1,
                    4,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 1,
                        },
                    },
                    FUSEDMB_CFG,
                ],
            ],
            # stage 3
            [
                [
                    "ir_k3",
                    80,
                    2,
                    1,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 2,
                        },
                    },
                    FUSEDMB_CFG,
                ],
                [
                    "ir_k3",
                    80,
                    1,
                    4,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 1,
                        },
                    },
                    FUSEDMB_CFG,
                ],
            ],
            # stage 4
            [
                ["ir_k3_se", 160, 2, 1, e4, IRF_ARGS],
                ["ir_k3_se", 160, 1, 6, e4, IRF_ARGS],
            ],
            # stage 5
            [
                ["ir_k3_se", 176, 1, 14, e6, IRF_ARGS],
            ],
            # stage 6
            [
                ["ir_k3_se", 304, 2, 1, e6, IRF_ARGS],
                ["ir_k3_se", 304, 1, 17, e6, IRF_ARGS],
            ],
            # stage 7
            [
                ["ir_k3_se", 512, 1, 5, e6, IRF_ARGS],
            ],
            # stage 8
            [["conv_k1", 1280, 1, 1]],
        ],
    },
    "eff_v2_l": {
        "input_size": 480,
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3", 32, 2, 1]],
            # stage 1
            [
                [
                    "ir_k3",
                    32,
                    1,
                    4,
                    FUSEDMB_e1,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 1,
                        },
                    },
                    FUSEDMB_CFG,
                ]
            ],
            # stage 2
            [
                [
                    "ir_k3",
                    64,
                    2,
                    1,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 2,
                        },
                    },
                    FUSEDMB_CFG,
                ],
                [
                    "ir_k3",
                    64,
                    1,
                    6,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 1,
                        },
                    },
                    FUSEDMB_CFG,
                ],
            ],
            # stage 3
            [
                [
                    "ir_k3",
                    96,
                    2,
                    1,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 2,
                        },
                    },
                    FUSEDMB_CFG,
                ],
                [
                    "ir_k3",
                    96,
                    1,
                    6,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 1,
                        },
                    },
                    FUSEDMB_CFG,
                ],
            ],
            # stage 4
            [
                ["ir_k3_se", 192, 2, 1, e4, IRF_ARGS],
                ["ir_k3_se", 192, 1, 9, e4, IRF_ARGS],
            ],
            # stage 5
            [
                ["ir_k3_se", 224, 1, 19, e6, IRF_ARGS],
            ],
            # stage 6
            [
                ["ir_k3_se", 384, 2, 1, e6, IRF_ARGS],
                ["ir_k3_se", 384, 1, 24, e6, IRF_ARGS],
            ],
            # stage 7
            [
                ["ir_k3_se", 640, 1, 7, e6, IRF_ARGS],
            ],
            # stage 8
            [["conv_k1", 1280, 1, 1]],
        ],
    },
    "eff_v2_xl": {
        "input_size": 512,
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3", 32, 2, 1]],
            # stage 1
            [
                [
                    "ir_k3",
                    32,
                    1,
                    4,
                    FUSEDMB_e1,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 1,
                        },
                    },
                    FUSEDMB_CFG,
                ]
            ],
            # stage 2
            [
                [
                    "ir_k3",
                    64,
                    2,
                    1,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 2,
                        },
                    },
                    FUSEDMB_CFG,
                ],
                [
                    "ir_k3",
                    64,
                    1,
                    7,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 1,
                        },
                    },
                    FUSEDMB_CFG,
                ],
            ],
            # stage 3
            [
                [
                    "ir_k3",
                    96,
                    2,
                    1,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 2,
                        },
                    },
                    FUSEDMB_CFG,
                ],
                [
                    "ir_k3",
                    96,
                    1,
                    7,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 1,
                        },
                    },
                    FUSEDMB_CFG,
                ],
            ],
            # stage 4
            [
                ["ir_k3_se", 192, 2, 1, e4, IRF_ARGS],
                ["ir_k3_se", 192, 1, 15, e4, IRF_ARGS],
            ],
            # stage 5
            [
                ["ir_k3_se", 256, 1, 24, e6, IRF_ARGS],
            ],
            # stage 6
            [
                ["ir_k3_se", 512, 2, 1, e6, IRF_ARGS],
                ["ir_k3_se", 512, 1, 31, e6, IRF_ARGS],
            ],
            # stage 7
            [
                ["ir_k3_se", 640, 1, 8, e6, IRF_ARGS],
            ],
            # stage 8
            [["conv_k1", 1280, 1, 1]],
        ],
    },
    "eff_v2_b0": {
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3", 32, 2, 1]],
            # stage 1
            [
                [
                    "ir_k3",
                    16,
                    1,
                    1,
                    FUSEDMB_e1,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 1,
                        },
                    },
                    FUSEDMB_CFG,
                ]
            ],
            # stage 2
            [
                [
                    "ir_k3",
                    32,
                    2,
                    1,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 2,
                        },
                    },
                    FUSEDMB_CFG,
                ],
                [
                    "ir_k3",
                    32,
                    1,
                    1,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 1,
                        },
                    },
                    FUSEDMB_CFG,
                ],
            ],
            # stage 3
            [
                [
                    "ir_k3",
                    48,
                    2,
                    1,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 2,
                        },
                    },
                    FUSEDMB_CFG,
                ],
                [
                    "ir_k3",
                    48,
                    1,
                    1,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 1,
                        },
                    },
                    FUSEDMB_CFG,
                ],
            ],
            # stage 4
            [
                ["ir_k3_se", 96, 2, 1, e4, IRF_ARGS],
                ["ir_k3_se", 96, 1, 2, e4, IRF_ARGS],
            ],
            # stage 5
            [
                ["ir_k3_se", 112, 1, 5, e6, IRF_ARGS],
            ],
            # stage 6
            [
                ["ir_k3_se", 192, 2, 1, e6, IRF_ARGS],
                ["ir_k3_se", 192, 1, 7, e6, IRF_ARGS],
            ],
            # stage 7
            [["conv_k1", 1280, 1, 1]],
        ],
    },
    "eff_v2_b1": {
        "input_size": 240,
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3", 32, 2, 1]],
            # stage 1
            [
                [
                    "ir_k3",
                    16,
                    1,
                    2,
                    FUSEDMB_e1,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 1,
                        },
                    },
                    FUSEDMB_CFG,
                ]
            ],
            # stage 2
            [
                [
                    "ir_k3",
                    32,
                    2,
                    1,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 2,
                        },
                    },
                    FUSEDMB_CFG,
                ],
                [
                    "ir_k3",
                    32,
                    1,
                    2,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 1,
                        },
                    },
                    FUSEDMB_CFG,
                ],
            ],
            # stage 3
            [
                [
                    "ir_k3",
                    48,
                    2,
                    1,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 2,
                        },
                    },
                    FUSEDMB_CFG,
                ],
                [
                    "ir_k3",
                    48,
                    1,
                    2,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 1,
                        },
                    },
                    FUSEDMB_CFG,
                ],
            ],
            # stage 4
            [
                ["ir_k3_se", 96, 2, 1, e4, IRF_ARGS],
                ["ir_k3_se", 96, 1, 3, e4, IRF_ARGS],
            ],
            # stage 5
            [
                ["ir_k3_se", 112, 1, 6, e6, IRF_ARGS],
            ],
            # stage 6
            [
                ["ir_k3_se", 192, 2, 1, e6, IRF_ARGS],
                ["ir_k3_se", 192, 1, 8, e6, IRF_ARGS],
            ],
            # stage 7
            [["conv_k1", 1280, 1, 1]],
        ],
    },
    "eff_v2_b2": {
        "input_size": 260,
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3", 32, 2, 1]],
            # stage 1
            [
                [
                    "ir_k3",
                    16,
                    1,
                    2,
                    FUSEDMB_e1,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 1,
                        },
                    },
                    FUSEDMB_CFG,
                ]
            ],
            # stage 2
            [
                [
                    "ir_k3",
                    32,
                    2,
                    1,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 2,
                        },
                    },
                    FUSEDMB_CFG,
                ],
                [
                    "ir_k3",
                    32,
                    1,
                    2,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 1,
                        },
                    },
                    FUSEDMB_CFG,
                ],
            ],
            # stage 3
            [
                [
                    "ir_k3",
                    56,
                    2,
                    1,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 2,
                        },
                    },
                    FUSEDMB_CFG,
                ],
                [
                    "ir_k3",
                    56,
                    1,
                    2,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 1,
                        },
                    },
                    FUSEDMB_CFG,
                ],
            ],
            # stage 4
            [
                ["ir_k3_se", 104, 2, 1, e4, IRF_ARGS],
                ["ir_k3_se", 104, 1, 3, e4, IRF_ARGS],
            ],
            # stage 5
            [
                ["ir_k3_se", 120, 1, 6, e6, IRF_ARGS],
            ],
            # stage 6
            [
                ["ir_k3_se", 208, 2, 1, e6, IRF_ARGS],
                ["ir_k3_se", 208, 1, 9, e6, IRF_ARGS],
            ],
            # stage 7
            [["conv_k1", 1280, 1, 1]],
        ],
    },
    "eff_v2_b3": {
        "input_size": 300,
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3", 40, 2, 1]],
            # stage 1
            [
                [
                    "ir_k3",
                    16,
                    1,
                    2,
                    FUSEDMB_e1,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 1,
                        },
                    },
                    FUSEDMB_CFG,
                ]
            ],
            # stage 2
            [
                [
                    "ir_k3",
                    40,
                    2,
                    1,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 2,
                        },
                    },
                    FUSEDMB_CFG,
                ],
                [
                    "ir_k3",
                    40,
                    1,
                    2,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 1,
                        },
                    },
                    FUSEDMB_CFG,
                ],
            ],
            # stage 3
            [
                [
                    "ir_k3",
                    56,
                    2,
                    1,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 2,
                        },
                    },
                    FUSEDMB_CFG,
                ],
                [
                    "ir_k3",
                    56,
                    1,
                    2,
                    e4,
                    {
                        "pw_args": {
                            "kernel_size": 3,
                            "padding": 1,
                            "stride": 1,
                        },
                    },
                    FUSEDMB_CFG,
                ],
            ],
            # stage 4
            [
                ["ir_k3_se", 112, 2, 1, e4, IRF_ARGS],
                ["ir_k3_se", 112, 1, 4, e4, IRF_ARGS],
            ],
            # stage 5
            [
                ["ir_k3_se", 136, 1, 7, e6, IRF_ARGS],
            ],
            # stage 6
            [
                ["ir_k3_se", 232, 2, 1, e6, IRF_ARGS],
                ["ir_k3_se", 232, 1, 11, e6, IRF_ARGS],
            ],
            # stage 7
            [["conv_k1", 1280, 1, 1]],
        ],
    },
}

MODEL_ARCH.register_dict(MODEL_ARCH_EFFICIENT_NET_V2)
