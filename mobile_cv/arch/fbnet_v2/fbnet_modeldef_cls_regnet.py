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


BASIC_ARGS = {
    "bias": False,
}

IRF_CFG = {
    "expansion": 1,
    "res_conn_args": "projection",
    "last_relu": True,
    "always_pw": True,
    "mid_expand_out": True,
    "zero_last_bn_gamma": False,
}


def create_regnet_arch(block_type, depth, width, group):
    assert len(depth) == len(width) == len(group)
    ret = {"input_size": 224, "basic_args": BASIC_ARGS, "blocks": []}
    ret["blocks"].append([["conv_k3", 32, 2, 1]])
    for d, w, g in zip(depth, width, group):
        stage = []
        stage.append([block_type, w, 2, 1, {"dw_group_ratio": g}, IRF_CFG])
        if d > 1:
            stage.append([block_type, w, 1, d - 1, {"dw_group_ratio": g}, IRF_CFG])
        ret["blocks"].append(stage)
    return ret


REGNET_X_INFOS = [
    {
        "name": "RegNetX_200M",
        "block_type": "ir_k3",
        "depth": [1, 1, 4, 7],
        "width": [24, 56, 152, 368],
        "group": [8, 8, 8, 8],
    },
    {
        "name": "RegNetX_400M",
        "block_type": "ir_k3",
        "depth": [1, 2, 7, 12],
        "width": [32, 64, 160, 384],
        "group": [16, 16, 16, 16],
    },
    {
        "name": "RegNetX_600M",
        "block_type": "ir_k3",
        "depth": [1, 3, 5, 7],
        "width": [48, 96, 240, 528],
        "group": [24, 24, 24, 24],
    },
    {
        "name": "RegNetX_800M",
        "block_type": "ir_k3",
        "depth": [1, 3, 7, 5],
        "width": [64, 128, 288, 672],
        "group": [16, 16, 16, 16],
    },
    {
        "name": "RegNetX_1.6G",
        "block_type": "ir_k3",
        "depth": [2, 4, 10, 2],
        "width": [72, 168, 408, 912],
        "group": [24, 24, 24, 24],
    },
    {
        "name": "RegNetX_3.2G",
        "block_type": "ir_k3",
        "depth": [2, 6, 15, 2],
        "width": [96, 192, 432, 1008],
        "group": [48, 48, 48, 48],
    },
    {
        "name": "RegNetX_4.0G",
        "block_type": "ir_k3",
        "depth": [2, 5, 14, 2],
        "width": [80, 240, 560, 1360],
        "group": [40, 40, 40, 40],
    },
    {
        "name": "RegNetX_6.4G",
        "block_type": "ir_k3",
        "depth": [2, 4, 10, 1],
        "width": [168, 392, 784, 1624],
        "group": [56, 56, 56, 56],
    },
    {
        "name": "RegNetX_8.0G",
        "block_type": "ir_k3",
        "depth": [2, 5, 15, 1],
        "width": [80, 240, 720, 1920],
        "group": [80, 120, 120, 120],
    },
    {
        "name": "RegNetX_12G",
        "block_type": "ir_k3",
        "depth": [2, 5, 11, 1],
        "width": [224, 448, 896, 2240],
        "group": [112, 112, 112, 112],
    },
    {
        "name": "RegNetX_16G",
        "block_type": "ir_k3",
        "depth": [2, 6, 13, 1],
        "width": [256, 512, 896, 2048],
        "group": [128, 128, 128, 128],
    },
    {
        "name": "RegNetX_32G",
        "block_type": "ir_k3",
        "depth": [2, 7, 13, 1],
        "width": [336, 672, 1344, 2520],
        "group": [168, 168, 168, 168],
    },
]


MODEL_ARCH_REGNETX = {}
for model_info in REGNET_X_INFOS:
    MODEL_ARCH_REGNETX[model_info["name"]] = create_regnet_arch(  # noqa
        model_info["block_type"],
        model_info["depth"],
        model_info["width"],
        model_info["group"],
    )
MODEL_ARCH.register_dict(MODEL_ARCH_REGNETX)


IRF_CFG.update(
    {
        "round_se_channels": True,
    }
)

REGNET_Y_INFOS = [
    {
        "name": "RegNetY_200M",
        "block_type": "ir_k3_se",
        "depth": [1, 1, 4, 7],
        "width": [24, 56, 152, 368],
        "group": [8, 8, 8, 8],
    },
    {
        "name": "RegNetY_400M",
        "block_type": "ir_k3_se",
        "depth": [1, 3, 6, 6],
        "width": [48, 104, 208, 440],
        "group": [8, 8, 8, 8],
    },
    {
        "name": "RegNetY_600M",
        "block_type": "ir_k3_se",
        "depth": [1, 3, 7, 4],
        "width": [48, 112, 256, 608],
        "group": [16, 16, 16, 16],
    },
    {
        "name": "RegNetY_800M",
        "block_type": "ir_k3_se",
        "depth": [1, 3, 8, 2],
        "width": [64, 128, 320, 768],
        "group": [16, 16, 16, 16],
    },
    {
        "name": "RegNetY_1.6G",
        "block_type": "ir_k3_se",
        "depth": [2, 6, 17, 2],
        "width": [48, 120, 336, 888],
        "group": [24, 24, 24, 24],
    },
    {
        "name": "RegNetY_3.2G",
        "block_type": "ir_k3_se",
        "depth": [2, 5, 13, 1],
        "width": [72, 216, 576, 1512],
        "group": [24, 24, 24, 24],
    },
    {
        "name": "RegNetY_4.0G",
        "block_type": "ir_k3_se",
        "depth": [2, 6, 12, 2],
        "width": [128, 192, 512, 1088],
        "group": [64, 64, 64, 64],
    },
    {
        "name": "RegNetY_6.4G",
        "block_type": "ir_k3_se",
        "depth": [2, 7, 14, 2],
        "width": [144, 288, 576, 1296],
        "group": [72, 72, 72, 72],
    },
    {
        "name": "RegNetY_8.0G",
        "block_type": "ir_k3_se",
        "depth": [2, 4, 10, 1],
        "width": [168, 448, 896, 2016],
        "group": [56, 56, 56, 56],
    },
    {
        "name": "RegNetY_12G",
        "block_type": "ir_k3_se",
        "depth": [2, 5, 11, 1],
        "width": [224, 448, 896, 2240],
        "group": [112, 112, 112, 112],
    },
    {
        "name": "RegNetY_16G",
        "block_type": "ir_k3_se",
        "depth": [2, 4, 11, 1],
        "width": [224, 448, 1232, 3024],
        "group": [112, 112, 112, 112],
    },
    {
        "name": "RegNetY_32G",
        "block_type": "ir_k3_se",
        "depth": [2, 5, 12, 1],
        "width": [232, 696, 1392, 3712],
        "group": [232, 232, 232, 232],
    },
]


MODEL_ARCH_REGNETY = {}
for model_info in REGNET_Y_INFOS:
    MODEL_ARCH_REGNETY[model_info["name"]] = create_regnet_arch(  # noqa
        model_info["block_type"],
        model_info["depth"],
        model_info["width"],
        model_info["group"],
    )
MODEL_ARCH.register_dict(MODEL_ARCH_REGNETY)
