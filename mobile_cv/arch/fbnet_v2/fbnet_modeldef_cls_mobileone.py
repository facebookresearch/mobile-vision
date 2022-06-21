#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .fbnet_modeldef_cls import MODEL_ARCH


BASIC_ARGS = {}


TRAIN_CFG = {
    "bias": False,
    "deploy": False,
}


DEPLOY_CFG = {
    "bias": True,
    "deploy": True,
}


MODEL_ARCH_MOBILEONE = {
    "MobileOne-S0-Train": {
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [("mobileone", 48, 2, 1, {"over_param_branches": 4}, TRAIN_CFG)],
            [("mobileone", 48, 2, 2, {"over_param_branches": 4}, TRAIN_CFG)],
            [("mobileone", 128, 2, 8, {"over_param_branches": 4}, TRAIN_CFG)],
            [("mobileone", 256, 2, 5, {"over_param_branches": 4}, TRAIN_CFG)],
            [("mobileone", 256, 1, 5, {"over_param_branches": 4}, TRAIN_CFG)],
            [("mobileone", 1024, 2, 1, {"over_param_branches": 4}, TRAIN_CFG)],
            [
                ("adaptive_avg_pool", 1024, 1, 1, {"output_size": 1}),
                ("conv_k1", 1024, 1, 1, {"bias": False}),
            ],
        ],
    },
    "MobileOne-S0-Deploy": {
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [("mobileone", 48, 2, 1, {"over_param_branches": 4}, DEPLOY_CFG)],
            [("mobileone", 48, 2, 2, {"over_param_branches": 4}, DEPLOY_CFG)],
            [("mobileone", 128, 2, 8, {"over_param_branches": 4}, DEPLOY_CFG)],
            [("mobileone", 256, 2, 5, {"over_param_branches": 4}, DEPLOY_CFG)],
            [("mobileone", 256, 1, 5, {"over_param_branches": 4}, DEPLOY_CFG)],
            [("mobileone", 1024, 2, 1, {"over_param_branches": 4}, DEPLOY_CFG)],
            [
                ("adaptive_avg_pool", 1024, 1, 1, {"output_size": 1}),
                ("conv_k1", 1024, 1, 1, {"bias": False}),
            ],
        ],
    },
    "MobileOne-S1-Train": {
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [("mobileone", 96, 2, 1, {"over_param_branches": 1}, TRAIN_CFG)],
            [("mobileone", 96, 2, 2, {"over_param_branches": 1}, TRAIN_CFG)],
            [("mobileone", 192, 2, 8, {"over_param_branches": 1}, TRAIN_CFG)],
            [("mobileone", 512, 2, 5, {"over_param_branches": 1}, TRAIN_CFG)],
            [("mobileone", 512, 1, 5, {"over_param_branches": 1}, TRAIN_CFG)],
            [("mobileone", 1280, 2, 1, {"over_param_branches": 1}, TRAIN_CFG)],
            [
                ("adaptive_avg_pool", 1280, 1, 1, {"output_size": 1}),
                ("conv_k1", 1280, 1, 1, {"bias": False}),
            ],
        ],
    },
    "MobileOne-S1-Deploy": {
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [("mobileone", 96, 2, 1, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 96, 2, 2, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 192, 2, 8, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 512, 2, 5, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 512, 1, 5, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 1280, 2, 1, {"over_param_branches": 1}, DEPLOY_CFG)],
            [
                ("adaptive_avg_pool", 1280, 1, 1, {"output_size": 1}),
                ("conv_k1", 1280, 1, 1, {"bias": False}),
            ],
        ],
    },
    "MobileOne-S2-Train": {
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [("mobileone", 96, 2, 1, {"over_param_branches": 1}, TRAIN_CFG)],
            [("mobileone", 96, 2, 2, {"over_param_branches": 1}, TRAIN_CFG)],
            [("mobileone", 256, 2, 8, {"over_param_branches": 1}, TRAIN_CFG)],
            [("mobileone", 640, 2, 5, {"over_param_branches": 1}, TRAIN_CFG)],
            [("mobileone", 640, 1, 5, {"over_param_branches": 1}, TRAIN_CFG)],
            [("mobileone", 2048, 2, 1, {"over_param_branches": 1}, TRAIN_CFG)],
            [
                ("adaptive_avg_pool", 2048, 1, 1, {"output_size": 1}),
                ("conv_k1", 2048, 1, 1, {"bias": False}),
            ],
        ],
    },
    "MobileOne-S2-Deploy": {
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [("mobileone", 96, 2, 1, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 96, 2, 2, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 256, 2, 8, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 640, 2, 5, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 640, 1, 5, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 2048, 2, 1, {"over_param_branches": 1}, DEPLOY_CFG)],
            [
                ("adaptive_avg_pool", 2048, 1, 1, {"output_size": 1}),
                ("conv_k1", 2048, 1, 1, {"bias": False}),
            ],
        ],
    },
    "MobileOne-S3-Train": {
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [("mobileone", 128, 2, 1, {"over_param_branches": 1}, TRAIN_CFG)],
            [("mobileone", 128, 2, 2, {"over_param_branches": 1}, TRAIN_CFG)],
            [("mobileone", 320, 2, 8, {"over_param_branches": 1}, TRAIN_CFG)],
            [("mobileone", 768, 2, 5, {"over_param_branches": 1}, TRAIN_CFG)],
            [("mobileone", 768, 1, 5, {"over_param_branches": 1}, TRAIN_CFG)],
            [("mobileone", 2048, 2, 1, {"over_param_branches": 1}, TRAIN_CFG)],
            [
                ("adaptive_avg_pool", 2048, 1, 1, {"output_size": 1}),
                ("conv_k1", 2048, 1, 1, {"bias": False}),
            ],
        ],
    },
    "MobileOne-S3-Deploy": {
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [("mobileone", 128, 2, 1, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 128, 2, 2, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 320, 2, 8, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 768, 2, 5, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 768, 1, 5, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 2048, 2, 1, {"over_param_branches": 1}, DEPLOY_CFG)],
            [
                ("adaptive_avg_pool", 2048, 1, 1, {"output_size": 1}),
                ("conv_k1", 2048, 1, 1, {"bias": False}),
            ],
        ],
    },
    # TODO(xfw): Add SE-ReLU in MobileOne-S4
    "MobileOne-S4-Train": {
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [("mobileone", 192, 2, 1, {"over_param_branches": 1}, TRAIN_CFG)],
            [("mobileone", 192, 2, 2, {"over_param_branches": 1}, TRAIN_CFG)],
            [("mobileone", 448, 2, 8, {"over_param_branches": 1}, TRAIN_CFG)],
            [("mobileone", 896, 2, 5, {"over_param_branches": 1}, TRAIN_CFG)],
            [("mobileone", 896, 1, 5, {"over_param_branches": 1}, TRAIN_CFG)],
            [("mobileone", 2048, 2, 1, {"over_param_branches": 1}, TRAIN_CFG)],
            [
                ("adaptive_avg_pool", 2048, 1, 1, {"output_size": 1}),
                ("conv_k1", 2048, 1, 1, {"bias": False}),
            ],
        ],
    },
    "MobileOne-S4-Deploy": {
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [("mobileone", 192, 2, 1, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 192, 2, 2, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 448, 2, 8, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 896, 2, 5, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 896, 1, 5, {"over_param_branches": 1}, DEPLOY_CFG)],
            [("mobileone", 2048, 2, 1, {"over_param_branches": 1}, DEPLOY_CFG)],
            [
                ("adaptive_avg_pool", 2048, 1, 1, {"output_size": 1}),
                ("conv_k1", 2048, 1, 1, {"bias": False}),
            ],
        ],
    },
}


MODEL_ARCH.register_dict(MODEL_ARCH_MOBILEONE)
