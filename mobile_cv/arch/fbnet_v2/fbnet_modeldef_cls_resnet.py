#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .fbnet_modeldef_cls import MODEL_ARCH


BASIC_ARGS = {}

NO_BIAS = {
    "bias": False,
}

RES_CFG = {
    "downsample_in_conv2": False,
    "bn_in_skip": True,
    "bias_in_skip": False,
    "bias": False,
}


RES_BN_CFG = {
    "bias": False,
    "bn_in_skip": True,
}


MODEL_ARCH_RESNET = {
    "ResNet18": {
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [
                ("conv_k7", 64, 2, 1, NO_BIAS),
                ("maxpool", 64, 2, 1, {"kernel_size": 3, "padding": 1}),
            ],
            [("res_k3", 64, 1, 2, RES_CFG)],
            [("res_k3", 128, 2, 2, RES_CFG)],
            [("res_k3", 256, 2, 2, RES_CFG)],
            [("res_k3", 512, 2, 2, RES_CFG)],
        ],
    },
    "ResNet34": {
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [
                ("conv_k7", 64, 2, 1, NO_BIAS),
                ("maxpool", 64, 2, 1, {"kernel_size": 3, "padding": 1}),
            ],
            [("res_k3", 64, 1, 3, RES_CFG)],
            [("res_k3", 128, 2, 4, RES_CFG)],
            [("res_k3", 256, 2, 6, RES_CFG)],
            [("res_k3", 512, 2, 3, RES_CFG)],
        ],
    },
    "ResNet50": {
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [
                ("conv_k7", 64, 2, 1, NO_BIAS),
                ("maxpool", 64, 2, 1, {"kernel_size": 3, "padding": 1}),
            ],
            [("res_block_k3", 256, 1, 3, RES_BN_CFG)],
            [("res_block_k3", 512, 2, 4, RES_BN_CFG)],
            [("res_block_k3", 1024, 2, 6, RES_BN_CFG)],
            [("res_block_k3", 2048, 2, 3, RES_BN_CFG)],
        ],
    },
    "ResNet101": {
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [
                ("conv_k7", 64, 2, 1, NO_BIAS),
                ("maxpool", 64, 2, 1, {"kernel_size": 3, "padding": 1}),
            ],
            [("res_block_k3", 256, 1, 3, RES_BN_CFG)],
            [("res_block_k3", 512, 2, 4, RES_BN_CFG)],
            [("res_block_k3", 1024, 2, 23, RES_BN_CFG)],
            [("res_block_k3", 2048, 2, 3, RES_BN_CFG)],
        ],
    },
    "ResNet152": {
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [
                ("conv_k7", 64, 2, 1, NO_BIAS),
                ("maxpool", 64, 2, 1, {"kernel_size": 3, "padding": 1}),
            ],
            [("res_block_k3", 256, 1, 3, RES_BN_CFG)],
            [("res_block_k3", 512, 2, 8, RES_BN_CFG)],
            [("res_block_k3", 1024, 2, 36, RES_BN_CFG)],
            [("res_block_k3", 2048, 2, 3, RES_BN_CFG)],
        ],
    },
}

MODEL_ARCH.register_dict(MODEL_ARCH_RESNET)
