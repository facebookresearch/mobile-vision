#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from . import modeldef_utils as mdu  # noqa
from .fbnet_modeldef_cls import MODEL_ARCH
from .modeldef_utils import _ex, e1, e6


BASIC_ARGS = {
    "width_divisor": 8,
}

IRF_CFG = {
    "less_se_channels": False,
    "zero_last_bn_gamma": True,
}

use_bias = {"bias": True}


MODEL_ARCH_FBNETV2_PAPER = {
    "FBNetV2_F0": {
        # nparams: 1.53, nflops 65.2,
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3_hs", 8, 2, 1]],
            # stage 1
            [["ir_k5_hs", 8, 1, 1, e1, IRF_CFG]],
            # stage 2
            [
                ["ir_k5_hs", 16, 2, 1, _ex(4), IRF_CFG],
                ["ir_k5_hs", 16, 1, 1, _ex(3), IRF_CFG],
            ],
            # stage 3
            [
                ["ir_k5_sehsig", 24, 2, 1, _ex(4), IRF_CFG],
                ["ir_k5_sehsig", 24, 1, 1, _ex(3), IRF_CFG],
            ],
            # stage 4
            [
                ["ir_k5_sehsig_hs", 48, 2, 1, _ex(4), IRF_CFG],
                ["ir_k3_sehsig_hs", 48, 1, 1, _ex(3), IRF_CFG],
                ["ir_k5_sehsig_hs", 64, 1, 1, _ex(3), IRF_CFG],
                ["ir_k5_sehsig_hs", 64, 1, 1, _ex(2), IRF_CFG],
            ],
            # stage 5
            [
                ["ir_k3_sehsig_hs", 128, 2, 1, _ex(3), IRF_CFG],
                ["ir_k5_sehsig_hs", 160, 1, 1, _ex(3), IRF_CFG],
                ["skip", 192, 1, 1, _ex(4), IRF_CFG],
            ],
            # stage 6
            [["ir_pool_hs", 1280, 1, 1, _ex(4)]],
        ],
    },
    "FBNetV2_F1": {
        # nparams: 5.999408, nflops 56.41056,
        # accuracy 68.3% at job f167001958 test scale 151
        "input_size": 128,
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3_hs", 8, 2, 1]],
            # stage 1
            [["ir_k5", 8, 1, 1, e1, IRF_CFG]],
            # stage 2
            [
                ["ir_k5", 24, 2, 1, _ex(5.4566), IRF_CFG],
                ["ir_k5", 24, 1, 1, _ex(4.7912), IRF_CFG],
            ],
            # stage 3
            [
                ["ir_k5_sehsig", 32, 2, 1, _ex(5.3501), IRF_CFG],
                ["ir_k5_sehsig", 24, 1, 1, _ex(4.5379), IRF_CFG],
            ],
            # stage 4
            [
                ["ir_k5_hs", 56, 2, 1, _ex(5.7133), IRF_CFG],
                ["ir_k3_hs", 56, 1, 1, _ex(4.1212), IRF_CFG],
                ["ir_k3_sehsig_hs", 56, 1, 1, _ex(5.1246), IRF_CFG],
                ["skip", 80, 1, 1, _ex(5.0333), IRF_CFG],
                ["ir_k5_sehsig_hs", 80, 1, 1, _ex(4.5070), IRF_CFG],
                ["ir_k5_sehsig_hs", 80, 1, 1, _ex(1.7712), IRF_CFG],
            ],
            # stage 5
            [
                ["ir_k3_sehsig_hs", 144, 2, 1, _ex(4.5685), IRF_CFG],
                ["ir_k5_sehsig_hs", 144, 1, 1, _ex(5.8400), IRF_CFG],
                ["ir_k5_sehsig_hs", 144, 1, 1, _ex(6.8754), IRF_CFG],
                ["skip", 224, 1, 1, _ex(6.5245), IRF_CFG],
            ],
            # stage 6
            [["ir_pool_hs", 1600, 1, 1, e6]],
        ],
    },
    "FBNetV2_F2": {
        # nparams: 5.999408, nflops 85.441392
        # accuracy 71.1% at job: f167001918
        "input_size": 160,
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3_hs", 8, 2, 1]],
            # stage 1
            [["ir_k5", 8, 1, 1, e1, IRF_CFG]],
            # stage 2
            [
                ["ir_k5", 24, 2, 1, _ex(5.4566), IRF_CFG],
                ["ir_k5", 24, 1, 1, _ex(4.7912), IRF_CFG],
            ],
            # stage 3
            [
                ["ir_k5_sehsig", 32, 2, 1, _ex(5.3501), IRF_CFG],
                ["ir_k5_sehsig", 24, 1, 1, _ex(4.5379), IRF_CFG],
            ],
            # stage 4
            [
                ["ir_k5_hs", 56, 2, 1, _ex(5.7133), IRF_CFG],
                ["ir_k3_hs", 56, 1, 1, _ex(4.1212), IRF_CFG],
                ["ir_k3_sehsig_hs", 56, 1, 1, _ex(5.1246), IRF_CFG],
                ["skip", 80, 1, 1, _ex(5.0333), IRF_CFG],
                ["ir_k5_sehsig_hs", 80, 1, 1, _ex(4.5070), IRF_CFG],
                ["ir_k5_sehsig_hs", 80, 1, 1, _ex(1.7712), IRF_CFG],
            ],
            # stage 5
            [
                ["ir_k3_sehsig_hs", 144, 2, 1, _ex(4.5685), IRF_CFG],
                ["ir_k5_sehsig_hs", 144, 1, 1, _ex(5.8400), IRF_CFG],
                ["ir_k5_sehsig_hs", 144, 1, 1, _ex(6.8754), IRF_CFG],
                ["skip", 224, 1, 1, _ex(6.5245), IRF_CFG],
            ],
            # stage 6
            [["ir_pool_hs", 1600, 1, 1, e6]],
        ],
    },
    "FBNetV2_F3": {
        # nparams: 6.899504, nflops 121.824
        # accuracy: 73.1% at job: f167001975
        "input_size": 192,
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3_hs", 8, 2, 1]],
            # stage 1
            [["ir_k5", 8, 1, 1, e1, IRF_CFG]],
            # stage 2
            [
                ["ir_k5", 24, 2, 1, _ex(5.4566), IRF_CFG],
                ["ir_k5", 24, 1, 1, _ex(4.7912), IRF_CFG],
            ],
            # stage 3
            [
                ["ir_k5_sehsig", 32, 2, 1, _ex(5.3501), IRF_CFG],
                ["ir_k5_sehsig", 24, 1, 1, _ex(4.5379), IRF_CFG],
            ],
            # stage 4
            [
                ["ir_k5_hs", 56, 2, 1, _ex(5.7133), IRF_CFG],
                ["ir_k3_hs", 56, 1, 1, _ex(4.1212), IRF_CFG],
                ["ir_k3_sehsig_hs", 56, 1, 1, _ex(5.1246), IRF_CFG],
                ["skip", 80, 1, 1, _ex(5.0333), IRF_CFG],
                ["ir_k5_sehsig_hs", 80, 1, 1, _ex(4.5070), IRF_CFG],
                ["ir_k5_sehsig_hs", 80, 1, 1, _ex(1.7712), IRF_CFG],
            ],
            # stage 5
            [
                ["ir_k3_sehsig_hs", 144, 2, 1, _ex(4.5685), IRF_CFG],
                ["ir_k5_sehsig_hs", 144, 1, 1, _ex(5.8400), IRF_CFG],
                ["ir_k5_sehsig_hs", 144, 1, 1, _ex(6.8754), IRF_CFG],
                ["skip", 224, 1, 1, _ex(6.5245), IRF_CFG],
            ],
            # stage 6
            [["ir_pool_hs", 1984, 1, 1, e6]],
        ],
    },
    "FBNetV2_F4": {
        # nparams: 6.994824, nflops 238.351984
        # 75.8% accuracy at job: f167001913 test scale 251
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3_hs", 16, 2, 1]],
            # stage 1
            [["ir_k3_hs", 16, 1, 1, e1, IRF_CFG]],
            # stage 2
            [
                ["ir_k5_hs", 24, 2, 1, _ex(5.4566), IRF_CFG],
                ["ir_k5_hs", 24, 1, 1, _ex(1.7912), IRF_CFG],
                ["ir_k5_hs", 24, 1, 1, _ex(1.7912), IRF_CFG],
            ],
            # stage 3
            [
                ["ir_k5_sehsig", 32, 2, 1, _ex(5.3501), IRF_CFG],
                ["ir_k5_sehsig", 32, 1, 1, _ex(3.5379), IRF_CFG],
                ["ir_k5_sehsig", 32, 1, 1, _ex(4.5379), IRF_CFG],
                ["ir_k5_sehsig", 32, 1, 1, _ex(4.5379), IRF_CFG],
            ],
            # stage 4
            [
                ["ir_k5_hs", 64, 2, 1, _ex(5.7133), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(2.1212), IRF_CFG],
                ["skip", 64, 1, 1, _ex(3.1246), IRF_CFG],
                ["ir_k3_hs", 104, 1, 1, _ex(5.0333), IRF_CFG],
                ["ir_k5_sehsig_hs", 104, 1, 1, _ex(2.5070), IRF_CFG],
                ["ir_k5_sehsig_hs", 104, 1, 1, _ex(1.7712), IRF_CFG],
                ["ir_k5_sehsig_hs", 112, 1, 1, _ex(3.7712), IRF_CFG],
            ],
            # stage 5
            [
                ["ir_k3_sehsig_hs", 184, 2, 1, _ex(5.5685), IRF_CFG],
                ["ir_k5_sehsig_hs", 184, 1, 1, _ex(2.8400), IRF_CFG],
                ["ir_k5_sehsig_hs", 184, 1, 1, _ex(4.8754), IRF_CFG],
                ["skip", 224, 1, 1, _ex(6.5245), IRF_CFG],
            ],
            # stage 6
            [["ir_pool_hs", 1984, 1, 1, e6]],
        ],
    },
    "FBNetV2_L1": {
        # accuracy 77.2% at job: f183461678
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [["conv_k3_hs", 16, 2, 1]],
            [["ir_k3_hs", 16, 1, 1, e1, IRF_CFG]],
            [
                ["ir_k5_hs", 24, 2, 1, _ex(5.4566), IRF_CFG],
                ["ir_k5_hs", 24, 1, 1, _ex(1.7912), IRF_CFG],
                ["ir_k3_hs", 24, 1, 1, _ex(1.7912), IRF_CFG],
                ["ir_k5_hs", 24, 1, 1, _ex(1.7912), IRF_CFG],
            ],
            [
                ["ir_k5_sehsig_hs", 40, 2, 1, _ex(5.3501), IRF_CFG],
                ["ir_k5_sehsig_hs", 32, 1, 1, _ex(3.5379), IRF_CFG],
                ["ir_k5_sehsig_hs", 32, 1, 1, _ex(4.5379), IRF_CFG],
                ["ir_k5_sehsig_hs", 32, 1, 1, _ex(4.5379), IRF_CFG],
            ],
            [
                ["ir_k5_hs", 64, 2, 1, _ex(5.7133), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(2.1212), IRF_CFG],
                ["skip", 64, 1, 1, _ex(3.1246), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(3.1246), IRF_CFG],
                ["ir_k3_hs", 112, 1, 1, _ex(5.0333), IRF_CFG],
                ["ir_k5_sehsig_hs", 112, 1, 1, _ex(2.5070), IRF_CFG],
                ["ir_k5_sehsig_hs", 112, 1, 1, _ex(1.7712), IRF_CFG],
                ["ir_k5_sehsig_hs", 112, 1, 1, _ex(2.7712), IRF_CFG],
                ["ir_k5_sehsig_hs", 112, 1, 1, _ex(3.7712), IRF_CFG],
                ["ir_k5_sehsig_hs", 112, 1, 1, _ex(3.7712), IRF_CFG],
            ],
            [
                ["ir_k3_sehsig_hs", 184, 2, 1, _ex(5.5685), IRF_CFG],
                ["ir_k5_sehsig_hs", 184, 1, 1, _ex(2.8400), IRF_CFG],
                ["ir_k5_sehsig_hs", 184, 1, 1, _ex(2.8400), IRF_CFG],
                ["ir_k5_sehsig_hs", 184, 1, 1, _ex(4.8754), IRF_CFG],
                ["ir_k5_sehsig_hs", 184, 1, 1, _ex(4.8754), IRF_CFG],
                ["skip", 224, 1, 1, _ex(6.5245), IRF_CFG],
            ],
            [["ir_pool_hs", 1984, 1, 1, e6]],
        ],
    },
    "FBNetV2_L2": {
        # nparams: 8.49652 nflops: 423.912768
        # accuracy 78.2% at job: f167001961
        "input_size": 256,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [["conv_k3_hs", 16, 2, 1]],
            [["ir_k3_hs", 16, 1, 1, e1, IRF_CFG]],
            [
                ["ir_k5_hs", 24, 2, 1, _ex(5.4566), IRF_CFG],
                ["ir_k5_hs", 24, 1, 1, _ex(1.7912), IRF_CFG],
                ["ir_k3_hs", 24, 1, 1, _ex(1.7912), IRF_CFG],
                ["ir_k5_hs", 24, 1, 1, _ex(1.7912), IRF_CFG],
            ],
            [
                ["ir_k5_sehsig_hs", 40, 2, 1, _ex(5.3501), IRF_CFG],
                ["ir_k5_sehsig_hs", 32, 1, 1, _ex(3.5379), IRF_CFG],
                ["ir_k5_sehsig_hs", 32, 1, 1, _ex(4.5379), IRF_CFG],
                ["ir_k5_sehsig_hs", 32, 1, 1, _ex(4.5379), IRF_CFG],
            ],
            [
                ["ir_k5_hs", 64, 2, 1, _ex(5.7133), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(2.1212), IRF_CFG],
                ["skip", 64, 1, 1, _ex(3.1246), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(3.1246), IRF_CFG],
                ["ir_k3_hs", 112, 1, 1, _ex(5.0333), IRF_CFG],
                ["ir_k5_sehsig_hs", 112, 1, 1, _ex(2.5070), IRF_CFG],
                ["ir_k5_sehsig_hs", 112, 1, 1, _ex(1.7712), IRF_CFG],
                ["ir_k5_sehsig_hs", 112, 1, 1, _ex(2.7712), IRF_CFG],
                ["ir_k5_sehsig_hs", 112, 1, 1, _ex(3.7712), IRF_CFG],
                ["ir_k5_sehsig_hs", 112, 1, 1, _ex(3.7712), IRF_CFG],
            ],
            [
                ["ir_k3_sehsig_hs", 184, 2, 1, _ex(5.5685), IRF_CFG],
                ["ir_k5_sehsig_hs", 184, 1, 1, _ex(2.8400), IRF_CFG],
                ["ir_k5_sehsig_hs", 184, 1, 1, _ex(2.8400), IRF_CFG],
                ["ir_k5_sehsig_hs", 184, 1, 1, _ex(4.8754), IRF_CFG],
                ["ir_k5_sehsig_hs", 184, 1, 1, _ex(4.8754), IRF_CFG],
                ["skip", 224, 1, 1, _ex(6.5245), IRF_CFG],
            ],
            [["ir_pool_hs", 1984, 1, 1, e6]],
        ],
    },
}

MODEL_ARCH.register_dict(MODEL_ARCH_FBNETV2_PAPER)
