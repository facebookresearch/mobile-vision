#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import mobile_cv.common.misc.registry as registry

from . import modeldef_utils as mdu
from .modeldef_utils import e1, e6


MODEL_ARCH = registry.Registry("cls_arch_factory")


MODEL_ARCH_DEFAULT = {
    "default": {
        "blocks": [
            # [op, c, s, n, ...]
            # stage 0
            [("conv_k3", 32, 2, 1)],
            # stage 1
            [("ir_k3", 16, 1, 1, e1)],
            # stage 2
            [("ir_k3", 24, 2, 2, e6)],
            # stage 3
            [("ir_k3", 32, 2, 3, e6)],
            # stage 4
            [("ir_k3", 64, 2, 4, e6), ("ir_k3", 96, 1, 3, e6)],
            # stage 5
            [("ir_k3", 160, 2, 3, e6), ("ir_k3", 320, 1, 1, e6)],
            # stage 6
            [("conv_k1", 1280, 1, 1)],
        ]
    },
}
MODEL_ARCH.register_dict(MODEL_ARCH_DEFAULT)
MODEL_ARCH.register_dict(mdu.get_i8f_models(MODEL_ARCH_DEFAULT))
