#!/usr/bin/env python3
from .fbnet_modeldef_cls import MODEL_ARCH
from .modeldef_utils import _ex

_BASIC_ARGS = {
    "dw_skip_bnrelu": True,
    "width_divisor": 8,
    # uncomment below (always_pw and bias) to match model definition of the
    # FBNetV1 builder.
    # "always_pw": True,
    # "bias": False,
    # temporarily disable zero_last_bn_gamma
    "zero_last_bn_gamma": False,
}

FD_ARCH = {
    "cham_e_fd_v2": {
        "trunk": [
            # [c, s, n, t]
            [("conv_k3", 8, 2, 1), ("ir_k3", 8, 1, 1, _ex(1))],
            [("ir_k3", 24, 2, 1, _ex(4))],
            [("ir_k3", 32, 2, 3, _ex(4))],
            [("ir_k3", 64, 2, 1, _ex(6)), ("ir_k3", 64, 1, 3, _ex(6))],
        ],
        "bbox": [("ir_k3", 128, 1, 1, _ex(4))],
        "rpn": [("ir_k3", 96, 1, 1, _ex(6))],
        "basic_args": _BASIC_ARGS,
    },
}

MODEL_ARCH.register_dict(FD_ARCH)
