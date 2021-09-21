#!/usr/bin/env python3

import mobile_cv.common.misc.registry as registry


MODEL_ARCH = registry.Registry("hr_arch_factory")

BASIC_ARGS = {"bias": False, "relu_args": None}

MODEL_ARCH_DEFAULT = {
    # arch_def explanation in `build_model` https://fburl.com/diff/0f7g763r
    "TestModel": {
        "basic_args": BASIC_ARGS,
        "stages": [
            # [op, c, s, n, ...]
            # original res
            [("conv_k3", 16, 1, 1)],
            [("skip", 16, 1, 1)],
            [("skip", 16, 1, 1)],
            # downsampled x2
            [("skip", 16, 1, 1)],
            [("skip", 16, 1, 1)],
            [("conv_k3", 16, 1, 1)],
        ],
    }
}

MODEL_ARCH.register_dict(MODEL_ARCH_DEFAULT)
