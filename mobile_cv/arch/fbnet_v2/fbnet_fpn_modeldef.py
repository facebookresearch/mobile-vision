#!/usr/bin/env python3

import mobile_cv.common.misc.registry as registry


MODEL_ARCH = registry.Registry("fpn_arch_factory")

MODEL_ARCH_DEFAULT = {
    "TestModel": {
        "stages": [
            [("conv_k1", 256, 1, 1, {"relu_args": None})],
            [("noop", 256, 1, 1)],
            [("skip", 256, 1, 1)],
            [("conv_k3", 256, 1, 1, {"relu_args": None})],
            [("upsample", 256, 2, 1)],
        ]
        * 4,
        "stage_combiners": ["add"] * 4,
        "combiner_path": "low_res",
    }
}

MODEL_ARCH.register_dict(MODEL_ARCH_DEFAULT)
