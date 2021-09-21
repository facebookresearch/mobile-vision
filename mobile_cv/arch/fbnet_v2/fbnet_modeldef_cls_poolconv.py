#!/usr/bin/env python3

# pyre-fixme[21]: Could not find name `fbnet_blocks_factory` in
#  `mobile_cv.arch.fbnet_v2`.
from . import fbnet_blocks_factory, fbnet_modeldef_cls, pw_conv


PRIMITIVES = {
    "pcirf_nearest": lambda C_in, C_out, expansion, stride, **kwargs: pw_conv.PoolConvIRF(
        C_in, C_out, expansion, stride, up_method="nearest", **kwargs
    ),
    # "pcirf_linear": lambda C_in, C_out, expansion, stride, **kwargs: pw_conv.PoolConvIRF(
    #     C_in, C_out, expansion, stride, up_method="linear", **kwargs
    # ),
    "pcirf_bilinear": lambda C_in, C_out, expansion, stride, **kwargs: pw_conv.PoolConvIRF(
        C_in, C_out, expansion, stride, up_method="bilinear", **kwargs
    ),
    "pcirf_deconv": lambda C_in, C_out, expansion, stride, **kwargs: pw_conv.PoolConvIRF(
        C_in, C_out, expansion, stride, up_method="conv_transpose", **kwargs
    ),
    # "pcirf_shuffle": lambda C_in, C_out, expansion, stride, **kwargs: pw_conv.PoolConvIRF(
    #     C_in, C_out, expansion, stride, up_method="shuffle", **kwargs
    # ),
}
# pyre-fixme[16]: Module `fbnet_v2` has no attribute `fbnet_blocks_factory`.
fbnet_blocks_factory.add_primitives(PRIMITIVES)


FBNET_CSE_BLOCKS = {
    "first": [16, 2],
    "stages": [
        # [t, c, n, s]
        # stage 0
        [[1, 16, 1, 1]],
        # stage 1
        [[6, 24, 1, 2], [1, 24, 3, 1]],
        # stage 2
        [[6, 32, 1, 2], [3, 32, 1, 1], [6, 32, 1, 1], [6, 32, 1, 1]],
        # stage 3
        [
            [6, 64, 1, 2],
            [3, 64, 1, 1],
            [6, 64, 1, 1],
            [6, 64, 1, 1],
            [6, 112, 1, 1],
            [6, 112, 1, 1],
            [6, 112, 1, 1],
            [3, 112, 1, 1],
        ],
        # stage 4
        [[6, 184, 1, 2], [6, 184, 1, 1], [6, 184, 1, 1], [6, 184, 1, 1]],
    ],
}


def _fbcse_block_op(name):
    ret = {
        "first": "conv_hs",
        "stages": [
            # stage 0
            [name],
            # stage 1
            [name, "skip", name, name],
            # stage 2
            ["ir_k5_sehsig", "ir_k5_sehsig", "ir_k5_sehsig", "ir_k3_sehsig"],
            # stage 3
            [
                "ir_k5_hs",
                "ir_k5_hs",
                "ir_k5_hs",
                "ir_k5_hs",
                "ir_k5_hs",
                "ir_k5_se_hs",
                "ir_k5_se_hs",
                "ir_k5_se_hs",
            ],
            # stage 4
            ["ir_k5_se_hs", "ir_k5_se_hs", "ir_k5_se_hs", "ir_k5_se_hs"],
        ],
        "last": "ir_pool_hs",
    }
    return ret


MODEL_ARCH_FBNET = {
    "fbnet_cse_pcirf_nearest": {
        "block_op_type": _fbcse_block_op("pcirf_nearest"),
        "block_cfg": FBNET_CSE_BLOCKS,
        "last": [6, 1984, 1, 1],
    },
    # "fbnet_cse_pcirf_shuffle": {
    #     "block_op_type": _fbcse_block_op("pcirf_shuffle"),
    #     "block_cfg": FBNET_CSE_BLOCKS,
    #     "last": [6, 1984, 1, 1],
    # },
    # "fbnet_cse_pcirf_linear": {
    #     "block_op_type": _fbcse_block_op("pcirf_linear"),
    #     "block_cfg": FBNET_CSE_BLOCKS,
    #     "last": [6, 1984, 1, 1],
    # },
    "fbnet_cse_pcirf_bilinear": {
        "block_op_type": _fbcse_block_op("pcirf_bilinear"),
        "block_cfg": FBNET_CSE_BLOCKS,
        "last": [6, 1984, 1, 1],
    },
    "fbnet_cse_pcirf_deconv": {
        "block_op_type": _fbcse_block_op("pcirf_deconv"),
        "block_cfg": FBNET_CSE_BLOCKS,
        "last": [6, 1984, 1, 1],
    },
}
# pyre-fixme[16]: Module `fbnet_modeldef_cls` has no attribute `add_archs`.
fbnet_modeldef_cls.add_archs(MODEL_ARCH_FBNET)
