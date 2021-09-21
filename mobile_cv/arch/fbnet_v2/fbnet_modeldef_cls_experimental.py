#!/usr/bin/env python3

import copy

from . import fbnet_modeldef_cls as fmc


def add_archs(archs):
    fmc.add_archs(archs)


RESIZE_METHODS = (
    "amp",
    # TODO: FractionalPool caused a stride error
    # 'fmp',
    "rnc",
    "rbc",
    "rnmp",
    "rbmp",
    "crn",
    "crb",
    "crnmp",
    "crbmp",
    # TODO: failure due to stride issue
    # 'p3d'
)


TEMPLATE_ARCH = {
    "mnv3": {
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                # stage 0
                ["ir_k3"],
                # stage 1
                ["ir_k3"] * 2,
                # stage 2
                ["ir_k5_sehsig"] * 3,
                # stage 3
                ["ir_k3_hs"] * 4 + ["ir_k3_se_hs"] * 2,
                # stage 4
                ["ir_k5_se_hs"] * 3,
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [[1, 16, 1, 1]],
                # stage 1
                [[4, 24, 1, 2], [3, 24, 1, 1]],
                # stage 2
                [[3, 40, 3, 2]],
                # stage 3
                [[6, 80, 1, 2], [2.5, 80, 1, 1], [2.3, 80, 2, 1], [6, 112, 2, 1]],
                # stage 4
                [[6, 160, 3, 2]],
            ],
            # [t, c, n, s]
            "last": [6, 1280, 1, 1],
        },
    },
}


SANITY_ARCH = {
    "mnv3_sanity_octconv_0_k3": {
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                # stage 0
                ["oct_ir_k3_0"],
                # stage 1
                ["oct_ir_k3_0"] * 2,
                # stage 2
                ["ir_k5_sehsig"] * 3,
                # stage 3
                ["ir_k3_hs"] * 4 + ["ir_k3_se_hs"] * 2,
                # stage 4
                ["ir_k5_se_hs"] * 3,
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [[1, 16, 1, 1]],
                # stage 1
                [[4, 24, 1, 2], [3, 24, 1, 1]],
                # stage 2
                [[3, 40, 3, 2]],
                # stage 3
                [[6, 80, 1, 2], [2.5, 80, 1, 1], [2.3, 80, 2, 1], [6, 112, 2, 1]],
                # stage 4
                [[6, 160, 3, 2]],
            ],
            # [t, c, n, s]
            "last": [6, 1280, 1, 1],
        },
    },
    "mnv3_sanity_octconv_0_k5": {
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                # stage 0
                ["ir_k3"],
                # stage 1
                ["ir_k3"] * 2,
                # stage 2
                ["oct_ir_k5_sehsig_0"] * 3,
                # stage 3
                ["ir_k3_hs"] * 4 + ["ir_k3_se_hs"] * 2,
                # stage 4
                ["ir_k5_se_hs"] * 3,
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [[1, 16, 1, 1]],
                # stage 1
                [[4, 24, 1, 2], [3, 24, 1, 1]],
                # stage 2
                [[3, 40, 3, 2]],
                # stage 3
                [[6, 80, 1, 2], [2.5, 80, 1, 1], [2.3, 80, 2, 1], [6, 112, 2, 1]],
                # stage 4
                [[6, 160, 3, 2]],
            ],
            # [t, c, n, s]
            "last": [6, 1280, 1, 1],
        },
    },
    "mnv3_sanity_octirf_0_k3": {
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                # stage 0
                ["oct_irf_k3_0"],
                # stage 1
                ["oct_irf_k3_0"] * 2,
                # stage 2
                ["ir_k5_sehsig"] * 3,
                # stage 3
                ["ir_k3_hs"] * 4 + ["ir_k3_se_hs"] * 2,
                # stage 4
                ["ir_k5_se_hs"] * 3,
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [[1, 16, 1, 1]],
                # stage 1
                [[4, 24, 1, 2], [3, 24, 1, 1]],
                # stage 2
                [[3, 40, 3, 2]],
                # stage 3
                [[6, 80, 1, 2], [2.5, 80, 1, 1], [2.3, 80, 2, 1], [6, 112, 2, 1]],
                # stage 4
                [[6, 160, 3, 2]],
            ],
            # [t, c, n, s]
            "last": [6, 1280, 1, 1],
        },
    },
    "mnv3_sanity_octirpw_0_k3": {
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                # stage 0
                ["oct_irpw_k3_0"],
                # stage 1
                ["oct_irpw_k3_0"] * 2,
                # stage 2
                ["ir_k5_sehsig"] * 3,
                # stage 3
                ["ir_k3_hs"] * 4 + ["ir_k3_se_hs"] * 2,
                # stage 4
                ["ir_k5_se_hs"] * 3,
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [[1, 16, 1, 1]],
                # stage 1
                [[4, 24, 1, 2], [3, 24, 1, 1]],
                # stage 2
                [[3, 40, 3, 2]],
                # stage 3
                [[6, 80, 1, 2], [2.5, 80, 1, 1], [2.3, 80, 2, 1], [6, 112, 2, 1]],
                # stage 4
                [[6, 160, 3, 2]],
            ],
            # [t, c, n, s]
            "last": [6, 1280, 1, 1],
        },
    },
    # TODO: ir_k3_cm1x is not defined
    # "mnv3_sanity_cm_k3": {
    #     "block_op_type": {
    #         "first": "conv_hs",
    #         "stages": [
    #             # stage 0
    #             ["ir_k3_cm1x"],
    #             # stage 1
    #             ["ir_k3_cm1x"] * 2,
    #             # stage 2
    #             ["ir_k5_sehsig"] * 3,
    #             # stage 3
    #             ["ir_k3_hs"] * 4 + ["ir_k3_se_hs"] * 2,
    #             # stage 4
    #             ["ir_k5_se_hs"] * 3,
    #         ],
    #         "last": "ir_pool_hs",
    #     },
    #     "block_cfg": {
    #         "first": [16, 2],
    #         "stages": [
    #             # [t, c, n, s]
    #             # stage 0
    #             [[1, 16, 1, 1]],
    #             # stage 1
    #             [[4, 24, 1, 2], [3, 24, 1, 1]],
    #             # stage 2
    #             [[3, 40, 3, 2]],
    #             # stage 3
    #             [[6, 80, 1, 2], [2.5, 80, 1, 1], [2.3, 80, 2, 1], [6, 112, 2, 1]],
    #             # stage 4
    #             [[6, 160, 3, 2]],
    #         ],
    #         # [t, c, n, s]
    #         "last": [6, 1280, 1, 1],
    #     },
    # },
}
add_archs(SANITY_ARCH)


EXPERIMENTAL_ARCH = {
    "mnv3_sk_k3": {
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                # stage 0
                ["ir_k3"],  # replacing all k3 in stage 0 + 1 -> 1.1k flops
                # stage 1
                ["ir_k3"] * 2,
                # stage 2
                ["ir_k5_sehsig"] * 3,
                # stage 3
                ["ir_k3_hs"] * 3 + ["ir_k3_sk_hs"] + ["ir_k3_se_hs"] * 2,
                # stage 4
                ["ir_k5_se_hs"] * 3,
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [[1, 16, 1, 1]],
                # stage 1
                [[4, 24, 1, 2], [3, 24, 1, 1]],
                # stage 2
                [[3, 40, 3, 2]],
                # stage 3
                [[6, 80, 1, 2], [2.5, 80, 1, 1], [2.3, 80, 2, 1], [6, 112, 2, 1]],
                # stage 4
                [[6, 160, 3, 2]],
            ],
            # [t, c, n, s]
            "last": [6, 1280, 1, 1],
        },
    },
    "mnv3_full_sk_k3": {
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                # stage 0
                ["ir_k3_sk"],
                # stage 1
                ["ir_k3_sk"] * 2,
                # stage 2
                ["ir_k5_sehsig"] * 3,
                # stage 3
                ["ir_k3_sk_hs"] * 4 + ["ir_k3_se_sk_hs"] * 2,
                # stage 4
                ["ir_k5_se_hs"] * 3,
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [[1, 16, 1, 1]],
                # stage 1
                [[4, 24, 1, 2], [3, 24, 1, 1]],
                # stage 2
                [[3, 40, 3, 2]],
                # stage 3
                [[6, 80, 1, 2], [2.5, 80, 1, 1], [2.3, 80, 2, 1], [6, 112, 2, 1]],
                # stage 4
                [[6, 160, 3, 2]],
            ],
            # [t, c, n, s]
            "last": [6, 1280, 1, 1],
        },
    },
    "mnv3_sk_k3_2stages": {
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                # stage 0
                ["ir_k3"],
                # stage 1
                ["ir_k3"] * 2,
                # stage 2
                ["ir_k5_sehsig"] * 3,
                # stage 3
                ["ir_k3_sk_hs"] * 4 + ["ir_k3_se_sk_hs"] * 2,
                # stage 4
                ["ir_k5_se_hs"] * 3,
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [[1, 16, 1, 1]],
                # stage 1
                [[4, 24, 1, 2], [3, 24, 1, 1]],
                # stage 2
                [[3, 40, 3, 2]],
                # stage 3
                [[6, 80, 1, 2], [2.5, 80, 1, 1], [2.3, 80, 2, 1], [6, 112, 2, 1]],
                # stage 4
                [[6, 160, 3, 2]],
            ],
            # [t, c, n, s]
            "last": [6, 1280, 1, 1],
        },
    },
    "mnv3_sk_k3_octconv_0.375_k3": {
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                # stage 0
                ["oct_ir_k3_0.375"],
                # stage 1
                ["oct_ir_k3_0.375"] * 2,
                # stage 2
                ["ir_k5_sehsig"] * 3,
                # stage 3
                ["ir_k3_sk_hs"] * 4 + ["ir_k3_se_sk_hs"] * 2,
                # stage 4
                ["ir_k5_se_hs"] * 3,
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [[1, 16, 1, 1]],
                # stage 1
                [[4, 24, 1, 2], [3, 24, 1, 1]],
                # stage 2
                [[3, 40, 3, 2]],
                # stage 3
                [[6, 80, 1, 2], [2.5, 80, 1, 1], [2.3, 80, 2, 1], [6, 112, 2, 1]],
                # stage 4
                [[6, 160, 3, 2]],
            ],
            # [t, c, n, s]
            "last": [6, 1280, 1, 1],
        },
    },
    "mnv3_octirpw_0.375_k3": {
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                # stage 0
                ["oct_irpw_k3_0.375"],
                # stage 1
                ["oct_irpw_k3_0.375"] * 2,
                # stage 2
                ["ir_k5_sehsig"] * 3,
                # stage 3
                ["ir_k3_hs"] * 4 + ["ir_k3_se_hs"] * 2,
                # stage 4
                ["ir_k5_se_hs"] * 3,
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [[1, 16, 1, 1]],
                # stage 1
                [[4, 24, 1, 2], [3, 24, 1, 1]],
                # stage 2
                [[3, 40, 3, 2]],
                # stage 3
                [[6, 80, 1, 2], [2.5, 80, 1, 1], [2.3, 80, 2, 1], [6, 112, 2, 1]],
                # stage 4
                [[6, 160, 3, 2]],
            ],
            # [t, c, n, s]
            "last": [6, 1280, 1, 1],
        },
    },
    "mnv3_amp_k3_stage3": {
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                # stage 0
                ["ir_k3"],
                # stage 1
                ["ir_k3_amp"] * 2,
                # stage 2
                ["ir_k5_sehsig"] * 3,
                # stage 3
                ["ir_k3_hs_amp"] * 3 + ["ir_k3_hs"] * 1 + ["ir_k3_se_hs"] * 2,
                # stage 4
                ["ir_k5_se_hs"] * 3,
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [[1, 16, 1, 1]],
                # stage 1
                [[4, 24, 1, 1.25], [3, 24, 1, 1.5]],
                # stage 2
                [[3, 40, 3, 2]],
                # stage 3
                [
                    [6, 80, 1, 1.5],
                    [2.5, 80, 1, 1.5],
                    [2.3, 80, 1, 1.1],
                    [2.3, 80, 1, 1],
                    [6, 112, 2, 1],
                ],
                # stage 4
                [[6, 160, 3, 2]],
            ],
            # [t, c, n, s]
            "last": [6, 1280, 1, 1],
        },
    },
    "mnv3_amp_k3_stage1": {
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                # stage 0
                ["ir_k3"],
                # stage 1
                ["ir_k3_amp"] * 2,
                # stage 2
                ["ir_k5_sehsig"] * 3,
                # stage 3
                ["ir_k3_hs"] * 4 + ["ir_k3_se_hs"] * 2,
                # stage 4
                ["ir_k5_se_hs"] * 3,
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [[1, 16, 1, 1]],
                # stage 1
                [[4, 24, 1, 1.75], [3, 24, 1, 1.25]],
                # stage 2
                [[3, 40, 3, 2]],
                # stage 3
                [[6, 80, 1, 2], [2.5, 80, 1, 1], [2.3, 80, 2, 1], [6, 112, 2, 1]],
                # stage 4
                [[6, 160, 3, 2]],
            ],
            # [t, c, n, s]
            "last": [6, 1280, 1, 1],
        },
    },
    "mnv3_rbc_k3_stage1": {
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                # stage 0
                ["ir_k3"],
                # stage 1
                ["ir_k3_rbc"] * 2,
                # stage 2
                ["ir_k5_sehsig"] * 3,
                # stage 3
                ["ir_k3_hs"] * 4 + ["ir_k3_se_hs"] * 2,
                # stage 4
                ["ir_k5_se_hs"] * 3,
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [[1, 16, 1, 1]],
                # stage 1
                [[4, 24, 1, 1.5], [3, 24, 1, 1.45]],
                # stage 2
                [[3, 40, 3, 2]],
                # stage 3
                [[6, 80, 1, 2], [2.5, 80, 1, 1], [2.3, 80, 2, 1], [6, 112, 2, 1]],
                # stage 4
                [[6, 160, 3, 2]],
            ],
            # [t, c, n, s]
            "last": [6, 1280, 1, 1],
        },
    },
    "mnv3_rbc_k3_mid": {
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                # stage 0
                ["ir_k3"],
                # stage 1
                ["ir_k3", "ir_k3_rbc"],
                # stage 2
                ["ir_k5_sehsig_rbc"] * 3,
                # stage 3
                ["ir_k3_hs_rbc"] * 4 + ["ir_k3_se_hs"] * 2,
                # stage 4
                ["ir_k5_se_hs"] * 3,
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [[1, 16, 1, 1]],
                # stage 1
                [[4, 24, 1, 2], [3, 24, 1, 1.45]],
                # stage 2
                [[3, 40, 3, 1.45]],
                # stage 3
                [[6, 80, 1, 2], [2.5, 80, 1, 1], [2.3, 80, 2, 1], [6, 112, 2, 1]],
                # stage 4
                [[6, 160, 3, 2]],
            ],
            # [t, c, n, s]
            "last": [6, 1280, 1, 1],
        },
    },
    "mnv3_rbc_k3_stage3": {
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                # stage 0
                ["ir_k3"],
                # stage 1
                ["ir_k3_rbc"] * 2,
                # stage 2
                ["ir_k5_sehsig"] * 3,
                # stage 3
                ["ir_k3_hs_rbc"] * 3 + ["ir_k3_hs"] * 1 + ["ir_k3_se_hs"] * 2,
                # stage 4
                ["ir_k5_se_hs"] * 3,
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [[1, 16, 1, 1]],
                # stage 1
                [[4, 24, 1, 1.25], [3, 24, 1, 1.5]],
                # stage 2
                [[3, 40, 3, 2]],
                # stage 3
                [
                    [6, 80, 1, 1.5],
                    [2.5, 80, 1, 1.5],
                    [2.3, 80, 1, 1.1],
                    [2.3, 80, 1, 1],
                    [6, 112, 2, 1],
                ],
                # stage 4
                [[6, 160, 3, 2]],
            ],
            # [t, c, n, s]
            "last": [6, 1280, 1, 1],
        },
    },
    "mnv3_rnc_k3_stage3": {
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                # stage 0
                ["ir_k3"],
                # stage 1
                ["ir_k3_rnc"] * 2,
                # stage 2
                ["ir_k5_sehsig"] * 3,
                # stage 3
                ["ir_k3_hs_rnc"] * 3 + ["ir_k3_hs"] * 1 + ["ir_k3_se_hs"] * 2,
                # stage 4
                ["ir_k5_se_hs"] * 3,
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [[1, 16, 1, 1]],
                # stage 1
                [[4, 24, 1, 1.25], [3, 24, 1, 1.5]],
                # stage 2
                [[3, 40, 3, 2]],
                # stage 3
                [
                    [6, 80, 1, 1.5],
                    [2.5, 80, 1, 1.5],
                    [2.3, 80, 1, 1.1],
                    [2.3, 80, 1, 1],
                    [6, 112, 2, 1],
                ],
                # stage 4
                [[6, 160, 3, 2]],
            ],
            # [t, c, n, s]
            "last": [6, 1280, 1, 1],
        },
    },
    "mnv3_rbmp_k3_stage3": {
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                # stage 0
                ["ir_k3"],
                # stage 1
                ["ir_k3_rbmp"] * 2,
                # stage 2
                ["ir_k5_sehsig"] * 3,
                # stage 3
                ["ir_k3_hs_rbmp"] * 3 + ["ir_k3_hs"] * 1 + ["ir_k3_se_hs"] * 2,
                # stage 4
                ["ir_k5_se_hs"] * 3,
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [[1, 16, 1, 1]],
                # stage 1
                [[4, 24, 1, 1.25], [3, 24, 1, 1.5]],
                # stage 2
                [[3, 40, 3, 2]],
                # stage 3
                [
                    [6, 80, 1, 1.5],
                    [2.5, 80, 1, 1.5],
                    [2.3, 80, 1, 1.1],
                    [2.3, 80, 1, 1],
                    [6, 112, 2, 1],
                ],
                # stage 4
                [[6, 160, 3, 2]],
            ],
            # [t, c, n, s]
            "last": [6, 1280, 1, 1],
        },
    },
    "mnv3_rnmp_k3_stage3": {
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                # stage 0
                ["ir_k3"],
                # stage 1
                ["ir_k3_rnmp"] * 2,
                # stage 2
                ["ir_k5_sehsig"] * 3,
                # stage 3
                ["ir_k3_hs_rnmp"] * 3 + ["ir_k3_hs"] * 1 + ["ir_k3_se_hs"] * 2,
                # stage 4
                ["ir_k5_se_hs"] * 3,
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [[1, 16, 1, 1]],
                # stage 1
                [[4, 24, 1, 1.25], [3, 24, 1, 1.5]],
                # stage 2
                [[3, 40, 3, 2]],
                # stage 3
                [
                    [6, 80, 1, 1.5],
                    [2.5, 80, 1, 1.5],
                    [2.3, 80, 1, 1.1],
                    [2.3, 80, 1, 1],
                    [6, 112, 2, 1],
                ],
                # stage 4
                [[6, 160, 3, 2]],
            ],
            # [t, c, n, s]
            "last": [6, 1280, 1, 1],
        },
    },
    "mnv3_rnc_k3_stage3_mags": {  # more aggressive subsampling
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                # stage 0
                ["ir_k3"],
                # stage 1
                ["ir_k3_rnc"] * 2,
                # stage 2
                ["ir_k5_sehsig"] * 3,
                # stage 3
                ["ir_k3_hs_rnc"] * 4 + ["ir_k3_se_hs"] * 2,
                # stage 4
                ["ir_k5_se_hs"] * 3,
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [[1, 16, 1, 1]],
                # stage 1
                [[4, 24, 1, 2.25], [3, 24, 1, 1.6]],
                # stage 2
                [[3, 40, 3, 2]],
                # stage 3
                [
                    [6, 80, 1, 1],
                    [2.5, 80, 1, 1],
                    [2.3, 80, 1, 1],
                    [2.3, 80, 1, 1.05],
                    [6, 112, 2, 1],
                ],
                # stage 4
                [[6, 160, 3, 2]],
            ],
            # [t, c, n, s]
            "last": [6, 1280, 1, 1],
        },
    },
    "mnv3_rbc_k3_stage3_mags": {  # more aggressive subsampling
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                # stage 0
                ["ir_k3"],
                # stage 1
                ["ir_k3_rbc"] * 2,
                # stage 2
                ["ir_k5_sehsig"] * 3,
                # stage 3
                ["ir_k3_hs_rbc"] * 4 + ["ir_k3_se_hs"] * 2,
                # stage 4
                ["ir_k5_se_hs"] * 3,
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [[1, 16, 1, 1]],
                # stage 1
                [[4, 24, 1, 2.25], [3, 24, 1, 1.6]],
                # stage 2
                [[3, 40, 3, 2]],
                # stage 3
                [
                    [6, 80, 1, 1],
                    [2.5, 80, 1, 1],
                    [2.3, 80, 1, 1],
                    [2.3, 80, 1, 1.05],
                    [6, 112, 2, 1],
                ],
                # stage 4
                [[6, 160, 3, 2]],
            ],
            # [t, c, n, s]
            "last": [6, 1280, 1, 1],
        },
    },
    "mnv3_rnmp_k3_stage3_mags": {  # more aggressive subsampling
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                # stage 0
                ["ir_k3"],
                # stage 1
                ["ir_k3_rnmp"] * 2,
                # stage 2
                ["ir_k5_sehsig"] * 3,
                # stage 3
                ["ir_k3_hs_rnmp"] * 4 + ["ir_k3_se_hs"] * 2,
                # stage 4
                ["ir_k5_se_hs"] * 3,
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [[1, 16, 1, 1]],
                # stage 1
                [[4, 24, 1, 2.25], [3, 24, 1, 1.6]],
                # stage 2
                [[3, 40, 3, 2]],
                # stage 3
                [
                    [6, 80, 1, 1],
                    [2.5, 80, 1, 1],
                    [2.3, 80, 1, 1],
                    [2.3, 80, 1, 1.05],
                    [6, 112, 2, 1],
                ],
                # stage 4
                [[6, 160, 3, 2]],
            ],
            # [t, c, n, s]
            "last": [6, 1280, 1, 1],
        },
    },
    "mnv3_rbmp_k3_stage3_mags": {  # more aggressive subsampling
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                # stage 0
                ["ir_k3"],
                # stage 1
                ["ir_k3_rbmp"] * 2,
                # stage 2
                ["ir_k5_sehsig"] * 3,
                # stage 3
                ["ir_k3_hs_rbmp"] * 4 + ["ir_k3_se_hs"] * 2,
                # stage 4
                ["ir_k5_se_hs"] * 3,
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [[1, 16, 1, 1]],
                # stage 1
                [[4, 24, 1, 2.25], [3, 24, 1, 1.6]],
                # stage 2
                [[3, 40, 3, 2]],
                # stage 3
                [
                    [6, 80, 1, 1],
                    [2.5, 80, 1, 1],
                    [2.3, 80, 1, 1],
                    [2.3, 80, 1, 1.05],
                    [6, 112, 2, 1],
                ],
                # stage 4
                [[6, 160, 3, 2]],
            ],
            # [t, c, n, s]
            "last": [6, 1280, 1, 1],
        },
    },
    "mnv3_amp_k3_stage3_mags": {  # more aggressive subsampling
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                # stage 0
                ["ir_k3"],
                # stage 1
                ["ir_k3_amp"] * 2,
                # stage 2
                ["ir_k5_sehsig"] * 3,
                # stage 3
                ["ir_k3_hs_amp"] * 4 + ["ir_k3_se_hs"] * 2,
                # stage 4
                ["ir_k5_se_hs"] * 3,
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [[1, 16, 1, 1]],
                # stage 1
                [[4, 24, 1, 2.25], [3, 24, 1, 1.6]],
                # stage 2
                [[3, 40, 3, 2]],
                # stage 3
                [
                    [6, 80, 1, 1],
                    [2.5, 80, 1, 1],
                    [2.3, 80, 1, 1],
                    [2.3, 80, 1, 1.05],
                    [6, 112, 2, 1],
                ],
                # stage 4
                [[6, 160, 3, 2]],
            ],
            # [t, c, n, s]
            "last": [6, 1280, 1, 1],
        },
    },
}

add_archs(EXPERIMENTAL_ARCH)


# [t, c, n, s]
fbnet2_block_cfg = {
    "first": [16, 2],
    "stages": [
        [[1, 16, 1, 1]],
        [[6, 24, 4, 2]],
        [[6, 32, 4, 2]],
        [[6, 64, 4, 2], [6, 112, 4, 1]],
        [[6, 184, 4, 2], [6, 352, 1, 1]],
    ],
    "last": [6, 1280, 1, 1],
}


FBNET2_EXPERIMENTAL_CHANNEL = {
    "fbnet2_layer4_channel4_e89": {  # 15 Mflops
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                ["ir_k3"],
                ["ir_k3"] * 2 + ["skip"] * 2,
                ["ir_k3"] * 4,
                ["ir_k3"] * 5 + ["skip"] + ["ir_k3"] * 2,
                ["ir_k3"] * 5,
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                [[1, 4, 1, 1]],
                [[3, 12, 1, 2], [1, 12, 1, 1], [6, 24, 1, 1], [6, 24, 1, 1]],
                [[1, 24, 1, 2], [3, 8, 1, 1], [3, 16, 1, 1], [3, 8, 1, 1]],
                [
                    [1, 48, 1, 2],
                    [1, 32, 1, 1],
                    [1, 32, 1, 1],
                    [3, 16, 1, 1],
                    [3, 56, 1, 1],
                    [6, 112, 1, 1],
                    [1, 112, 1, 1],
                    [3, 28, 1, 1],
                ],
                [
                    [6, 46, 1, 2],
                    [6, 46, 1, 1],
                    [6, 46, 1, 1],
                    [6, 46, 1, 1],
                    [6, 88, 1, 1],
                ],
            ],
            # [t, c, n, s]
            "last": [6, 1280, 1, 1],
        },
        "source": "/mnt/vol/gfsai-east/aml/mobile-vision/gpu/alvinwan/20190805/search-fbnet-v41-224-1.0-e0.6.al_mmo13/epoch_89_0/arch_config.json",  # noqa
    },
    "fbnet2_layer4_channel1_e89": {  # 59 Mflops
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                ["ir_k3"],
                ["ir_k3", "ir_k3", "ir_k3", "ir_k3"],
                ["ir_k5", "ir_k3", "ir_k3", "ir_k3"],
                [
                    "ir_k5",
                    "ir_k5",
                    "ir_k5",
                    "ir_k3",
                    "ir_k5",
                    "ir_k3",
                    "ir_k5",
                    "ir_k3",
                ],
                ["ir_k5", "ir_k5", "ir_k5", "ir_k5", "ir_k5"],
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": fbnet2_block_cfg,
        "source": "/mnt/vol/gfsai-east/aml/mobile-vision/gpu/alvinwan/20190805/search-fbnet-v41-224-1.0-e0.6.02r1zf2_/epoch_89_0/arch_config.json",  # noqa
    },
    "fbnet2_layer4_channel1_e89_manual": {  # 59 Mflops
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                ["ir_k3"],
                ["ir_k3", "ir_k3", "ir_k3", "ir_k3"],
                ["ir_k5_sehsig", "ir_k3_sehsig", "ir_k3_sehsig", "ir_k3_sehsig"],
                [
                    "ir_k5_hs",
                    "ir_k5_hs",
                    "ir_k5_hs",
                    "ir_k3_hs",
                    "ir_k5_hs",
                    "ir_k3_se_hs",
                    "ir_k5_se_hs",
                    "ir_k3_se_hs",
                ],
                ["ir_k5_se_hs", "ir_k5_se_hs", "ir_k5_se_hs", "ir_k5_se_hs", "ir_k5"],
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": fbnet2_block_cfg,
        "source": "/mnt/vol/gfsai-east/aml/mobile-vision/gpu/alvinwan/20190805/search-fbnet-v41-224-1.0-e0.6.02r1zf2_/epoch_89_0/arch_config.json",  # noqa
    },
    "mnv3_layer4_channel1_e14": {  # 441 Mflops
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                ["ir_k3_se"],
                ["ir_k3_se", "ir_k3", "ir_k3", "ir_k3_se"],
                ["ir_k5_se", "ir_k5_se", "ir_k5_se", "ir_k5_se"],
                [
                    "ir_k5_se",
                    "ir_k5_se",
                    "ir_k5_se",
                    "ir_k5_se",
                    "ir_k5_se",
                    "ir_k5_se",
                    "ir_k5_se",
                    "ir_k5_se",
                ],
                ["ir_k3", "ir_k5_se", "ir_k5_se", "ir_k5_se", "ir_k3"],
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": fbnet2_block_cfg,
        "source": "/mnt/vol/gfsai-east/aml/mobile-vision/gpu/alvinwan/20190806/dnas_mnv3_channel1.g62u9haw/epoch_29_0/arch_config.json",  # noqa
    },
    "mnv3_channel16_e89": {
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                ["ir_k3"],
                ["ir_k3", "ir_k3", "ir_k3", "ir_k3"],
                ["ir_k3", "ir_k3", "ir_k3", "ir_k3"],
                ["ir_k3", "ir_k3", "ir_k3", "ir_k3", "ir_k3", "skip", "ir_k3", "ir_k3"],
                ["ir_k3", "ir_k3", "ir_k3", "ir_k3", "ir_k3"],
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                [[1, 5, 1, 1]],
                [[1, 7, 1, 2], [1, 12, 1, 1], [1, 12, 1, 1], [1, 1, 1, 1]],
                [[1, 10, 1, 2], [1, 16, 1, 1], [1, 10, 1, 1], [1, 10, 1, 1]],
                [
                    [1, 32, 1, 2],
                    [1, 20, 1, 1],
                    [1, 20, 1, 1],
                    [1, 4, 1, 1],
                    [1, 35, 1, 1],
                    [6, 112, 1, 1],
                    [1, 7, 1, 1],
                    [1, 35, 1, 1],
                ],
                [
                    [1, 92, 1, 2],
                    [1, 92, 1, 1],
                    [1, 92, 1, 1],
                    [1, 92, 1, 1],
                    [1, 176, 1, 1],
                ],
            ],
            "last": [6, 1280, 1, 1],
        },
    },
    "fbnet2_layer3_dsp_step8_e89": {
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                ["ir_k5_e6"],
                ["ir_k5_e6", "ir_k5_e6", "ir_k5_e6", "ir_k5_e1"],
                ["ir_k5_e6", "ir_k5_e6", "ir_k5_e6", "ir_k5_e6"],
                [
                    "ir_k5_e6",
                    "ir_k5_e6",
                    "ir_k5_e6",
                    "ir_k5_e6",
                    "ir_k5_e6",
                    "ir_k5_e6",
                    "ir_k5_e6",
                    "ir_k3_e1",
                ],
                ["ir_k5_e6", "ir_k5_e6", "ir_k5_e1", "ir_k5_e1", "ir_k5_e6"],
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                [[1, 1, 1, 1]],
                [[6, 9, 1, 2], [6, 1, 1, 1], [6, 1, 1, 1], [6, 9, 1, 1]],
                [[6, 25, 1, 2], [6, 17, 1, 1], [6, 17, 1, 1], [6, 17, 1, 1]],
                [
                    [6, 41, 1, 2],
                    [6, 33, 1, 1],
                    [6, 49, 1, 1],
                    [6, 49, 1, 1],
                    [6, 81, 1, 1],
                    [6, 57, 1, 1],
                    [6, 65, 1, 1],
                    [6, 33, 1, 1],
                ],
                [
                    [6, 129, 1, 2],
                    [6, 97, 1, 1],
                    [6, 97, 1, 1],
                    [6, 9, 1, 1],
                    [6, 185, 1, 1],
                ],
            ],
            "last": [6, 1280, 1, 1],
        },
    },
    "fbnet2_layer3_dsp_step8_e74": {
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                ["ir_k5_e6"],
                ["ir_k5_e6", "ir_k5_e6", "ir_k5_e6", "ir_k5_e6"],
                ["ir_k5_e6", "ir_k5_e6", "ir_k5_e6", "ir_k3_e6"],
                [
                    "ir_k5_e6",
                    "ir_k5_e6",
                    "ir_k5_e6",
                    "ir_k5_e1",
                    "ir_k5_e6",
                    "ir_k5_e6",
                    "ir_k5_e6",
                    "ir_k5_e1",
                ],
                ["ir_k5_e6", "ir_k5_e6", "ir_k3_e6", "ir_k5_e6", "ir_k5_e6"],
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                [[1, 1, 1, 1]],
                [[6, 9, 1, 2], [6, 1, 1, 1], [6, 9, 1, 1], [6, 1, 1, 1]],
                [[6, 25, 1, 2], [6, 17, 1, 1], [6, 17, 1, 1], [6, 17, 1, 1]],
                [
                    [6, 41, 1, 2],
                    [6, 33, 1, 1],
                    [6, 33, 1, 1],
                    [6, 33, 1, 1],
                    [6, 81, 1, 1],
                    [6, 57, 1, 1],
                    [6, 73, 1, 1],
                    [6, 65, 1, 1],
                ],
                [
                    [6, 129, 1, 2],
                    [6, 97, 1, 1],
                    [6, 113, 1, 1],
                    [6, 113, 1, 1],
                    [6, 185, 1, 1],
                ],
            ],
            "last": [6, 1280, 1, 1],
        },
    },
    "mnv3_step8_e89": {
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                ["ir_k3_e6_se"],
                ["ir_k5_e6_se", "ir_k3_e6_se", "ir_k5_e3_se", "ir_k5_e6_se"],
                ["ir_k5_e6_se", "ir_k5_e6_se", "ir_k5_e6_se", "ir_k3"],
                [
                    "ir_k5_e6_se",
                    "ir_k5_e6_se",
                    "ir_k3_e3_se",
                    "ir_k3_e6_se",
                    "ir_k5_e6_se",
                    "ir_k5_e6_se",
                    "ir_k5_e3_se",
                    "ir_k5_e1_se",
                ],
                [
                    "ir_k5_e6_se",
                    "ir_k5_e6_se",
                    "ir_k5_e3_se",
                    "ir_k5_e3_se",
                    "ir_k5_e6_se",
                ],
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                [[1, 1, 1, 1]],
                [[6, 17, 1, 2], [6, 1, 1, 1], [6, 17, 1, 1], [6, 1, 1, 1]],
                [[6, 17, 1, 2], [6, 17, 1, 1], [6, 17, 1, 1], [6, 9, 1, 1]],
                [
                    [6, 41, 1, 2],
                    [6, 17, 1, 1],
                    [6, 33, 1, 1],
                    [6, 33, 1, 1],
                    [6, 81, 1, 1],
                    [6, 57, 1, 1],
                    [6, 73, 1, 1],
                    [6, 73, 1, 1],
                ],
                [
                    [6, 129, 1, 2],
                    [6, 81, 1, 1],
                    [6, 137, 1, 1],
                    [6, 89, 1, 1],
                    [6, 217, 1, 1],
                ],
            ],
            "last": [6, 1280, 1, 1],
        },
    },
    "fbnet2_layer3_dsp_step1_e59": {
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                ["ir_k5_e3"],
                ["ir_k5_e6", "ir_k5_e6", "ir_k5_e3", "ir_k3_e6"],
                ["ir_k5_e6", "ir_k5_e6", "ir_k5_e1", "ir_k5_e3"],
                [
                    "ir_k5_e6",
                    "ir_k3_e6",
                    "ir_k5_e3",
                    "ir_k5_e1",
                    "ir_k5_e6",
                    "ir_k5_e6",
                    "ir_k3_e6",
                    "ir_k5_e1",
                ],
                ["ir_k5_e6", "ir_k5_e6", "ir_k5_e3", "ir_k3_e1", "ir_k5_e6"],
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                [[1, 7, 1, 1]],
                [[6, 15, 1, 2], [6, 13, 1, 1], [6, 12, 1, 1], [6, 15, 1, 1]],
                [[6, 23, 1, 2], [6, 13, 1, 1], [6, 23, 1, 1], [6, 17, 1, 1]],
                [
                    [6, 32, 1, 2],
                    [6, 25, 1, 1],
                    [6, 32, 1, 1],
                    [6, 17, 1, 1],
                    [6, 66, 1, 1],
                    [6, 66, 1, 1],
                    [6, 48, 1, 1],
                    [6, 91, 1, 1],
                ],
                [
                    [6, 94, 1, 2],
                    [6, 19, 1, 1],
                    [6, 53, 1, 1],
                    [6, 38, 1, 1],
                    [6, 34, 1, 1],
                ],
            ],
            "last": [6, 1280, 1, 1],
        },
    },
    "fbnet2_layer3_dsp_step1_e74": {
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                ["ir_k3_e6"],
                ["ir_k5_e6", "ir_k5_e6", "ir_k5_e1", "ir_k5_e6"],
                ["ir_k5_e6", "ir_k5_e6", "ir_k5_e3", "ir_k3_e1"],
                [
                    "ir_k5_e6",
                    "ir_k5_e6",
                    "ir_k5_e3",
                    "ir_k3_e3",
                    "ir_k5_e6",
                    "ir_k5_e3",
                    "ir_k3_e6",
                    "ir_k5_e3",
                ],
                ["ir_k5_e6", "ir_k3_e3", "ir_k3_e1", "ir_k5_e6", "ir_k5_e6"],
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                [[1, 8, 1, 1]],
                [[6, 15, 1, 2], [6, 14, 1, 1], [6, 18, 1, 1], [6, 9, 1, 1]],
                [[6, 23, 1, 2], [6, 17, 1, 1], [6, 17, 1, 1], [6, 16, 1, 1]],
                [
                    [6, 32, 1, 2],
                    [6, 40, 1, 1],
                    [6, 32, 1, 1],
                    [6, 40, 1, 1],
                    [6, 53, 1, 1],
                    [6, 31, 1, 1],
                    [6, 79, 1, 1],
                    [6, 66, 1, 1],
                ],
                [
                    [6, 83, 1, 2],
                    [6, 105, 1, 1],
                    [6, 140, 1, 1],
                    [6, 16, 1, 1],
                    [6, 192, 1, 1],
                ],
            ],
            "last": [6, 1280, 1, 1],
        },
    },
    "mnv3_ch8_exp1_e89_manual": {
        "block_op_type": {
            "first": "conv_hs",
            "stages": [
                ["ir_k5_hs"],
                ["ir_k3_se_hs", "ir_k5_se_hs", "ir_k5_hs", "ir_k5_hs"],
                ["ir_k3_hs", "ir_k5_hs", "ir_k5_hs", "ir_k5_hs"],
                [
                    "ir_k5_hs",
                    "ir_k5_se_hs",
                    "ir_k5_hs",
                    "ir_k5_se_hs",
                    "ir_k3_hs",
                    "ir_k3_se_hs",
                    "ir_k5_se_hs",
                    "ir_k3_hs",
                ],
                ["ir_k3_hs", "ir_k3_hs", "ir_k3_se_hs", "ir_k3_hs", "ir_k5_se_hs"],
            ],
            "last": "ir_pool_hs",
        },
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                [[1, 4, 1, 1]],
                [[4, 4, 1, 2], [1, 9, 1, 1], [1, 4, 1, 1], [3, 9, 1, 1]],
                [[4, 17, 1, 2], [3, 17, 1, 1], [1, 9, 1, 1], [4, 9, 1, 1]],
                [
                    [4, 25, 1, 2],
                    [1, 17, 1, 1],
                    [4, 25, 1, 1],
                    [3, 9, 1, 1],
                    [1, 65, 1, 1],
                    [1, 73, 1, 1],
                    [3, 41, 1, 1],
                    [3, 49, 1, 1],
                ],
                [
                    [2, 105, 1, 2],
                    [4, 33, 1, 1],
                    [2, 1, 1, 1],
                    [3, 81, 1, 1],
                    [5, 121, 1, 1],
                ],
            ],
            "last": [6, 1280, 1, 1],
        },
    },
}

add_archs(FBNET2_EXPERIMENTAL_CHANNEL)


# Less aggressive subsampling

MNV3_1LAYER_LAGS_BLOCK_CFG = copy.deepcopy(TEMPLATE_ARCH["mnv3"]["block_cfg"])
# pyre-fixme[16]: `int` has no attribute `__getitem__`.
MNV3_1LAYER_LAGS_BLOCK_CFG["stages"][1][0][3] = 1

add_archs(
    {
        "mnv3_1layer_lags": {
            "block_op_type": TEMPLATE_ARCH["mnv3"]["block_op_type"],
            "block_cfg": MNV3_1LAYER_LAGS_BLOCK_CFG,
        }
    }
)

BLOCK_CFG_1STAGE_LAGS = copy.deepcopy(TEMPLATE_ARCH["mnv3"]["block_cfg"])
BLOCK_CFG_1STAGE_LAGS["stages"][1][0][3] = 1.5
BLOCK_CFG_1STAGE_LAGS["stages"][1][1][3] = 1.33

BLOCK_CFG_1LAYER_LAGS = copy.deepcopy(TEMPLATE_ARCH["mnv3"]["block_cfg"])
BLOCK_CFG_1LAYER_LAGS["stages"][1][0][3] = 1.5

BLOCK_CFG_NET_LAGS = copy.deepcopy(TEMPLATE_ARCH["mnv3"]["block_cfg"])
BLOCK_CFG_NET_LAGS["stages"][1][0][3] = 1.5
BLOCK_CFG_NET_LAGS["stages"][2][0][3] = 1.5
BLOCK_CFG_NET_LAGS["stages"][4][0][3] = 1.5

# More aggressive subsampling

BLOCK_CFG_1STAGE_MAGS = copy.deepcopy(TEMPLATE_ARCH["mnv3"]["block_cfg"])
BLOCK_CFG_1STAGE_MAGS["stages"][1][0][3] = 2.5
BLOCK_CFG_1STAGE_MAGS["stages"][1][1][3] = 0.8

BLOCK_CFG_1LAYER_MAGS = copy.deepcopy(TEMPLATE_ARCH["mnv3"]["block_cfg"])
BLOCK_CFG_1LAYER_MAGS["stages"][1][0][3] = 2.5

BLOCK_CFG_NET_MAGS = copy.deepcopy(TEMPLATE_ARCH["mnv3"]["block_cfg"])
BLOCK_CFG_NET_MAGS["stages"][1][0][3] = 2.5
BLOCK_CFG_NET_MAGS["stages"][2][0][3] = 2.5
BLOCK_CFG_NET_MAGS["stages"][4][0][3] = 2.5


for method in RESIZE_METHODS:
    arch = "mnv3_{{}}_{{}}_{}_k3".format(method)
    block_op_type = copy.deepcopy(TEMPLATE_ARCH["mnv3"]["block_op_type"])
    # pyre-fixme[16]: `str` has no attribute `__setitem__`.
    block_op_type["stages"][1] = ["ir_k3_{}".format(method)] * 2
    for name, ags, block_cfg in (
        ("1stage", "lags", BLOCK_CFG_1STAGE_LAGS),
        ("1layer", "lags", BLOCK_CFG_1LAYER_LAGS),
        ("1stage", "mags", BLOCK_CFG_1STAGE_MAGS),
        ("1layer", "mags", BLOCK_CFG_1LAYER_MAGS),
    ):
        add_archs(
            {
                arch.format(name, ags): {
                    "block_op_type": block_op_type,
                    "block_cfg": block_cfg,
                }
            }
        )

    block_op_type = copy.deepcopy(TEMPLATE_ARCH["mnv3"]["block_op_type"])
    block_op_type["stages"][1] = ["ir_k3_{}".format(method)] * 2
    block_op_type["stages"][2] = ["ir_k5_sehsig_{}".format(method)] * 3
    block_op_type["stages"][4] = ["ir_k5_se_hs_{}".format(method)] * 3

    add_archs(
        {
            arch.format("net", "lags"): {
                "block_op_type": block_op_type,
                "block_cfg": BLOCK_CFG_NET_LAGS,
            },
            arch.format("net", "mags"): {
                "block_op_type": block_op_type,
                "block_cfg": BLOCK_CFG_NET_MAGS,
            },
        }
    )

    block_op_type = copy.deepcopy(TEMPLATE_ARCH["mnv3"]["block_op_type"])
    block_op_type["stages"][1] = ["ir_k3_{}".format(method), "ir_k3"]
    block_op_type["stages"][2] = ["ir_k5_sehsig_{}".format(method)] * 3
    block_op_type["stages"][4] = ["ir_k5_se_hs_{}".format(method)] * 3

    add_archs(
        {
            "mnv3_sanity_{}".format(method): {
                "block_op_type": block_op_type,
                "block_cfg": TEMPLATE_ARCH["mnv3"]["block_cfg"],
            }
        }
    )
