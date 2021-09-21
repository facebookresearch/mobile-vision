#!/usr/bin/env python3

"""
FBNet model building blocks factory
"""

import mobile_cv.arch.fbnet.fbnet_blocks_factory as bf
import mobile_cv.arch.fbnet.utils_blocks as ub


def mix_irf(c_in, c_out, expansion, stride, **kwargs):
    # names = ["skip", "ir_k3", "ir_k5", "ir_k7"]
    names = ["ir_k3", "ir_k5", "ir_k7"]
    return ub.create_mix_blocks(
        names, bf.PRIMITIVES, c_in, c_out, expansion, stride, **kwargs
    )


bf.add_primitives(
    {
        "mix_irf": lambda C_in, C_out, expansion, stride, **kwargs: mix_irf(
            C_in, C_out, expansion, stride, **kwargs
        )
    }
)
