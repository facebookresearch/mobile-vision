#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
FBNet model building blocks factory for 3d convs
"""

import mobile_cv.arch.utils.helper as hp

from . import basic_blocks as bb, blocks_factory


_PRIMITIVES = {
    "conv3d": lambda in_channels, out_channels, stride, **kwargs: bb.ConvBNRelu(
        in_channels,
        out_channels,
        **hp.merge(
            conv_args={"name": "conv3d", "stride": stride},
            bn_args={"name": "bn3d"},
            kwargs=kwargs,
        )
    )
}
blocks_factory.PRIMITIVES.register_dict(_PRIMITIVES)
