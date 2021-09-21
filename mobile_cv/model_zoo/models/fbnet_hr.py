#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
FBNet_HR models

Example code to create the model:
    from mobile_cv.model_zoo.models.fbnet_hr import fbnet_hr
    model = fbnet_hr("TestModel")
    model.eval()

All suported architectures could be found in:
    mobile_cv/arch/fbnet_v2/fbnet_hr_modeldef.py

Architectures with pretrained weights could be found in:
    mobile_cv/model_zoo/models/model_info/fbnet_hr/*.json
"""

from typing import Dict, Tuple

import mobile_cv.arch.fbnet_v2.fbnet_hr_modeldef as modeldef
import torch.nn as nn
from mobile_cv.arch.fbnet_v2.fbnet_builder import FBNetBuilder
from mobile_cv.arch.fbnet_v2.fbnet_hr import FBNetHRBuilder


def _create_builder(arch_name: str) -> Tuple[FBNetHRBuilder, Dict]:
    """Creates a FBNetHR builder and generates arch def given arch_name"""
    assert arch_name in modeldef.MODEL_ARCH, (
        f"Invalid arch name {arch_name}, "
        f"available names: {modeldef.MODEL_ARCH.keys()}"
    )
    arch_def = modeldef.MODEL_ARCH[arch_name]

    scale_factor = 1.0
    width_divisor = 1
    bn_info = {"name": "bn"}
    builder = FBNetBuilder(
        width_ratio=scale_factor, bn_args=bn_info, width_divisor=width_divisor
    )
    builder.last_depth = 3
    builder_hr = FBNetHRBuilder(builder)
    return builder_hr, arch_def


def fbnet_hr(arch_name: str) -> nn.Module:
    """
    Constructs an hourglass architecture named `arch_name`

    Args:
        arch_name (str): Architecture name
    """
    builder, arch_def = _create_builder(arch_name)
    return builder.build_model(arch_def)
