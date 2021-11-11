#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
FBNet classification models

Example code to create the model:
    from mobile_cv.model_zoo.models.fbnet_v2 import fbnet
    model = fbnet("fbnet_c", pretrained=True)
    model.eval()

Full example code is available at `examples/run_fbnet_v2.py`.

All suported architectures could be found in:
    mobile_cv/arch/fbnet_v2/fbnet_modeldef_cls*.py

Architectures with pretrained weights could be found in:
    mobile_cv/model_zoo/models/model_info/fbnet_v2/*.json
"""

import json
import typing

import torch
import torch.nn as nn
from mobile_cv.arch.fbnet_v2 import (
    fbnet_builder as mbuilder,
    fbnet_modeldef_cls as modeldef,
)
from mobile_cv.common import utils_io
from mobile_cv.model_zoo.models import model_zoo_factory, utils


def _load_pretrained_info():
    folder_name = utils.get_model_info_folder("fbnet_v2")
    ret = utils.load_model_info_all(folder_name)
    return ret


PRETRAINED_MODELS = _load_pretrained_info()

NAME_MAPPING = {
    # external name : internal name
    "FBNet_a": "fbnet_a",
    "FBNet_b": "fbnet_b",
    "FBNet_c": "fbnet_c",
    "MobileNetV3": "mnv3",
    "FBNetV2_F5": "FBNetV2_L2",
}


def _load_fbnet_state_dict(file_name, progress=True, ignore_prefix="module."):
    path_manager = utils_io.get_path_manager()
    with path_manager.open(file_name, "rb") as h_in:
        state_dict = torch.load(h_in, map_location="cpu")

    if "model_ema" in state_dict and state_dict["model_ema"] is not None:
        state_dict = state_dict["model_ema"]
    else:
        state_dict = state_dict["state_dict"]
    ret = {}
    for name, val in state_dict.items():
        if name.startswith(ignore_prefix):
            name = name[len(ignore_prefix) :]
        ret[name] = val
    return ret


def _create_builder(arch_name_or_def: typing.Union[str, dict], unify_block_names=None):
    if isinstance(arch_name_or_def, str) and arch_name_or_def in modeldef.MODEL_ARCH:
        arch_def = modeldef.MODEL_ARCH[arch_name_or_def]
    elif isinstance(arch_name_or_def, str):
        try:
            arch_def = json.loads(arch_name_or_def)
            assert isinstance(arch_def, dict), f"Invalid arch type {arch_name_or_def}"
        except ValueError:
            assert arch_name_or_def in modeldef.MODEL_ARCH, (
                f"Invalid arch name {arch_name_or_def}, "
                f"available names: {modeldef.MODEL_ARCH.keys()}"
            )
    else:
        assert isinstance(arch_name_or_def, dict)
        arch_def = arch_name_or_def

    if unify_block_names is None:
        unify_block_names = ["blocks"]
    assert isinstance(unify_block_names, (list, tuple))
    arch_def = mbuilder.unify_arch_def(arch_def, unify_block_names)

    builder = mbuilder.FBNetBuilder()
    builder.add_basic_args(**arch_def.get("basic_args", {}))

    return builder, arch_def


class ClsConvHead(nn.Module):
    """Global average pooling + conv head for classification"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        # global avg pool of arbitrary feature map size
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(input_dim, output_dim, 1)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return x


class FBNetBackbone(nn.Module):
    def __init__(self, arch_name, dim_in=3, stage_indices=None):
        super().__init__()

        builder, arch_def = _create_builder(arch_name)

        self.stages = builder.build_blocks(
            arch_def["blocks"], dim_in=dim_in, stage_indices=stage_indices
        )
        self.out_channels = builder.last_depth
        self.arch_def = arch_def

    def forward(self, x):
        y = self.stages(x)
        return y


class FBNet(nn.Module):
    def __init__(self, arch_name, dim_in=3, num_classes=1000, stage_indices=None):
        super().__init__()
        self.backbone = FBNetBackbone(
            arch_name, dim_in=dim_in, stage_indices=stage_indices
        )
        self.head = ClsConvHead(self.backbone.out_channels, num_classes)

    def forward(self, x):
        y = self.backbone(x)
        y = self.head(y)
        return y

    @property
    def arch_def(self):
        return self.backbone.arch_def


def _load_pretrained_weight(
    arch_name, model, progress, ignore_prefix="module.", strict=True
):
    assert (
        arch_name in PRETRAINED_MODELS
    ), f"Invalid arch {arch_name}, supported arch {PRETRAINED_MODELS.keys()}"
    model_info = PRETRAINED_MODELS[arch_name]
    model_path = model_info["model_path"]
    state_dict = _load_fbnet_state_dict(
        model_path, progress=progress, ignore_prefix=ignore_prefix
    )
    model.load_state_dict(state_dict, strict=strict)
    model.model_info = model_info


@model_zoo_factory.MODEL_ZOO_FACTORY.register("fbnet_v2")
def fbnet(arch_name, pretrained=False, progress=True, **kwargs):
    """
    Constructs a FBNet architecture named `arch_name` with classification head

    Args:
        arch_name (str): Architecture name
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if isinstance(arch_name, str) and arch_name in NAME_MAPPING:
        arch_name = NAME_MAPPING[arch_name]

    model = FBNet(arch_name, **kwargs)
    if pretrained:
        _load_pretrained_weight(arch_name, model, progress)
    return model


@model_zoo_factory.MODEL_ZOO_FACTORY.register("fbnet_v2_backbone")
def fbnet_backbone(
    arch_name, pretrained=False, progress=True, stage_indices=None, **kwargs
):
    """
    Constructs a FBNet backbone architecture named `arch_name`

    Args:
        arch_name (str): Architecture name
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        stage_indices (list): Indices of stages to use, None to use all stages
    """
    if isinstance(arch_name, str) and arch_name in NAME_MAPPING:
        arch_name = NAME_MAPPING[arch_name]

    model = FBNetBackbone(arch_name, stage_indices=stage_indices, **kwargs)
    if pretrained:
        _load_pretrained_weight(
            arch_name,
            model,
            progress,
            ignore_prefix="module.backbone.",
            strict=False,
        )
    return model
