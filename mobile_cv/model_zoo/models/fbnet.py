#!/usr/bin/env python3

import typing

import torch
import torch.nn as nn
from mobile_cv.arch.fbnet import (
    fbnet_builder as mbuilder,
    fbnet_modeldef_cls as modeldef,
)
from mobile_cv.common import utils_io
from mobile_cv.model_zoo.models import model_zoo_factory, utils


NAME_MAPPING = {
    # external name : internal name
    "EfficientNet_B3": "eff_3",
    "FBNetV2_L1": "FBNetV2_L1",
    "FBNetV2_L2": "FBNetV2_L2",
    "FBNetV2_L3": "FBNetV2_L3",
}


def _load_pretrained_info():
    folder_name = utils.get_model_info_folder("fbnet")
    ret = utils.load_model_info_all(folder_name)
    return ret


PRETRAINED_MODELS = _load_pretrained_info()


def _load_fbnet_state_dict(file_name):
    path_manager = utils_io.get_path_manager()
    with path_manager.open(file_name, "rb") as h_in:
        state_dict = torch.load(h_in, map_location="cpu")

    if "model_ema" in state_dict and state_dict["model_ema"] is not None:
        state_dict = state_dict["model_ema"]
    else:
        state_dict = state_dict["state_dict"]
    ret = {}
    for name, val in state_dict.items():
        if name.startswith("module."):
            name = name[len("module."):]
        ret[name] = val
    return ret


def _create_builder(arch_name_or_def: typing.Union[str, dict]):
    if isinstance(arch_name_or_def, str):
        assert arch_name_or_def in modeldef.MODEL_ARCH, (
            f"Invalid arch name {arch_name_or_def}, "
            f"available names: {modeldef.MODEL_ARCH.keys()}"
        )
        arch_def = modeldef.MODEL_ARCH[arch_name_or_def]
    else:
        assert isinstance(arch_name_or_def, dict)
        arch_def = arch_name_or_def

    arch_def = mbuilder.unify_arch_def(arch_def)

    scale_factor = 1.0
    width_divisor = arch_def.get("width_divisor", 8)
    bn_info = {"bn_type": "bn", "momentum": 0.003}
    drop_out = 0.2
    dw_skip_bnrelu = arch_def.get("dw_skip_bnrelu", False)

    builder = mbuilder.FBNetBuilder(
        width_ratio=scale_factor,
        # pyre-fixme[6]: Expected `str` for 2nd param but got `Dict[str,
        #  typing.Union[float, str]]`.
        bn_type=bn_info,
        width_divisor=width_divisor,
        dw_skip_bn=dw_skip_bnrelu,
        dw_skip_relu=dw_skip_bnrelu,
        dropout_ratio=drop_out,
    )

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
    def __init__(self, arch_name, dim_in=3):
        super().__init__()

        builder, arch_def = _create_builder(arch_name)

        self.first = builder.add_first(arch_def["first"], dim_in=dim_in)
        self.stages = builder.add_blocks(arch_def["stages"])
        self.last = builder.add_last(arch_def["last"])
        self.dropout = builder.add_dropout()
        self.output_channels = builder.last_depth
        self.arch_def = arch_def

    def forward(self, x):
        y = self.first(x)
        y = self.stages(y)
        y = self.last(y)
        if self.dropout is not None:
            y = self.dropout(y)
        return y


class FBNet(nn.Module):
    def __init__(self, arch_name, dim_in=3, num_classes=1000):
        super().__init__()
        self.backbone = FBNetBackbone(arch_name, dim_in)
        self.head = ClsConvHead(self.backbone.output_channels, num_classes)

    def forward(self, x):
        y = self.backbone(x)
        y = self.head(y)
        return y


@model_zoo_factory.MODEL_ZOO_FACTORY.register("fbnet_v1")
def fbnet(arch_name, pretrained=False, progress=True, **kwargs):
    """
    Constructs a FBNet architecture named `arch_name`

    Args:
        arch_name (str): Architecture name, supports [
            "fbnet_a", "fbnet_b", "fbnet_c",
            "fbnet_ase", "fbnet_bse", "fbnet_cse",
        ]
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if isinstance(arch_name, str) and arch_name in NAME_MAPPING:
        arch_name = NAME_MAPPING[arch_name]

    model = FBNet(arch_name, **kwargs)
    if pretrained:
        assert (
            arch_name in PRETRAINED_MODELS
        ), f"Invalid arch {arch_name}, supported arch {PRETRAINED_MODELS.keys()}"
        model_info = PRETRAINED_MODELS[arch_name]
        model_path = model_info["model_path"]
        state_dict = _load_fbnet_state_dict(model_path)
        model.load_state_dict(state_dict)
        model.model_info = model_info
    return model
