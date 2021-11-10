#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from collections import defaultdict
from typing import List, Dict, Any

import torch


def collect_model_attributes(model: torch.nn.Module, attr_names: List[str]):
    ret = defaultdict(dict)
    for name, module in model.named_modules():
        for attr in attr_names:
            if hasattr(module, attr):
                ret[name][attr] = getattr(module, attr)
    return ret


def apply_model_attributes(
    model: torch.nn.Module, attributes_map: Dict[str, Dict[str, Any]]
):
    for module_name, module_attrs in attributes_map.items():
        try:
            module = model.get_submodule(module_name)
        except AttributeError:
            continue

        for attr, val in module_attrs.items():
            setattr(module, attr, val)


def copy_model_attributes(
    model_source: torch.nn.Module, model_target: torch.nn.Module, attr_names: List[str]
):
    attrs = collect_model_attributes(model_source, attr_names)
    apply_model_attributes(model_target, attrs)
    return attrs
