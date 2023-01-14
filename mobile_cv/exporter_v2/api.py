# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.


from typing import List

import torch.nn as nn
from mobile_cv.exporter_v2.converter import ConverterBase


def export_models(
    converter: ConverterBase, output_dir: str, export_targets: List[str]
) -> List[str]:
    """
    # Vanilla implementation:
    paths = []
    for export_target in export_targets:
        convert_target, backend, additional_flags = _parse(export_target)
        model = _convert(converter, convert_target)
        path = _save(model, backend, additional_flags)
        paths.append(path)
    return paths
    """

    raise NotImplementedError()


def load_exported_model(path: str) -> nn.Module:
    raise NotImplementedError()
