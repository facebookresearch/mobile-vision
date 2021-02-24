#!/usr/bin/env python3

import logging
import os

import torch
import torch.nn as nn
from fvcore.common.file_io import PathManager


logger = logging.getLogger(__name__)


def load_model(model_info, model_root):
    model_path = os.path.join(model_root, model_info.path)
    model_path = os.path.abspath(model_path)
    logger.info("Loading {} model from {} ...".format(model_path, model_info.type))

    if model_info.type.startswith("torchscript"):
        extra_files = {}
        with PathManager.open(os.path.join(model_path, "model.jit"), "rb") as f:
            ts = torch.jit.load(f, _extra_files=extra_files)

        return TorchscriptWrapper(ts)

    else:
        raise RuntimeError("Unsupported model type {}".format(model_info.type))


class TorchscriptWrapper(nn.Module):
    """"""

    def __init__(self, module, int8_backend=None):
        super().__init__()
        self.module = module
        self.int8_backend = int8_backend

    def forward(self, *args, **kwargs):
        # TODO: set int8 backend accordingly if needed
        return self.module(*args, **kwargs)
