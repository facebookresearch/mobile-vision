#!/usr/bin/env python3

import logging
import os

import torch
import torch.nn as nn
from mobile_cv.common.fb import utils_io

path_manager = utils_io.get_path_manager()
logger = logging.getLogger(__name__)


def load_model(model_path, model_type):
    logger.info("Loading {} model from {} ...".format(model_path, model_type))

    if model_type.startswith("torchscript"):
        extra_files = {}
        # NOTE: may support loading extra_file specified by model_info
        # extra_files["predictor_info.json"] = ""

        with path_manager.open(os.path.join(model_path, "model.jit"), "rb") as f:
            ts = torch.jit.load(f, _extra_files=extra_files)

        return TorchscriptWrapper(ts)

    else:
        raise RuntimeError("Unsupported model type {}".format(model_type))


class TorchscriptWrapper(nn.Module):
    """"""

    def __init__(self, module, int8_backend=None):
        super().__init__()
        self.module = module
        self.int8_backend = int8_backend

    def forward(self, *args, **kwargs):
        # TODO: set int8 backend accordingly if needed
        return self.module(*args, **kwargs)
