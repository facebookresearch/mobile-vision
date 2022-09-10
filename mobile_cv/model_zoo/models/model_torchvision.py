#!/usr/bin/env python3

"""
Models from torchvision, providing the same interface for loading in model zoo
"""


import torchvision
from mobile_cv.model_zoo.models.hub_utils import pretrained_download

# to register for model_zoo
from . import model_zoo_factory  # noqa


if hasattr(torchvision.models, "list_models"):
    # TorchVision >= v0.14
    tv_models = torchvision.models.list_models(torchvision.models)
else:
    tv_models = [
        v.__name__
        for k, v in torchvision.models.__dict__.items()
        if callable(v) and k[0].islower() and k[0] != "_" and k != "get_weight"
    ]

for model_name in tv_models:
    if model_name not in model_zoo_factory.MODEL_ZOO_FACTORY:
        model_zoo_factory.MODEL_ZOO_FACTORY.register(
            model_name,
            pretrained_download(torchvision.models.__dict__[model_name]),
        )
