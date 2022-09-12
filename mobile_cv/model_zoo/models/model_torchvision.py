#!/usr/bin/env python3

"""
Models from torchvision, providing the same interface for loading in model zoo
"""


import torchvision
from mobile_cv.model_zoo.models.hub_utils import pretrained_download
from pkg_resources import parse_version

# to register for model_zoo
from . import model_zoo_factory  # noqa


_tv_version = parse_version(torchvision.__version__)


if _tv_version >= parse_version("0.14.0a0"):
    _tv_models = torchvision.models.list_models(torchvision.models)
    _get_model_builder = torchvision.models.get_model_builder
else:
    _tv_models = [
        v.__name__
        for k, v in torchvision.models.__dict__.items()
        if callable(v) and k[0].islower() and k[0] != "_" and k != "get_weight"
    ]
    _get_model_builder = lambda model_name: torchvision.models.__dict__[model_name]


def _get_builder(model_name):
    def wrapped(**kwargs):
        if _tv_version >= parse_version("0.13"):
            if kwargs.pop("pretrained", False):
                kwargs["weights"] = "IMAGENET1K_V1"
        model_fn = _get_model_builder(model_name)
        return model_fn(**kwargs)

    return wrapped


for model_name in _tv_models:
    if model_name not in model_zoo_factory.MODEL_ZOO_FACTORY:
        model_zoo_factory.MODEL_ZOO_FACTORY.register(
            model_name,
            pretrained_download(_get_builder(model_name)),
        )
