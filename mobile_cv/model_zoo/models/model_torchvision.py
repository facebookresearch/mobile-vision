#!/usr/bin/env python3

"""
Models from torchvision, providing the same interface for loading in model zoo
"""


import torchvision
from mobile_cv.model_zoo.models.hub_utils import pretrained_download

# to register for model_zoo
from . import model_zoo_factory  # noqa


model_zoo_factory.MODEL_ZOO_FACTORY.register(
    # pyre-fixme[16]: Module `models` has no attribute `resnet18`.
    "resnet18",
    pretrained_download(torchvision.models.resnet18),
)
model_zoo_factory.MODEL_ZOO_FACTORY.register(
    # pyre-fixme[16]: Module `models` has no attribute `resnet34`.
    "resnet34",
    pretrained_download(torchvision.models.resnet34),
)
model_zoo_factory.MODEL_ZOO_FACTORY.register(
    # pyre-fixme[16]: Module `models` has no attribute `resnet50`.
    "resnet50",
    pretrained_download(torchvision.models.resnet50),
)
model_zoo_factory.MODEL_ZOO_FACTORY.register(
    # pyre-fixme[16]: Module `models` has no attribute `resnet101`.
    "resnet101",
    pretrained_download(torchvision.models.resnet101),
)
model_zoo_factory.MODEL_ZOO_FACTORY.register(
    # pyre-fixme[16]: Module `models` has no attribute `resnet152`.
    "resnet152",
    pretrained_download(torchvision.models.resnet152),
)
model_zoo_factory.MODEL_ZOO_FACTORY.register(
    # pyre-fixme[16]: Module `models` has no attribute `resnext50_32x4d`.
    "resnext50_32x4d",
    pretrained_download(torchvision.models.resnext50_32x4d),
)
model_zoo_factory.MODEL_ZOO_FACTORY.register(
    # pyre-fixme[16]: Module `models` has no attribute `resnext101_32x8d`.
    "resnext101_32x8d",
    pretrained_download(torchvision.models.resnext101_32x8d),
)
