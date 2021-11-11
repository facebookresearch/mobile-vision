#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Model zoo factory to create a model based on model builder name and arch name

To register a model builder, add the builder to __init__.py and
  from mobile_cv.model_zoo.models import model_zoo_factory
  @model_zoo_factory.MODEL_ZOO_FACTORY.register("fbnet_v2")
  def fbnet(...):
      ...

To create a model, use
  from mobile_cv.model_zoo.models import model_zoo_factory
  model_zoo_factory.get_model(builder, pretrained, ...)
"""

import mobile_cv.common.misc.registry as registry


MODEL_ZOO_FACTORY = registry.Registry("model_zoo_factory")


def get_model(builder, pretrained=False, progress=True, **kwargs):
    model_builder = MODEL_ZOO_FACTORY.get(builder)
    return model_builder(pretrained=pretrained, progress=progress, **kwargs)
