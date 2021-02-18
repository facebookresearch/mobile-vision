#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Model zoo factory to create a data loader based on builder name

To register a dataset builder, add the builder to __init__.py and
  from mobile_cv.model_zoo.datasets import dataset_factory
  @dataset_factory.DATASET_FACTORY.register("classy")
  def classy_dataloader(...):
      ...

To create a data loader from a dataset, use
  from mobile_cv.model_zoo.datasets import dataset_factory
  dataset_factory.get_dataset(builder, ...)
"""

import mobile_cv.common.misc.registry as registry


DATASET_FACTORY = registry.Registry("dataset_factory")


def get(builder, *args, **kwargs):
    """Return a data loader from the given builder name
    The returned object could be an iterable, a pytorch data loader etc.
    """
    data_loader_builder = DATASET_FACTORY.get(builder)
    return data_loader_builder(*args, **kwargs)
