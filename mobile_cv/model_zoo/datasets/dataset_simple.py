#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Simple dummy datasets
"""

import torch

# to register for model_zoo
from mobile_cv.model_zoo.datasets import dataset_factory  # noqa


@dataset_factory.DATASET_FACTORY.register("tensor_shape")
def dataset_from_shape(input_shapes, **kwargs):
    assert isinstance(input_shapes, (tuple, list))
    assert all(isinstance(x, (list, tuple)) for x in input_shapes)

    ret = [torch.zeros(x) for x in input_shapes]

    # return an iterable
    return [ret]
