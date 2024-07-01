#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from mobile_cv.common.misc.oss_utils import fb_overwritable


# to register datasets
@fb_overwritable()
def register_datasets():
    from mobile_cv.model_zoo.datasets import dataset_simple  # noqa


register_datasets()
