#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from mobile_cv.common.misc.py import import_matching_modules

# to register datasets
import_matching_modules("mobile_cv.model_zoo.datasets", "dataset_simple")
# @fb-only: import_matching_modules("mobile_cv.model_zoo.datasets.fb", "*") 
