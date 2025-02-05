#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from mobile_cv.common.misc.py import import_matching_modules

# to register for model_zoo
import_matching_modules("mobile_cv.model_zoo.models", "fbnet_v2")
import_matching_modules("mobile_cv.model_zoo.models", "model_jit")
import_matching_modules("mobile_cv.model_zoo.models", "model_torchvision")
# @fb-only[end= ]: import_matching_modules("mobile_cv.model_zoo.models.fb", "*")
