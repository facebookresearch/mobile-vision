#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from mobile_cv.common.misc.py import import_matching_modules


import_matching_modules("mobile_cv.model_zoo.tasks", "task_general")
# @fb-only: import_matching_modules("mobile_cv.model_zoo.tasks.fb", "*") 
