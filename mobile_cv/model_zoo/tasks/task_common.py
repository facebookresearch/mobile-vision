#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Task with some common task implementation
"""

from mobile_cv.common.misc.oss_utils import fb_overwritable
from mobile_cv.model_zoo.tasks.task_base import TaskBase


# task with common implementations
@fb_overwritable()
class TaskCommon(TaskBase):
    pass
