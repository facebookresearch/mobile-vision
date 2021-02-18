#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Task factory to create a task object based builder name and arch name

To register a task builder, add the builder to __init__.py and
  from mobile_cv.model_zoo.tasks import task_factory
  @task_factory.TASK_FACTORY.register("classy")
  def classy_task(...):
      ...

To create a task, use
  from mobile_cv.model_zoo.tasks import task_factory
  task_factory.get(builder, ...)
"""

import mobile_cv.common.misc.registry as registry


TASK_FACTORY = registry.Registry("task_factory")


def get(builder, *args, **kwargs):
    task_builder = TASK_FACTORY.get(builder)
    return task_builder(*args, **kwargs)
