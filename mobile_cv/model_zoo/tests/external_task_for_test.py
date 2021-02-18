#!/usr/bin/env python3

import torch
from mobile_cv.model_zoo.tasks import task_factory
from mobile_cv.model_zoo.tasks.task_base import TaskBase


@task_factory.TASK_FACTORY.register(
    "test_task@mobile_cv.model_zoo.tests.external_task_for_test"
)
def ext_task(weight_file=None, **kwargs):
    return ExtTask()


class ExtTask(TaskBase):
    def get_model(self):
        return torch.nn.Identity()

    def get_dataloader(self):
        return [[torch.Tensor(1)]]
