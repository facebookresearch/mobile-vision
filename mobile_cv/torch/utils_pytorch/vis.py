#!/usr/bin/env python3

import datetime
import os
import random
import string
from contextlib import contextmanager
from typing import Any, List

import torch


"""
Usage:
"//mobile-vision/mobile_cv/mobile_cv/torch:main"

from mobile_cv.torch.utils_pytorch.vis import vis_model
vis_model(name, model, data)
"""


def _rand_str(str_len):
    letters = string.ascii_letters
    return "".join(random.choice(letters) for i in range(str_len))


def get_tensorboard_base_name(name):
    BASE_DIR = "mobile_cv/vis"
    date_string = datetime.date.today().strftime("%Y%m%d")
    time_string = datetime.datetime.now().strftime("%H%M%S")
    rand_str = _rand_str(6)
    ret = os.path.join(
        BASE_DIR, os.environ["USER"], date_string, time_string, rand_str, name or "tb"
    )
    return ret


def get_tensorboard(name=None, log_dir=None):
    from fblearner.flow.util.visualization_utils import summary_writer

    base_name = get_tensorboard_base_name(name)
    tb_logger = summary_writer(workflow_name=base_name, log_dir=log_dir)
    return tb_logger


@contextmanager
def tensorboard(name=None, log_dir=None):
    logger = get_tensorboard(name, log_dir)
    try:
        yield logger
    finally:
        logger.close()
        print(get_on_demand_url(logger.log_dir))


def get_on_demand_url(log_dir):
    ret = f"https://our.internmc.facebook.com/intern/tensorboard/?dir={log_dir}"
    return ret


def vis_model(name: str, model: torch.nn.Module, data: List[Any], log_dir=None):
    writer = get_tensorboard(name, log_dir)
    writer.add_graph(model, data)
    writer.close()
    log_dir = writer.log_dir
    model_url = get_on_demand_url(log_dir)
    print(f"Model visualization url: {model_url}")
    return log_dir, model_url
