#!/usr/bin/env python3

import os
from functools import wraps
from typing import Any, Tuple, Type, Union

import mobile_cv.common.misc.iter_utils as iu
import numpy as np
import torch
import torch.distributed as dist


def skip_if_no_gpu(func):
    """Decorator that can be used to skip GPU tests on non-GPU machines."""
    func.skip_if_no_gpu = True

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            return
        if torch.cuda.device_count() <= 0:
            return

        return func(*args, **kwargs)

    return wrapper


def enable_ddp_env(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        dist.init_process_group("gloo", rank=0, world_size=1)
        ret = func(*args, **kwargs)
        dist.destroy_process_group()
        return ret

    return wrapper


def run_and_compare(model_before, model_after, inputs, device="cpu"):
    model_before.to(device)
    model_after.to(device)
    output_before = model_before(inputs)
    output_after = model_after(inputs)

    for item in iu.recursive_iterate(
        iu.create_pair(output_before, output_after), wait_on_send=False
    ):
        obefore = item.lhs
        oafter = item.rhs
        np.testing.assert_allclose(
            obefore.to("cpu").detach(),
            oafter.to("cpu").detach(),
            rtol=0,
            atol=1e-4,
        )


def find_modules(
    model: torch.nn.Module,
    module_type: Union[Type[Any], Tuple[Type[Any], ...]],
    exact_match=False,
):
    if not isinstance(module_type, tuple):
        module_type = (module_type,)
    for x in model.modules():
        if not exact_match:
            if isinstance(x, module_type):
                return True
        else:
            if type(x) in module_type:
                return True
    return False
