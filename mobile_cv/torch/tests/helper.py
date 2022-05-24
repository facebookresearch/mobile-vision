import uuid
from contextlib import contextmanager
from functools import wraps
from typing import Optional

from torch import distributed as dist
from torch.multiprocessing import Pool


def run_func_dist(world_size: int, func, func_args):
    """Run the given function in `world_size` new processes with distrbuted env setup
    func(rank, func_args)
    """
    uid = uuid.uuid4().hex
    with Pool(processes=world_size) as pool:
        results = pool.starmap(
            _run_process,
            [(idx, world_size, uid, func, func_args) for idx in range(world_size)],
        )
    return results


@contextmanager
def enable_dist(world_size: int = 1, rank: int = 0, uid: Optional[str] = None):
    """Enable pytorch distributed enviroment"""
    if uid is None:
        uid = uuid.uuid4().hex
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"file:///tmp/mcv_test_ddp_init_{uid}",
    )
    yield
    dist.destroy_process_group()


def _run_process(rank: int, world_size: int, uid: str, func, func_args):
    with enable_dist(world_size=world_size, rank=rank, uid=uid):
        ret = func(rank, func_args)
    return ret


def enable_dist_env(func):
    """Decorator to enable ddp enviroment, only world size=1 supported"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with enable_dist():
            ret = func(*args, **kwargs)
        return ret

    return wrapper
