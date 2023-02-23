#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Helper for distributed training in pytorch
adapted from d2/d2go
"""

import contextlib
import functools
import logging
import os
import pickle
import tempfile
import time
import types
from datetime import timedelta
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

import mobile_cv.torch.utils_pytorch.comm as comm
import torch
import torch.distributed as dist
import torch.distributed.launcher as pet
import torch.multiprocessing as mp
from mobile_cv.common.misc.py import PicklableWrapper

logger = logging.getLogger(__name__)
_RT = TypeVar("_RT")  # return type

DEFAULT_TIMEOUT = timedelta(minutes=30)
DEFAULT_UNITTEST_TIMEOUT = timedelta(minutes=1)


def get_mp_context():
    """spawn is used for launching processes in `launch`"""
    return mp.get_context("spawn")


class DistributedParams(object):
    """store information about ranks and sizes"""

    LOCAL_RANK_KEY: str = "LOCAL_RANK"
    RANK_KEY: str = "RANK"
    GROUP_RANK_KEY: str = "GROUP_RANK"
    LOCAL_WORLD_SIZE_KEY: str = "LOCAL_WORLD_SIZE"
    WORLD_SIZE_KEY: str = "WORLD_SIZE"

    def __init__(
        self,
        local_rank: int,
        global_rank: int,
        machine_rank: int,
        num_processes_per_machine: int,
        world_size: int,
    ):
        self.local_rank: int = local_rank
        self.global_rank: int = global_rank
        self.machine_rank: int = machine_rank
        self.num_processes_per_machine: int = num_processes_per_machine
        self.world_size: int = world_size
        self.validate()

    def validate(self):
        # assume same number of processes per machine
        if (
            self.global_rank
            != self.machine_rank * self.num_processes_per_machine + self.local_rank
        ):
            raise ValueError(f"{self} is not valid!")

    @classmethod
    def from_environ(cls) -> "DistributedParams":
        # Read environment variables according to the contract in:
        # https://pytorch.org/elastic/0.2.0rc1/distributed.html
        # Note that this is a superset of required environment variables of:
        # https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization

        def _get_key(key, default):
            if key not in os.environ:
                logger.warning(
                    f"Can't find {key} in os.environ, use default: {default}"
                )
            return os.environ.get(key, default)

        local_rank = int(_get_key(cls.LOCAL_RANK_KEY, 0))
        global_rank = int(_get_key(cls.RANK_KEY, 0))
        machine_rank = int(_get_key(cls.GROUP_RANK_KEY, 0))
        num_processes_per_machine = int(_get_key(cls.LOCAL_WORLD_SIZE_KEY, 1))
        world_size = int(_get_key(cls.WORLD_SIZE_KEY, 1))

        logger.info(
            "Loaded distributed params from os.environ:\n"
            f"    local_rank={local_rank}\n"
            f"    global_rank={global_rank}\n"
            f"    machine_rank={machine_rank}\n"
            f"    num_processes_per_machine={num_processes_per_machine}\n"
            f"    world_size={world_size}\n"
        )

        return DistributedParams(
            local_rank=local_rank,
            global_rank=global_rank,
            machine_rank=machine_rank,
            num_processes_per_machine=num_processes_per_machine,
            world_size=world_size,
        )

    @classmethod
    def set_environ(cls, params: "DistributedParams") -> None:
        def _set_env_key(key: str, value: str):
            if key in os.environ and (curr_value := os.environ[key]) != value:
                logger.warning(
                    f"Key {key} already set in OS environ. "
                    f"Current value {curr_value}, overwriting with {value}."
                )
            os.environ[key] = value

        _set_env_key(cls.LOCAL_RANK_KEY, str(params.local_rank))
        _set_env_key(cls.RANK_KEY, str(params.global_rank))
        _set_env_key(cls.GROUP_RANK_KEY, str(params.machine_rank))
        _set_env_key(cls.LOCAL_WORLD_SIZE_KEY, str(params.num_processes_per_machine))
        _set_env_key(cls.WORLD_SIZE_KEY, str(params.world_size))


@contextlib.contextmanager
def enable_dist_process_groups(
    backend: str,
    init_method: Optional[str],
    dist_params: DistributedParams,
    timeout: timedelta = DEFAULT_TIMEOUT,
):
    assert backend.lower() in ["nccl", "gloo"]
    try:
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=dist_params.world_size,
            rank=dist_params.global_rank,
            timeout=timeout,
        )
    except Exception as e:
        logger.error("Process group URL: {}".format(init_method))
        raise e

    if backend.lower() in ["nccl"]:
        torch.cuda.set_device(dist_params.local_rank)
    # synchronize is needed here to prevent a possible timeout after calling
    # init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()

    with _enable_local_process_group(comm, dist_params):
        yield
    dist.destroy_process_group()


def _get_filename_for_rank(prefix: str, rank: int) -> str:
    return f"{prefix}.rank{rank}"


def save_return_deco(
    return_save_file: Optional[str], rank: int
) -> Callable[[Callable[..., _RT]], Callable[..., _RT]]:
    def deco(func: Callable[..., _RT]) -> Callable[..., _RT]:
        """warp a function to save its return to the filename"""

        @functools.wraps(func)
        def new_func(*args, **kwargs) -> _RT:
            ret = func(*args, **kwargs)
            if return_save_file is not None:
                filename = _get_filename_for_rank(return_save_file, rank)
                logger.info(
                    f"Save {func.__module__}.{func.__name__} return to: {filename}"
                )
                # NOTE: test if the return of func is pickable, if not, wrap it
                # so that the results can be saved by torch.save.
                try:
                    pickle.dumps(ret)
                except pickle.PicklingError as e:
                    logger.info(
                        f"Can't pickle the return of function `{func}` due to error `{str(e)}`,"
                        f" try wrapping it with PicklableWrapper to make it pickable."
                    )
                    ret = PicklableWrapper(ret)
                torch.save(ret, filename)
            return ret

        return new_func

    return deco


def default_distributed_worker(
    main_func: Callable[..., _RT],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    backend: str,
    init_method: Optional[str] = None,
    dist_params: Optional[DistributedParams] = None,
    return_save_file: Optional[str] = None,
    timeout: timedelta = DEFAULT_TIMEOUT,
    shared_context: Optional[comm.BaseSharedContext] = None,
) -> _RT:
    if shared_context:
        comm.set_shared_context(
            shared_context
        )  # set the global shared context from the args passed in by mp spawn
    dist_params = dist_params or DistributedParams.from_environ()
    with enable_dist_process_groups(backend, init_method, dist_params, timeout):
        deco = save_return_deco(return_save_file, dist_params.global_rank)
        return deco(main_func)(*args, **kwargs)


def _non_distributed_worker(
    main_func: Callable[..., _RT],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    shared_context: Optional[comm.BaseSharedContext] = None,
) -> _RT:
    if shared_context:
        comm.set_shared_context(
            shared_context
        )  # set the global shared context from the args passed in by mp spawn
    return main_func(*args, **kwargs)


def launch(
    main_func: Callable[..., _RT],
    num_processes_per_machine: int,
    num_machines: int = 1,
    machine_rank: int = 0,
    dist_url: Optional[str] = None,
    backend: str = "NCCL",
    always_spawn: bool = False,
    launch_method: str = "multiprocessing",
    shared_context: Optional[comm.BaseSharedContext] = None,
    timeout: timedelta = DEFAULT_TIMEOUT,
    args: Tuple[Any, ...] = (),
    kwargs: Dict[str, Any] = None,
    # NOTE: API of "distributed worker" is not finalized, please reach out if you want
    # to use customized "distributed worker".
    _distributed_worker: Callable[..., _RT] = default_distributed_worker,
) -> Dict[int, _RT]:
    """Run the `main_func` using multiple processes/nodes
    main_func(*args, **kwargs)
    """

    if kwargs is None:
        kwargs = {}

    if dist_url is None:
        dist_url = f"file:///tmp/mcvdh_dist_file_{time.time()}"

    logger.info(
        f"Launch with num_processes_per_machine: {num_processes_per_machine},"
        f" num_machines: {num_machines}, machine_rank: {machine_rank},"
        f" dist_url: {dist_url}, backend: {backend}, launch_method: {launch_method}."
    )

    if backend == "NCCL":
        assert (
            num_processes_per_machine <= torch.cuda.device_count()
        ), "num_processes_per_machine is greater than device count: {} vs {}".format(
            num_processes_per_machine, torch.cuda.device_count()
        )

    local_ranks = range(
        num_processes_per_machine * machine_rank,
        num_processes_per_machine * (machine_rank + 1),
    )
    world_size = num_machines * num_processes_per_machine
    if world_size > 1 or always_spawn:
        if launch_method not in ["multiprocessing", "elastic"]:
            raise ValueError(f"Invalid launch_method: {launch_method}")

        if launch_method == "elastic":
            lc = pet.LaunchConfig(
                min_nodes=num_machines,
                max_nodes=num_machines,
                nproc_per_node=num_processes_per_machine,
                rdzv_backend="zeus",
                # run_id just has to be globally unique
                run_id=str(hash(dist_url)),  # can't have special character
                # for fault tolerance; set it to 0 for single-node (no fault tolerance)
                max_restarts=3 if num_machines > 1 else 0,
                start_method="spawn",
            )
            results = pet.elastic_launch(lc, entrypoint=_distributed_worker)(
                main_func,
                args,
                kwargs,
                backend,
                None,  # init_method is env by default
                None,  # dist_params is inferred from env
                None,  # return_save_file is None since no return file is needed
                timeout,
                shared_context,
            )
            return results
        else:
            prefix = f"mcvdh_{main_func.__module__}.{main_func.__name__}_return"
            with tempfile.NamedTemporaryFile(prefix=prefix, suffix=".pth") as f:
                return_file = f.name
                if dist_url.startswith("env://"):
                    # FIXME (tsahi): This branch is not necessary, it doesn't launch
                    # anything, we should simply call distributed_worker_elastic_launch
                    return _distributed_worker(
                        main_func,
                        args,
                        kwargs,
                        backend,
                        dist_url,
                        None,  # dist_params is inferred from env
                        return_file,  # is this needed?
                        timeout,
                        shared_context,
                    )
                else:
                    mp.spawn(
                        _mp_spawn_helper,
                        nprocs=num_processes_per_machine,
                        args=(
                            _distributed_worker,
                            main_func,
                            args,
                            kwargs,
                            backend,
                            dist_url,
                            return_file,
                            timeout,
                            shared_context,
                            world_size,
                            num_processes_per_machine,
                            machine_rank,
                        ),
                        daemon=False,
                    )
                    return {
                        local_rank: torch.load(
                            _get_filename_for_rank(return_file, local_rank)
                        )
                        for local_rank in local_ranks
                    }
    else:
        return {0: _non_distributed_worker(main_func, args, kwargs, shared_context)}


def _mp_spawn_helper(
    local_rank: int,  # first position required by mp.spawn
    distributed_worker: Callable[..., _RT],
    main_func: Callable,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    backend: str,
    init_method: Optional[str],
    return_save_file: Optional[str],
    timeout: timedelta,
    shared_context: Optional[comm.BaseSharedContext],
    world_size: int,
    num_processes_per_machine: int,
    machine_rank: int,
) -> _RT:
    global_rank = machine_rank * num_processes_per_machine + local_rank
    # Write into env variables rather than passing the args explicitly
    DistributedParams.set_environ(
        DistributedParams(
            local_rank=local_rank,
            machine_rank=machine_rank,
            global_rank=global_rank,
            num_processes_per_machine=num_processes_per_machine,
            world_size=world_size,
        )
    )

    return distributed_worker(
        main_func=main_func,
        args=args,
        kwargs=kwargs,
        backend=backend,
        init_method=init_method,
        dist_params=None,  # dist_params will be inferred from env
        return_save_file=return_save_file,
        timeout=timeout,
        shared_context=shared_context,
    )


@contextlib.contextmanager
def _enable_local_process_group(
    comm_: types.ModuleType,
    dist_params: DistributedParams,
):
    # Setup the local process group (which contains ranks within the same machine)
    assert comm_._LOCAL_PROCESS_GROUP is None
    num_machines = dist_params.world_size // dist_params.num_processes_per_machine
    for i in range(num_machines):
        ranks_on_i = list(
            range(
                i * dist_params.num_processes_per_machine,
                (i + 1) * dist_params.num_processes_per_machine,
            )
        )
        pg = dist.new_group(ranks_on_i)
        if i == dist_params.machine_rank:
            comm_._LOCAL_PROCESS_GROUP = pg

    comm.synchronize()
    yield

    torch.distributed.destroy_process_group(pg)
    comm_._LOCAL_PROCESS_GROUP = None


def launch_deco(
    num_processes: int = 1,
    backend: str = "GLOO",
    always_spawn: bool = True,
    launch_method: str = "multiprocessing",
    timeout: timedelta = DEFAULT_UNITTEST_TIMEOUT,
    shared_context: Optional[comm.BaseSharedContext] = None,
) -> Callable[[Callable[..., _RT]], Callable[..., Dict[int, _RT]]]:
    """
    A helper decorator to run the instance method via `launch`. This is convenient
    to converte a unittest to distributed version.
    """

    def deco(func: Callable[..., _RT]) -> Callable[..., Dict[int, _RT]]:
        # use functools.wraps to preserve information like __name__, __doc__, which are
        # very useful for unittest.
        @functools.wraps(func)
        def _launch_func(self, *args, **kwargs) -> Dict[int, _RT]:
            results = launch(
                # make func pickable for the sake of multiprocessing.spawn
                PicklableWrapper(func),
                num_processes_per_machine=num_processes,
                backend=backend,
                always_spawn=always_spawn,
                launch_method=launch_method,
                shared_context=shared_context,
                timeout=timeout,
                # multiprocessing.spawn also requires `args` to be pickable, however
                # the unittest.TestCase instance (i.e. `self`) is not pickable,
                # therefore we also need to wrap it.
                args=(PicklableWrapper(self), *args),
                kwargs=kwargs,
            )
            return results

        return _launch_func

    return deco


@contextlib.contextmanager
def process_group_with_timeout(timeout, backend=None):
    """
    A helper contextmanager to create a temporary process group using custom timeout
    without changing the global timeout value set by during dist.init_process_group (
    default value is 30 minutes). This is useful when doing heavy communication that the
    default timeout might not be enough.
    """
    pg = torch.distributed.new_group(
        ranks=list(range(comm.get_world_size())),
        timeout=timeout,
        backend=backend,
    )
    yield pg
    torch.distributed.destroy_process_group(pg)


@contextlib.contextmanager
def interleave_by_rank(concurrency_limit: int = 1):
    """
    A helper contextmanager to interleave code execution by rank.

    Args:
        concurrency_limit: number of concurrent execution.
    """
    rank = comm.get_rank()
    total_size = comm.get_world_size()

    if not concurrency_limit > 0:
        raise ValueError("`concurrency_limit` must be positive")

    for i in range(0, total_size, concurrency_limit):
        if i <= rank and rank < i + concurrency_limit:
            logger.info(f"[interleave_by_rank] rank{rank}/{total_size} starts")
            yield
            logger.info(f"[interleave_by_rank] rank{rank}/{total_size} ends")
        comm.synchronize()
