#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Helper for distributed training in pytorch
adapted from d2/d2go
"""

import logging
import os
import tempfile
import time

import mobile_cv.torch.utils_pytorch.comm as comm
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def get_mp_context():
    """spawn is used for launching processes in `launch`"""
    return mp.get_context("spawn")


def launch(
    main_func,
    num_processes_per_machine,
    num_machines=1,
    machine_rank=0,
    dist_url=None,
    backend="NCCL",
    always_spawn=False,
    args=(),
):
    """Run the `main_func` using multiple processes/nodes
    main_func(*args)
    """

    if dist_url is None:
        dist_url = f"file:///tmp/mcvdh_dist_file_{time.time()}"

    logger = logging.getLogger(__name__)
    logger.info(
        f"Launch with num_processes_per_machine: {num_processes_per_machine},"
        f" num_machines: {num_machines}, machine_rank: {machine_rank},"
        f" dist_url: {dist_url}, backend: {backend}."
    )

    if backend == "NCCL":
        assert (
            num_processes_per_machine <= torch.cuda.device_count()
        ), "num_processes_per_machine is greater than device count: {} vs {}".format(
            num_processes_per_machine, torch.cuda.device_count()
        )

    world_size = num_machines * num_processes_per_machine
    if world_size > 1 or always_spawn:
        prefix = f"mcvdh_{main_func.__module__}.{main_func.__name__}_return"
        with tempfile.NamedTemporaryFile(prefix=prefix, suffix=".pth") as f:
            return_file = f.name
            if dist_url.startswith("env://"):
                _run_with_dist_env(
                    main_func,
                    world_size,
                    num_processes_per_machine,
                    machine_rank,
                    dist_url,
                    backend,
                    return_file,
                    args,
                )
            else:
                mp.spawn(
                    _distributed_worker,
                    nprocs=num_processes_per_machine,
                    args=(
                        main_func,
                        world_size,
                        num_processes_per_machine,
                        machine_rank,
                        dist_url,
                        backend,
                        return_file,
                        args,
                    ),
                    daemon=False,
                )
            if machine_rank == 0:
                return torch.load(return_file)
    else:
        return main_func(*args)


def _run_with_dist_env(
    main_func,
    world_size,
    num_processes_per_machine,
    machine_rank,
    dist_url,
    backend,
    return_file,
    args,
):
    assert dist_url.startswith("env://")

    # Read torch.distributed params from env according to the contract in
    # https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    num_processes_per_machine = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    machine_rank = int(os.environ.get("GROUP_RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    num_machines = int(world_size / num_processes_per_machine)

    logger = logging.getLogger(__name__)
    logger.info(
        "Loaded distributed params from env."
        f" Run with num_processes_per_machine: {num_processes_per_machine},"
        f" num_machines: {num_machines}, machine_rank: {machine_rank},"
    )

    _distributed_worker(
        local_rank,
        main_func,
        world_size,
        num_processes_per_machine,
        machine_rank,
        dist_url,
        backend,
        return_file,
        args,
    )


def _distributed_worker(
    local_rank,
    main_func,
    world_size,
    num_processes_per_machine,
    machine_rank,
    dist_url,
    backend,
    return_file,
    args,
):
    assert backend in ["NCCL", "GLOO"]

    logger = logging.getLogger(__name__)

    global_rank = machine_rank * num_processes_per_machine + local_rank
    try:
        dist.init_process_group(
            backend=backend,
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
        )
    except Exception as e:
        logger.error("Process group URL: {}".format(dist_url))
        raise e

    # Setup the local process group (which contains ranks within the same machine)
    assert comm._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_processes_per_machine
    for i in range(num_machines):
        ranks_on_i = list(
            range(i * num_processes_per_machine, (i + 1) * num_processes_per_machine)
        )
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg

    if backend in ["NCCL"]:
        torch.cuda.set_device(local_rank)
    # synchronize is needed here to prevent a possible timeout after calling
    # init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()

    ret = main_func(*args)
    if global_rank == 0:
        logger.info(
            "Save {}.{} return to: {}".format(
                main_func.__module__, main_func.__name__, return_file
            )
        )
        torch.save(ret, return_file)
