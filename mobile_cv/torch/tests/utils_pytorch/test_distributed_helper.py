#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import time
import unittest
from dataclasses import dataclass
from datetime import timedelta

import mobile_cv.torch.utils_pytorch.comm as comm
import mobile_cv.torch.utils_pytorch.distributed_helper as dh
import torch
from mobile_cv.common.misc.oss_utils import is_oss
from mobile_cv.common.misc.py import PicklableWrapper
from mobile_cv.torch.utils_pytorch.comm import BaseSharedContext


@dataclass
class SharedContext(BaseSharedContext):
    value: int


def _test_func(value):
    rank = comm.get_rank()
    data = {
        "a": torch.tensor([2.0 + value + rank]),
        "b": torch.tensor([3.0 + value + rank]),
    }
    ret = comm.reduce_dict(data)
    return ret


def _do_workload_with_interleave(concurrency_limit: int, workload_sec: float) -> float:
    with dh.interleave_by_rank(concurrency_limit=concurrency_limit):
        time.sleep(workload_sec)
        return time.perf_counter()


def _all_gather_offset_by_200ms(self: unittest.TestCase, timeout_ms: float):
    results = comm.all_gather(comm.get_rank())
    self.assertEqual(results, list(range(comm.get_world_size())))

    # If the `timeout_ms` is less than 200ms, calling `all_gather` will cause timeout
    # error because there's a 200ms gap between two processes at the beginning of
    # `all_gather`.
    with dh.process_group_with_timeout(
        timeout=timedelta(milliseconds=timeout_ms)
    ) as pg:
        if comm.get_rank() == 0:
            time.sleep(0.2)
        torch.distributed.monitored_barrier(group=pg)
        results = comm.all_gather(comm.get_rank(), group=pg)
    self.assertEqual(results, list(range(comm.get_world_size())))


class TestUtilsPytorchDistributedHelper(unittest.TestCase):
    def setUp(self):
        self.magic_value = 42

    def get_magic_value(self):
        return 42

    def test_distributed_helper_launch(self):
        results = dh.launch(
            _test_func,
            num_processes_per_machine=2,
            backend="GLOO",
            args=(3,),
        )
        self.assertEqual(
            results[0], {"a": torch.tensor([5.5]), "b": torch.tensor([6.5])}
        )
        self.assertEqual(
            results[1], {"a": torch.tensor([6.0]), "b": torch.tensor([7.0])}
        )

    @dh.launch_deco(num_processes=2)
    def test_distributed_helper_launch_deco(self):
        rank = comm.get_rank()
        value = 3
        data = {
            "a": torch.tensor([2.0 + value + rank]),
            "b": torch.tensor([3.0 + value + rank]),
        }
        result = comm.reduce_dict(data)
        if rank == 0:
            self.assertEqual(
                result, {"a": torch.tensor([5.5]), "b": torch.tensor([6.5])}
            )
        else:
            self.assertEqual(
                result, {"a": torch.tensor([6.0]), "b": torch.tensor([7.0])}
            )

    @dh.launch_deco(num_processes=2)
    def _test_launch_deco_with_args(self, inputs, outputs):
        self.assertEqual(len(inputs), comm.get_world_size())
        self.assertEqual(len(outputs), comm.get_world_size())
        rank = comm.get_rank()
        results = comm.all_gather(inputs[rank])
        self.assertEqual(results, outputs)

    def test_launch_deco_with_args(self):
        inputs = outputs = [1, 2]
        self._test_launch_deco_with_args(inputs, outputs)

    def test_launch_deco_with_kwargs(self):
        inputs = outputs = [1, 2]
        self._test_launch_deco_with_args(inputs=inputs, outputs=outputs)

    @dh.launch_deco(num_processes=1)
    def test_launch_deco_access_member_variables(self):
        self.assertEqual(self.magic_value, 42)
        self.assertEqual(self.get_magic_value(), 42)

    @unittest.skipIf(is_oss(), "rdzv_backend is not available")
    @dh.launch_deco(num_processes=2, launch_method="elastic")  # elastic is slow
    def test_elastic_launch(self):
        rank = comm.get_rank()
        ranks = comm.all_gather(rank)
        self.assertEqual(ranks, [0, 1])
        world_size = comm.get_world_size()
        self.assertEqual(world_size, 2)
        # test local process group
        local_rank = comm.get_local_rank()
        local_ranks = comm.all_gather(local_rank)
        self.assertEqual(local_ranks, [0, 1])
        local_world_size = comm.get_local_size()
        self.assertEqual(local_world_size, 2)

    def test_custom_timeout(self):
        # with 500ms timeout, the `all_gather` should run successfully.
        dh.launch(
            _all_gather_offset_by_200ms,
            num_processes_per_machine=2,
            backend="GLOO",
            kwargs={
                "self": PicklableWrapper(self),
                "timeout_ms": 500,
            },
        )

        # with 100ms timeout, the `all_gather` should fail.
        with self.assertRaises(torch.multiprocessing.ProcessRaisedException):
            dh.launch(
                _all_gather_offset_by_200ms,
                num_processes_per_machine=2,
                backend="GLOO",
                kwargs={
                    "self": PicklableWrapper(self),
                    "timeout_ms": 100,
                },
            )

    @dh.launch_deco(num_processes=2, shared_context=SharedContext(10))
    def test_shared_context(self):
        # check that subprocess gets the correct shared_context
        self.assertEqual(comm.get_shared_context().value, 10)

    def test_interleave_by_rank_no_concurrency(self):
        results = dh.launch(
            _do_workload_with_interleave,
            num_processes_per_machine=3,
            backend="GLOO",
            kwargs={
                "concurrency_limit": 1,
                "workload_sec": 0.1,
            },
        )
        self.assertGreater(results[2] - results[1], 0.1)
        self.assertGreater(results[1] - results[0], 0.1)

    def test_interleave_by_rank_with_concurrency(self):
        results = dh.launch(
            _do_workload_with_interleave,
            num_processes_per_machine=5,
            backend="GLOO",
            kwargs={
                "concurrency_limit": 2,
                "workload_sec": 0.1,
            },
        )
        epsilon = 0.1  # assume concurrent processes start within epsilon time window.
        self.assertLess(abs(results[1] - results[0]), epsilon)
        self.assertLess(abs(results[3] - results[2]), epsilon)
        self.assertGreater(results[2] - results[1], 0.1)
        self.assertGreater(results[4] - results[3], 0.1)

    def test_interleave_by_rank_max_concurrency(self):
        results = dh.launch(
            _do_workload_with_interleave,
            num_processes_per_machine=3,
            backend="GLOO",
            kwargs={
                "concurrency_limit": 5,  # it's legal to set limit larger than nproc
                "workload_sec": 0.1,
            },
        )
        epsilon = 0.1  # assume concurrent processes start within epsilon time window.
        self.assertLess(abs(results[2] - results[1]), epsilon)
