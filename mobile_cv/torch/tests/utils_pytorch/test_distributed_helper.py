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

    @dh.launch_deco(num_processes=2, timeout=timedelta(milliseconds=100))
    def test_timeout(self):
        results = comm.all_gather(comm.get_rank())
        self.assertEqual(results, [0, 1])

        # the global timeout is set to 100ms, calling `all_gather` will cause timeout
        # error because there's a 200ms gap between two processes at the beginning of
        # `all_gather`. Here we use `process_group_with_timeout` to increate the timeout
        # temporarily, so the `all_gather` can run successfully.
        with dh.process_group_with_timeout(timeout=timedelta(milliseconds=1000)) as pg:
            if comm.get_rank() == 0:
                time.sleep(0.2)
            comm.all_gather(comm.get_rank(), group=pg)
        self.assertEqual(results, [0, 1])

    @dh.launch_deco(num_processes=2, shared_context=SharedContext(10))
    def test_shared_context(self):
        # check that subprocess gets the correct shared_context
        self.assertEqual(comm.get_shared_context().value, 10)
