#!/usr/bin/env python3

import unittest

import mobile_cv.torch.utils_pytorch.comm as comm
import mobile_cv.torch.utils_pytorch.distributed_helper as dh
import torch


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
        result = dh.launch(
            _test_func,
            num_processes_per_machine=2,
            backend="GLOO",
            args=(3,),
        )
        self.assertEqual(result, {"a": torch.tensor([5.5]), "b": torch.tensor([6.5])})

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

    @dh.launch_deco(num_processes=1)
    def test_launch_deco_access_member_variables(self):
        self.assertEqual(self.magic_value, 42)
        self.assertEqual(self.get_magic_value(), 42)

    # @dh.launch_deco(num_processes=2, launch_method="elastic")  # elastic is slow
    # def test_elastic_launch(self):
    #     rank = comm.get_rank()
    #     ranks = comm.all_gather(rank)
    #     self.assertEqual(ranks, [0, 1])
    #     world_size = comm.get_world_size()
    #     self.assertEqual(world_size, 2)
    #     # test local process group
    #     local_rank = comm.get_local_rank()
    #     local_ranks = comm.all_gather(local_rank)
    #     self.assertEqual(local_ranks, [0, 1])
    #     local_world_size = comm.get_local_size()
    #     self.assertEqual(local_world_size, 2)
