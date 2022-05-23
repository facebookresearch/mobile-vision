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
    def test_distributed_helper_launch(self):
        result = dh.launch(
            _test_func,
            num_processes_per_machine=2,
            backend="GLOO",
            args=(3,),
        )
        self.assertEqual(result, {"a": torch.tensor([5.5]), "b": torch.tensor([6.5])})
