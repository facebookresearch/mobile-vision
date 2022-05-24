#!/usr/bin/env python3

import unittest

import mobile_cv.torch.tests.helper as helper
import mobile_cv.torch.utils_pytorch.comm as comm
import torch


def _reduce_dict(rank, adict):
    ret = comm.reduce_dict(adict[rank])
    return ret


def _run_gather(rank, data):
    return comm.all_gather(data[rank])


class TestUtilsPytorchCommUtils(unittest.TestCase):
    def test_comm_utils_reduce_dict(self):
        result = helper.run_func_dist(
            2,
            _reduce_dict,
            [
                {"a": torch.tensor([2.0]), "b": torch.tensor([3.0])},
                {"a": torch.tensor([3.0]), "b": torch.tensor([4.0])},
            ],
        )
        self.assertEqual(
            result[0], {"a": torch.tensor([2.5]), "b": torch.tensor([3.5])}
        )

    def test_comm_utils_all_gather(self):
        result = helper.run_func_dist(
            2,
            _run_gather,
            [
                123,
                456,
            ],
        )
        self.assertEqual(result[0], [123, 456])

    @helper.enable_dist_env
    def test_get_world_size(self):
        self.assertEqual(comm.get_world_size(), 1)
