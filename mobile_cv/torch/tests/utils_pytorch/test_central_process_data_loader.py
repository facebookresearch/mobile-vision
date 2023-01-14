#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import time
import unittest

import mobile_cv.torch.utils_pytorch.central_process_data_loader as cpdl
import mobile_cv.torch.utils_pytorch.comm as comm
import mobile_cv.torch.utils_pytorch.distributed_helper as dh
from mobile_cv.common.misc.oss_utils import is_oss


def get_data_loader(num_items):
    for item in range(num_items):
        yield item


def _test_func(num_items, single_process=False):
    print("start test function")
    rank = comm.get_rank()
    dl = cpdl.CentralProcessDataLoader(
        get_data_loader,
        (num_items,),
        comm.get_local_group(),
        single_process=single_process,
    )
    ret = list(dl)
    print(f"Rank {rank} result: {ret}")
    return ret


def _test_func_unbalanced(num_items):
    print("start test unbalanced function")
    rank = comm.get_rank()
    dl = cpdl.CentralProcessDataLoader(
        get_data_loader, (num_items,), comm.get_local_group()
    )
    ret = []
    dl_iter = iter(dl)
    time.sleep(rank * 2)
    for item in dl_iter:
        print(f"get item {item}")
        ret.append(item)
        time.sleep(rank * 2)
    time.sleep(rank * 2)
    print(f"Rank {rank} result: {ret}")
    return ret


def _test_func_max_qsize(num_items, max_qsize):
    print("start test max queue size function")
    assert max_qsize > 0

    rank = comm.get_rank()
    dl = cpdl.CentralProcessDataLoader(
        get_data_loader, (num_items,), comm.get_local_group(), max_qsize=max_qsize
    )
    ret = []
    dl_iter = iter(dl)
    for item in dl_iter:
        print(f"get item {item}")
        ret.append(item)
        time.sleep(0.1)
        # the queue should have at most max_qsize data
        assert dl_iter.data_queue.qsize() <= max_qsize, dl_iter.data_queue.qsize()
    print(f"Rank {rank} result: {ret}")
    return ret


class TestUtilsPytorchCentralProcessDataLoader(unittest.TestCase):
    def test_central_process_data_loader_single_process(self):
        results = dh.launch(
            _test_func,
            num_processes_per_machine=1,
            backend="GLOO",
            args=(10,),
        )
        result = results[0]
        self.assertEqual(result, list(range(10)))

    def test_central_process_data_loader_single_process_single_enqueue(self):
        # the enqueue func will run in main process as well
        single_process = True
        results = dh.launch(
            _test_func,
            num_processes_per_machine=1,
            backend="GLOO",
            args=(10, single_process),
        )
        result = results[0]
        self.assertEqual(result, list(range(10)))

    def test_central_process_data_loader_multi_process(self):
        results = dh.launch(
            _test_func,
            num_processes_per_machine=2,
            backend="GLOO",
            args=(10,),
        )
        result = results[0]
        self.assertEqual(result, sorted(result))
        self.assertLess(len(result), 10)
        self.assertTrue(all(x in range(10) for x in result))

    # FIXME: this test fails on OSS
    @unittest.skipIf(is_oss(), "this test fails on OSS")
    def test_central_process_data_loader_multi_process_unbalanced(self):
        results = dh.launch(
            _test_func_unbalanced,
            num_processes_per_machine=2,
            backend="GLOO",
            args=(10,),
        )
        result = results[0]
        self.assertEqual(result, sorted(result))
        self.assertEqual(len(result), 9)
        self.assertTrue(all(x in range(10) for x in result))

    def test_central_process_data_loader_maxqsize(self):
        # test the queue size is 1
        max_qsize = 1
        results = dh.launch(
            _test_func_max_qsize,
            num_processes_per_machine=1,
            backend="GLOO",
            args=(10, max_qsize),
        )
        result = results[0]
        self.assertEqual(result, list(range(10)))


"""
buck2 run @mode/dev-nosan mobile-vision/mobile_cv/mobile_cv/torch/tests:test_utils_pytorch_central_process_data_loader
"""
