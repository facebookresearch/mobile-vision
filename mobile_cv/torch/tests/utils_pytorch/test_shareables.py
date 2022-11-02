#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import time
import unittest

import mobile_cv.torch.utils_pytorch.comm as comm
import mobile_cv.torch.utils_pytorch.distributed_helper as dh
import numpy as np
from mobile_cv.common.misc.py import PicklableWrapper
from mobile_cv.torch.utils_pytorch.shareables import (
    share_numpy_array_locally,
    SharedList,
)


class TestInMemorySharedNumpyArray(unittest.TestCase):
    @dh.launch_deco(num_processes=2)
    def test_shared_numpy_array(self):
        # create numpy array on master process
        if comm.get_rank() == 0:
            data = np.array([1, 2, 3], dtype=np.dtype("float32"))
        else:
            data = None

        # share the numpy array for all processes
        data, shm_ref = share_numpy_array_locally(data)
        self.assertEqual(data.shape, (3,))
        self.assertEqual(data[0], 1.0)
        self.assertEqual(data[1], 2.0)
        self.assertEqual(data[2], 3.0)
        comm.synchronize()


def _check_and_modify(self, shared_lst):
    # the shared list should be available on all ranks
    self.assertEqual(len(shared_lst), 3)
    self.assertEqual(shared_lst[0], "old")
    self.assertEqual(shared_lst[1], 2)
    self.assertEqual(shared_lst[2], (3,))

    # make sure other ranks have finished above checks before rank N-1 modifies the list
    comm.synchronize()
    # modify the list from rank N-1
    if comm.get_rank() == comm.get_world_size() - 1:
        shared_lst[0] = "new"
        self.assertEqual(shared_lst[0], "new")
        # setting different sized object is illegal
        with self.assertRaises(ValueError):
            shared_lst[0] = "long enough string"
        self.assertEqual(shared_lst[0], "new")
    # make sure rank N-1 has modified the list before rank0 does the next check
    comm.synchronize()

    # now the list should be updated on rank 0 as well since they're shared
    self.assertEqual(shared_lst[0], "new")


class TestInMemoryShareables(unittest.TestCase):
    def test_shared_list_single_process(self):
        lst = ["old", 2, (3,)]
        shared_lst = SharedList(lst, _allow_inplace_update=True)
        del lst
        _check_and_modify(self, shared_lst)

    @dh.launch_deco(num_processes=2)
    def test_shared_list_shared_among_peers(self):
        """
        This test mimics that one GPU worker creates a large dataset and wants to shared
        it with other GPU workers without copying the memory.
        """
        # only create the list from rank 0
        if comm.get_rank() == 0:
            lst = ["old", 2, (3,)]
        else:
            lst = "whatever, this won't be used"

        # create the shared list
        shared_lst = SharedList(lst, _allow_inplace_update=True)
        del lst

        # now the list should be available on all ranks
        _check_and_modify(self, shared_lst)

    def test_shared_list_pass_to_child_processes(self):
        """
        This test mimics that a large dataset is created on GPU worker, the dataset is
        then passed to the data loader worker without copying the memory.
        """
        lst = ["old", 2, (3,)]
        parent_proc_shared_lst = SharedList(lst, _allow_inplace_update=True)
        del lst

        # launch child processes and pass the shared object
        dh.launch(
            _check_and_modify,
            num_processes_per_machine=2,
            backend="GLOO",
            args=(PicklableWrapper(self), parent_proc_shared_lst),
        )

        # since child process share the object with parent, the object should be modified
        self.assertEqual(parent_proc_shared_lst[0], "new")

    @dh.launch_deco(num_processes=2)
    def test_unevenly_close(self):
        """
        Test the SharedList can still be accessed from non-master process after master
        process has finished using it.
        """
        lst = SharedList([1, 2, 3])
        # creation of SharedList should handle synchronization, don't need to call it here.
        rank = comm.get_rank()
        time.sleep(rank)  # mimic a workload that master process finishes first
        # rank 1 can still access the list even after the list has been deleted from rank 0.
        print(f"rank {rank} got: {(x := lst[0])}")
        self.assertEqual(x, 1)
        del lst
