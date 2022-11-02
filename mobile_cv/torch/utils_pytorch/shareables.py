#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import os
import pickle
from multiprocessing import shared_memory
from typing import Any, List, Tuple, Union

import mobile_cv.torch.utils_pytorch.comm as comm
import numpy as np

logger = logging.getLogger(__name__)


class _SharedMemoryRef(object):
    """Deal with the clean up of shared memory"""

    def __init__(self, shm: shared_memory.SharedMemory, owner_pid: int):
        self.shm = shm
        self.owner_pid = owner_pid

    def __del__(self):
        self.shm.close()  # all instances should call close()
        if os.getpid() == self.owner_pid:
            self.shm.unlink()  # destroy the underlying shared memory block


def share_numpy_array_locally(
    data: Union[np.ndarray, None],
) -> Tuple[np.ndarray, _SharedMemoryRef]:
    """
    Helper function to create memory-shared numpy array.

    Args:
        data: the original data, the data provided by non-local master process will
            be discarded.
    Returns:
        new_data: a shared numpy array equal to the original one provided from master
            process. Note that the memory of numpy array might still be copied if
            passing it to child process, in those case it's better to user `shm`.
        shm: the underlying shared memory, the caller needs to hold this object in
            order to prevent it from GC-ed.
    """

    if not isinstance(data, (np.ndarray, type(None))):
        raise TypeError(f"Unexpected data type: {type(data)}")

    new_data = None
    shm = None
    master_rank_pid = None

    if comm.get_local_rank() == 0:
        if not isinstance(data, np.ndarray):
            raise ValueError(
                f"Data must be provided from local master rank (rank: {comm.get_rank()}"
            )
        # create a new shared memory using the original data
        shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
        master_rank_pid = os.getpid()
        logger.info(f"Moving data to shared memory ({shm}) ...")
        new_data = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
        new_data[:] = data[:]
        shared_data_info = (data.shape, data.dtype, shm.name, master_rank_pid)
        # maybe release the memory held by the original data?
    else:
        if data is not None:
            raise ValueError(
                f"Data must be None for non local master rank (rank: {comm.get_rank()}"
            )
        shared_data_info = None

    # broadcast the shared memory name
    shared_data_info_list = comm.all_gather(shared_data_info)
    local_master_rank = (
        comm.get_rank() // comm.get_local_size()
    ) * comm.get_local_size()
    shared_data_info = shared_data_info_list[local_master_rank]
    assert shared_data_info is not None

    # create new data from shared memory
    if not comm.get_local_rank() == 0:
        shape, dtype, name, master_rank_pid = shared_data_info
        shm = shared_memory.SharedMemory(name=name)
        logger.info(f"Attaching to the existing shared memory ({shm}) ...")
        new_data = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    # synchronize before returning to make sure data are usable
    comm.synchronize()
    assert isinstance(new_data, np.ndarray)
    assert isinstance(shm, shared_memory.SharedMemory)
    assert isinstance(master_rank_pid, int)
    return new_data, _SharedMemoryRef(shm, master_rank_pid)


class SharedList(object):
    """
    List-like read-only object shared between all (local) processes, backed by
    multiprocessing.shared_memory (requires Python 3.8+).
    """

    def __init__(
        self, lst: Union[List[Any], Any], *, _allow_inplace_update: bool = False
    ):
        """
        Args:
            lst (list or None): a list of serializable objects.
        """

        self._allow_inplace_update = _allow_inplace_update

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        logger.info(
            "Serializing {} elements to byte tensors and concatenating them all ...".format(
                len(lst)
            )
        )
        if comm.get_local_rank() == 0:
            lst = [_serialize(x) for x in lst]
            addr = np.asarray([len(x) for x in lst], dtype=np.int64)
            addr = np.cumsum(addr)
            lst = np.concatenate(lst)
            logger.info(
                "Serialized dataset takes {:.2f} MiB".format(len(lst) / 1024**2)
            )
        else:
            addr = None
            lst = None

        logger.info("Moving serialized dataset to shared memory ...")
        # keep the returned shared memory to prevent it from GC-ed
        _, self._lst_shm_ref = share_numpy_array_locally(lst)
        self._addr, self._addr_shm_ref = share_numpy_array_locally(addr)
        logger.info("Finished moving to shared memory")

    def _calculate_addr_range(self, idx: int) -> Tuple[int, int]:
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        return start_addr, end_addr

    def __len__(self):
        return len(self._addr)

    def __getitem__(self, idx):
        start_addr, end_addr = self._calculate_addr_range(idx)
        # @lint-ignore PYTHONPICKLEISBAD
        return pickle.loads(self._lst_shm_ref.shm.buf[start_addr:end_addr])

    def __setitem__(self, idx, value):
        # Normally user shouldn't update the stored data since this class is designed to
        # be read-only, in rare cases where user knows that the size of data would be
        # the same, it might be helpful to update the stored data.
        if not self._allow_inplace_update:
            raise RuntimeError("Update item from SharedList is not allowed!")
        # NOTE: Currently user should be responsible for dealing with race-condition.
        start_addr, end_addr = self._calculate_addr_range(idx)
        nbytes = end_addr - start_addr
        new_bytes = pickle.dumps(value, protocol=-1)
        if len(new_bytes) != nbytes:
            raise ValueError(
                f"Can't replace the original object ({nbytes} bytes) with one that has"
                f" different size ({len(new_bytes)} bytes)!"
            )
        self._lst_shm_ref.shm.buf[start_addr:end_addr] = new_bytes


class SharedDict(object):
    """
    Dict-like read-only object shared between all (local) processes, backed by
    multiprocessing.shared_memory (requires Python 3.8+).
    """

    # TODO: we can support dict in a similar way if needed
