# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

from typing import Any, Callable, Iterable, Optional, Tuple

import mobile_cv.torch.utils_pytorch.comm as comm
import mobile_cv.torch.utils_pytorch.distributed_helper as dist_helper
import torch.distributed as dist
from mobile_cv.common.misc.logger_utils import setup_logger
from torch import multiprocessing as mp


class QueueEnd(object):
    pass


def _enqueue_data_proc(
    rank: int,
    build_dataset_func: Callable[..., Iterable[Any]],
    build_dataset_func_args: Tuple,
    queue: mp.Queue,
    local_world_size: int,
):
    logger = setup_logger(None, distributed_rank=0, name="central_process_data_loader")

    logger.info(f"Starting data queue process for {local_world_size} processes")

    dataset = build_dataset_func(*build_dataset_func_args)
    if hasattr(dataset, "set_logger"):
        dataset.set_logger(logger)

    logger.info(f"Dataset built: {dataset}")

    for idx, item in enumerate(dataset):
        queue.put(item)
        if idx % 50 == 0:
            logger.info(f"data queue size {queue.qsize()}")

    for _ in range(local_world_size):
        queue.put(QueueEnd())
    logger.info("All data has sent to data queue")


class CentralProcessDataLoader(object):
    """
    Create a data loader wrapper that the underlying data loader will run
    on a separate process and the data are shared across processes in the same
    node of the machine through a queue.

    The class could be used as follow on each process:
    ```
        data_loader = CentralProcessDataLoader(
            build_dataset_func, (data_loader_args1,), comm
        )
        for item in data_loader:
            # process item
    ```
    The underlying data loader created by `build_dataset_func` needs to be sharded
    based on the node index, and a ProcessGroup representing the processes in the
    same node is required for communication.

    This class is helpful to wrap a data loader that is expensive/has limits to
    connect but could load data in batch.
    """

    def __init__(
        self,
        build_dataset_func: Callable[..., Iterable[Any]],
        build_dataset_func_args: Tuple,
        local_process_group: Optional[dist.ProcessGroup],
        single_process: bool = False,
        max_qsize: int = 0,
    ):
        """
        * `build_dataset_func` returns the actual data loader. It needs to be sharded
        based on the index of the *node*.
        * `build_dataset_func_args` is a tuple of arguments for `build_dataset_func`
        * `local_process_group` is the ProcessGroup that represents processes in
        the same node. See `_LOCAL_PROCESS_GROUP` in distributed_helper.py for how
        it is built.
        * `single_process` whether to run the data loader in the same process
        * `max_qsize` maximum queue size for the underlying data loader
        """
        self.build_dataset_func = build_dataset_func
        self.build_dataset_func_args = build_dataset_func_args
        self.local_process_group = local_process_group
        self.single_process = single_process
        self.max_qsize = max_qsize

        if local_process_group is not None:
            assert isinstance(local_process_group, dist.ProcessGroup), type(
                local_process_group
            )

    def __iter__(self):
        return CentralProcessDataLoaderIter(self)


class CentralProcessDataLoaderIter(object):
    def __init__(self, cdl: CentralProcessDataLoader):
        self.cdl = cdl
        build_dataset_func = cdl.build_dataset_func
        build_dataset_func_args = cdl.build_dataset_func_args
        group_helper = comm.LocalGroupHelper(cdl.local_process_group)
        single_process = cdl.single_process
        max_qsize = cdl.max_qsize

        if group_helper.get_local_rank() == 0:
            mp_manager = dist_helper.get_mp_context().Manager()
            data_queue = mp_manager.Queue(max_qsize)
            data_queue_list = [data_queue]
        else:
            data_queue_list = [None]

        # broadcast the queue to each process inside the same node
        if group_helper.get_local_size() > 1:
            dist_helper.dist.broadcast_object_list(
                data_queue_list,
                src=group_helper.get_node_rank() * group_helper.get_local_size(),
                group=group_helper.get_local_group(),
            )
            data_queue = data_queue_list[0]
        assert data_queue is not None
        self.data_queue = data_queue

        self.load_proc = None
        if group_helper.get_local_rank() == 0:
            # launch the data loading process
            local_world_size = group_helper.get_local_size()
            if not single_process:
                self.load_proc = mp.spawn(
                    _enqueue_data_proc,
                    args=(
                        build_dataset_func,
                        build_dataset_func_args,
                        data_queue,
                        local_world_size,
                    ),
                    join=False,
                )
            else:
                _enqueue_data_proc(
                    -1,
                    build_dataset_func,
                    build_dataset_func_args,
                    data_queue,
                    local_world_size,
                )

    def __iter__(self):
        return self

    def __next__(self):
        item = self.data_queue.get()
        if type(item) == QueueEnd:
            # wait on all processes to finish, the mp_manager/data_queue may have
            # been destoryed if the main process ends before other processes
            comm.synchronize()
            raise StopIteration
        return item
