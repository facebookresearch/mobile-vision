#!/usr/bin/env python3

import multiprocessing
import os
import random
import tempfile
import time
import unittest

import torch
from mobile_cv.common.misc import local_cache


ITER = 100
READ_TIME = 0.1
READ_DATA_SIZE = int(12 * 1024 * 1024 / 4)


def _load_func(value):
    time.sleep(READ_TIME)
    ret = torch.zeros([READ_DATA_SIZE])
    ret[0] = int(value)
    return ret


def func_overlap(cache_dir, index, total_processes, iter, shards, sample_range):
    cache = local_cache.LocalCache(cache_dir, use_timer=True, num_shards=shards)
    random.seed(index)
    for _idx in range(iter):
        read_path = random.randrange(sample_range)
        cache.load(_load_func, str(read_path))
    cache.print_stat(f"pid {os.getpid()}")


def func_unique(cache_dir, index, total_processes, iter, shards):
    cache = local_cache.LocalCache(cache_dir, use_timer=True, num_shards=shards)
    random.seed(index)
    for idx in range(iter):
        read_idx = index * iter + idx
        cache.load(_load_func, str(read_idx))
    cache.print_stat(f"pid {os.getpid()}")


def test_cache(self, test_func, num_procs, iter, shards=1, **kwargs):

    with tempfile.TemporaryDirectory() as tmpdir:
        if num_procs == 0:
            test_func(
                tmpdir, index=0, total_processes=1, iter=iter, shards=shards, **kwargs
            )

        else:
            mp = multiprocessing.get_context("spawn")
            procs = []
            for idx in range(num_procs):
                pp = mp.Process(
                    target=test_func,
                    args=(tmpdir, idx, num_procs, iter, shards),
                    kwargs=kwargs,
                )
                procs.append(pp)

            [x.start() for x in procs]
            [x.join() for x in procs]

        cache = local_cache.LocalCache(tmpdir, num_shards=shards)

        total_count = 0
        for path in cache.cache:
            total_count += 1
            self.assertEqual(int(path), cache.cache[path][0])

        # check cache is not empty (child process may fail)
        self.assertGreater(total_count, 0)


class TestLocalCache(unittest.TestCase):
    def test_local_cache_mp_unique_single_process(self):
        test_cache(self, func_unique, 0, 10, shards=16)

    def test_local_cache_mp_unique(self):
        test_cache(self, func_unique, 16, 10, shards=16)

    def test_local_cache_mp_overlap_single_process(self):
        test_cache(self, func_overlap, 0, 20, shards=4, sample_range=80)

    def test_local_cache_mp_overlap(self):
        test_cache(self, func_overlap, 4, 20, shards=4, sample_range=80)


if __name__ == "__main__":
    unittest.main()
