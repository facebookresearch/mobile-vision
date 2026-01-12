#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import shutil
import tempfile
import unittest

from mobile_cv.common.utils_io import get_path_manager
from mobile_cv.torch.utils_pytorch import comm, distributed_helper as dh
from mobile_cv.torch.utils_pytorch.result_cache import ResultCache


class TestUtilsPytorchResultCache(unittest.TestCase):
    @dh.launch_deco(num_processes=2)
    def test_result_cache(self):
        """
        buck2 run @mode/dev-nosan //mobile-vision/mobile_cv/mobile_cv/torch/tests:utils_pytorch_test_result_cache
        """
        path_manager = get_path_manager()
        if comm.is_main_process():
            cache_dir = tempfile.mkdtemp()
        else:
            cache_dir = None
        cache_dir = comm.all_gather(cache_dir)[0]

        rc = ResultCache(
            cache_dir, "test_cache", logger=None, path_manager=path_manager
        )
        self.assertFalse(rc.has_cache())
        rc.save({"data": f"data_{comm.get_rank()}"})
        comm.synchronize()
        self.assertTrue(rc.has_cache())
        out = rc.load(gather=True)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["data"], "data_0")
        self.assertEqual(out[1]["data"], "data_1")

        if comm.is_main_process():
            shutil.rmtree(cache_dir)
        comm.synchronize()
