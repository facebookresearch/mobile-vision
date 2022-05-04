# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import tempfile
import unittest

import mobile_cv.common.misc.logger_utils as lu
from mobile_cv.common.utils_io import get_path_manager


def has_content(file_name: str, content: str):
    all_data = open(file_name).read()
    return content in all_data


class TestLoggerUtils(unittest.TestCase):
    def test_logger_utils(
        self,
    ):
        pm = get_path_manager()

        with tempfile.TemporaryDirectory() as output_dir:
            logger = lu.setup_logger(output_dir, path_manager=pm)
            logger.info("abcd")

            log_file = os.path.join(output_dir, "log.txt")
            self.assertTrue(os.path.exists(log_file))
            self.assertTrue(has_content(log_file, "abcd"))

            cur_logger = lu.logging.getLogger(__name__)
            cur_logger.warning("efgh")
            self.assertTrue(has_content(log_file, "efgh"))
            self.assertTrue(
                has_content(log_file, "mobile_cv.common.tests.test_misc_logger_utils")
            )

            logger1 = lu.setup_logger(
                output_dir, distributed_rank=1, name="abcd", path_manager=pm
            )
            logger1.info("kwexy")
            log_file1 = os.path.join(output_dir, "log.txt.rank1")
            self.assertTrue(os.path.exists(log_file1))
            self.assertTrue(has_content(log_file1, "kwexy"))
