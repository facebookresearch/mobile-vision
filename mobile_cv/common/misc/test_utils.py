#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Helper utilities for unit test
"""

import glob
import itertools
import os
import unittest

import pkg_resources
from mobile_cv.common.misc.oss_utils import fb_overwritable


class SubPackageInitFileTestMixin(object):
    """
    A helper for creating test to check every subpackage has an __init__.py file.

    Example Usage:
        class MyTest(SubPackageInitFileTestMixin, unittest.TestCase):
            def get_pacakge_name(self) -> str:
                return "my_package"
    """

    def get_pacakge_name(self) -> str:
        raise NotImplementedError()

    def test_has_init_files(self):
        package_name = self.get_pacakge_name()
        root = pkg_resources.resource_filename(package_name, "")

        all_py_files = glob.glob(f"{root}/**/*.py", recursive=True)
        all_package_dirs = [os.path.dirname(f) for f in all_py_files]
        all_package_dirs = sorted(set(all_package_dirs))  # dedup
        all_package_dirs = [os.path.relpath(d, root) for d in all_package_dirs]  # rel

        # include all dirs between root/ and root/a/b/c/e/ (eg. root/b/)
        all_package_dirs = [
            x
            for d in all_package_dirs
            for x in itertools.accumulate(d.split(os.sep), os.path.join)
        ]
        all_package_dirs = sorted(set(all_package_dirs))  # dedup

        init_files = [os.path.join(d, "__init__.py") for d in all_package_dirs]
        print(
            "Checking following files under root: {} ...\n{}".format(
                root, "\n".join(init_files)
            )
        )
        missing_init_files = [
            f for f in init_files if not os.path.isfile(os.path.join(root, f))
        ]
        self.assertTrue(
            len(missing_init_files) == 0,
            "Missing following __init__.py files:\n{}".format(
                "\n".join(missing_init_files)
            ),
        )


@fb_overwritable()
def no_complaints_skip_if(condition, reason):
    return unittest.skipIf(condition, reason)
