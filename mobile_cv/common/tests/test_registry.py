#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import sys
import unittest
import uuid

from mobile_cv.common.misc.file_utils import make_temp_directory
from mobile_cv.common.misc.registry import Registry


def test_func():
    return "test_func"


TEST_LAZY_REGISTRY = Registry("test_lazy")


class TestRegistry(unittest.TestCase):
    def test_registry(self):
        REG_LIST = Registry("test")
        REG_LIST.register("test_func", test_func)

        @REG_LIST.register()
        def test_func1():
            return "test_func_1"

        self.assertEqual(len(REG_LIST.get_names()), 2)

        out = REG_LIST.get("test_func")()
        self.assertEqual(out, "test_func")
        out1 = REG_LIST.get("test_func1")()
        self.assertEqual(out1, "test_func_1")

    def test_lazy_registration(self):
        random_package_name = f"random_package_{str(uuid.uuid4().hex)[:8]}"
        self.assertTrue(random_package_name not in sys.modules)

        # we can even do lazy register before the code exists
        TEST_LAZY_REGISTRY.register("lazy_func1", f"{random_package_name}.lazy_func1")

        with make_temp_directory("test_lazy_registration") as root:
            os.makedirs(os.path.join(root, random_package_name))
            with open(os.path.join(root, random_package_name, "__init__.py"), "w") as f:
                f.write(
                    f"""
from {__package__}.test_registry import TEST_LAZY_REGISTRY
@TEST_LAZY_REGISTRY.register()
def lazy_func1():
    return "lazy_func1"
"""
                )
            # make test files import-able
            sys.path.append(root)

            # resolve the lazy registered object
            lazy_func1 = TEST_LAZY_REGISTRY.get("lazy_func1")
            self.assertEqual(lazy_func1(), "lazy_func1")
