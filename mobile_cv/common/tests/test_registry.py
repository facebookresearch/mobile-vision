#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import shutil
import sys
import tempfile
import unittest
import uuid

from mobile_cv.common.misc.registry import LazyRegisterable, Registry


def test_func():
    return "test_func"


TEST_LAZY_REGISTRY = Registry("test_lazy")


def _write_file(filename, content):
    os.makedirs(os.path.dirname(filename))
    with open(filename, "w") as f:
        f.write(content)


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.import_root = tempfile.mkdtemp(prefix="test_lazy_registration_")
        self.addCleanup(shutil.rmtree, self.import_root)
        sys.path.append(self.import_root)

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

    def test_lazy_registration_with_obj_name(self):
        package_name = f"random_package_{str(uuid.uuid4().hex)[:8]}"
        self.assertTrue(package_name not in sys.modules)

        # we can even do lazy register before the code exists
        TEST_LAZY_REGISTRY.register(
            "lazy_func1",
            LazyRegisterable(module=package_name, name="lazy_func1"),
        )

        _write_file(
            os.path.join(self.import_root, package_name, "__init__.py"),
            f"""
from {__package__}.test_registry import TEST_LAZY_REGISTRY
@TEST_LAZY_REGISTRY.register()
def lazy_func1():
    return "lazy_func1"
            """,
        )

        # resolve the lazy registered object
        lazy_func1 = TEST_LAZY_REGISTRY.get("lazy_func1")
        self.assertEqual(lazy_func1(), "lazy_func1")

    def test_lazy_registration_without_obj_name(self):
        """
        The registry might be used as Dict to hold arbitrary python object other than
        function/class, those object might not have __name__.
        """
        package_name = f"random_package_{str(uuid.uuid4().hex)[:8]}"
        self.assertTrue(package_name not in sys.modules)

        TEST_LAZY_REGISTRY.register(
            "arch1",
            LazyRegisterable(module=package_name),  # only module, no name
        )

        _write_file(
            os.path.join(self.import_root, package_name, "__init__.py"),
            f"""
from {__package__}.test_registry import TEST_LAZY_REGISTRY
MODEL_ARCH_FBNETV100 = {{
    "arch1": "fbnet50",
    "arch2": "fbnet100",
}}
TEST_LAZY_REGISTRY.register_dict(MODEL_ARCH_FBNETV100)
            """,
        )

        # arch2 is not lazy-registered
        with self.assertRaises(KeyError):
            TEST_LAZY_REGISTRY.get("arch2")
        self.assertEqual(TEST_LAZY_REGISTRY.get("arch1"), "fbnet50")
        # now arch2 is also registred during resolving arch1
        self.assertEqual(TEST_LAZY_REGISTRY.get("arch2"), "fbnet100")
