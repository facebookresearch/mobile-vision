#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import sys
import unittest

from mobile_cv.common.misc.file_utils import make_temp_directory
from mobile_cv.common.misc.py import dynamic_import, import_file


class TestMisc(unittest.TestCase):
    def test_import_from_file(self):
        with make_temp_directory("test_import_from_file") as test_dir:
            filename = os.path.join(test_dir, "my_module.py")
            with open(filename, "w") as f:
                f.write("ANSWER = 42\n")

            my_module = import_file("my_module", filename)
            self.assertEqual(my_module.ANSWER, 42)

    def test_dynamic_import(self):
        func = dynamic_import("mobile_cv.common.misc.py.dynamic_import")
        self.assertEqual(func, dynamic_import)

    def test_dynamic_import_forward_circular_dependency(self):
        with make_temp_directory("test_dynamic_import") as test_dir:
            os.makedirs(os.path.join(test_dir, "test_package/lib"), exist_ok=True)
            with open(os.path.join(test_dir, "test_package/__init__.py"), "w") as f:
                f.write("")
            with open(os.path.join(test_dir, "test_package/lib/__init__.py"), "w") as f:
                f.write("import test_package.lib.a\n")
            with open(os.path.join(test_dir, "test_package/lib/a.py"), "w") as f:
                f.write(
                    "from mobile_cv.common.misc.py import dynamic_import\n"
                    'bar = dynamic_import("test_package.lib.b.foo")\n'
                )
            with open(os.path.join(test_dir, "test_package/lib/b.py"), "w") as f:
                f.write("foo = 42\n")

            sys.path.append(test_dir)
            from test_package.lib.a import bar

            self.assertEqual(bar, 42)
