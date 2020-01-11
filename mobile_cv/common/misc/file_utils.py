#!/usr/bin/env python3

import contextlib
import shutil
import tempfile


@contextlib.contextmanager
def make_temp_directory(prefix):
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)
