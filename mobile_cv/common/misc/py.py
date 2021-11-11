#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

""" python utils """

import fcntl
import logging
import os
import pdb
import sys
import threading
import traceback


logger = logging.getLogger(__name__)


# from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path  # NOQA
def import_file(module_name, file_path, make_importable=False):
    import importlib
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[module_name] = module
    return module


def dynamic_import(obj_full_name):
    """
    Dynamically import an object (class or function or global variable).

    Args:
        obj_full_name: full name of the object, eg. ExampleClass/example_foo is defined
        inside a/b.py, the obj_full_name is "a.b.ExampleClass" or "a.b.example_foo".
    Returns:
        The imported object.
    """
    import importlib
    import pydoc

    ret = pydoc.locate(obj_full_name)
    if ret is None:
        # pydoc.locate imports in forward order, sometimes causing circular import,
        # fallback to use importlib if pydoc.locate doesn't work
        module_name, obj_name = obj_full_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        ret = getattr(module, obj_name)
    return ret


class FolderLock:
    """
    Locks a folder to synchronize potentially conflicting operations.
    The default mode of operation is to wait for the lock. If fail_fast
    flag is set, we fail immediately if the lock cannot be acquired.
    """

    def __init__(self, folder, fail_fast=False) -> None:
        self.folder = folder
        self.thread_lock = threading.Lock()
        self.fail_fast = fail_fast

    def __enter__(self) -> "FolderLock":
        logger.info("Acquiring lock for " + self.folder)
        self.handle = open(self.folder + "/lockfile", "w+")
        operations = fcntl.LOCK_EX
        if self.fail_fast:
            operations |= fcntl.LOCK_NB

        fcntl.lockf(self.handle, operations)
        self.thread_lock.acquire()
        return self

    def __exit__(self, *args) -> None:
        logger.info("Releasing lock for " + self.folder)
        global _CREATED_FILES
        _CREATED_FILES = set()
        self.thread_lock.release()
        self.handle.close()


# https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess  # noqa
class MultiprocessingPdb(pdb.Pdb):

    _stdin_fd = sys.stdin.fileno()
    _stdin = None

    def __init__(self, lock) -> None:
        pdb.Pdb.__init__(self, nosigint=True)
        self._stdin_lock = lock

    def _cmdloop(self):
        stdin_bak = sys.stdin
        with self._stdin_lock:
            logger.info("Entered MultiprocessingPdb (first come first serve)")
            try:
                if not self._stdin:
                    self._stdin = os.fdopen(self._stdin_fd)
                sys.stdin = self._stdin
                self.cmdloop()
            finally:
                sys.stdin = stdin_bak


def post_mortem_if_fail(pdb_=None):
    def post_mortem(pdb_, t=None):
        # handling the default
        if t is None:
            # sys.exc_info() returns (type, value, traceback) if an exception is
            # being handled, otherwise it returns None
            t = sys.exc_info()[2]
        if t is None:
            raise ValueError(
                "A valid traceback must be passed if no " "exception is being handled"
            )

        pdb_ = pdb_ or pdb.Pdb()
        pdb_.reset()
        pdb_.interaction(None, t)

    def decorator(func):
        def new_func(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                extype, value, tb = sys.exc_info()
                traceback.print_exc()
                post_mortem(pdb_, tb)

        return new_func

    return decorator
