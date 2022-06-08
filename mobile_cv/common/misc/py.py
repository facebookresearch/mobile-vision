#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

""" python utils """

import logging
import os
import pdb
import sys
import threading
import traceback
from unittest import mock

import cloudpickle

try:
    import fcntl
except ImportError:
    # fcntl is not available on windows, skip the import since it's for using the file
    # lock from MultiprocessingPdb, which is an optional feature.
    pass


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


# Copied from detectron2/utils/serialize.py
class PicklableWrapper(object):
    """
    Wrap an object to make it more picklable, note that it uses
    heavy weight serialization libraries that are slower than pickle.
    It's best to use it only on closures (which are usually not picklable).

    This is a simplified version of
    https://github.com/joblib/joblib/blob/master/joblib/externals/loky/cloudpickle_wrapper.py
    """

    def __init__(self, obj):
        while isinstance(obj, PicklableWrapper):
            # Wrapping an object twice is no-op
            obj = obj._obj
        self._obj = obj

    def __reduce__(self):
        s = cloudpickle.dumps(self._obj)
        return cloudpickle.loads, (s,)

    def __call__(self, *args, **kwargs):
        return self._obj(*args, **kwargs)

    def __getattr__(self, attr):
        # Ensure that the wrapped object can be used seamlessly as the previous object.
        if attr not in ["_obj"]:
            return getattr(self._obj, attr)
        return getattr(self, attr)


class MoreMagicMock(mock.MagicMock):
    """
    Subclass MagicMock to provide more features, such as: comparison, inheritance, ...
    """

    def __init__(self, *args, **kwargs):
        # we don't pass `args` and `kwargs` so that it starts with a fresh Mock object
        super(MoreMagicMock, self).__init__()

        # resolve the `mocked_obj_info`
        if (
            # Speical handling for inheritance: B = MoreMagicMock(); Class A(B)
            # is equivalent to calling: type(B)("A", (B,), {...})
            len(args) == 3
            and isinstance(args[0], str)
            and isinstance(args[1], tuple)
            and isinstance(args[1][0], MoreMagicMock)
            and isinstance(args[2], dict)
        ):
            class_name = args[0]
            assert "__module__" in args[2], args[2]
            assert "__qualname__" in args[2], args[2]
            self._mocked_obj_info = {
                "__name__": class_name,
                "__module__": args[2]["__module__"],
                "__qualname__": args[2]["__qualname__"],
            }
        else:
            self._mocked_obj_info = None

        # modify the default implementation for some of the "Magic Methods"
        # https://docs.python.org/3/library/unittest.mock.html#magic-mock
        self.__lt__ = self
        self.__gt__ = self
        self.__le__ = self
        self.__ge__ = self

        # support some commonly used dunber methods (those are just ordinary member
        # variables/methods that happen to have "__", not "Magic Methods")
        self.__version__ = self

    @property
    def mocked_obj_info(self):
        return self._mocked_obj_info
