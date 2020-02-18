#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
from os import path

from setuptools import find_packages, setup


def get_version():
    init_py_path = path.join(
        path.abspath(path.dirname(__file__)), "mobile_cv", "__init__.py"
    )
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][
        0
    ]
    version = version_line.split("=")[-1].strip().strip("'\"")

    # Used by CI to build nightly packages. Users should never use it.
    # To build a nightly wheel, run:
    # BUILD_NIGHTLY=1 python setup.py sdist
    if os.getenv("BUILD_NIGHTLY", "0") == "1":
        from datetime import datetime

        date_str = datetime.today().strftime("%y%m%d")
        version = version + ".dev" + date_str

        new_init_py = [l for l in init_py if not l.startswith("__version__")]
        new_init_py.append('__version__ = "{}"\n'.format(version))
        with open(init_py_path, "w") as f:
            f.write("".join(new_init_py))
    return version


setup(
    name="mobile_cv",
    version=get_version(),
    author="Mobile Vision",
    license="CC BY-NC",
    url="https://github.com/facebookresearch/mobile-vision",
    description="FBNet model builder and model zoo",
    python_requires=">=3.6",
    install_requires=[],
    extras_require={},
    packages=find_packages(),
)
