#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import contextlib
import shutil
import tempfile
import zipfile

from iopath.common.file_io import g_pathmgr


@contextlib.contextmanager
def make_temp_directory(prefix):
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)


def local_recompress_export(inputfile: str, outputfile: str) -> None:
    with zipfile.ZipFile(inputfile, "r") as zip_input:
        with zipfile.ZipFile(
            outputfile,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as zip_output:
            for filename in zip_input.namelist():
                with zip_input.open(filename, "r") as f_input:
                    data = f_input.read()
                with zip_output.open(filename, "w") as f_output:
                    f_output.write(data)


def recompress_export(inputfile: str, outputfile: str) -> None:
    local_inputfile = g_pathmgr.get_local_path(inputfile, force=True)
    with tempfile.TemporaryDirectory() as tempdir:
        local_outputfile = f"{tempdir}/out.zip"
        local_recompress_export(local_inputfile, local_outputfile)
        g_pathmgr.copy_from_local(local_outputfile, outputfile)
