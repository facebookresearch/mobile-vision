#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import logging

from caffe2.python import workspace


logger = logging.getLogger(__name__)


class ScopedWS(object):
    def __init__(self, ws_name, is_reset, is_cleanup=False):
        self.ws_name = ws_name
        self.is_reset = is_reset
        self.is_cleanup = is_cleanup
        self.org_ws = ""

    def __enter__(self):
        self.org_ws = workspace.CurrentWorkspace()
        if self.ws_name is not None:
            workspace.SwitchWorkspace(self.ws_name, True)
        if self.is_reset:
            workspace.ResetWorkspace()

        return workspace

    def __exit__(self, *args):
        if self.is_cleanup:
            workspace.ResetWorkspace()
        if self.ws_name is not None:
            workspace.SwitchWorkspace(self.org_ws)


def fetch_any_blob(name):
    bb = None
    try:
        bb = workspace.FetchBlob(name)
    except TypeError:
        bb = workspace.FetchInt8Blob(name)
    except Exception as e:
        logger.error("Get blob {} error: {}".format(name, e))

    return bb
