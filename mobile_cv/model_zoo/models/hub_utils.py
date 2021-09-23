#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import errno
import hashlib
import logging
import os
from types import TracebackType
from typing import Optional, Type

import torch


logger = logging.getLogger(__name__)


def download_file(url, model_dir=None, progress=True):
    """Download url to `model_dir`
    Append hash of the url to the filename to make it unique.
    """
    # Issue warning to move data if old env is set
    if os.getenv("TORCH_MODEL_ZOO"):
        logger.warning(
            "TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead"
        )

    if model_dir is None:
        torch_home = torch.hub._get_torch_home()
        model_dir = os.path.join(torch_home, "checkpoints")

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = torch.hub.urlparse(url)
    filename = os.path.basename(parts.path)
    filename = hashlib.sha256(url.encode("utf-8")).hexdigest() + "_" + filename
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        logger.info('Downloading: "{}" to {}\n'.format(url, cached_file))
        torch.hub.download_url_to_file(url, cached_file, None, progress=progress)

    return cached_file


class DictModifier(object):
    """Convenience class to modify a dict when entered, restore to original
    after it is exited
    """

    def __init__(self, dict_to_modify, new_dict):
        """dict_to_modify: the dict that it's content will be modify by new_dict
                        and recover later
        new_dict: the dict that it will override dict_to_modify
        """
        self.dict_to_modify = dict_to_modify
        self.new_dict = new_dict

    def __enter__(self):
        # backup current envs
        self.old_vars = {
            x: self.dict_to_modify[x]
            for x in self.new_dict.keys()
            if x in self.dict_to_modify
        }
        # additional variables that are written to self.dict_to_modify
        self.extra_vars = [x for x in self.new_dict if x not in self.old_vars.keys()]
        # apply new environments
        for x, val in self.new_dict.items():
            self.dict_to_modify[x] = val

    def __exit__(
        self,
        atype: Optional[Type[BaseException]],
        avalue: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        # pyre-fixme[16]: `DictModifier` has no attribute `old_vars`.
        for x, y in self.old_vars.items():
            self.dict_to_modify[x] = y
        # pyre-fixme[16]: `DictModifier` has no attribute `extra_vars`.
        for x in self.extra_vars:
            del self.dict_to_modify[x]


def pretrained_download(builder):
    """Convenience function to download pretrained weights from https"""

    def func(*args, **kwargs):
        # no need to handle for oss version
        return builder(*args, **kwargs)

    return func
