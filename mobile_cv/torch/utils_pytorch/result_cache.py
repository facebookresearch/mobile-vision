import os
from typing import Any, Optional

import torch
from mobile_cv.torch.utils_pytorch import comm


class ResultCache(object):
    def __init__(self, cache_dir: Optional[str], cache_name: str, logger, path_manager):
        """A utility class to handle save/load cache data across processes"""
        self.cache_dir = cache_dir
        self.cache_name = cache_name
        self.logger = logger
        self.path_manager = path_manager

    @property
    def cache_file(self):
        if self.cache_dir is None:
            return None
        return os.path.join(
            self.cache_dir, f"_result_cache_{self.cache_name}.{comm.get_rank()}.pkl"
        )

    def has_cache(self):
        return self.path_manager.isfile(self.cache_file)

    def load(self, gather=False):
        """
        Load cache results.
        gather (bool): gather cache results arcoss ranks to a list
        """
        if self.cache_file is None or not self.path_manager.exists(self.cache_file):
            ret = None
        else:
            with self.path_manager.open(self.cache_file, "rb") as fp:
                ret = torch.load(fp)
            if self.logger is not None:
                self.logger.info(f"Loaded from checkpoint {self.cache_file}")

        if gather:
            ret = comm.all_gather(ret)
        return ret

    def save(self, data: Any):
        if self.cache_file is None:
            return

        self.path_manager.mkdirs(os.path.dirname(self.cache_file))
        with self.path_manager.open(self.cache_file, "wb") as fp:
            torch.save(data, fp)
        if self.logger is not None:
            self.logger.info(f"Saved checkpoint to {self.cache_file}")
