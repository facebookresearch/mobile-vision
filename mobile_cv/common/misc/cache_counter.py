#!/usr/bin/env python3

from collections import Counter
from datetime import datetime, timedelta
from enum import Enum


DEFAULT_LOG_FREQ = timedelta(minutes=30)


class CacheStat(Enum):
    TOTAL = 0
    READ_CACHED = 1
    READ_NOT_CACHABLE = 2

    @classmethod
    def cache_hit_rate(cls, counter):
        if cls.read_total(counter) == 0:
            return 0.0
        return cls.read_cached(counter) / cls.read_total(counter)

    @classmethod
    def read_total(cls, counter):
        return counter[cls.TOTAL]

    @classmethod
    def read_cached(cls, counter):
        return counter[cls.READ_CACHED]

    @classmethod
    def read_not_cachable(cls, counter):
        return counter[cls.READ_NOT_CACHABLE]

    @classmethod
    def get_stat_str(cls, counter):
        ret = (
            f"Hit rate {cls.cache_hit_rate(counter):.2f} "
            + f"({cls.read_cached(counter)} / {cls.read_total(counter)}), "
            + f"not cachable rate {cls.read_not_cachable(counter) / cls.read_total(counter):.2f} "
            + f"({cls.read_not_cachable(counter)} / {cls.read_total(counter)})."
        )
        return ret


class DownloadStat(Enum):
    TOTAL = 0
    FAILED = 1

    @classmethod
    def failure_rate(cls, counter):
        if counter[cls.TOTAL] == 0:
            return 0.0
        return counter[cls.FAILED] / counter[cls.TOTAL]

    @classmethod
    def get_stat_str(cls, counter):
        ret = (
            f"Download failure rate {cls.failure_rate(counter):.2f} "
            + f"({counter[cls.FAILED]} / {counter[cls.TOTAL]})."
        )
        return ret


class CacheCounter(object):
    def __init__(
        self,
        name,
        # cache_stat: Enum class with a `get_stat_str()` function
        cache_stat: type(Enum),
        log_func=None,
        log_freq: timedelta = DEFAULT_LOG_FREQ,
    ):
        self.name = name
        self.counter = Counter()
        self.cache_stat = cache_stat
        self.log_func = log_func
        self.log_freq = log_freq
        self.cur_time = None

    def add(self, stat):
        self.counter.update([stat])
        if stat == self.cache_stat.TOTAL and self._check_and_update_timer():
            self.log_stat()

    def _check_and_update_timer(self):
        if isinstance(self.log_freq, int):
            return self._check_int_freq()

        if self.cur_time is None:
            self.cur_time = datetime.now()
        cur_time = datetime.now()
        if (cur_time - self.cur_time) > self.log_freq:
            self.cur_time = cur_time
            return True
        return False

    def _check_int_freq(self):
        if self.counter[self.cache_stat.TOTAL] % self.log_freq == 0:
            return True
        return False

    def log_stat(self):
        if self.log_func is None:
            return
        prefix = ""
        if self.name is not None:
            prefix = f"{self.name}: "
        self.log_func(prefix + self.cache_stat.get_stat_str(self.counter))
