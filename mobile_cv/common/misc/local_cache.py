#!/usr/bin/env python3

import contextlib
import os
import tempfile
import time
from contextlib import contextmanager

import diskcache


class Timer(object):
    def __init__(self):
        self.reset()

    @property
    def average_time(self):
        return self.total_time / self.calls if self.calls > 0 else 0.0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.add(time.time() - self.start_time)
        if average:
            return self.average_time
        else:
            return self.diff

    def add(self, time_diff):
        self.diff = time_diff
        self.total_time += self.diff
        self.calls += 1

    def reset(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0


class TimerDict(object):
    def __init__(self):
        self.timers = {}

    @contextmanager
    def timed(self, name):
        self.tic(name)
        yield
        self.toc(name)

    def tic(self, name):
        if name not in self.timers:
            self.timers[name] = Timer()

        self.timers[name].tic()

    def toc(self, name, average=False):
        assert name in self.timers
        return self.timers[name].toc(average)

    def print(self):
        for name, item in self.timers.items():
            print(f"{name}: {item.average_time} ({item.total_time}, {item.calls})")


class LocalCache(object):
    @classmethod
    def Create(cls, prefix, num_shards, use_timer=False):
        tmp_dir = tempfile.gettempdir()
        cache_dir = os.path.join(tmp_dir, prefix)
        return cls(cache_dir, num_shards, use_timer)

    def __init__(self, cache_dir, num_shards, use_timer=False):
        self.cache_dir = cache_dir
        self.use_fanout = num_shards > 1
        if not self.use_fanout:
            self.cache = diskcache.Cache(
                cache_dir,
                timeout=60,
                size_limit=int(1e12),
                cull_limit=0,
                # statistics=True,
                eviction_policy="none",
            )
        else:
            self.cache = diskcache.FanoutCache(
                cache_dir,
                shards=num_shards,
                size_limit=int(1e12),
                cull_limit=0,
                # statistics=True,
                eviction_policy="none",
            )
        print(f"cache dir {cache_dir}, current cache size {len(self.cache)}")
        self.timer = TimerDict() if use_timer else None
        self.counter_total = 0
        self.counter_hit = 0
        self.conflict_hit = 0

        self.enable = True

    def set_enable(self, use):
        self.enable = use

    def load(self, load_func, path, *args, **kwargs):
        if not self.enable:
            return load_func(path, *args, **kwargs)

        assert isinstance(path, str), f"Invalid path type {path}"

        self.counter_total += 1

        with contextlib.ExitStack() as stack:
            if self.timer is not None:
                stack.enter_context(self.timer.timed(load_func.__name__))

            ret = self.cache.get(path, None, retry=True)
            if ret is not None:
                self.counter_hit += 1
                return ret

            ret = load_func(path, *args, **kwargs)
            # other process may write to `path` at this point, but may not be
            #  a big issue as it should not happen often
            suc = self.cache.add(path, ret, retry=True)
            if not suc:
                self.conflict_hit += 1
            return ret

    def print_stat(self, prefix):
        total = self.counter_total
        hit_rate = self.counter_hit / max(total, 1)
        conflict_rate = self.conflict_hit / max(total, 1)
        print(
            f"{prefix}: Cache hit rate {hit_rate} "
            f"({self.counter_hit} / {total}), "
            f"conflict rate {conflict_rate}, "
            f"volume {self.cache.volume() / 1024.0 / 1024.0 / 1024.0} G."
        )
        if self.timer:
            self.timer.print()
