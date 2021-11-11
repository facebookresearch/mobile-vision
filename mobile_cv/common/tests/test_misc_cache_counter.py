#!/usr/bin/env python3
import time
import unittest
from datetime import timedelta

from mobile_cv.common.misc.cache_counter import CacheCounter, CacheStat, DownloadStat


class TestCacheCounter(unittest.TestCase):
    def test_count(self):
        call_counter = 0

        def log_func(log_str):
            nonlocal call_counter
            call_counter += 1

        cc = CacheCounter(
            "test",
            DownloadStat,
            log_func=log_func,
            log_freq=3,
        )

        for idx in range(9):
            cc.add(DownloadStat.TOTAL)
            if idx % 2 == 0:
                cc.add(DownloadStat.FAILED)

        self.assertEqual(call_counter, 3)
        self.assertEqual(cc.counter[DownloadStat.TOTAL], 9)
        self.assertEqual(cc.counter[DownloadStat.FAILED], 5)
        self.assertEqual(DownloadStat.failure_rate(cc.counter), 5 / 9)

    def test_count_timer(self):
        call_counter = 0

        def log_func(log_str):
            nonlocal call_counter
            call_counter += 1

        cc = CacheCounter(
            "test",
            DownloadStat,
            log_func=log_func,
            log_freq=timedelta(seconds=2),
        )

        for idx in range(9):
            cc.add(DownloadStat.TOTAL)
            if idx % 2 == 0:
                cc.add(DownloadStat.FAILED)
            time.sleep(1)

        self.assertEqual(call_counter, 4)
        self.assertEqual(cc.counter[DownloadStat.TOTAL], 9)
        self.assertEqual(cc.counter[DownloadStat.FAILED], 5)
        self.assertEqual(DownloadStat.failure_rate(cc.counter), 5 / 9)

    def test_count_cache_stat(self):
        call_counter = 0

        def log_func(log_str):
            nonlocal call_counter
            call_counter += 1

        cc = CacheCounter(
            "test",
            CacheStat,
            log_func=log_func,
            log_freq=3,
        )

        for idx in range(9):
            cc.add(CacheStat.TOTAL)
            if idx % 2 == 0:
                cc.add(CacheStat.READ_CACHED)

        self.assertEqual(call_counter, 3)
        self.assertEqual(cc.counter[CacheStat.TOTAL], 9)
        self.assertEqual(cc.counter[CacheStat.READ_CACHED], 5)
        self.assertEqual(CacheStat.cache_hit_rate(cc.counter), 5 / 9)
