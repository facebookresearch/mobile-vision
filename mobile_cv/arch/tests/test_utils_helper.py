#!/usr/bin/env python3

import unittest

import mobile_cv.arch.utils.helper as helper


class TestUtilsHelper(unittest.TestCase):
    def test_filter_kwargs(self):
        def _func1(arg1, arg2):
            pass

        res_args = helper.filter_kwargs(
            _func1, {"arg1": 2, "arg2": 3, "non_args": 4}
        )
        self.assertEqual(res_args, {"arg1": 2, "arg2": 3})

        def _func2(arg1, arg2, *, arg3=4, arg4=4):
            pass

        res_args = helper.filter_kwargs(
            _func2, {"arg1": 2, "arg2": 3, "non_args": 4, "arg3": 5}
        )
        self.assertEqual(res_args, {"arg1": 2, "arg2": 3, "arg3": 5})

        def _func3(arg1, arg2, *, arg3=4, arg4=4, **kwargs):
            pass

        res_args = helper.filter_kwargs(
            _func3, {"arg1": 2, "arg2": 3, "non_args": 4, "arg3": 5}
        )
        self.assertEqual(
            res_args, {"arg1": 2, "arg2": 3, "non_args": 4, "arg3": 5}
        )

        def _func4(arg1, arg2, arg3=4, arg4=4, *args, **kwargs):
            pass

        res_args = helper.filter_kwargs(
            _func4, {"arg1": 2, "arg2": 3, "non_args": 4, "arg3": 5}
        )
        self.assertEqual(
            res_args, {"arg1": 2, "arg2": 3, "non_args": 4, "arg3": 5}
        )
