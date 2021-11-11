#!/usr/bin/env python3
import unittest

import mobile_cv.common.misc.iter_utils as iu


class TestIterUtils(unittest.TestCase):
    def test_recursive_iter_simple(self):
        self.assertEqual(next(iter(iu.recursive_iterate(1))), 1)
        self.assertEqual(next(iter(iu.recursive_iterate("str"))), "str")
        self.assertEqual(list(iu.recursive_iterate(1)), [1])
        self.assertEqual(list(iu.recursive_iterate("str")), ["str"])
        # special cases for empty list and dict, the function will return an
        # empty iterable
        self.assertEqual(list(iu.recursive_iterate([])), [])
        self.assertEqual(list(iu.recursive_iterate({})), [])

    def test_recursive_iter_simple_no_wait_on_send(self):
        self.assertEqual(list(iu.recursive_iterate(None, wait_on_send=False)), [None])
        self.assertEqual(
            list(iu.recursive_iterate([1, 2, None], wait_on_send=False)),
            [1, 2, None],
        )

    def test_recursive_iter_dict(self):
        values = {"k1": "v1", "k2": ["v2", 3, "v4"], "k3": 5}
        val_list = list(iu.recursive_iterate(values))
        self.assertEqual(val_list, ["v1", "v2", 3, "v4", 5])

        val_list = list(iu.recursive_iterate(values, iter_types=str))
        self.assertEqual(val_list, ["v1", "v2", "v4"])

    def test_recursive_iter_dict_with_name(self):
        values = {"k1": "v1", "k2": ["v2", 3, "v4"], "k3": 5}
        val_list = list(iu.recursive_iterate(values, yield_name=True))
        self.assertEqual(
            val_list,
            [("k1", "v1"), ("k2.0", "v2"), ("k2.1", 3), ("k2.2", "v4"), ("k3", 5)],
        )

        val_list = list(iu.recursive_iterate(values, iter_types=str, yield_name=True))
        self.assertEqual(val_list, [("k1", "v1"), ("k2.0", "v2"), ("k2.2", "v4")])

    def test_recursive_iter_send(self):
        values = {"k1": "v1", "k2": ["v2", 3, "v4"], "k3": 5}
        gt_values = {"k1": "v1_ret", "k2": ["v2_ret", 4, "v4_ret"], "k3": 6}

        iters = iu.recursive_iterate(values)
        for x in iters:
            iters.send(x + "_ret" if isinstance(x, str) else x + 1)
        result = iters.value
        self.assertEqual(result, gt_values)

    def test_recursive_iter_send_None(self):
        values = {"k1": "v1", "k2": ["v2", 3, "v4"], "k3": 5}
        gt_values = {
            "k1": "v1_ret",
            "k2": ["v2_ret", None, "v4_ret"],
            "k3": None,
        }

        iters = iu.recursive_iterate(values, wait_on_send=True)
        for x in iters:
            iters.send(x + "_ret" if isinstance(x, str) else None)
        result = iters.value
        self.assertEqual(result, gt_values)

    def test_recursive_iter_send_with_name(self):
        values = {"k1": "v1", "k2": ["v2", 3, "v4"], "k3": 5}
        gt_values = {
            "k1": "v1_ret",
            "k2": ["v2_ret", 4, "v4_ret"],
            "k3": 7,
        }

        iters = iu.recursive_iterate(values, wait_on_send=True, yield_name=True)
        for name, x in iters:
            iters.send(
                x + "_ret" if isinstance(x, str) else (x + 2 if name == "k3" else x + 1)
            )
        result = iters.value
        self.assertEqual(result, gt_values)

    def test_recursive_iter_seq_check_func(self):
        def _is_int_list(x):
            return isinstance(x, list) and all(isinstance(y, int) for y in x)

        values = [{"k1": [1, 2, 3], "k2": ["v2", 3, "v4"], "k3": 5}]
        # List of ints are not considered as a list
        val_list = list(
            iu.recursive_iterate(values, seq_check_func=lambda x: not _is_int_list(x))
        )
        self.assertEqual(val_list, [[1, 2, 3], "v2", 3, "v4", 5])

    def test_recursive_iter_map_check_func(self):
        def _is_int_dict_key(x):
            return isinstance(x, dict) and all(isinstance(yk, int) for yk in x.keys())

        values = [{"k1": {1: 1, 2: 2, 3: 3}, "k2": ["v2", 3, "v4"], "k3": 5}]
        # dict where keys are ints are not considered as a map
        val_list = list(
            iu.recursive_iterate(
                values,
                iter_types=dict,
                map_check_func=lambda x: not _is_int_dict_key(x),
            )
        )
        self.assertEqual(val_list, [{1: 1, 2: 2, 3: 3}])

    def test_paired(self):
        self.assertIsInstance(iu.PairedDict({}, {}), iu.cabc.Mapping)
        self.assertIsInstance(iu.PairedSeq([], []), iu.cabc.Sequence)

        lhs = {"alist": [1, 2, 3], "bdict": {"c": "d", "e": ["f", "g", "h"]}}
        rhs = {"alist": ["1", "2", "3"], "bdict": {"c": 4, "e": [5, 6, 7]}}
        paired = iu.create_pair(lhs, rhs)
        self.assertIsInstance(paired, iu.PairedDict)

        paired_list = paired["alist"]
        self.assertIsInstance(paired_list, iu.PairedSeq)

        paired_obj = paired_list[0]
        self.assertIsInstance(paired_obj, iu.Pair)

        iter = iu.recursive_iterate(paired)
        for x in iter:
            iter.send(x.to_tuple())

        merged_gt = {
            "alist": [(1, "1"), (2, "2"), (3, "3")],
            "bdict": {"c": ("d", 4), "e": [("f", 5), ("g", 6), ("h", 7)]},
        }
        self.assertEqual(iter.value, merged_gt)


if __name__ == "__main__":
    unittest.main()
