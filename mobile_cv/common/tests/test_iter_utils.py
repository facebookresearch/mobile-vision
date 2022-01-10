#!/usr/bin/env python3
import unittest
from dataclasses import dataclass
from typing import Any, List, NamedTuple

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

    def test_recursive_iter_send_int(self):
        values = {"k1": "v1", "k2": ["v2", 3, "v4"], "k3": 5}
        gt_values = {"k1": "v1", "k2": ["v2", 4, "v4"], "k3": 6}

        iters = iu.recursive_iterate(values, iter_types=int)
        for x in iters:
            iters.send(x + 1)
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
            iu.recursive_iterate(
                values, seq_check_func=lambda x: iu.is_seq(x) and not _is_int_list(x)
            )
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
                map_check_func=lambda x: iu.is_map(x) and not _is_int_dict_key(x),
            )
        )
        self.assertEqual(val_list, [{1: 1, 2: 2, 3: 3}])

    def test_named_tuple(self):
        class Type(NamedTuple):
            field1: str
            field2: int

        space = [Type("t1", 2), Type("t2", 3), 3]

        # NamedTuple will be treated as a sequence by default
        ret = list(
            iu.recursive_iterate(
                space,
                iter_types=Type,
                map_check_func=lambda x: iu.is_map(x, strict=True),
                seq_check_func=lambda x: iu.is_seq(x, strict=True),
            )
        )
        self.assertEqual(ret, [space[0], space[1]])

        ret = list(iu.recursive_iterate(space, iter_types=Type, yield_container=True))
        self.assertEqual(ret, [space[0], space[1]])

    def test_yield_container(self):
        data = {
            "ab": {"cd": 123, "ef": "ga"},
            "cd": [1, 2, 3],
            "j": 6,
        }
        # basic case
        ret = list(iu.recursive_iterate(data, yield_container=True))
        self.assertEqual(
            ret, [123, "ga", {"cd": 123, "ef": "ga"}, 1, 2, 3, [1, 2, 3], 6, data]
        )

        # yield with name
        ret = list(iu.recursive_iterate(data, yield_name=True, yield_container=True))
        self.assertEqual(
            ret,
            [
                ("ab.cd", 123),
                ("ab.ef", "ga"),
                ("ab", {"cd": 123, "ef": "ga"}),
                ("cd.0", 1),
                ("cd.1", 2),
                ("cd.2", 3),
                ("cd", [1, 2, 3]),
                ("j", 6),
                ("", data),
            ],
        )

    def test_yield_container_send(self):
        data = {
            "ab": {"cd": 123, "ef": "ga"},
            "cd": [1, 2, 3],
            "j": 6,
        }
        # yield container with send
        riter = iu.recursive_iterate(data, yield_container=True)
        for item in riter:
            if iu.is_container(item):
                sent_item = riter.sent_value()
                if item == data:
                    # for the last item (the full object), replace the sent object
                    riter.send(sent_item)
                else:
                    # replace the container with its first item
                    if iu.is_map(sent_item):
                        riter.send(tuple(list(sent_item.items())[0]))
                    else:
                        riter.send(sent_item[0])
            else:
                riter.send(item)
        self.assertEqual(
            riter.value,
            {"ab": ("cd", 123), "cd": 1, "j": 6},
        )

    def test_linearize_remap(self):
        """Convert a specific types of objects to a list and reconstruct the list
        to the original structure"""

        @dataclass
        class Type1(object):
            name: str

        @dataclass
        class Type2(object):
            name: str

        adict = {
            "ab": "cd",
            "t1": Type1("t1"),
            "t2": Type2("t2"),
            "dict": {
                "t3": Type1("t3"),
                "list": ["val", Type2("t4"), Type1("t5"), 10],
                "value": "value",
            },
        }
        # convert to a list
        ret = list(iu.recursive_iterate(adict, iter_types=(Type1, Type2)))
        self.assertEqual(
            ret,
            [
                adict["t1"],
                adict["t2"],
                adict["dict"]["t3"],
                adict["dict"]["list"][1],
                adict["dict"]["list"][2],
            ],
        )

        # process on the list
        ret = [x.name + str(idx) for idx, x in enumerate(ret)]

        # reconstruct the original structure with the new values
        riter = iu.recursive_iterate(adict, iter_types=(Type1, Type2))
        # riter needs to come first in zip
        for _value, new_value in zip(riter, ret):
            # replace value with new_value
            riter.send(new_value)
        out = riter.value
        self.assertEqual(
            out,
            {
                "ab": "cd",
                "t1": "t10",
                "t2": "t21",
                "dict": {
                    "t3": "t32",
                    "list": ["val", "t43", "t54", 10],
                    "value": "value",
                },
            },
        )

    def test_linearize_remap_nested(self):
        """Convert a specific types of objects to a list and reconstruct the list
        to the original structure"""

        @dataclass
        class Type1(object):
            name: str

            def process(self, _):
                return self.name + "_t1"

        @dataclass
        class Type2(object):
            name: str

            def process(self, _):
                return self.name + "_t2"

        @dataclass
        class TypeChoice(object):
            name: str
            data: List[Any]

            def __iter__(self):
                return iter(self.data)

            def process(self, data):
                return data[1]

        adict = {
            "ab": "cd",
            "t3": Type1("t3"),
            "choice": TypeChoice("choice", ["val", Type2("t4"), Type1("t5"), 10]),
        }
        # convert to a list
        ret = list(
            iu.recursive_iterate(
                adict,
                iter_types=(Type1, Type2, TypeChoice),
                seq_check_func=lambda x: isinstance(x, (list, tuple, TypeChoice)),
                yield_container=True,
            )
        )
        self.assertEqual(
            ret,
            [
                adict["t3"],
                adict["choice"].data[1],
                adict["choice"].data[2],
                adict["choice"],
            ],
        )

        # reconstruct the original structure with the new values
        riter = iu.recursive_iterate(
            adict,
            iter_types=(Type1, Type2, TypeChoice),
            seq_check_func=lambda x: isinstance(x, (list, tuple, TypeChoice)),
            yield_container=True,
        )
        for _value, new_value in zip(riter, ret):
            riter.send(new_value.process(riter.sent_value()))

        out = riter.value
        self.assertEqual(
            out,
            {
                "ab": "cd",
                "t3": "t3_t1",
                "choice": "t4_t2",
            },
        )

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
