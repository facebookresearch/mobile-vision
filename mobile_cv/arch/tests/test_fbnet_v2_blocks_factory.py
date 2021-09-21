#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import unittest
from typing import List, Tuple, Union

import mobile_cv.arch.fbnet_v2.fbnet_builder as fbnet_builder
import mobile_cv.common.misc.iter_utils as iu
import numpy as np
import torch


def _create_single_input(input_dims: List[int], device: str):
    assert isinstance(input_dims, list)
    nchw = np.prod(input_dims)
    ret = (torch.arange(nchw, dtype=torch.float32) - (nchw / 2.0)) / nchw
    ret = ret.reshape(*input_dims)
    ret = ret.to(device)
    return ret


def _create_input(input_dims: Union[List[int], Tuple[List[int], ...]], device: str):
    """Could be List or Tuple of Lists"""
    if isinstance(input_dims, tuple):
        return tuple(_create_single_input(x, device) for x in input_dims)
    assert isinstance(input_dims, list)
    return _create_single_input(input_dims, device)


def _compare_output_shape(self, outputs, gt_shapes):
    for item in iu.recursive_iterate(iu.create_pair(outputs, gt_shapes)):
        self.assertEqual(item.lhs.shape, torch.Size(item.rhs))


def _compare_outputs(outputs, gt_outputs, error_msg):
    riter = iu.recursive_iterate(iu.create_pair(outputs, gt_outputs))
    for item in riter:
        np.testing.assert_allclose(
            item.lhs, item.rhs, rtol=0, atol=1e-4, err_msg=error_msg
        )


def _to_str(data):
    if isinstance(data, str):
        return data
    if isinstance(data, list):
        return "[" + ", ".join(_to_str(x) for x in data) + "]"
    if isinstance(data, tuple):
        return "(" + ", ".join(_to_str(x) for x in data) + ")"
    return str(data)


OP_CFGS_DEFAULT = {
    "in_channels": 4,
    "out_channels": 4,
    "stride": 2,
    "_inputs_": [1, 4, 4, 4],
    "_gt_shape_": [1, 4, 2, 2],
    "bias": True,
}


OP_CFGS = {
    "default": OP_CFGS_DEFAULT,
    "conv_cfg": {**OP_CFGS_DEFAULT, "kernel_size": 3, "padding": 1, "bias": False},
    "irf_cfg": {**OP_CFGS_DEFAULT, "expansion": 4, "bias": False},
    "irf_cfg_sefc": {
        **OP_CFGS_DEFAULT,
        "expansion": 4,
        "bias": False,
        "se_args": "se_fc",
        "width_divisor": 1,
    },
    "irf_cfg_seconvhsig": {
        **OP_CFGS_DEFAULT,
        "expansion": 4,
        "bias": False,
        "width_divisor": 1,
    },
    ("frac_ds", "default"): {
        "in_channels": 4,
        "out_channels": 4,
        "stride": 1,
        "si": 1,
        "so": 1,
        "_inputs_": [1, 4, 4, 4],
        "_gt_shape_": [1, 4, 4, 4],
    },
    ("frac_ds2", "default"): {
        "in_channels": 4,
        "out_channels": 4,
        "stride": 1,
        "si": 1,
        "so": 1,
        "_inputs_": [1, 4, 4, 4],
        "_gt_shape_": [1, 4, 4, 4],
    },
    ("ir2dp1_k3", "default"): {
        "in_channels": 3,
        "out_channels": 2,
        "stride": 1,
        "_inputs_": [1, 3, 2, 2, 2],
        "_gt_shape_": [1, 2, 2, 2, 2],
    },
    ("ir2dp1_k5", "default"): {
        "in_channels": 3,
        "out_channels": 2,
        "stride": 1,
        "_inputs_": [1, 3, 2, 2, 2],
        "_gt_shape_": [1, 2, 2, 2, 2],
    },
    ("ir3d", "default"): {
        "in_channels": 3,
        "out_channels": 2,
        "stride": 1,
        "_inputs_": [1, 3, 2, 2, 2],
        "_gt_shape_": [1, 2, 2, 2, 2],
    },
    ("ir3d_k3", "default"): {
        "in_channels": 3,
        "out_channels": 2,
        "stride": 1,
        "_inputs_": [1, 3, 2, 2, 2],
        "_gt_shape_": [1, 2, 2, 2, 2],
    },
    ("ir3d_k133", "default"): {
        "in_channels": 3,
        "out_channels": 2,
        "stride": 1,
        "_inputs_": [1, 3, 2, 2, 2],
        "_gt_shape_": [1, 2, 2, 2, 2],
    },
    ("ir3d_k5", "default"): {
        "in_channels": 3,
        "out_channels": 2,
        "stride": 1,
        "_inputs_": [1, 3, 2, 2, 2],
        "_gt_shape_": [1, 2, 2, 2, 2],
    },
    ("ir3d_k155", "default"): {
        "in_channels": 3,
        "out_channels": 2,
        "stride": 1,
        "_inputs_": [1, 3, 2, 2, 2],
        "_gt_shape_": [1, 2, 2, 2, 2],
    },
    ("ir3d_pool", "default"): {
        "in_channels": 3,
        "out_channels": 2,
        "stride": 1,
        "_inputs_": [1, 3, 2, 2, 2],
        "_gt_shape_": [1, 2, 1, 1, 1],
    },
    ("conv3d", "default"): {
        "in_channels": 4,
        "out_channels": 4,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "_inputs_": [1, 4, 6, 4, 4],
        "_gt_shape_": [1, 4, 6, 4, 4],
    },
    ("noop", "default"): {
        "in_channels": 4,
        "out_channels": 4,
        "stride": 2,
        "_inputs_": [1, 4, 4, 4],
        "_gt_shape_": None,
    },
    ("upsample", "default"): {
        "in_channels": 4,
        "out_channels": 4,
        "stride": 2,
        "_inputs_": [1, 4, 4, 4],
        "_gt_shape_": [1, 4, 8, 8],
    },
    ("unsqueeze", "default"): {
        "in_channels": 1,
        "out_channels": 1,
        "stride": 1,
        "dim": 2,
        "_inputs_": [1, 2, 3, 4],
        "_gt_shape_": [1, 2, 1, 3, 4],
    },
    ("conv_k3_tuple_left", "default"): {
        "in_channels": 4,
        "out_channels": 4,
        "stride": 2,
        "_inputs_": ([1, 4, 4, 4], [1, 2, 3, 3]),
        "_gt_shape_": ([1, 4, 2, 2], [1, 2, 3, 3]),
    },
    ("conv_k5_tuple_left", "default"): {
        "in_channels": 4,
        "out_channels": 4,
        "stride": 2,
        "_inputs_": ([1, 4, 4, 4], [1, 2, 3, 3]),
        "_gt_shape_": ([1, 4, 2, 2], [1, 2, 3, 3]),
    },
    ("conv_tuple_left", "default"): {
        "in_channels": 4,
        "out_channels": 4,
        "stride": 2,
        "_inputs_": ([1, 4, 4, 4], [1, 2, 3, 3]),
        "_gt_shape_": ([1, 4, 2, 2], [1, 2, 3, 3]),
    },
    ("reshape_to_batch", "default"): {
        "in_channels": 32,
        "out_channels": 8,
        "stride": 1,
        "_inputs_": [3, 32, 4, 4],
        "_gt_shape_": [12, 8, 4, 4],
    },
    ("reshape_to_channel", "default"): {
        "in_channels": 8,
        "out_channels": 32,
        "stride": 1,
        "_inputs_": [8, 8, 4, 4],
        "_gt_shape_": [2, 32, 4, 4],
    },
    ("adaptive_avg_pool", "default"): {
        "in_channels": 4,
        "out_channels": 4,
        "stride": 1,
        "output_size": (1, 1),
        "_inputs_": [2, 4, 4, 4],
        "_gt_shape_": [2, 4, 1, 1],
    },
}

# fmt: off
# key: (op_name, cfg_name), default cfg will be used if only op_name is provided
TEST_OP_EXPECTED_OUTPUT = {
    ("skip", "default"): ([1, 4, 2, 2], [0.91831, 0.881, 0.76907, 0.73176, 0.39075, 0.40302, 0.43984, 0.45211, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # noqa
    ("conv_hs", "conv_cfg"): ([1, 4, 2, 2], [-0.15096, -0.1892, -0.17661, -0.18275, -0.07184, -0.10703, -0.0634, -0.09789, 0.00342, 0.09698, -0.03241, -0.07256, -0.03632, -0.04269, -0.08453, -0.05676]),  # noqa
    ("maxpool", "default"): ([1, 4, 1, 1], [-0.3438, -0.0938, 0.1562, 0.4062]),  # noqa

    ("ir_k3", "irf_cfg"): ([1, 4, 2, 2], [0.01408, 0.07465, -0.06185, 0.05435, 0.0878, 0.09416, 0.07732, 0.0464, 0.11598, 0.17861, 0.09616, 0.22626, 0.16809, 0.23102, 0.26872, 0.27185]),  # noqa
    ("ir_k5", "irf_cfg"): ([1, 4, 2, 2], [0.00059, 0.01012, 0.12793, 0.07516, 0.04054, 0.00832, -0.03221, -0.00779, 0.00193, 0.00041, -0.02473, -0.04499, -0.41635, -0.18052, -0.26376, -0.18981]),  # noqa
    ("ir_k3", "irf_cfg_sefc"): ([1, 4, 2, 2], [0.03449, 0.03763, -0.0264, -0.01943, 0.15323, 0.19681, 0.18142, 0.26796, -0.02263, -0.07074, -0.00743, -0.10626, -0.07885, -0.11029, -0.0243, -0.05389]),  # noqa
    ("ir_k3_se", "default"): ([1, 4, 2, 2], [-0.00262, -0.00662, -0.02462, -0.03795, 0.015, 0.00297, -0.00137, -0.00965, 0.03152, 0.02507, 0.01809, 0.01829, 0.01728, 0.00427, 0.04378, 0.04499]),  # noqa
    ("ir_k3_sehsig", "irf_cfg_seconvhsig"): ([1, 4, 2, 2], [-0.17406, -0.24675, -0.22749, -0.38714, -0.12084, -0.14361, -0.12785, -0.16972, 0.01255, 0.01106, 0.11432, 0.09041, 0.04951, 0.08074, 0.06063, 0.12473]),  # noqa
    ("ir_k5_sehsig", "default"): ([1, 4, 2, 2], [-0.01736, -0.0139, -0.01012, -0.05196, 0.02194, 0.03361, 0.03642, 0.0707, -0.02973, -0.03315, -0.0269, -0.01883, -0.00826, -0.00196, -0.0171, 0.02342]),  # noqa
    ("ir_pool", "default"): ([1, 4, 1, 1], [0.0, 0.1414, 0.49571, 0.43462]),  # noqa
    ("ir_pool_hs", "default"): ([1, 4, 1, 1], [-0.12981, 0.09867, -0.08033, 0.0543]),  # noqa

    ("ir3d_k3", "default"): ([1, 2, 2, 2, 2], [0.02443, 0.00738, 0.06039, 0.06596, 0.05445, 0.00319, 0.04746, 0.01919, 0.05831, -0.00115, 0.05368, -0.02315, 0.08671, 0.06718, 0.01821, 0.05173]),  # noqa

    ("conv_k3_tuple_left", "default"): (([1, 4, 2, 2], [0.0, 0.0, 0.27391, 0.0, 0.0, 0.0, 0.0221, 0.49272, 0.0, 0.0, 0.0, 0.0, 0.20517, 0.12133, 0.29797, 0.42078]), ([1, 2, 3, 3], [-0.5, -0.44444, -0.38889, -0.33333, -0.27778, -0.22222, -0.16667, -0.11111, -0.05556, 0.0, 0.05556, 0.11111, 0.16667, 0.22222, 0.27778, 0.33333, 0.38889, 0.44444])),  # noqa

    ("gb_k3_r2", "default"): ([1, 4, 2, 2], [0.0, 0.0, 0.0, 0.08396, 0.0108, 0.0, 0.0, 0.0, 0.01115, 0.01652, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # noqa
}
# fmt: on


def _get_computed_tensor_to_list(op_name, op_cfg_name, result):
    riter = iu.recursive_iterate(result)
    for item in riter:
        item_str = (
            f"({list(item.shape)}, "
            f"{[float('%.5f' % o) for o in item.contiguous().view(-1).tolist()]})"
        )
        riter.send(item_str)
    ret = f'("{op_name}", "{op_cfg_name}"): {_to_str(riter.value)},  # noqa'
    return ret


def _get_expected_output_to_tensor(outputs):
    # a valid tensor is represented by a tuple with two lists (shape and value)
    def _is_shape_value_tuple(obj):
        if isinstance(obj, tuple) and len(obj) == 2:
            if isinstance(obj[0], list) and isinstance(obj[1], list):
                return True
        return False

    # get all tensors
    riter = iu.recursive_iterate(
        outputs, seq_check_func=lambda x: not _is_shape_value_tuple(x)
    )
    for item in riter:
        item_tensor = torch.FloatTensor(item[1]).reshape(item[0])
        riter.send(item_tensor)

    # get all tensor shapes
    shape_iter = iu.recursive_iterate(
        outputs, seq_check_func=lambda x: not _is_shape_value_tuple(x)
    )
    for item in shape_iter:
        shape_iter.send(item[0])

    return riter.value, shape_iter.value


class TestFBNetV2BlocksFactory(unittest.TestCase):
    def _get_op_cfgs(self, op_name, op_cfg_name):
        assert op_cfg_name in OP_CFGS, f"op cfg name {op_cfg_name} not existed."
        op_cfg = OP_CFGS.get((op_name, op_cfg_name), OP_CFGS[op_cfg_name])
        op_cfg = copy.deepcopy(op_cfg)
        input_dims = op_cfg.pop("_inputs_")
        gt_shape = op_cfg.pop("_gt_shape_", None)

        output = TEST_OP_EXPECTED_OUTPUT.get((op_name, op_cfg_name), None)
        if output is None:
            assert op_cfg_name == "default"
            output = TEST_OP_EXPECTED_OUTPUT.get(op_name, None)
        if output is not None:
            gt_value, gt_shape = _get_expected_output_to_tensor(output)
        else:
            gt_value = None

        return op_cfg, input_dims, gt_shape, gt_value

    def _test_primitive_check_output(self, device, op_name, op_cfg_name):
        torch.manual_seed(0)

        op_args, op_input_dims, gt_shape, gt_value = self._get_op_cfgs(
            op_name, op_cfg_name
        )
        op_func = fbnet_builder.PRIMITIVES.get(op_name)
        op = op_func(**op_args).to(device).eval()

        inputs = _create_input(op_input_dims, device)
        with torch.no_grad():
            output = op(inputs)

        if op_name == "noop":
            self.assertEqual(gt_shape, None)
            self.assertEqual(output, None)
            return
        _compare_output_shape(self, output, gt_shape)

        computed_str = _get_computed_tensor_to_list(op_name, op_cfg_name, output)
        if gt_value is not None:
            _compare_outputs(output, gt_value, error_msg=computed_str)
        else:
            print(
                f"Ground truth output for op {op_name} and cfg {op_cfg_name} "
                f"not provided. Computed output: \n{computed_str}"
            )

    def test_primitives_check_output(self):
        """Make sures the primitives produce expected results"""
        op_names = list(TEST_OP_EXPECTED_OUTPUT.keys())
        op_names = {
            (x, "default") if isinstance(x, str) else x for x in sorted(op_names)
        }

        for op_name_info in op_names:
            op_name, op_cfg_name = op_name_info
            with self.subTest(op=op_name, cfg_name=op_cfg_name):
                print(f"Testing {op_name} with config {op_cfg_name}")
                self._test_primitive_check_output("cpu", op_name, op_cfg_name)

    def test_primitives_check_shape(self):
        """Make sures the primitives runs"""
        op_names = [x for (x, y) in list(TEST_OP_EXPECTED_OUTPUT.keys())]
        op_names = {(x, "default") for x in sorted(op_names)}

        for op_name_info in op_names:
            op_name, op_cfg_name = op_name_info
            with self.subTest(op=op_name, cfg_name=op_cfg_name):
                print(f"Testing {op_name} with config {op_cfg_name}")
                self._test_primitive_check_output("cpu", op_name, op_cfg_name)


if __name__ == "__main__":
    unittest.main()
