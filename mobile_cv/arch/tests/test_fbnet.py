#!/usr/bin/env python3

import math
import unittest

import mobile_cv.arch.fbnet.fbnet_builder as fbnet_builder
import mobile_cv.arch.fbnet.fbnet_building_blocks as fbnet_building_blocks
import mobile_cv.arch.fbnet.fbnet_modeldef as fbnet_modeldef
import mobile_cv.arch.fbnet.fbnet_modeldef_cls as fbnet_modeldef_cls
import numpy as np
import torch
import torch.nn as nn
from parameterized import parameterized


TEST_CUDA = torch.cuda.is_available()


def _test_primitive(self, device, op_name, op_func, N, C_in, C_out, expand, stride):
    if op_name in ["swish", "upsample", "conv_nb_nr_bs_dw"]:
        C_in = C_out
    if op_name in ["upsample"]:
        stride = -2
    op = op_func(C_in, C_out, expand, stride).to(device)
    input = torch.rand([N, C_in, 8, 8], dtype=torch.float32).to(device)
    output = op(input)
    self.assertEqual(
        output.shape[:2],
        torch.Size([N, C_out]),
        "Primitive {} failed for shape {}.".format(op_name, input.shape),
    )


class TestFBNetPrimitives(unittest.TestCase):
    def test_identity(self):
        id_op = fbnet_building_blocks.Identity(20, 20, 1)
        input = torch.rand([10, 20, 7, 7], dtype=torch.float32)
        output = id_op(input)
        np.testing.assert_array_equal(np.array(input), np.array(output))

        id_op = fbnet_building_blocks.Identity(20, 40, 2)
        input = torch.rand([10, 20, 7, 7], dtype=torch.float32)
        output = id_op(input)
        np.testing.assert_array_equal(output.shape, [10, 40, 4, 4])

    def test_torchadd(self):
        input_a = torch.rand([3, 3], dtype=torch.float32)
        input_b = torch.rand([3, 3], dtype=torch.float32)
        add_op = fbnet_building_blocks.TorchAdd()
        result = input_a + input_b
        result_fbb = add_op(input_a, input_b)
        np.testing.assert_array_equal(np.array(result), np.array(result_fbb))

    def test_primitives(self):
        """Make sures the primitives runs"""
        for op_name, op_func in fbnet_builder.PRIMITIVES.items():
            with self.subTest(op=op_name):
                print("Testing {}".format(op_name))
                _test_primitive(
                    self,
                    "cpu",
                    op_name,
                    op_func,
                    N=20,
                    C_in=16,
                    C_out=32,
                    expand=4,
                    stride=1,
                )

    @unittest.skipIf(not TEST_CUDA, "no CUDA detected")
    def test_primitives_cuda(self):
        """Make sures the primitives runs on cuda"""
        for op_name, op_func in fbnet_builder.PRIMITIVES.items():
            with self.subTest(op=op_name):
                print("Testing {}".format(op_name))
                _test_primitive(
                    self,
                    "cuda",
                    op_name,
                    op_func,
                    N=20,
                    C_in=16,
                    C_out=32,
                    expand=4,
                    stride=1,
                )

    @unittest.skip("Skip for now as many new ops could not pass")
    def test_primitives_empty_batch(self):
        """Make sures the primitives runs"""
        for op_name, op_func in fbnet_builder.PRIMITIVES.items():
            with self.subTest(op=op_name):
                print("Testing {}".format(op_name))
                # test empty batch size
                _test_primitive(
                    self,
                    "cpu",
                    op_name,
                    op_func,
                    N=0,
                    C_in=16,
                    C_out=32,
                    expand=4,
                    stride=1,
                )

    @unittest.skip("Skip for now as many new ops could not pass")
    @unittest.skipIf(not TEST_CUDA, "no CUDA detected")
    def test_primitives_cuda_empty_batch(self):
        """Make sures the primitives runs"""
        for op_name, op_func in fbnet_builder.PRIMITIVES.items():
            with self.subTest(op=op_name):
                print("Testing {}".format(op_name))
                # test empty batch size
                _test_primitive(
                    self,
                    "cuda",
                    op_name,
                    op_func,
                    N=0,
                    C_in=16,
                    C_out=32,
                    expand=4,
                    stride=1,
                )


class TestFBNetBuildBlockHelpers(unittest.TestCase):
    def test_build_relu(self):
        "Check different options return activation ops"
        activations = {
            None: fbnet_building_blocks.PassThrough,
            "": fbnet_building_blocks.PassThrough,
            "relu": nn.ReLU,
            "relu6": nn.ReLU6,
            "hswish": fbnet_building_blocks.HSwish,
            "leakyrelu": nn.LeakyReLU,
            "sig": nn.Sigmoid,
        }
        for name, gt_op in activations.items():
            op = fbnet_building_blocks.build_relu(name)
            assert isinstance(op, gt_op)


class TestFBNetPrimitiviesCheckOutput(unittest.TestCase):
    OP_SHAPE_DEFAULT = {
        "C_in": 4,
        "C_out": 4,
        "expand": 4,
        "stride": 2,
        "N": 1,
        "hw": 4,
    }
    OP_SHAPES = {
        "oct_ir_k3_0.375": {
            "C_in": 4,
            "C_out": 2,
            "expand": 4,
            "stride": 1,
            "N": 1,
            "hw": 4,
        },
        "oct_ir_k3_0.5": {
            "C_in": 4,
            "C_out": 2,
            "expand": 4,
            "stride": 1,
            "N": 1,
            "hw": 4,
        },
        "aa_ir_k3": {"C_in": 4, "C_out": 2, "expand": 4, "stride": 2, "N": 1, "hw": 8},
        "aa_ir_k5": {"C_in": 4, "C_out": 2, "expand": 4, "stride": 2, "N": 1, "hw": 8},
    }

    # fmt: off
    SKIPPED_OPS = [
        "ir_k3_e1", "ir_k3_e3", "ir_k3_e6",
        "ir_k5_e1", "ir_k5_e3", "ir_k5_e6", "ir_k5_s4",
        "ir_k3_e1_se", "ir_k3_e3_se", "ir_k3_e6_se",
        "ir_k5_e1_se", "ir_k5_e3_se", "ir_k5_e6_se", "ir_k5_s4_se",
        "ir_k5_sehsig", "ir_k7_sehsig", "ir_k5_seconvhsig",
        "ir_k5_hs", "ir_k7_hs", "ir_k5_se_hs", "ir_k7_se_hs",
        "ir_k5_s2", "ir_k3_s2_se",
        "ir_k33_e1", "ir_k33_e3",
        "ir_k7_e1", "ir_k7_e3", "ir_k7_e6", "ir_k7_se",
        "ir_k7_sep", "ir_k7_sep_e1", "ir_k7_sep_e3", "ir_k7_sep_e6",
        "mix_irf", "oct_ir_k3_0.875",
    ]
    TEST_OP_EXPECTED_OUTPUT = {
        "skip": ([1, 4, 2, 2], [1.21165, 0.72699, 0.0, 0.0, 0.0, 0.0, 0.72199, 1.20332, 0.0, 0.0, 0.67622, 1.12703, 0.0, 0.0, 0.72742, 1.21237]), # noqa
        "conv": ([1, 4, 2, 2], [1.63705, 0.0, 0.0, 0.0, 0.73916, 0.0, 1.19383, 0.0, 0.11499, 1.53941, 0.0, 0.0, 0.99802, 0.6674, 0.0, 0.0]), # noqa
        "conv_hs": ([1, 4, 2, 2], [1.26518, -0.33303, -0.04969, -0.22011, 0.46064, -0.36264, 0.83445, -0.26974, 0.0597, 1.16467, -0.20044, -0.3577, 0.66501, 0.40794, -0.37364, -0.03666]), # noqa
        "ir_k3": ([1, 4, 2, 2], [-0.27444, 0.00614, -1.26024, 1.52854, 0.72921, 0.81073, 0.1315, -1.67143, -0.11324, -0.32587, -1.15603, 1.59515, -0.31567, -0.03981, -1.20709, 1.56258]), # noqa
        "ir_k5": ([1, 4, 2, 2], [0.04597, 0.70837, 0.89002, -1.64435, -0.44674, -1.26645, 0.25286, 1.46033, 1.21518, 0.71088, -0.67746, -1.2486, -1.46606, -0.00653, 1.3553, 0.11729]), # noqa
        "ir_k7": ([1, 4, 2, 2], [-1.19829, -0.16337, -0.21642, 1.57808, -0.06501, -1.27109, -0.19436, 1.53046, -0.00137, -1.30323, 1.50394, -0.19934, 0.97475, 0.30557, 0.39348, -1.6738]),  # noqa
        "ir_k357": ([1, 4, 2, 2], [0.80096, -1.71488, 0.45909, 0.45483, -0.84537, 1.67755, -0.1662, -0.66598, 0.54951, -0.11353, 1.12181, -1.55779, -1.54019, -0.04185, 1.22382, 0.35823]),  # noqa
        "ir_k3_se": ([1, 4, 2, 2], [1.00829, 0.14804, 0.49254, -1.64888, 0.3508, 0.69977, -1.71593, 0.66536, -0.43575, -0.4919, -0.78875, 1.7164, -0.8786, -0.81134, 1.60054, 0.0894]),  # noqa
        "ir_k3_sehsig": ([1, 4, 2, 2], [1.00825, 0.14807, 0.49257, -1.64889, 0.35101, 0.70002, -1.71596, 0.66492, -0.43575, -0.49191, -0.78874, 1.7164, -0.87861, -0.81129, 1.60056, 0.08934]),  # noqa
        "ir_k3_seconvhsig": ([1, 4, 2, 2], [1.07372, 0.33734, 0.2266, -1.63766, -0.10855, -0.65601, -0.89552, 1.66007, 0.68671, 0.23328, 0.77507, -1.69506, -1.20954, -0.25699, 1.5686, -0.10207]),  # noqa
        "ir_k5_se": ([1, 4, 2, 2], [1.24743, -1.46813, -0.25279, 0.4735, -1.41054, -0.47015, 0.83858, 1.04211, -0.77477, -1.02195, 0.2896, 1.50712, -1.07438, -0.639, 1.55292, 0.16046]),  # noqa
        "ir_k3_hs": ([1, 4, 2, 2], [-1.2386, -0.52011, 0.30979, 1.44892, 0.84044, 0.98524, -0.33997, -1.48572, 1.5551, -0.91209, -0.84251, 0.1995, -1.54891, -0.20953, 0.8054, 0.95304]),  # noqa
        "ir_k3_se_hs": ([1, 4, 2, 2], [1.3871, 0.21401, -0.18889, -1.41222, 1.10158, 0.63934, -1.52236, -0.21856, -0.70774, -0.86637, -0.08172, 1.65584, -0.31647, -0.55309, 1.7028, -0.83324]),  # noqa
        "ir_k3_s2": ([1, 4, 2, 2], [0.78395, 0.78395, -1.66152, 0.09362, -0.86879, -0.86879, 1.56908, 0.1685, -1.26467, -0.69305, 0.93823, 1.01948, 1.40815, -1.39518, -0.19265, 0.17969]),  # noqa
        "ir_k5_s2_se": ([1, 4, 2, 2], [1.71618, -0.5742, -0.75672, -0.38527, -0.08195, 0.04356, 1.4303, -1.39191, -1.72945, 0.66493, 0.53226, 0.53226, -0.30373, 1.70759, -0.70193, -0.70193]),  # noqa
        "ir_pool": ([1, 4, 1, 1], [0.60653, 0.7793, 0.0, 0.03284]),  # noqa
        "ir_pool_hs": ([1, 4, 1, 1], [0.1295, 0.18161, -0.02259, -0.01122]),  # noqa
        "ir_k3_sc": ([1, 4, 2, 2], [-1.485478, -0.720486, -0.533615, 2.73958 , -0.479584, 0.085451, 0.856775, -0.462642, 1.099065, 0.401511, -1.883419, 0.382843, -1.527896, -0.767148, -0.479762, 2.774806]),  # noqa
        "ir_k3_d2": ([1, 4, 2, 2], [-1.26349, 0.98796, -0.69577, 0.9713, -0.28509, -0.47345, 1.68443, -0.92589, 0.18336, -0.83992, -0.90569, 1.56225, -1.04272, 1.64067, -0.1538, -0.44415]),  # noqa
        "ir_k3_d3": ([1, 4, 2, 2], [-1.66446, 0.26932, 0.39442, 1.00072, -0.69753, -1.12383, 1.45462, 0.36675, 1.5124, -1.18977, -0.51126, 0.18863, 1.35137, 0.16712, -1.46385, -0.05464]),  # noqa
        "ir_k3_d123": ([1, 4, 2, 2], [0.48252, -1.40059, 1.29093, -0.37286, 0.13844, 0.13293, 1.26545, -1.53681, -0.97841, 1.22295, -0.99325, 0.7487, -1.36274, -0.36037, 0.3475, 1.37561]),  # noqa
        "ir_k1": ([1, 4, 4, 4], [2.2009, 1.77839, 1.35589, 0.93339, 0.51089, 0.08838, -0.33412, -0.75662, -0.93716, -0.87572, -0.81429, -0.75285, -0.69142, -0.62998, -0.56855, -0.50712,  # noqa
                                 0.02521, -0.14964, -0.32448, -0.49932, -0.67416, -0.849, -1.02385, -1.19869, -1.05201, -0.58379, -0.11558, 0.35264, 0.82085, 1.28906, 1.75727, 2.22549,  # noqa
                                -1.70241, -1.4636, -1.2248, -0.986, -0.7472, -0.5084, -0.2696, -0.03079, 0.18586, 0.38035, 0.57485, 0.76935, 0.96385, 1.15835, 1.35285, 1.54735, 1.6744,  # noqa
                                1.44379, 1.21319, 0.98258, 0.75197, 0.52137, 0.29076, 0.06015, -0.15666, -0.3597, -0.56273, -0.76576, -0.96879, -1.17182, -1.37485, -1.57789]),  # noqa
        "shuffle": ([1, 4, 2, 2], [0.96677, 0.94993, -0.55459, -1.36212, -1.45744, -0.33763, 0.62303, 1.17204, -1.72829, 0.68328, 0.51713, 0.52788, 0.69777, 1.24193, -0.75833, -1.18137]),  # noqa
        "basic_block": ([1, 4, 2, 2], [0.07659, -0.64224, 1.59069, -1.02503, -1.43952, -0.09378, 1.37597, 0.15733, -0.05296, -1.03841, -0.5312, 1.62257, 0.46016, -1.12598, 1.4031, -0.73728]),  # noqa
        "shift_5x5": ([1, 4, 2, 2], [1.28765, 0.6607, -1.03365, -0.91469, -1.25778, -0.6983, 0.88619, 1.0699, -1.24442, -0.70905, 0.83439, 1.11908, -1.29331, -0.53118, 1.34719, 0.4773]),  # noqa
        "ir_k3_s4": ([1, 4, 2, 2], [0.96677, 0.94993, -0.55459, -1.36212, -1.45744, -0.33763, 0.62303, 1.17204, -1.72829, 0.68328, 0.51713, 0.52788, 0.69777, 1.24193, -0.75833, -1.18137]),  # noqa
        "ir_k3_s4_se": ([1, 4, 2, 2], [-0.88783, -0.90381, 0.26755, 1.52409, 1.72166, -0.42574, -0.56403, -0.73188, -0.746, 0.86426, 1.10046, -1.21872, -0.26906, 1.70277, -0.72879, -0.70491]),  # noqa
        "ir_k3_sep": ([1, 4, 2, 2], [0.51501, 0.41564, 0.78545, -1.71611, 0.37362, -1.53592, 1.22374, -0.06144, -1.40206, 0.72108, -0.46026, 1.14124, -0.90951, 1.20373, -1.06378, 0.76955]),  # noqa
        "ir_k33_e6": ([1, 4, 2, 2], [1.34431, -0.3949, -1.36526, 0.41586, 1.41803, -1.25586, 0.36547, -0.52764, 0.64791, 0.81663, -1.69167, 0.22713, -1.5721, 1.14404, 0.46706, -0.03901]),  # noqa
        "oct_ir_k3_0.375": ([1, 2, 4, 4], [-0.17214, 2.17864, 2.096, 0.87559, -0.39273, 0.69286, 0.52646, -0.10504, -0.3845, -1.08403, -0.50685, -0.32666, -0.63596, -1.29775, -1.11592, -0.34796,  # noqa
                                           1.44531, 2.36959, 0.66756, -0.33041, 0.83286, 0.88984, -0.63707, -1.33025, -0.57503, -0.91547, -0.61953, -0.92774, -1.15677, -0.28885, 0.08838, 0.48758]),  # noqa
        "oct_ir_k3_0.5": ([1, 2, 4, 4], [-1.59667, -1.00565, 0.75513, 0.86774, -1.41442, -0.34988, 1.73913, 1.83536, -0.20946, -0.08027, 0.70189, 0.411, -0.43776, -1.1122, -0.52844, 0.42451,  # noqa
                                         -2.2229, -0.73207, -0.04227, 0.17553, -2.00204, 0.02375, 0.65175, 0.18436, 0.2, 0.62204, -0.34799, -0.27163, 1.14956, 1.75122, 0.997, -0.13631]),  # noqa
    }
    # fmt: on

    @classmethod
    def _get_input_shape(cls, op_name):
        return cls.OP_SHAPES.get(op_name, cls.OP_SHAPE_DEFAULT)

    @classmethod
    def _get_input(cls, op_name):
        ins = cls._get_input_shape(op_name)
        nchw = ins["N"] * ins["C_in"] * ins["hw"] * ins["hw"]
        ret = (torch.arange(nchw, dtype=torch.float32) - (nchw / 2.0)) / (nchw)
        ret = ret.reshape(ins["N"], ins["C_in"], ins["hw"], ins["hw"])
        return ret

    def _test_primitive_check_output(self, device, op_name, op_func):
        torch.manual_seed(0)

        ins = self._get_input_shape(op_name)
        op = op_func(ins["C_in"], ins["C_out"], ins["expand"], ins["stride"]).to(device)
        input = self._get_input(op_name).to(device)
        output = op(input)

        self.assertIn(
            op_name,
            self.TEST_OP_EXPECTED_OUTPUT.keys(),
            f"Ground truth output for op {op_name} not provided. "
            "Computed output: \n"
            f'"{op_name}": ({list(output.shape)}, '
            f"{[float('%.5f' % o) for o in output.view(-1).tolist()]})"
            ",  # noqa",
        )

        gt_shape = self.TEST_OP_EXPECTED_OUTPUT[op_name][0]
        gt_value = self.TEST_OP_EXPECTED_OUTPUT[op_name][1]
        gt_output = torch.FloatTensor(gt_value).reshape(gt_shape)
        np.testing.assert_allclose(output.detach(), gt_output, rtol=0, atol=1e-4)

    def test_primitives_check_output(self):
        """Make sures the primitives produce expected results"""
        for op_name, op_func in fbnet_builder.PRIMITIVES.items():
            if op_name in self.SKIPPED_OPS:
                continue
            if op_name not in self.TEST_OP_EXPECTED_OUTPUT.keys():
                print(f"WARNING: {op_name} doesn't have gt output, skipping")
                continue
            with self.subTest(op=op_name):
                print("Testing {}".format(op_name))
                self._test_primitive_check_output("cpu", op_name, op_func)


def _build_model(arch_def):
    arch_def = fbnet_builder.unify_arch_def(arch_def)
    torch.manual_seed(0)
    builder = fbnet_builder.FBNetBuilder(1.0)
    ops = []
    ops.append(builder.add_first(arch_def["first"]))
    ops.append(builder.add_blocks(arch_def["stages"]))
    model = torch.nn.Sequential(*ops)
    model.eval()
    return model


def _get_input(n, c, h, w):
    nchw = n * c * h * w
    input = (torch.arange(nchw, dtype=torch.float32) - (nchw / 2.0)) / (nchw)
    input = input.reshape(n, c, h, w)
    return input


def _run_model_archs(self, model_defs):
    print("")
    for arch_name, arch in model_defs.items():
        with self.subTest(arch=arch_name):
            print("Testing {}".format(arch_name))
            model = _build_model(arch)
            # input = _get_input(2, 3, 8, 8)
            # input = _get_input(1, 3, 224, 224)
            arch_def = fbnet_builder.unify_arch_def(arch)
            max_stride = fbnet_builder.count_strides(arch_def)
            print("    arch_def stride: {}".format(max_stride))

            # anti-alias requires doubling resolution
            if any("aa_" in s["block_op_type"] for s in arch_def["stages"]):
                max_stride *= 2

            # for fractional stride, make it an integer
            if max_stride != int(max_stride):
                for mul in range(2, int(224 / max_stride) + 1):
                    test_stride = max_stride * mul
                    if test_stride == int(test_stride):
                        max_stride = test_stride
                        break
                else:
                    max_stride = 224

            input = _get_input(1, 3, int(max_stride), int(max_stride))
            print("    input resolution: {}".format(input.shape))
            output = model(input)
            self.assertEqual(output.shape[0], 1)


class TestFBNetBuilder(unittest.TestCase):
    def test_unify_arch(self):
        arch_def_new = {
            "block_op_type": {
                "first": "conv",
                "stages": [
                    # stage 0
                    ["ir_k3"],
                    # stage 1
                    ["ir_k5"],
                ],
            },
            "block_cfg": {
                "first": [32, 2],
                "stages": [
                    # [t, c, n, s]
                    # stage 0
                    [[1, 16, 1, 1]],
                    # stage 1
                    [[6, 2, 1, 2]],
                ],
                # [c, channel_scale]
                "last": [100, 0.0],
            },
        }
        arch_def_old = {
            "block_op_type": [
                # stage 0
                ["ir_k3"],
                # stage 1
                ["ir_k5"],
            ],
            "block_cfg": {
                "first": [32, 2],
                "stages": [
                    # [t, c, n, s]
                    # stage 0
                    [[1, 16, 1, 1]],
                    # stage 1
                    [[6, 2, 1, 2]],
                ],
                # [c, channel_scale]
                "last": [100, 0.0],
            },
        }
        arch_def_new = fbnet_builder.unify_arch_def(arch_def_new)
        arch_def_old = fbnet_builder.unify_arch_def(arch_def_old)
        self.assertEqual(
            arch_def_old,
            arch_def_new,
            "Unmatchd model for unify_arch_def, old: {}, new: {}".format(
                arch_def_old, arch_def_new
            ),
        )

    def test_fbnet_builder_check_output(self):
        arch_def = {
            "block_op_type": [
                # stage 0
                ["ir_k3"],
                # stage 1
                ["ir_k5"],
            ],
            "block_cfg": {
                "first": [32, 2],
                "stages": [
                    # [t, c, n, s]
                    # stage 0
                    [[1, 16, 1, 1]],
                    # stage 1
                    [[6, 2, 1, 2]],
                ],
                # [c, channel_scale]
                "last": [100, 0.0],
            },
        }

        model = _build_model(arch_def)
        input = _get_input(2, 3, 8, 8)
        output = model(input)
        self.assertEqual(output.shape, torch.Size([2, 2, 2, 2]))

        gt_value = [
            -0.006989,
            -0.001088,
            -0.003623,
            -0.000779,
            -0.002274,
            0.008948,
            -0.004563,
            -0.004967,
            -0.002268,
            -0.002927,
            -0.008885,
            -0.002523,
            0.002622,
            0.003849,
            0.004999,
            0.007137,
        ]  # noqa
        gt_shape = [2, 2, 2, 2]
        gt_output = torch.FloatTensor(gt_value).reshape(gt_shape)
        self.assertIsNotNone(
            gt_output,
            "Ground truth output not provided. "
            "Computed output: {}\n{}".format(
                output.shape, [float("%.5f" % o) for o in output.view(-1).tolist()]
            ),
        )

        np.testing.assert_allclose(output.detach(), gt_output, rtol=0, atol=1e-4)


def get_chunks(size):
    CHUNK_SIZE = 15
    num_chunks = math.ceil(size / CHUNK_SIZE)
    ret = []
    for i in range(num_chunks):
        start = i * CHUNK_SIZE
        end = min((i + 1) * CHUNK_SIZE, size)
        name = "{}to{}".format(start, end)
        ret.append([name, start, end])
    # eg; [["0to15", 0, 15], ["16to25", 16, 25]] when size is 25
    return ret


class TestFBNetModels(unittest.TestCase):
    @parameterized.expand(get_chunks(len(fbnet_modeldef.MODEL_ARCH)))
    def test_fbnet_build_all_archs_trunk(self, name, start, end):
        name_to_test = sorted(fbnet_modeldef.MODEL_ARCH.keys())[start:end]
        _run_model_archs(self, {x: fbnet_modeldef.MODEL_ARCH[x] for x in name_to_test})

    @parameterized.expand(get_chunks(len(fbnet_modeldef_cls.MODEL_ARCH)))
    def test_fbnet_build_all_archs_cls(self, name, start, end):
        name_to_test = sorted(fbnet_modeldef_cls.MODEL_ARCH.keys())[start:end]
        _run_model_archs(
            self, {x: fbnet_modeldef_cls.MODEL_ARCH[x] for x in name_to_test}
        )

    def test_fbnet_build_all_archs_cls_single(self):
        name_to_test = ["default", "default_se"]
        _run_model_archs(
            self, {x: fbnet_modeldef_cls.MODEL_ARCH[x] for x in name_to_test}
        )


if __name__ == "__main__":
    unittest.main()
