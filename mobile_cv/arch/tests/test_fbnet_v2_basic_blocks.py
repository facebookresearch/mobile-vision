#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import mobile_cv.arch.fbnet_v2.basic_blocks as bb
import numpy as np
import torch


def _create_input(input_dims):
    assert isinstance(input_dims, (tuple, list))
    nchw = np.prod(input_dims)
    ret = (torch.arange(nchw, dtype=torch.float32) - (nchw / 2.0)) / (nchw)
    ret = ret.reshape(*input_dims)
    return ret


def _test_op(self, op, inputs, gt_out_shape, trace=True, script=True):
    print(op)
    out = op(inputs)
    self.assertEqual(out.shape, gt_out_shape)

    # trace the op
    if trace:
        traced_op = torch.jit.trace(op, [inputs])
        traced_out = traced_op(inputs)
        self.assertEqual(traced_out.shape, gt_out_shape)

    # script the op
    if script:
        script_op = torch.jit.script(op)
        script_out = script_op(inputs)
        self.assertEqual(script_out.shape, gt_out_shape)


class TestFBNetV2BasicBlocks(unittest.TestCase):
    def test_hsigmoid(self):
        input = _create_input([1, 2, 2, 2])
        op = bb.HSigmoid()
        output = op(input)
        gt_output = torch.tensor(
            [
                0.416667,
                0.4375,
                0.458333,
                0.479167,
                0.5000,
                0.5208,
                0.5417,
                0.5625,
            ]
        ).reshape([1, 2, 2, 2])
        np.testing.assert_allclose(output, gt_output, rtol=0, atol=1e-4)

    def test_hswish(self):
        input = _create_input([1, 2, 2, 2])
        op = bb.HSwish()
        output = op(input)
        gt_output = torch.tensor(
            [-0.2083, -0.1641, -0.1146, -0.0599, 0.0000, 0.0651, 0.1354, 0.2109]
        ).reshape([1, 2, 2, 2])
        np.testing.assert_allclose(output, gt_output, rtol=0, atol=1e-4)

    def test_chooserightpath(self):
        input_r = torch.randn(1, 2, 2, 2)
        input_l = torch.randn(1, 2, 2, 2)
        op = bb.ChooseRightPath()
        output = op(input_l, input_r)
        np.testing.assert_equal(output.numpy(), input_r.numpy())

    def test_conv_bn_relu_upsample(self):
        # currently empty batch for dw conv is not supported
        op = bb.ConvBNRelu(4, 4, stride=-2, kernel_size=3, padding=1)

        input_size = [1, 4, 4, 4]
        inputs = torch.rand(input_size)
        output = op(inputs)
        self.assertEqual(output.shape, torch.Size([1, 4, 8, 8]))

    def test_channel_shuffle(self):
        # currently empty batch for dw conv is not supported
        op = bb.ChannelShuffle(2)

        #  4D tensor
        x = torch.tensor(
            [
                [
                    [[1, 2], [3, 4]],
                    [[5, 6], [7, 8]],
                    [[9, 10], [11, 12]],
                    [[13, 14], [15, 16]],
                ]
            ]
        )
        y_ref = torch.tensor(
            [
                [
                    [[1, 2], [3, 4]],
                    [[9, 10], [11, 12]],
                    [[5, 6], [7, 8]],
                    [[13, 14], [15, 16]],
                ]
            ]
        )
        y = op(x)
        np.testing.assert_equal(y.numpy(), y_ref.numpy())

    def test_torchwhere(self):
        """Check that the torchwhere wrapper is the same as the torch.where"""
        op = bb.TorchWhere()
        input0 = torch.randn(5, 3)
        input1 = torch.randn(5, 3)
        condition = input0 > 0
        np.testing.assert_equal(
            op(condition, input0, input1).numpy(),
            torch.where(condition, input0, input1).numpy(),
        )

    def test_ignorewhereselectx1(self):
        """Check that module returns x1"""
        op = bb.IgnoreWhereSelectX1()
        input0 = torch.randn(5, 3)
        input1 = torch.randn(5, 3)
        condition = input0 > 0
        np.testing.assert_equal(op(condition, input0, input1).numpy(), input0.numpy())

    def test_se(self):
        """Test 2d se module"""

        with self.subTest("se"):
            op = bb.build_se("se", 8, 2).eval()
            input0 = torch.randn(2, 8, 4, 4)
            gt_shape = torch.Size([2, 8, 4, 4])
            _test_op(self, op, input0, gt_shape)

        with self.subTest("se_fc"):
            op = bb.build_se("se", 8, 2, fc=True).eval()
            _test_op(self, op, input0, gt_shape)

        with self.subTest("se_hsig"):
            op = bb.build_se("se_hsig", 8, 2).eval()
            _test_op(self, op, input0, gt_shape)

    def test_se3d(self):
        """Test 3d se module"""
        with self.subTest("se3d"):
            op = bb.build_se("se3d", 8, 2).eval()
            input0 = torch.randn(2, 8, 4, 4, 4)
            gt_shape = torch.Size([2, 8, 4, 4, 4])
            _test_op(self, op, input0, gt_shape)

        with self.subTest("se3d_fc"):
            op = bb.build_se("se3d", 8, 2, fc=True).eval()
            _test_op(self, op, input0, gt_shape)

        with self.subTest("se3d_hsig"):
            op = bb.build_se("se3d_hsig", 8, 2).eval()
            _test_op(self, op, input0, gt_shape)


class TestFBNetV2BasicBlocksEmptyInput(unittest.TestCase):
    def test_conv_bn_relu_empty_input(self):
        # currently empty batch for dw conv is not supported
        op = bb.ConvBNRelu(4, 4, stride=2, kernel_size=3, padding=1, groups=4)

        input_size = [0, 4, 4, 4]
        inputs = torch.rand(input_size)
        output = op(inputs)
        self.assertEqual(output.shape, torch.Size([0, 4, 2, 2]))

        input_size = [2, 4, 4, 4]
        inputs = torch.rand(input_size)
        output = op(inputs)
        self.assertEqual(output.shape, torch.Size([2, 4, 2, 2]))

    def test_conv_bn_relu_upsample_empty_input(self):
        op = bb.ConvBNRelu(4, 4, stride=-2, kernel_size=3, padding=1)

        input_size = [0, 4, 4, 4]
        inputs = torch.rand(input_size)
        output = op(inputs)
        self.assertEqual(output.shape, torch.Size([0, 4, 8, 8]))

    def test_dw_conv_empty_input(self):
        op = torch.nn.Conv2d(4, 4, stride=2, kernel_size=3, padding=1, groups=4)

        input_size = [0, 4, 4, 4]
        inputs = torch.rand(input_size)
        output = op(inputs)
        self.assertEqual(output.shape, torch.Size([0, 4, 2, 2]))

        input_size = [2, 4, 4, 4]
        inputs = torch.rand(input_size)
        output = op(inputs)
        self.assertEqual(output.shape, torch.Size([2, 4, 2, 2]))

    def test_bn_empty_input(self):
        op = torch.nn.BatchNorm2d(1)

        input_size = [0, 1, 4, 4]
        inputs = torch.rand(input_size)
        output = op(inputs)
        self.assertEqual(output.shape, torch.Size([0, 1, 4, 4]))

        input_size = [2, 1, 4, 4]
        inputs = torch.rand(input_size)
        output = op(inputs)
        self.assertEqual(output.shape, torch.Size([2, 1, 4, 4]))

    def test_interpolate_empty_input(self):
        input_size = [0, 1, 4, 4]
        inputs = torch.rand(input_size)
        output = torch.nn.functional.interpolate(inputs, scale_factor=0.5)
        self.assertEqual(output.shape, torch.Size([0, 1, 2, 2]))

        input_size = [2, 1, 4, 4]
        inputs = torch.rand(input_size)
        output = torch.nn.functional.interpolate(inputs, scale_factor=0.5)
        self.assertEqual(output.shape, torch.Size([2, 1, 2, 2]))
