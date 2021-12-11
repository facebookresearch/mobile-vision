#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import mobile_cv.arch.fbnet_v2.irf_3d_block as irf_3d_block
import mobile_cv.lut.lib.pt.flops_utils as flops_utils
import torch


def create_test_irf_2dp1(self, out_channels, op_args, input_shape, gt_output_dim):
    N, C_in, D, H, W = input_shape
    op = irf_3d_block.IRF2dP1Block(
        in_channels=C_in, out_channels=out_channels, **op_args
    )

    inputs = torch.rand(input_shape, dtype=torch.float32)
    output = flops_utils.print_model_flops(op, [inputs])

    self.assertEqual(output.shape, torch.Size([N, out_channels, *gt_output_dim]))

    return op


def create_test_irf_3d(self, out_channels, op_args, input_shape, gt_output_dim):
    N, C_in, D, H, W = input_shape
    op = irf_3d_block.IRF3dBlock(in_channels=C_in, out_channels=out_channels, **op_args)

    inputs = torch.rand(input_shape, dtype=torch.float32)
    output = flops_utils.print_model_flops(op, [inputs])

    self.assertEqual(output.shape, torch.Size([N, out_channels, *gt_output_dim]))

    return op


class TestIRFBlocks(unittest.TestCase):
    def test_irf_2dp1_block(self):
        N, C_in, C_out, D = 2, 3, 16, 8
        input_dim = 7

        create_test_irf_2dp1(
            self,
            C_out,
            op_args={
                "expansion": 6,
                "kernel_size": 3,
                "stride": 1,
                "kernel_size_temporal": 3,
                "stride_temporal": 1,
            },
            input_shape=[N, C_in, D, input_dim, input_dim],
            gt_output_dim=[D, input_dim, input_dim],
        )

    def test_irf_2dp1_block_s2(self):
        N, C_in, C_out, D = 2, 3, 16, 8
        input_dim = 8

        create_test_irf_2dp1(
            self,
            C_out,
            op_args={
                "expansion": 6,
                "kernel_size": 3,
                "stride": 2,
                "kernel_size_temporal": 3,
                "stride_temporal": 2,
            },
            input_shape=[N, C_in, D, input_dim, input_dim],
            gt_output_dim=[D // 2, input_dim // 2, input_dim // 2],
        )

    def test_irf_2dp1_block_res(self):
        N, C_in, C_out, D = 2, 16, 16, 8
        input_dim = 8

        op = create_test_irf_2dp1(
            self,
            C_out,
            op_args={
                "expansion": 6,
                "kernel_size": 3,
                "stride": 1,
                "kernel_size_temporal": 3,
                "stride_temporal": 1,
            },
            input_shape=[N, C_in, D, input_dim, input_dim],
            gt_output_dim=[D, input_dim, input_dim],
        )
        self.assertTrue(hasattr(op, "res_conn"))

    def test_irf_3d_block_stride(self):
        N, C_in, C_out, D = 2, 3, 16, 8
        input_dim = 8

        create_test_irf_3d(
            self,
            C_out,
            op_args={
                "expansion": 4,
                "kernel_size": 3,
                "stride": 2,
            },
            input_shape=[N, C_in, D, input_dim, input_dim],
            gt_output_dim=[D, input_dim // 2, input_dim // 2],
        )

    def test_irf_3d_block_fusedMBConv(self):
        N, C_in, C_out, D = 2, 3, 16, 8
        input_dim = 8

        create_test_irf_3d(
            self,
            C_out,
            op_args={
                "skip_dw": True,
                "always_pw": True,
                "expansion": 2,
                "kernel_size": 3,
                "stride": 2,
                "kernel_size_temporal": 3,
                "stride_temporal": 2,
                "pw_args": {
                    "kernel_size": (3, 3, 3),
                    "padding": (1, 1, 1),
                    "stride": (2, 2, 2),
                },
            },
            input_shape=[N, C_in, D, input_dim, input_dim],
            gt_output_dim=[D // 2, input_dim // 2, input_dim // 2],
        )

        create_test_irf_3d(
            self,
            C_out,
            op_args={
                "skip_dw": True,
                "always_pw": True,
                "skip_pwl": True,
                "mid_expand_out": True,
                "expansion": 1,
                "kernel_size": 3,
                "stride": 2,
                "pw_args": {
                    "kernel_size": (3, 3, 3),
                    "padding": (1, 1, 1),
                    "stride": (1, 2, 2),
                },
            },
            input_shape=[N, C_in, D, input_dim, input_dim],
            gt_output_dim=[D, input_dim // 2, input_dim // 2],
        )
