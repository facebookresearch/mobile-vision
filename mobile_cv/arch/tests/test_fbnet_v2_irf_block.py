#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import mobile_cv.arch.fbnet_v2.irf_block as irf_block
import torch


TEST_CUDA = torch.cuda.is_available()


def _create_op_irf(in_channels, out_channels, op_args):
    op = irf_block.IRFBlock(
        in_channels=in_channels, out_channels=out_channels, **op_args
    )
    return op


def _create_op_ir_pool(in_channels, out_channels, op_args):
    op = irf_block.IRPoolBlock(
        in_channels=in_channels, out_channels=out_channels, **op_args
    )
    return op


def create_test_irf(
    self,
    out_channels,
    op_args,
    input_shape,
    gt_output_dim,
    op_create_func=_create_op_irf,
    trace=True,
    script=True,
):
    N, C_in, H, W = input_shape
    op = op_create_func(in_channels=C_in, out_channels=out_channels, op_args=op_args)
    print(op)

    inputs = torch.rand(input_shape, dtype=torch.float32)
    output = op(inputs)
    gt_shape = torch.Size([N, out_channels, gt_output_dim, gt_output_dim])

    self.assertEqual(output.shape, gt_shape)

    # trace the op
    if trace:
        traced_op = torch.jit.trace(op, [inputs])
        traced_out = traced_op(inputs)
        self.assertEqual(traced_out.shape, gt_shape)

    # script the op
    if script:
        script_op = torch.jit.script(op)
        script_out = script_op(inputs)
        self.assertEqual(script_out.shape, gt_shape)

    return op


class TestIRFBlocks(unittest.TestCase):
    def test_irf_block(self):
        N, C_in, C_out = 2, 16, 16
        input_dim = 7

        for bn_args in [
            "bn",
            {"name": "bn", "momentum": 0.003},
            {"name": "sync_bn", "momentum": 0.003},
        ]:
            with self.subTest(f"bn={bn_args}"):
                create_test_irf(
                    self,
                    C_out,
                    op_args={
                        "expansion": 6,
                        "kernel_size": 3,
                        "stride": 1,
                        "bn_args": bn_args,
                    },
                    input_shape=[N, C_in, input_dim, input_dim],
                    gt_output_dim=input_dim,
                    # need to fuse sync bn before we could script the op
                    script=(bn_args == "bn" or bn_args["name"] == "bn"),
                )

        with self.subTest("skip_bnrelu=True"):
            create_test_irf(
                self,
                C_out,
                op_args={
                    "expansion": 6,
                    "kernel_size": 3,
                    "stride": 1,
                    "bn_args": "bn",
                    "dw_skip_bnrelu": True,
                },
                input_shape=[N, C_in, input_dim, input_dim],
                gt_output_dim=input_dim,
            )

    def test_irf_block_res_conn(self):
        N, C_in, C_out = 2, 16, 32
        input_dim = 8

        create_test_irf(
            self,
            C_out,
            op_args={"expansion": 6, "kernel_size": 3, "stride": 1},
            input_shape=[N, C_in, input_dim, input_dim],
            gt_output_dim=input_dim,
        )

        create_test_irf(
            self,
            C_out,
            op_args={"expansion": 6, "kernel_size": 3, "stride": 2},
            input_shape=[N, C_in, input_dim, input_dim],
            gt_output_dim=input_dim // 2,
        )

    def test_irf_block_se(self):
        N, C_in, C_out = 2, 16, 32
        input_dim = 8

        create_test_irf(
            self,
            C_out,
            op_args={
                "expansion": 6,
                "kernel_size": 3,
                "stride": 1,
                "se_args": "se_fc",
            },
            input_shape=[N, C_in, input_dim, input_dim],
            gt_output_dim=input_dim,
        )

        create_test_irf(
            self,
            C_out,
            op_args={
                "expansion": 6,
                "kernel_size": 3,
                "stride": 2,
                "se_args": {"name": "se_hsig", "relu_args": "hswish"},
            },
            input_shape=[N, C_in, input_dim, input_dim],
            gt_output_dim=input_dim // 2,
        )

    def test_irf_block_upsample(self):
        N, C_in, C_out = 2, 16, 32
        input_dim = 8

        create_test_irf(
            self,
            C_out,
            op_args={
                "expansion": 6,
                "kernel_size": 3,
                "stride": -2,
                "se_args": "se_fc",
            },
            input_shape=[N, C_in, input_dim, input_dim],
            gt_output_dim=input_dim * 2,
        )

    def test_irf_block_width_divisor(self):
        N, C_in, C_out = 2, 3, 4
        input_dim = 8

        op = create_test_irf(
            self,
            C_out,
            op_args={
                "expansion": 5,
                "kernel_size": 3,
                "stride": 2,
                "width_divisor": 8,
            },
            input_shape=[N, C_in, input_dim, input_dim],
            gt_output_dim=input_dim // 2,
        )
        print(op)
        self.assertEqual(op.dw.out_channels, 16)

    def test_irf_block_fusedMBConv(self):
        N, C_in, C_out = 2, 16, 32
        input_dim = 8

        create_test_irf(
            self,
            C_out,
            op_args={
                "stride": 2,
                "kernel_size": 3,
                "expansion": 4,
                "always_pw": True,
                "skip_dw": True,
                "pw_args": {
                    "kernel_size": 3,
                    "padding": 1,
                    "stride": 2,
                },
            },
            input_shape=[N, C_in, input_dim, input_dim],
            gt_output_dim=input_dim // 2,
        )

        create_test_irf(
            self,
            C_out,
            op_args={
                "stride": 2,
                "kernel_size": 5,
                "expansion": 1,
                "always_pw": True,
                "skip_dw": True,
                "skip_pwl": True,
                "mid_expand_out": True,
                "pw_args": {
                    "kernel_size": 5,
                    "padding": 2,
                    "stride": 2,
                },
            },
            input_shape=[N, C_in, input_dim, input_dim],
            gt_output_dim=input_dim // 2,
        )

    def test_irpool_block(self):
        N, C_in, C_out = 2, 16, 32
        input_dim = 8

        create_test_irf(
            self,
            C_out,
            op_args={"expansion": 6, "stride": 1},
            input_shape=[N, C_in, input_dim, input_dim],
            gt_output_dim=1,
            op_create_func=_create_op_ir_pool,
        )
