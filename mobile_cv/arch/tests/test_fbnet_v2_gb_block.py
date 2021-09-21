#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import mobile_cv.arch.fbnet_v2.gb_block as gb_block
import mobile_cv.arch.utils.helper as hp
import torch


TEST_CUDA = torch.cuda.is_available()


def create_test_gb(self, out_channels, ratio, op_args, input_shape, gt_output_dim):
    N, C_in, H, W = input_shape
    op = gb_block.GhostBottleneckBlock(
        in_channels=C_in, out_channels=out_channels, ratio=ratio, **op_args
    )
    print(op)

    input = torch.rand(input_shape, dtype=torch.float32)
    output = op(input)
    gt_out_channels = hp.get_divisible_by(out_channels, ratio)

    self.assertEqual(
        output.shape, torch.Size([N, gt_out_channels, gt_output_dim, gt_output_dim])
    )

    return op


class TestGBBlocks(unittest.TestCase):
    def test_gb_block(self):
        N, C_in, C_out, ratio = 2, 16, 16, 2
        input_dim = 7

        for bn_args in [
            "bn",
            {"name": "bn", "momentum": 0.003},
            {"name": "sync_bn", "momentum": 0.003},
        ]:
            with self.subTest(f"bn={bn_args}"):
                create_test_gb(
                    self,
                    C_out,
                    ratio,
                    op_args={
                        "expansion": 6,
                        "kernel_size": 3,
                        "stride": 1,
                        "bn_args": bn_args,
                    },
                    input_shape=[N, C_in, input_dim, input_dim],
                    gt_output_dim=input_dim,
                )

        with self.subTest(f"skip_bnrelu=True"):  # noqa
            create_test_gb(
                self,
                C_out,
                ratio,
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

    def test_gb_block_res_conn(self):
        N, C_in, C_out, ratio = 2, 16, 32, 2
        input_dim = 8

        create_test_gb(
            self,
            C_out,
            ratio,
            op_args={"expansion": 6, "kernel_size": 3, "stride": 1},
            input_shape=[N, C_in, input_dim, input_dim],
            gt_output_dim=input_dim,
        )

        create_test_gb(
            self,
            C_out,
            ratio,
            op_args={"expansion": 6, "kernel_size": 3, "stride": 2},
            input_shape=[N, C_in, input_dim, input_dim],
            gt_output_dim=input_dim // 2,
        )

    def test_gb_block_se(self):
        N, C_in, C_out, ratio = 2, 16, 32, 2
        input_dim = 8

        create_test_gb(
            self,
            C_out,
            ratio,
            op_args={
                "expansion": 6,
                "kernel_size": 3,
                "stride": 1,
                "se_args": "se_fc",
            },
            input_shape=[N, C_in, input_dim, input_dim],
            gt_output_dim=input_dim,
        )

        create_test_gb(
            self,
            C_out,
            ratio,
            op_args={
                "expansion": 6,
                "kernel_size": 3,
                "stride": 2,
                "se_args": {"name": "se_hsig", "relu_args": "hswish"},
            },
            input_shape=[N, C_in, input_dim, input_dim],
            gt_output_dim=input_dim // 2,
        )

    def test_gb_block_upsample(self):
        N, C_in, C_out, ratio = 2, 16, 32, 2
        input_dim = 8

        create_test_gb(
            self,
            C_out,
            ratio,
            op_args={
                "expansion": 6,
                "kernel_size": 3,
                "stride": -2,
                "se_args": "se_fc",
            },
            input_shape=[N, C_in, input_dim, input_dim],
            gt_output_dim=input_dim * 2,
        )

    def test_gb_block_ratio(self):
        N, C_in, C_out, ratio = 2, 16, 32, 3
        input_dim = 8

        create_test_gb(
            self,
            C_out,
            ratio,
            op_args={
                "expansion": 6,
                "kernel_size": 3,
                "stride": 1,
            },
            input_shape=[N, C_in, input_dim, input_dim],
            gt_output_dim=input_dim,
        )

        N, C_in, C_out, ratio = 2, 16, 36, 4
        input_dim = 8

        create_test_gb(
            self,
            C_out,
            ratio,
            op_args={
                "expansion": 6,
                "kernel_size": 3,
                "stride": 1,
            },
            input_shape=[N, C_in, input_dim, input_dim],
            gt_output_dim=input_dim,
        )

        N, C_in, C_out, ratio = 2, 16, 128, 5
        input_dim = 8

        create_test_gb(
            self,
            C_out,
            ratio,
            op_args={
                "expansion": 6,
                "kernel_size": 3,
                "stride": 1,
            },
            input_shape=[N, C_in, input_dim, input_dim],
            gt_output_dim=input_dim,
        )

    def test_gb_block_width_divisor(self):
        N, C_in, C_out, ratio = 2, 3, 4, 2
        input_dim = 8

        op = create_test_gb(
            self,
            C_out,
            ratio,
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
