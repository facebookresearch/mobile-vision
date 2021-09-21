#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import mobile_cv.arch.fbnet_v2.fbnet_builder as fbnet_builder
import mobile_cv.arch.fbnet_v2.fbnet_modeldef_cls as fbnet_modeldef_cls
import mobile_cv.arch.fbnet_v2.sg_block as sg_block
import mobile_cv.lut.lib.pt.flops_utils as flops_utils
import torch


def create_test_sg(self, out_channels, op_args, input_shape, gt_output_dim):
    N, C_in, H, W = input_shape
    op = sg_block.SGBlock(in_channels=C_in, out_channels=out_channels, **op_args).eval()
    inputs = torch.rand(input_shape, dtype=torch.float32)
    output = flops_utils.print_model_flops(op, [inputs])

    self.assertEqual(
        output.shape, torch.Size([N, out_channels, gt_output_dim, gt_output_dim])
    )

    return op


def _create_and_run(self, arch_name, model_arch):
    arch = fbnet_builder.unify_arch_def(model_arch, ["blocks"])
    builder = fbnet_builder.FBNetBuilder(basic_args=arch.get("basic_args", None))
    model = builder.build_blocks(arch["blocks"], dim_in=3)
    model.eval()
    res = model_arch.get("input_size", 224)
    inputs = (torch.zeros([1, 3, res, res]),)
    output = flops_utils.print_model_flops(model, inputs)
    self.assertEqual(output.shape[0], 1)


class TestSGBlocks(unittest.TestCase):
    def test_sg_block(self):
        N, C_in, C_out = 2, 96, 96
        input_dim = 7

        for bn_args in [
            "bn",
            {"name": "bn", "momentum": 0.003},
            {"name": "sync_bn", "momentum": 0.003},
        ]:
            with self.subTest(f"bn={bn_args}"):
                create_test_sg(
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
                )

        with self.subTest("skip_bnrelu=True"):
            create_test_sg(
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

    def test_sg_block_res_conn(self):
        N, C_in, C_out = 2, 96, 48
        input_dim = 8

        create_test_sg(
            self,
            C_out,
            op_args={"expansion": 6, "kernel_size": 3, "stride": 1},
            input_shape=[N, C_in, input_dim, input_dim],
            gt_output_dim=input_dim,
        )

        create_test_sg(
            self,
            C_out,
            op_args={"expansion": 6, "kernel_size": 3, "stride": 2},
            input_shape=[N, C_in, input_dim, input_dim],
            gt_output_dim=input_dim // 2,
        )

    def test_sg_block_se(self):
        N, C_in, C_out = 2, 96, 48
        input_dim = 8

        create_test_sg(
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

        create_test_sg(
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

    def test_sg_block_upsample(self):
        N, C_in, C_out = 2, 96, 48
        input_dim = 8

        create_test_sg(
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

    def test_sg_block_width_divisor(self):
        N, C_in, C_out = 2, 15, 20
        input_dim = 8

        op = create_test_sg(
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
        self.assertEqual(op.pwl.out_channels, 8)

    def test_sg_block_e1(self):
        N, C_in, C_out = 2, 32, 32
        input_dim = 8

        create_test_sg(
            self,
            C_out,
            op_args={
                "expansion": 1,
                "kernel_size": 5,
                "stride": 1,
            },
            input_shape=[N, C_in, input_dim, input_dim],
            gt_output_dim=input_dim,
        )

        create_test_sg(
            self,
            C_out,
            op_args={
                "expansion": 1,
                "kernel_size": 3,
                "stride": 2,
            },
            input_shape=[N, C_in, input_dim, input_dim],
            gt_output_dim=input_dim // 2,
        )

    def test_sg_arches(self):
        arch_factory = fbnet_modeldef_cls.MODEL_ARCH
        selected_archs = [
            # "default",
            # "sg_default",
            # "FBNetV3_ASG",
            # "FBNetV3_B",
            # "FBNetV3_BSG",
            # "FBNetV3_C",
            "FBNetV3_CSG",
        ]

        for name in selected_archs:
            with self.subTest(arch=name):
                print("Testing {}".format(name))
                model_arch = arch_factory.get(name)
                _create_and_run(self, name, model_arch)
