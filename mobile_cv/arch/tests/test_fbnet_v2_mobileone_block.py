#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import mobile_cv.arch.fbnet_v2.mobileone_block as mobileone_block
import torch


class TestMobileOneBlock(unittest.TestCase):
    def test_mobileone_block_for_train_stride_1(self):
        for k in [1, 2, 4]:
            conv = mobileone_block.MobileOneBlock(
                in_channels=3,
                out_channels=3,
                stride=1,
                over_param_branches=k,
            )

            data = torch.ones((2, 3, 4, 4))
            out = conv(data)

            self.assertEqual(out.shape, (2, 3, 4, 4))

    def test_mobileone_block_for_train_stride_2(self):
        for k in [1, 2, 4]:
            conv = mobileone_block.MobileOneBlock(
                in_channels=3,
                out_channels=6,
                stride=2,
                over_param_branches=k,
            )

            data = torch.ones((2, 3, 8, 8))
            out = conv(data)

            self.assertEqual(out.shape, (2, 6, 4, 4))

    def test_mobileone_block_for_deploy_stride_1(self):
        conv = mobileone_block.MobileOneBlock(
            in_channels=3,
            out_channels=3,
            stride=1,
            deploy=True,
        )

        data = torch.ones((2, 3, 4, 4))
        out = conv(data)

        self.assertEqual(out.shape, (2, 3, 4, 4))

    def test_mobileone_block_for_deploy_stride_2(self):
        conv = mobileone_block.MobileOneBlock(
            in_channels=3,
            out_channels=6,
            stride=2,
            deploy=True,
        )

        data = torch.ones((2, 3, 8, 8))
        out = conv(data)

        self.assertEqual(out.shape, (2, 6, 4, 4))
