#!/usr/bin/env python3

import unittest

import mobile_cv.arch.fbnet.antialias as antialias
import torch


TEST_CUDA = torch.cuda.is_available()


class TestAntiAlias(unittest.TestCase):
    def test_anti_alias(self):
        N, C_in, w, h = 10, 32, 14, 14
        op = antialias.Downsample(stride=2, channels=32)
        input = torch.rand([N, C_in, h, w], dtype=torch.float32)
        output = op(input)

        self.assertEqual(
            output.shape[:2],
            torch.Size([N, C_in]),
            "antialias.Downsample failed for shape {}.".format(input.shape),
        )

        self.assertEqual(
            output.shape[2:],
            torch.Size([h // 2, w // 2]),
            "antialias.Downsample failed for shape {}.".format(input.shape),
        )
