#!/usr/bin/env python3

import unittest

import mobile_cv.arch.fbnet.oct_conv as oct_conv
import torch


TEST_CUDA = torch.cuda.is_available()


class TestOctConv(unittest.TestCase):
    def test_oct_conv(self):
        N, C_in, C_out = 10, 32, 64
        op = oct_conv.OctIRFBlock(C_in, C_out, 6, 1, 0.25)
        input = torch.rand([N, C_in, 14, 14], dtype=torch.float32)
        output = op(input)

        self.assertEqual(
            output.shape[:2],
            torch.Size([N, C_out]),
            "OctIRFBlock failed for shape {}.".format(input.shape),
        )
