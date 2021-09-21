#!/usr/bin/env python3

import unittest

import mobile_cv.arch.fbnet.shift_conv as shift_conv
import torch


TEST_CUDA = torch.cuda.is_available()


class TestShiftConv(unittest.TestCase):
    def test_shift_irf(self):
        torch.manual_seed(0)
        N, C_in, C_out = 10, 32, 64
        op = shift_conv.ShiftConvIRF(C_in, C_out, 6, 1, shift_count=1)
        input = torch.rand([N, C_in, 14, 14], dtype=torch.float32)
        output = op(input)

        self.assertEqual(
            output.shape[:2],
            torch.Size([N, C_out]),
            "ShiftConvIRF failed for shape {}.".format(input.shape),
        )
