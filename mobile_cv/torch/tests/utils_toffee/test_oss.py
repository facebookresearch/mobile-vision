#!/usr/bin/env python3

import unittest

import torch
from mobile_cv.torch.utils_toffee.alias import alias


class TestOSS(unittest.TestCase):
    def test_alias(self):
        """test the function can be imported, and acts as identity function"""
        x = torch.tensor(1)
        y = alias(x, name="x")
        torch.testing.assert_close(y, x)
