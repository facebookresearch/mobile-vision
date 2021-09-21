#!/usr/bin/env python3
# usage: buck test mobile-vision/mobile_cv/mobile_cv/arch/tests:test_fbnet_utils


import unittest

import common.utils_pytorch.model_utils as model_utils
import numpy as np
import torch
import torch.nn as nn
from mobile_cv.arch.fbnet.fbnet_building_blocks import ConvBNRelu, QuantConvBNRelu
from mobile_cv.arch.utils.fbnet_utils import fuse_convbnrelu


class TestFBNetUtils(unittest.TestCase):
    def test_fuse_convbnrelu_removed_bn(self):
        """Check that batch norm is removed"""
        model_A = ConvBNRelu(1, 1, 1)
        model_B = fuse_convbnrelu(model_A)
        assert model_utils.has_module(model_A, nn.BatchNorm2d)
        assert not model_utils.has_module(model_B, nn.BatchNorm2d)
        fuse_convbnrelu(model_A, inplace=True)
        assert not model_utils.has_module(model_A, nn.BatchNorm2d)

    def test_fuse_convbnrelu_same_output(self):
        """Check fused model produces the same output"""

        def _check_same_output(model_A, c_in):
            """Wrapper to apply fuse_convbnrelu and check if models are the same"""
            # hack until we change quantconv2d so that weight q is disabled
            def fill_quantconv2d_bitwidth(model):
                if "QuantConv2d" in str(type(model)):
                    model.tempBitW = 32

            model_A.apply(fill_quantconv2d_bitwidth)

            model_A.apply(model_utils.init_params_tenths)
            model_B = fuse_convbnrelu(model_A)
            data = torch.from_numpy(0.01 * np.ones((1, c_in, 32, 32), dtype=np.float32))
            with torch.no_grad():
                model_A.eval()
                model_B.eval()
                assert torch.allclose(model_A(data), model_B(data), atol=1e-05)

        _check_same_output(ConvBNRelu(3, 2, 1), 3)
        _check_same_output(ConvBNRelu(2, 3, 1), 2)
        _check_same_output(ConvBNRelu(3, 3, 2), 3)
        _check_same_output(nn.Sequential(ConvBNRelu(3, 2, 2), ConvBNRelu(2, 3, 3)), 3)
        _check_same_output(QuantConvBNRelu(3, 2, 3), 3)
        _check_same_output(QuantConvBNRelu(3, 2, 3, use_relu=None), 3)
        _check_same_output(QuantConvBNRelu(3, 2, 3, bn_type=None), 3)
        _check_same_output(QuantConvBNRelu(3, 2, 3, use_relu=None, bn_type=None), 3)
