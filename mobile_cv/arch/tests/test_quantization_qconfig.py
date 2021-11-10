#!/usr/bin/env python3

import unittest

import mobile_cv.arch.fbnet_v2.basic_blocks as bb
import torch
from mobile_cv.arch.quantization.observer import (
    FixedMinMaxObserver,
    UpdatableSymmetricMovingAverageMinMaxObserver,
    UpdateableReLUMovingAverageMinMaxObserver,
)
from mobile_cv.arch.quantization.qconfig import (
    fixed_minmax_config,
    updateable_relu_moving_avg_minmax_config,
    updateable_symmetric_moving_avg_minmax_config,
)
from torch.ao.quantization.observer import MinMaxObserver


class TestQconfig(unittest.TestCase):
    def test_qconfig_conv(self):
        """Check that qconfig is applied to conv"""
        for qconfig, act_obs_type, weight_obs_type in [
            [fixed_minmax_config, FixedMinMaxObserver, MinMaxObserver],
            [
                updateable_symmetric_moving_avg_minmax_config,
                UpdatableSymmetricMovingAverageMinMaxObserver,
                MinMaxObserver,
            ],
            [
                updateable_relu_moving_avg_minmax_config,
                UpdateableReLUMovingAverageMinMaxObserver,
                MinMaxObserver,
            ],
        ]:
            model = bb.ConvBNRelu(1, 1, "conv", None, None)
            model.qconfig = qconfig
            qat_model = torch.ao.quantization.prepare_qat(model, inplace=False)
            self.assertTrue(
                isinstance(
                    qat_model.conv.activation_post_process.activation_post_process,
                    act_obs_type,
                )
            )
            self.assertTrue(
                isinstance(
                    qat_model.conv.weight_fake_quant.activation_post_process,
                    weight_obs_type,
                )
            )
