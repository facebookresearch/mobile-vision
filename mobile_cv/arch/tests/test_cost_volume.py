#!/usr/bin/env python3

import random
import unittest

import torch
import torch.nn.functional as F
from mobile_cv.arch.fbnet_v2.cost_volume import CorrVolume1DBlock


class TestCorrVolume(object):
    def _mult_with_shape(self, shape):
        """Corr volume is element wise multiplication if
        * k = 0
        * displacement = 0
        """
        assert shape[1] == 1
        model = self.get_corr_volume_block()(k=0, d=0)
        x1 = torch.randn(*shape)
        x2 = torch.randn(*shape)
        output = model(x1, x2)
        self.assertEqual(output.shape, torch.Size([*shape]))
        torch.testing.assert_allclose(output, x1 * x2)

    def _dot_with_shape(self, shape):
        """Corr volume is element wise dot product along channel dimension if
        * k = 0
        * displacement = 0
        """
        n, c, h, w = shape
        model = self.get_corr_volume_block()(k=0, d=0)
        x1 = torch.randn(*shape)
        x2 = torch.randn(*shape)
        output = model(x1, x2)
        gt = (x1 * x2).sum(dim=1).unsqueeze(1) / c
        self.assertEqual(output.shape, torch.Size([n, 1, h, w]))
        torch.testing.assert_allclose(output, gt)

    def test_mult_scalar(self):
        """Check mult case with scalar input"""
        self._mult_with_shape((1, 1, 1, 1))

    def test_mult_1d(self):
        """Check mult case with 1d input"""
        w = random.randint(2, 10)
        self._mult_with_shape((1, 1, 1, w))

    def test_mult_2d(self):
        """Check mult case with 2d input"""
        w = random.randint(2, 10)
        h = random.randint(2, 10)
        self._mult_with_shape((1, 1, h, w))

    def test_mult_2d_batch(self):
        """Check mult case with 2d input with batch"""
        w = random.randint(1, 10)
        h = random.randint(1, 10)
        n = random.randint(2, 10)
        self._mult_with_shape((n, 1, h, w))

    def test_dot_nchannels_scalar(self):
        """Check dot case nchannels with scalar input"""
        c = random.randint(2, 10)
        self._dot_with_shape((1, c, 1, 1))

    def test_dot_nchannels_1d(self):
        """Check dot case nchannels with 1d input"""
        c = random.randint(2, 10)
        w = random.randint(2, 10)
        self._dot_with_shape((1, c, 1, w))

    def test_dot_nchannels_2d(self):
        """Check dot case nchannels with 2d input"""
        c = random.randint(2, 10)
        w = random.randint(1, 10)
        h = random.randint(2, 10)
        self._dot_with_shape((1, c, h, w))

    def test_dot_nchannels_2d_batch(self):
        """Check dot case nchannels with batch"""
        c = random.randint(2, 10)
        w = random.randint(1, 10)
        h = random.randint(1, 10)
        n = random.randint(2, 10)
        self._dot_with_shape((n, c, h, w))

    def test_displacement_scalar(self):
        """Displacement > 0 with scalar inputs

        Given displacement d=1, we have (1+1) = 2 output channels
        Scalar input means we are taking correlations outside of the feature
        map bounds. There is implicit zero padding so these values should
        be zero.
        """
        d = random.randint(1, 10)
        model = self.get_corr_volume_block()(k=0, d=d)

        x1 = torch.randn(1, 1, 1, 1)
        x2 = torch.randn(1, 1, 1, 1)
        output = model(x1, x2)
        gt = F.pad(x1 * x2, (0, 0, 0, 0, 0, d))

        self.assertEqual(output.shape, torch.Size([1, d + 1, 1, 1]))
        torch.testing.assert_allclose(output, gt)

    def test_displacement_scalar_batch_channels(self):
        """Displacement with batch size and channels > 1"""
        d = random.randint(1, 10)

        model = self.get_corr_volume_block()(k=0, d=d)

        n = random.randint(1, 10)
        c = random.randint(1, 10)
        x1 = torch.randn(n, c, 1, 1)
        x2 = torch.randn(n, c, 1, 1)
        output = model(x1, x2)

        gt = (x1 * x2).sum(dim=1).unsqueeze(1) / c
        gt = F.pad(gt, (0, 0, 0, 0, 0, d))
        self.assertEqual(output.shape, torch.Size([n, d + 1, 1, 1]))
        torch.testing.assert_allclose(output, gt)

    def test_displacement_1d(self):
        """Displacement > 0 with 1d inputs

        Given displacement d=1,
        and 1d inputs we have:

            inputs = [x00 x01], [y00 y01]
            output[0, :, 0, 0] = [0       0
                                  x00*y00 x01*y01]
            output[0, :, 0, 1] = [ 0       0
                                  x01*y00  0]
        """
        d = 1
        model = self.get_corr_volume_block()(k=0, d=d)
        x1 = torch.randn(1, 1, 1, 2)
        x2 = torch.randn(1, 1, 1, 2)
        output = model(x1, x2)

        gt = torch.zeros(1, d + 1, 1, 2)

        gt[0, 0, 0, 0] = x1[0, 0, 0, 0] * x2[0, 0, 0, 0]  # disparity=0, x=0
        gt[0, 0, 0, 1] = x1[0, 0, 0, 1] * x2[0, 0, 0, 1]  # disparity=0, x=1
        gt[0, 1, 0, 1] = x1[0, 0, 0, 1] * x2[0, 0, 0, 0]  # disparity=1, x=0

        self.assertEqual(output.shape, torch.Size([1, d + 1, 1, 2]))
        torch.testing.assert_allclose(output, gt)

    def test_displacement_2d(self):
        """Displacement > 0 with 2d inputs"""
        d = 1
        model = self.get_corr_volume_block()(k=0, d=d)
        x1 = torch.randn(1, 1, 2, 2)
        x2 = torch.randn(1, 1, 2, 2)
        output = model(x1, x2)

        gt = torch.zeros(1, d + 1, 2, 2)
        gt[0, 0, :, 0] = x1[0, 0, :, 0] * x2[0, 0, :, 0]  # disparity=0, x=0
        gt[0, 0, :, 1] = x1[0, 0, :, 1] * x2[0, 0, :, 1]  # disparity=0, x=1
        gt[0, 1, :, 1] = x1[0, 0, :, 1] * x2[0, 0, :, 0]  # disparity=1, x=0

        self.assertEqual(output.shape, torch.Size([1, d + 1, 2, 2]))
        torch.testing.assert_allclose(output, gt)

    def test_displacement_batch_1d(self):
        """Check displacement with batch > 1"""
        b = random.randint(2, 10)
        d = 1
        model = self.get_corr_volume_block()(k=0, d=d)
        x1 = torch.randn(b, 1, 1, 2)
        x2 = torch.randn(b, 1, 1, 2)
        output = model(x1, x2)
        gt = torch.zeros(b, d + 1, 1, 2)
        gt[:, 0, 0, 0] = x1[:, 0, 0, 0] * x2[:, 0, 0, 0]  # disparity=0, x=0
        gt[:, 0, 0, 1] = x1[:, 0, 0, 1] * x2[:, 0, 0, 1]  # disparity=0, x=1
        gt[:, 1, 0, 1] = x1[:, 0, 0, 1] * x2[:, 0, 0, 0]  # disparity=1, x=0

        self.assertEqual(output.shape, torch.Size([b, d + 1, 1, 2]))
        torch.testing.assert_allclose(output, gt)

    def test_displacement_channel_1d(self):
        """Check displacement with channels > 1"""
        c = random.randint(2, 10)
        d = 1
        model = self.get_corr_volume_block()(k=0, d=d)
        x1 = torch.randn(1, c, 1, 2)
        x2 = torch.randn(1, c, 1, 2)
        output = model(x1, x2)

        gt = torch.zeros(1, d + 1, 1, 2)
        gt[0, 0, 0, 0] = (
            torch.dot(x1[0, :, 0, 0], x2[0, :, 0, 0]) / c
        )  # disparity=0, x=0
        gt[0, 0, 0, 1] = (
            torch.dot(x1[0, :, 0, 1], x2[0, :, 0, 1]) / c
        )  # disparity=0, x=1
        gt[0, 1, 0, 1] = (
            torch.dot(x1[0, :, 0, 1], x2[0, :, 0, 0]) / c
        )  # disparity=1, x=0

        self.assertEqual(output.shape, torch.Size([1, d + 1, 1, 2]))
        torch.testing.assert_allclose(output, gt)

    def get_corr_volume_block(self):
        raise NotImplementedError


class TestCorrVolume1DBlock(unittest.TestCase, TestCorrVolume):
    def get_corr_volume_block(self):
        return CorrVolume1DBlock
