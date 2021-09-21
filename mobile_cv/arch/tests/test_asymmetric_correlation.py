#!/usr/bin/env python3

import random
import unittest

import torch
import torch.nn.functional as F
from mobile_cv.arch.fbnet_v2.asymmetric_correlation import (
    NaiveAsymmetricCorrelationBlock,
)


class TestAsymmetricCorrelation(object):
    def _mult_with_shape(self, shape):
        """Correlation is element wise multiplication if
        * k = 0
        * displacement = 0 in all axises
        * stride = 1
        * channels = 1
        """
        assert shape[1] == 1
        model = self.get_asymmetric_correlation_block()(
            k=0, dw_pos=0, dw_neg=0, dh_pos=0, dh_neg=0, s1=1, s2=1
        )
        x1 = torch.randn(*shape)
        x2 = torch.randn(*shape)
        output = model(x1, x2)
        self.assertEqual(output.shape, torch.Size([*shape]))
        torch.testing.assert_allclose(output, x1 * x2)

    def _dot_with_shape(self, shape):
        """Correlation is element wise dot product along channel dimension if
            * k = 0
            * displacement = 0
            * stride = 1

        There is some numerical differences (e.g., 0.10231099 vs. 0.10231098), could
        be because this runs mult then whereas naivecorrelation uses dot. We use
        allclose rather than eq
        """
        n, c, h, w = shape
        model = self.get_asymmetric_correlation_block()(
            k=0, dw_pos=0, dw_neg=0, dh_pos=0, dh_neg=0, s1=1, s2=1
        )
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

        Given displacement dw_pos = dw_neg = dh_pos = dh_neg=1,
        we have (1+1+1)*(1+1+1) = 9 output channels
        Scalar input means we are taking correlations outside of the feature
        map bounds. There is implicit zero padding so these values should
        be zero.

            inputs = [x00], [y00]
            output = [ 0    0    0
                       0 x00*y00 0
                       0    0    0]

        There are (dw_pos+dw_neg+1)(dh_pos+dh_neg+1) - 1 zero values.
        """
        dw_pos = random.randint(1, 10)
        dw_neg = random.randint(1, 10)
        dh_pos = random.randint(1, 10)
        dh_neg = random.randint(1, 10)
        Dw = dw_pos + dw_neg + 1
        Dh = dh_pos + dh_neg + 1

        model = self.get_asymmetric_correlation_block()(
            k=0, dw_pos=dw_pos, dw_neg=dw_neg, dh_pos=dh_pos, dh_neg=dh_neg, s1=1, s2=1
        )

        x1 = torch.randn(1, 1, 1, 1)
        x2 = torch.randn(1, 1, 1, 1)
        output = model(x1, x2)

        gt = F.pad(x1 * x2, (0, 0, 0, 0, dh_neg * Dw + dw_neg, dh_pos * Dw + dw_pos))
        self.assertEqual(output.shape, torch.Size([1, Dw * Dh, 1, 1]))
        torch.testing.assert_allclose(output, gt)

    def test_displacement_scalar_batch_channels(self):
        """Displacement with batch size and channels > 1"""
        dw_pos = random.randint(1, 10)
        dw_neg = random.randint(1, 10)
        dh_pos = random.randint(1, 10)
        dh_neg = random.randint(1, 10)
        Dw = dw_pos + dw_neg + 1
        Dh = dh_pos + dh_neg + 1

        model = self.get_asymmetric_correlation_block()(
            k=0, dw_pos=dw_pos, dw_neg=dw_neg, dh_pos=dh_pos, dh_neg=dh_neg, s1=1, s2=1
        )

        n = random.randint(1, 10)
        c = random.randint(1, 10)
        x1 = torch.randn(n, c, 1, 1)
        x2 = torch.randn(n, c, 1, 1)
        output = model(x1, x2)
        gt = (x1 * x2).sum(dim=1).unsqueeze(1) / c
        gt = F.pad(gt, (0, 0, 0, 0, dh_neg * Dw + dw_neg, dh_pos * Dw + dw_pos))
        self.assertEqual(output.shape, torch.Size([n, Dw * Dh, 1, 1]))
        torch.testing.assert_allclose(output, gt)

    def test_displacement_1d(self):
        """Displacement > 0 with 1d inputs

        Given displacement dw_pos = dw_neg = dh_pos = dh_neg=1,
        and 1d inputs we have:

            inputs = [x00 x01], [y00 y01]
            output[0, :, 0, 0] = [0          0       0
                                  0      x00*y00 x00*y01
                                  0          0       0]
            output[0, :, 0, 1] = [0          0       0
                                  x01*y00 x01*y01    0
                                  0          0       0]
        """
        dw_pos = dw_neg = dh_pos = dh_neg = d = 1
        D = 2 * d + 1
        model = self.get_asymmetric_correlation_block()(
            k=0, dw_pos=dw_pos, dw_neg=dw_neg, dh_pos=dh_pos, dh_neg=dh_neg, s1=1, s2=1
        )
        x1 = torch.randn(1, 1, 1, 2)
        x2 = torch.randn(1, 1, 1, 2)
        output = model(x1, x2)
        corr = torch.matmul(torch.transpose(x1, 2, 3), x2)
        gt = torch.zeros(1, 9, 1, 2)
        gt[0, 4:6, 0, 0] = corr[0, 0, 0, :]
        gt[0, 3:5, 0, 1] = corr[0, 0, 1, :]
        self.assertEqual(output.shape, torch.Size([1, D ** 2, 1, 2]))
        torch.testing.assert_allclose(output, gt)

    def test_asymmetric_displacement_1d(self):
        """asymmetric Displacement >= 0 with 1d inputs

        Given displacement dw_neg = 1, dw_pos = dh_pos = dh_neg = 0,
        and 1d inputs we have:

            inputs = [x00 x01], [y00 y01]
            output[0, :, 0, 0] = [0       x00*y00]
            output[0, :, 0, 1] = [x01*y00 x01*y01]
        """
        dw_neg = 1
        dw_pos = dh_pos = dh_neg = 0
        Dw = dw_pos + dw_neg + 1
        Dh = dh_pos + dh_neg + 1
        model = self.get_asymmetric_correlation_block()(
            k=0, dw_pos=dw_pos, dw_neg=dw_neg, dh_pos=dh_pos, dh_neg=dh_neg, s1=1, s2=1
        )
        x1 = torch.randn(1, 1, 1, 2)
        x2 = torch.randn(1, 1, 1, 2)
        output = model(x1, x2)
        corr = torch.matmul(torch.transpose(x1, 2, 3), x2)
        gt = torch.zeros(1, 2, 1, 2)
        gt[0, 1:2, 0, 0] = corr[0, 0, 0, 0:1]
        gt[0, 0:2, 0, 1] = corr[0, 0, 1, :]
        self.assertEqual(output.shape, torch.Size([1, Dw * Dh, 1, 2]))
        torch.testing.assert_allclose(output, gt)

    def test_displacement_2d(self):
        """Displacement > 0 with 2d inputs

        Given displacement dw_pos = dw_neg = dh_pos = dh_neg=1,
        and 2d inputs we have:

            inputs = [x00 x01   [y00 y01
                      x10 x11],  y10 y11]
            output[0, :, 0, 0] = [0          0       0
                                  0      x00*y00 x00*y01
                                  0      x00*y10 x00*y11]
            output[0, :, 0, 1] = [0          0       0
                                  x01*y00 x01*y01    0
                                  x01*y10 x01*y11    0]
            output[0, :, 1, 0] = [0       x10*y00 x10*y01
                                  0       x10*y10 x10*y11
                                  0          0       0]
            output[0, :, 1, 2] = [x11*y00 x11*y01    0
                                  x11*y10 x11*y11    0
                                  0          0       0]
        """
        dw_pos = dw_neg = dh_pos = dh_neg = d = 1
        D = 2 * d + 1
        model = self.get_asymmetric_correlation_block()(
            k=0, dw_pos=dw_pos, dw_neg=dw_neg, dh_pos=dh_pos, dh_neg=dh_neg, s1=1, s2=1
        )
        x1 = torch.randn(1, 1, 2, 2)
        x2 = torch.randn(1, 1, 2, 2)
        output = model(x1, x2)
        corr = torch.matmul(x1.view(4, 1), x2.view(1, 4))
        corr = torch.cat([corr[:, :2], torch.zeros(4, 1), corr[:, 2:]], dim=1)
        gt = torch.zeros(1, D ** 2, 2, 2)
        gt[0, 4:, 0, 0] = corr[0, :]
        gt[0, 3:8, 0, 1] = corr[1, :]
        gt[0, 1:6, 1, 0] = corr[2, :]
        gt[0, :5, 1, 1] = corr[3, :]
        self.assertEqual(output.shape, torch.Size([1, D ** 2, 2, 2]))
        torch.testing.assert_allclose(output, gt)

    def test_displacement_batch_1d(self):
        """Check displacement with batch > 1"""
        b = random.randint(2, 10)
        dw_pos = dw_neg = dh_pos = dh_neg = d = 1
        D = 2 * d + 1
        model = self.get_asymmetric_correlation_block()(
            k=0, dw_pos=dw_pos, dw_neg=dw_neg, dh_pos=dh_pos, dh_neg=dh_neg, s1=1, s2=1
        )
        x1 = torch.randn(b, 1, 1, 2)
        x2 = torch.randn(b, 1, 1, 2)
        output = model(x1, x2)
        gt = torch.zeros(b, 9, 1, 2)

        for i in range(b):
            corr = torch.matmul(torch.transpose(x1[i, :], 1, 2), x2[i, :])
            gt[i, 4:6, 0, 0] = corr[0, 0, :]
            gt[i, 3:5, 0, 1] = corr[0, 1, :]
        self.assertEqual(output.shape, torch.Size([b, D ** 2, 1, 2]))
        torch.testing.assert_allclose(output, gt)

    def test_displacement_channel_1d(self):
        """Check displacement with channels > 1"""
        c = random.randint(2, 10)
        dw_pos = dw_neg = dh_pos = dh_neg = d = 1
        D = 2 * d + 1
        model = self.get_asymmetric_correlation_block()(
            k=0, dw_pos=dw_pos, dw_neg=dw_neg, dh_pos=dh_pos, dh_neg=dh_neg, s1=1, s2=1
        )
        x1 = torch.randn(1, c, 1, 2)
        x2 = torch.randn(1, c, 1, 2)
        output = model(x1, x2)
        gt = torch.zeros(1, 9, 1, 2)
        gt[0, 4, 0, 0] = torch.dot(x1[0, :, 0, 0], x2[0, :, 0, 0]) / c
        gt[0, 5, 0, 0] = torch.dot(x1[0, :, 0, 0], x2[0, :, 0, 1]) / c
        gt[0, 3, 0, 1] = torch.dot(x1[0, :, 0, 1], x2[0, :, 0, 0]) / c
        gt[0, 4, 0, 1] = torch.dot(x1[0, :, 0, 1], x2[0, :, 0, 1]) / c
        self.assertEqual(output.shape, torch.Size([1, D ** 2, 1, 2]))
        torch.testing.assert_allclose(output, gt)

    def test_stride2_1dx(self):
        """Stride2 > 1 quantizes the correlations in the x2 input

        Given displacement dw_pos = dw_neg = dh_pos = dh_neg = 2,
        s2 = 2, w=3

            inputs = [x00 x01, x02], [y00 y01, y02]
            output[0, :, 0, 0] = [0          0       0
                                  0       x00*y00 x00*y02
                                  0          0       0]
            output[0, :, 0, 1] = [0          0       0
                                  0       x01*y01    0
                                  0          0       0]
            output[0, :, 0, 1] = [0          0       0
                                  x02*y00 x02*y02    0
                                  0          0       0]
        """
        s2 = 2
        dw_pos = dw_neg = dh_pos = dh_neg = d = 2
        D = 2 * d // s2 + 1
        model = self.get_asymmetric_correlation_block()(
            k=0, dw_pos=dw_pos, dw_neg=dw_neg, dh_pos=dh_pos, dh_neg=dh_neg, s1=1, s2=s2
        )
        x1 = torch.randn(1, 1, 1, 3)
        x2 = torch.randn(1, 1, 1, 3)
        output = model(x1, x2)
        corr = torch.matmul(torch.transpose(x1, 2, 3), x2)
        gt = torch.zeros(1, 9, 1, 3)
        gt[0, 4, 0, 0] = corr[0, 0, 0, 0]
        gt[0, 5, 0, 0] = corr[0, 0, 0, 2]
        gt[0, 4, 0, 1] = corr[0, 0, 1, 1]
        gt[0, 3, 0, 2] = corr[0, 0, 2, 0]
        gt[0, 4, 0, 2] = corr[0, 0, 2, 2]
        self.assertEqual(output.shape, torch.Size([1, D ** 2, 1, 3]))
        torch.testing.assert_allclose(output, gt)

    def test_stride2_1dy(self):
        """Stride2 > 1 quantizes the correlations in the x2 input

        Given displacement dw_pos = dw_neg = dh_pos = dh_neg = 2, s2 = 2, h=3

            inputs = [x00   [y00
                      x01    y01
                      x02],  y02]
            output[0, :, 0, 0] = [0          0       0
                                  0       x00*y00    0
                                  0       x00*y02    0]
            output[0, :, 1, 0] = [0          0       0
                                  0       x01*y01    0
                                  0          0       0]
            output[0, :, 2, 0] = [0       x02*y00    0
                                  0       x02*y02    0
                                  0          0       0]
        """
        s2 = 2
        dw_pos = dw_neg = dh_pos = dh_neg = d = 2
        D = 2 * d // s2 + 1
        model = self.get_asymmetric_correlation_block()(
            k=0, dw_pos=dw_pos, dw_neg=dw_neg, dh_pos=dh_pos, dh_neg=dh_neg, s1=1, s2=s2
        )
        x1 = torch.randn(1, 1, 3, 1)
        x2 = torch.randn(1, 1, 3, 1)
        output = model(x1, x2)
        corr = torch.matmul(x1, torch.transpose(x2, 2, 3))
        gt = torch.zeros(1, D ** 2, 3, 1)
        gt[0, 4, 0, 0] = corr[0, 0, 0, 0]
        gt[0, 7, 0, 0] = corr[0, 0, 0, 2]
        gt[0, 4, 1, 0] = corr[0, 0, 1, 1]
        gt[0, 1, 2, 0] = corr[0, 0, 2, 0]
        gt[0, 4, 2, 0] = corr[0, 0, 2, 2]
        self.assertEqual(output.shape, torch.Size([1, D ** 2, 3, 1]))
        torch.testing.assert_allclose(output, gt)

    def test_stride2_batch_1dx(self):
        """Check stride > 1, batch > 1"""
        b = random.randint(2, 10)
        s2 = 2
        dw_pos = dw_neg = dh_pos = dh_neg = d = 2
        D = 2 * d // s2 + 1
        model = self.get_asymmetric_correlation_block()(
            k=0, dw_pos=dw_pos, dw_neg=dw_neg, dh_pos=dh_pos, dh_neg=dh_neg, s1=1, s2=s2
        )
        x1 = torch.randn(b, 1, 1, 3)
        x2 = torch.randn(b, 1, 1, 3)
        output = model(x1, x2)
        gt = torch.zeros(b, D ** 2, 1, 3)
        for i in range(b):
            corr = torch.matmul(torch.transpose(x1[i, :], 1, 2), x2[i, :])
            gt[i, 4, 0, 0] = corr[0, 0, 0]
            gt[i, 5, 0, 0] = corr[0, 0, 2]
            gt[i, 4, 0, 1] = corr[0, 1, 1]
            gt[i, 3, 0, 2] = corr[0, 2, 0]
            gt[i, 4, 0, 2] = corr[0, 2, 2]
        self.assertEqual(output.shape, torch.Size([b, D ** 2, 1, 3]))
        torch.testing.assert_allclose(output, gt)

    def test_stride2_channel_1dx(self):
        """Check channel > 1, stride > 1

        Given displacement d = 2, s2 = 2, w=3

            inputs = [x00 x01, x02], [y00 y01, y02]
            output[0, :, 0, 0] = [0             0          0
                                  0         <x00, y00> <x00, y02>
                                  0             0          0] / c
            output[0, :, 0, 1] = [0             0          0
                                  0         <x01, y01>     0
                                  0             0          0] / c
            output[0, :, 0, 1] = [0             0          0
                                  <x02, y00> <x02, y02>    0
                                  0             0          0] . c
        """
        c = random.randint(2, 10)
        s2 = 2
        dw_pos = dw_neg = dh_pos = dh_neg = d = 2
        D = 2 * d // s2 + 1
        model = self.get_asymmetric_correlation_block()(
            k=0, dw_pos=dw_pos, dw_neg=dw_neg, dh_pos=dh_pos, dh_neg=dh_neg, s1=1, s2=s2
        )
        x1 = torch.randn(1, c, 1, 3)
        x2 = torch.randn(1, c, 1, 3)
        output = model(x1, x2)
        gt = torch.zeros(1, D ** 2, 1, 3)
        gt[0, 4, 0, 0] = torch.dot(x1[0, :, 0, 0], x2[0, :, 0, 0]) / c
        gt[0, 5, 0, 0] = torch.dot(x1[0, :, 0, 0], x2[0, :, 0, 2]) / c
        gt[0, 4, 0, 1] = torch.dot(x1[0, :, 0, 1], x2[0, :, 0, 1]) / c
        gt[0, 3, 0, 2] = torch.dot(x1[0, :, 0, 2], x2[0, :, 0, 0]) / c
        gt[0, 4, 0, 2] = torch.dot(x1[0, :, 0, 2], x2[0, :, 0, 2]) / c
        self.assertEqual(output.shape, torch.Size([1, D ** 2, 1, 3]))
        torch.testing.assert_allclose(output, gt)

    def test_stride1_1dx(self):
        """Stride1 > 1 changes the size of the output

        Given displacement s1 = 2, d = 0, s2 = 1, w=3

            inputs = [x00 x01, x02], [y00 y01, y02]
            output[0, 0, 0, 0] = [x00*y00]
            output[0, 0, 0, 1] = [x02*y02]
        """

        s1 = 2
        model = self.get_asymmetric_correlation_block()(
            k=0, dw_pos=0, dw_neg=0, dh_pos=0, dh_neg=0, s1=s1, s2=1
        )
        x1 = torch.randn(1, 1, 1, 3)
        x2 = torch.randn(1, 1, 1, 3)
        output = model(x1, x2)
        gt = torch.zeros(1, 1, 1, 2)
        corr = x1 * x2
        gt[0, 0, 0, 0] = corr[0, 0, 0, 0]
        gt[0, 0, 0, 1] = corr[0, 0, 0, 2]

        self.assertEqual(output.shape, torch.Size([1, 1, 1, 2]))
        torch.testing.assert_allclose(output, gt)

    def test_stride1_1dy(self):
        """Stride1 > 1 changes the size of the output

        Given displacement s1 = 2, d = 0, s2 = 1, w=3

            inputs = [x00   [y00
                      x10    y10
                      x20],  y20]
            output[0, 0, 0, 0] = [x00*y00]
            output[0, 0, 1, 0] = [x20*y20]
        """

        s1 = 2
        model = self.get_asymmetric_correlation_block()(
            k=0, dw_pos=0, dw_neg=0, dh_pos=0, dh_neg=0, s1=s1, s2=1
        )
        x1 = torch.randn(1, 1, 3, 1)
        x2 = torch.randn(1, 1, 3, 1)
        output = model(x1, x2)
        gt = torch.zeros(1, 1, 2, 1)
        corr = x1 * x2
        gt[0, 0, 0, 0] = corr[0, 0, 0, 0]
        gt[0, 0, 1, 0] = corr[0, 0, 2, 0]

        self.assertEqual(output.shape, torch.Size([1, 1, 2, 1]))
        torch.testing.assert_allclose(output, gt)

    def get_asymmetric_correlation_block(self):
        raise NotImplementedError


class TestNaiveAsymmetricCorrelationBlock(unittest.TestCase, TestAsymmetricCorrelation):
    def get_asymmetric_correlation_block(self):
        return NaiveAsymmetricCorrelationBlock
