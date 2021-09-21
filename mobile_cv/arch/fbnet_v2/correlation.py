#!/usr/bin/env python3

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.quantized import FloatFunctional


torch.ops.load_library("//caffe2/fb/custom_ops/correlation:correlation")


class NaiveCorrelationBlock(nn.Module):
    """Naive implementation of correlation block.

    This can be used to debug other implementations as this should be the
    simplest.

    Warnings:
        * kernel size needs to be implemented
        * implicit zero padding
        * this is really just dot product multiplication between feature
          vectors as there is no mean or std removal. output is normalized
          by the number of input channels.
    """

    def __init__(self, k: int, d: int, s1: int, s2: int):
        """
        Inputs:
            k (int): kernel size, correlation calculated using patch region (-k, k)
            d (int): displacement, correlations returned in neighborhood (-d, d)
            s1 (int): stride along input1 (reference feature map), this effectively
                      downsamples the output resolution
            s2 (int): stride along input2, this effectively decreases the number of
                      returned correlation channels
        """
        super().__init__()
        assert k == 0
        self.k = k
        self.d = d
        self.s1 = s1
        self.s2 = s2
        self.div = FloatFunctional()

    def forward(self, x1, x2):
        """Calculates the correlation between x1, x2 in a neighborhood

        Correlation if calulating outside of feature map is 0 (implicit zero padding)
        Ignores kernel size.

        Inputs:
            x1 (tensor): feature map1, size (n, c, h, w)
            x2 (tensor): feature map2, size (n, c, h, w)
        Returns:
            corr (tensor): correlations, size (n, (2 * d / s2 + 1) ** 2, h / s1, w / s1)
                           where D = 2 * d + 1
        """
        assert x1.shape == x2.shape

        n, c, h, w = x1.shape
        d, s1, s2 = self.d, self.s1, self.s2
        out_c = (2 * d // s2 + 1) ** 2
        corr = torch.empty(n, out_c, math.ceil(h / s1), math.ceil(w / s1)).to(x1.device)

        for _n in range(n):
            # keep track of the output channel index
            _d = 0
            for _dh in range(-d, d + 1, s2):
                for _dw in range(-d, d + 1, s2):
                    _outh = 0
                    for _h in range(0, h, s1):
                        _outw = 0
                        for _w in range(0, w, s1):
                            # implicit zero padding by checking if we are computing
                            # correlation of point outside of bounds and returning 0
                            if (
                                _h + _dh < 0
                                or _h + _dh >= h
                                or _w + _dw < 0
                                or _w + _dw >= w
                            ):
                                corr[_n, _d, _outh, _outw] = 0.0
                            else:
                                corr[_n, _d, _outh, _outw] = self.div.mul_scalar(
                                    torch.dot(
                                        x1[_n, :, _h, _w], x2[_n, :, _h + _dh, _w + _dw]
                                    ),
                                    float(1.0 / c),
                                )
                            _outw += 1
                        _outh += 1
                    _d += 1
        return corr


class MatMulCorrelationBlock(nn.Module):
    """Faster implementation of correlation block.

    This implementation should behave the same as NaiveCorrelationBlock.
    TODO akashbapat: Replace the use of grid_sample.

    Algorithm: This class calculates all-pair correlations using matrix
    multiplication within the inputs x1 and x2. Then the result is sampled
    according to the stride and displacement parameters. For fast sampling, the
    sampling points are calculated before sampling via grid_sample.

    Warnings:
        * kernel size needs to be implemented
        * implicit zero padding
        * this is really just dot product multiplication between feature
          vectors as there is no mean or std removal. output is normalized
          by the number of input channels.
    """

    def __init__(self, k: int, d: int, s1: int, s2: int):
        super().__init__()
        assert k == 0
        self.k = k
        self.displacementH = d
        self.displacementW = d
        self.stride1W = s1
        self.stride1H = s1
        self.stride2H = s2
        self.stride2W = s2
        self.float_functional = FloatFunctional()

    def forward(self, x1, x2):

        batch, dim, ht, wd = x1.shape

        # Calculate all-pairs correlation between x1 and x2.
        corr = self.all_pairs_correlation(x1, x2)

        batch, h1, w1, dim, h2, w2 = corr.shape

        # Compute sampling deltas in X and Y according to strides.
        dx = torch.linspace(
            -self.displacementW * self.stride1W,
            self.displacementW * self.stride1W,
            (2 * self.displacementW // self.stride2W + 1),
        )
        dy = torch.linspace(
            -self.displacementH * self.stride1H,
            self.displacementH * self.stride1H,
            (2 * self.displacementH // self.stride2H + 1),
        )

        # Compute the sampling center-points using a grid.
        coords = self.coords_grid(
            batch=batch, ht=ht, wd=wd, strideH=self.stride1H, strideW=self.stride1W
        ).to(corr.device)
        coords = coords.permute(0, 2, 3, 1)

        # Add together the center-points and deltas to get the final sampling
        # co-ordinates.
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)
        delta = delta.view(1, dy.shape[0], dx.shape[0], 2)
        centroid = coords.reshape(-1, 1, 1, 2)
        coords_w_delta = centroid + delta

        # Downsample the the correlation. The sampling co-ordinates calculated above
        # assume that the input stride is already applied.
        corr = corr[:, 0 : ht : self.stride1H, 0 : wd : self.stride1W, :, :, :]
        corr = corr.view(-1, dim, h2, w2)

        # Sample the all-pair correlations to return what we need.
        sampled_corr = self.sample_correlation(corr, coords_w_delta)

        h1_out = math.ceil(h1 / self.stride1H)
        w1_out = math.ceil(w1 / self.stride1W)

        # Do a bunch of reshapes, views and permutes so that the data within
        # sampled_corr is in the right order. Note that the number of elements
        # in sampled_corr and result are the same.
        out = sampled_corr.view(batch, h1_out, w1_out, -1)
        out1 = out.permute(0, 3, 1, 2).view(
            batch, coords_w_delta.shape[1], coords_w_delta.shape[2], h1_out, w1_out
        )

        out2 = out1.transpose(1, 2)
        result = out2.reshape(
            batch, coords_w_delta.shape[1] * coords_w_delta.shape[2], h1_out, w1_out
        )

        return result

    def all_pairs_correlation(self, x1, x2):
        """Compute all all-pairs correlation between reference x1 and x2."""
        batch, dim, ht, wd = x1.shape
        x1 = x1.view(batch, dim, ht * wd)
        x2 = x2.view(batch, dim, ht * wd)

        corr = torch.matmul(x1.transpose(1, 2), x2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return self.float_functional.mul_scalar(
            corr, float(1.0 / torch.tensor(dim).float())
        )

    @staticmethod
    def coords_grid(batch, ht, wd, strideH, strideW):
        coords = torch.meshgrid(
            torch.arange(start=0, end=ht, step=strideH),
            torch.arange(start=0, end=wd, step=strideW),
        )
        coords = torch.stack(coords[::-1], dim=0).float()
        return coords[None].repeat(batch, 1, 1, 1)

    def sample_correlation_for(self, corr, coords):
        """
        Samples the all-pairs correlation using for loops. This is
        approximately 100-200 times slower than using grid_sample.
        """
        _, C, H, W = corr.shape
        xgrid, ygrid = coords.int().split([1, 1], dim=-1)

        grid = torch.cat([xgrid, ygrid], dim=-1)
        batch, Hout, Wout, _ = grid.shape

        corr2 = torch.zeros(batch, C, Hout, Wout).to(corr.device)

        for ni in range(batch):
            for hi in range(Hout):
                for wi in range(Wout):
                    xi = xgrid[ni, hi, wi, 0]
                    yi = ygrid[ni, hi, wi, 0]
                    if xi >= 0 and yi >= 0 and xi < W and yi < H:
                        corr2[ni, :, hi, wi] = corr[ni, :, yi, xi]

        return corr2

    def sample_correlation(self, corr, coords):
        """Samples the all-pairs correlation using grid-sample."""
        H, W = corr.shape[-2:]

        # We need to have this if block because grid_sample has a bug for
        # H = 1 and W = 1.
        if H == 1 or W == 1:
            return self.sample_correlation_for(corr, coords)

        xgrid, ygrid = coords.split([1, 1], dim=-1)

        xgrid = 2 * xgrid / (W - 1) - 1
        ygrid = 2 * ygrid / (H - 1) - 1

        grid = torch.cat([xgrid, ygrid], dim=-1)
        return F.grid_sample(corr, grid, align_corners=True, padding_mode="zeros")


class UnfoldCorrelationBlock(nn.Module):
    """Very faster implementation using nn.unfold and nn.fold. See
    https://pytorch.org/docs/master/generated/torch.nn.Unfold.html
    https://pytorch.org/docs/master/generated/torch.nn.Fold.html

    """

    def __init__(self, k: int, d: int, s1: int, s2: int):
        super().__init__()
        assert k == 0
        self.k = k
        self.d = d
        self.s1 = s1
        self.s2 = s2

    def forward(self, x1, x2):
        assert x1.shape == x2.shape
        d = self.d
        s1 = self.s1
        s2 = self.s2
        n, c, h, w = x1.shape
        out_h = (h - 1) // s1 + 1
        out_w = (w - 1) // s1 + 1
        out_k = 2 * d // s2 + 1
        result = torch.zeros(n, out_k ** 2, out_h, out_w, device=x1.device)
        normalizer = float(1.0 / c)
        for _c in range(0, c):
            # the current channel of x1, down sampled
            x1_channel = torch.unsqueeze(x1[:, _c, 0:h:s1, 0:w:s1], 1)
            # the current channel of x2, without sampling
            x2_channel = torch.unsqueeze(x2[:, _c, :, :], 1)
            # extract a set of sliding windows based on the stride over image (s1) and stride over sliding window (s2)
            blocks = nn.functional.unfold(
                x2_channel,
                kernel_size=(out_k, out_k),
                stride=s1,
                dilation=s2,
                padding=d,
            )
            # convert to the correct shape
            combined = nn.functional.fold(blocks, (out_h, out_w), kernel_size=(1, 1))
            # element wise product for this channel, and sum over for dot product (on channel dimension)
            result = result + x1_channel * combined
        return normalizer * result


class TorchscriptUnfoldCorrelation(nn.Module):
    """Wrapper around torchscript implementation of unfold correlation"""

    def __init__(self, k, d, s1, s2):
        super().__init__()
        self.k = k
        self.d = d
        self.s1 = s1
        self.s2 = s2

    def forward(self, x1, x2):
        return torch.ops.fb.correlation(x1, x2, self.k, self.d, self.s1, self.s2)

    @classmethod
    def from_module(cls, mod):
        return cls(mod.k, mod.d, mod.s1, mod.s2)


class TorchscriptNaiveCorrelation(nn.Module):
    """Wrapper around torchscript implementation of naive correlation"""

    def __init__(self, k, d, s1, s2):
        super().__init__()
        self.k = k
        self.d = d
        self.s1 = s1
        self.s2 = s2

    def forward(self, x1, x2):
        print(x1.shape, x2.shape)
        return torch.ops.fb.NaiveCorrelation(x1, x2, self.k, self.d, self.s1, self.s2)

    @classmethod
    def from_module(cls, mod):
        return cls(mod.k, mod.d, mod.s1, mod.s2)
