#!/usr/bin/env python3

import math

import torch
import torch.nn as nn
from torch.nn.quantized import FloatFunctional


class NaiveAsymmetricCorrelationBlock(nn.Module):
    """Naive implementation of asymmetric correlation block.

    This can be used to debug other implementations as this should be the
    simplest.

    The correlation block defined in correlation.py in this directory have the same
    displacement d in all four directions (+x, -x, +y, -y). This is inconvenient for
    some applications like stereo, where correlated patches can only be located in
    one of the two x directions when the stereo pairs are rectified well.

    This class implement asymmetric correlation where the displacement along each
    direction (dw_pos, dw_neg, dh_pos, dh_neg) can be set separately. If all four
    directions are set equally, this is equivalent to correlation.py

    For example, for inputs x1, x2 of dimensions n x c x h x w,
    and dw_pos = 0, dw_neg = d, dh_pos = 0, dh_neg = 0, this build a standard stereo
    correlation (cost) volume of dimension n x (dw_neg+1) x h x w

    Warnings:
        * simimlar to correlation.py, kernel size is not implemented yet
        * implicit zero padding
        * this is really just dot product multiplication between feature
          vectors as there is no mean or std removal. output is normalized
          by the number of input channels.
    """

    def __init__(
        self,
        k: int,
        dw_pos: int,
        dw_neg: int,
        dh_pos: int,
        dh_neg: int,
        s1: int,
        s2: int,
    ):
        """
        Inputs:
            k (int): kernel size, correlation calculated using patch region (-k, k)
            dw_pos (int): displacement in the positive x direction
            dw_neg (int): displacement in the negative x direction
            dh_pos (int): displacement in the positive y direction
            dh_neg (int): displacement in the negative y direction
            correlations returned in neighborhood:
                                        x in (-dw_neg, dw_pos)
                                        y in (-dh_neg, dh_pos)
            s1 (int): stride along input1 (reference feature map), this effectively
                      downsamples the output resolution
            s2 (int): stride along input2, this effectively decreases the number of
                      returned correlation channels
        """
        super().__init__()
        assert k == 0
        self.k = k
        self.dw_pos = dw_pos
        self.dw_neg = dw_neg
        self.dh_pos = dh_pos
        self.dh_neg = dh_neg
        self.s1 = s1
        self.s2 = s2
        self.div = FloatFunctional()

    def forward(self, x1, x2):
        """Calculates the correlation between x1, x2 in a neighborhood of
        (-dw_neg, dw_pos, -dh_neg, dh_pos)

        Correlation if calulating outside of feature map is 0 (implicit zero padding)
        Ignores kernel size.

        Inputs:
            x1 (tensor): feature map1, size (n, c, h, w)
            x2 (tensor): feature map2, size (n, c, h, w)
        Returns:
            corr (tensor): correlations, size
                        (n, ((dw_pos + dw_neg)/s2 + 1)*((dh_pos + dh_neg)/s2 + 1), h / s1, w / s1)
        """
        assert x1.shape == x2.shape

        n, c, h, w = x1.shape
        dw_pos, dw_neg, dh_pos, dh_neg = (
            self.dw_pos,
            self.dw_neg,
            self.dh_pos,
            self.dh_neg,
        )
        s1, s2 = self.s1, self.s2
        out_c = ((dw_pos + dw_neg) // s2 + 1) * ((dh_pos + dh_neg) // s2 + 1)
        corr = torch.empty(n, out_c, math.ceil(h / s1), math.ceil(w / s1)).to(x1.device)

        for _n in range(n):
            # keep track of the output channel index
            _d = 0
            for _dh in range(-dh_neg, dh_pos + 1, s2):
                for _dw in range(-dw_neg, dw_pos + 1, s2):
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
