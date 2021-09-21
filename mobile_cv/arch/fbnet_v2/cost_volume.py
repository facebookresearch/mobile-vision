#!/usr/bin/env python3

import torch
import torch.nn as nn


class CorrVolume1DBlock(nn.Module):
    def __init__(self, k: int, d: int):
        """
        Simple implementation of stereo correlation volume (1D single-direction correlation)
        also known as disparity space image (DSI)

        k must equal 0
        d: maximum disparity search range
        x1: left image feature with shape:  batch x feature_channel x height x width
        x2: right image feature with shape: batch x feature_channel x height x width

        output: correlation volume (cv) with shape: batch x d+1 x height x width

        This function compares the similarity between the left and right feature vectors using dot product
        """
        super().__init__()
        assert k == 0
        self.d = d

    def forward(self, x1, x2):
        n, c, h, w = x2.shape
        normalizer = float(
            1.0 / c
        )  # TODO: check why normalize this way. This is copy and pasted from correlation.py

        # correlation volume disparity search range is from 0 (depth infinity) to d
        # out-of-range entries are set to 0: least similar
        result = torch.zeros(n, self.d + 1, h, w, device=x1.device)

        for i in range(self.d + 1):
            shifted_x2 = torch.zeros(n, c, h, w, device=x1.device)
            shifted_x2[:, :, :, i:w] = x2[
                :, :, :, : max(0, w - i)
            ]  # shift x2 by disparity i

            # compute the similarity between x1 and shifted_x2
            result[:, i, :, :] = torch.sum(x1 * shifted_x2, 1)

        return result * normalizer
