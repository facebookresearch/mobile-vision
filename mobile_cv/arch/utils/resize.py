#!/usr/bin/env python3


def space2depth(x, scale_factor):
    """Resample spatial dimensions by scale_factor.

    Resample channel dimension by 1/scale_factor. Reshape in a valid way.
    """
    n, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)
    h2 = int(round(h * scale_factor))
    w2 = int(round(w * scale_factor))
    c2 = int(h * w * c / (h2 * w2))
    return x.view((n, c2, h2, w2))
