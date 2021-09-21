#!/usr/bin/env python3

"""Taken from http://github.com/alvinwan/uncertainty-based-downsampling

Non-uniform subsampling methods using PyTorch. Provides `warp_using_entropy`
with the following dimensions:

 * `image_hwc`: Height x width x channels. Image is assumed to contain only
    positive values. May be [0, 1] or [0, 255], floats or integers.
 * `entropy_hw`: Height x width.
 * `logits_hwk`: Height x width x classes. Is not necessarily a valid
     probability distribution.
"""

import numpy as np
import torch
import torch.nn.functional as F


###########
# ENTROPY #
###########


def prop2cmf(prop, unnorm=True, bin_size=1):
    """
    Proportional sizes/unnormalized distribution => warping pixel index values
    Ex: [1,1,1,1],True => [-1, -1/3, 1/3, 1]
        uniformly spaced
    Ex: [3,1,1,1],True => [-1, 0, .5, 1]
        space between first "two pixels" is twice as big as space between
        others [(3+1)/2=2] vs [(1+1)/2=1] this is proportionally reflected in
        the output cmf [0-(-1)=1] vs [.5-0=.5]
    Steps
      1. do a length-2 running average filter
      2. normalize to make a PMF
      3. cumsum to make a CMF in [0,1]
      4. (if unnorm) renormalize to make pixel indexing in [-1,1]

    >>> def test_prop2cmf(lst, **kwargs):
    ...     tensor = torch.Tensor(lst).float()
    ...     return np.array(prop2cmf(tensor, **kwargs))
    >>> pprint = lambda arr: [round(a, 2) for a in arr.tolist()]
    >>> pprint(test_prop2cmf([1, 1, 1, 1]) * 3)
    [-3.0, -1.0, 1.0, 3.0]
    >>> pprint(test_prop2cmf([3, 1, 1, 1]) * 2)
    [-2.0, 0.0, 1.0, 2.0]
    >>> pprint(test_prop2cmf([1, 1, 1, 1], unnorm=False) * 3)
    [0.0, 1.0, 2.0, 3.0]
    >>> pprint(test_prop2cmf([3, 1, 1, 1], unnorm=False) * 2)
    [0.0, 1.0, 1.5, 2.0]
    """
    pmf = prop.float()
    pmf = 0.5 * (pmf[1:] + pmf[:-1])
    cmf = torch.tensor([0.0], device=pmf.device)
    cmf = torch.cat(
        (cmf, torch.cumsum(torch.flatten(pmf / torch.sum(pmf)), dim=0).float()), dim=0
    )

    if unnorm:
        return (cmf - 0.5) * 2
    else:
        return cmf


def prop2inversecmf(prop, unnorm=True, bin_size=1):
    """
    Proportional sizes/unnormalized distribution => unwarping pixel index values
    Ex: [1,1,1,1] => [-1, -1/3, 1/3, 1]
        uniformly spaced will still be uniformly spaced
    Ex: [3,1,1,1] => [-1, -5/9, 1/9, 1]
        First two pixels are fattened in warping operation. They are skinnier
        in the unwarping operation.
    Steps
      1. get CMF
      2. get a CMF for uniformly spaced distribution (CMF-uniform)
      3. for every value in CMF-uniform, find the index it lies in the CMF
      4. (if unnorm) renormalize to make pixel indexing in [-1,1]

    >>> def test_prop2inversecmf(lst, **kwargs):
    ...     tensor = torch.Tensor(lst).float()
    ...     return np.array(prop2inversecmf(tensor, **kwargs))
    >>> pprint = lambda arr: [round(a, 2) for a in arr.tolist()]
    >>> pprint(test_prop2inversecmf([1, 1, 1, 1]) * 3)
    [-3.0, -1.0, 1.0, 3.0]
    >>> pprint(test_prop2inversecmf([3, 1, 1, 1]) * 9)
    [-9.0, -5.0, 1.0, 9.0]
    >>> pprint(test_prop2inversecmf([1, 1, 1, 1], unnorm=False) * 3)
    [0.0, 1.0, 2.0, 3.0]
    >>> pprint(test_prop2inversecmf([3, 1, 1, 1], unnorm=False) * 9)
    [0.0, 2.0, 5.0, 9.0]
    >>> pprint(test_prop2inversecmf([1, 1, 2, 1, 1], unnorm=False) * 5)
    [0.0, 1.46, 2.5, 3.54, 5.0]
    """

    L = prop.nelement()
    cmf = prop2cmf(prop, unnorm=False)  # CMF array between [0,1]
    # uniformly spaced array between [0,1]
    cmf_uni = prop2cmf(torch.ones(L, device=cmf.device), unnorm=False)

    eps = 1e-12
    cond = cmf[:, None] >= cmf_uni[None, :]
    a = torch.arange(L, device=cmf.device)[:, None]
    inds_more, _ = torch.min(a * cond.long() + (L - 1) * (1 - cond.long()), dim=0)
    inds_less = inds_more - 1
    inds_less[inds_less < 0] = 0
    ds_more = cmf[inds_more] - cmf_uni
    ds_less = cmf_uni - cmf[inds_less]
    inds = inds_less.float() + 1.0 * (ds_less) / (ds_more + ds_less + eps)
    inds[0] = 0

    inds = inds / inds[-1]  # normalize indices between [0,1]
    if unnorm:
        return (inds - 0.5) * 2
    else:
        return inds


def arr2indices(inds):
    # Since we're doing 1-D signals but are using 2-D functions, add 0 index
    # for the other dimension
    indices = [[0, a] for a in inds]
    return torch.Tensor(indices).reshape(1, len(indices), 1, 2)


def compute_inverse_widths(entropy_nhw):
    """
    >>> entropy = torch.Tensor([[[1, 2], [1, 1]]])
    >>> compute_inverse_widths(entropy)
    tensor([[0.8000, 1.2000]])
    >>> entropy = torch.Tensor([[[1, 2, 1], [1, 1, 1]]])
    >>> compute_inverse_widths(entropy)
    tensor([[0.8571, 1.2857, 0.8571]])
    """
    _, height, width = entropy_nhw.shape
    row_nw = torch.sum(entropy_nhw, dim=1)
    widths_nw = row_nw / (torch.sum(row_nw, dim=1)[:, None] + 1e-10) * width
    return widths_nw


def compute_new_columns(entropy_nhw, bin_size=1):
    """Returns new column starts positions, for forward warp

    >>> entropy = torch.Tensor([[[0.5, 2.25, 3.5, 4]]])
    >>> compute_new_columns(entropy)
    tensor([[-1.0000, -0.0338,  0.5259,  1.0000]])
    """
    widths_nw = compute_inverse_widths(entropy_nhw)

    cmfs = []
    for widths_w in widths_nw:
        cmfs.append(prop2inversecmf(widths_w, bin_size=bin_size))

    columns = torch.stack(cmfs)
    if entropy_nhw.is_cuda:
        columns = columns.cuda()
    return columns


def restrict_entropy(entropy_nhw, max_distortion=np.inf, min_distortion=0):
    """Restrict entropy by capping distortions and smoothing.

    >>> entropy0 = torch.Tensor(np.array([[[75, 1, 1, 1, 1], [1, 1, 1, 1, 1]]]))
    >>> compute_max_distortion(entropy0)
    tensor(4.5238)
    >>> entropy1 = torch.Tensor(np.array([[[10, 1, 1, 1, 1], [1, 1, 1, 1, 1]]]))
    >>> compute_max_distortion(entropy1)
    tensor(2.8947)
    >>> entropy = torch.cat((entropy0, entropy1), dim=0)
    >>> compute_max_distortion(entropy)
    tensor(4.5238)
    >>> entropy2 = restrict_entropy(entropy, 2)
    >>> compute_max_distortion(entropy2)
    tensor(2.)
    >>> max([compute_max_distortion(entropy2[i][None]) for i in range(2)])
    tensor(2.)
    >>> entropy3 = restrict_entropy(entropy2, 3)
    >>> compute_max_distortion(entropy3)
    tensor(2.)
    >>> compute_min_distortion(entropy3)
    tensor(0.7500)
    >>> entropy4 = restrict_entropy(entropy3, np.inf, 0.8)
    >>> compute_min_distortion(entropy4), compute_max_distortion(entropy4)
    (tensor(0.8000), tensor(1.8000))
    >>> entropy5 = restrict_entropy(entropy4, 1.2, 0.9)
    >>> compute_min_distortion(entropy5), compute_max_distortion(entropy5)
    (tensor(0.9500), tensor(1.2000))
    >>> entropy6 = restrict_entropy(entropy4, 1, 1)
    >>> compute_min_distortion(entropy6), compute_max_distortion(entropy6)
    (tensor(1.), tensor(1.))
    >>> entropy7 = restrict_entropy(entropy4, 1.00001, 0.999999)
    >>> compute_min_distortion(entropy7), compute_max_distortion(entropy7)
    (tensor(1.0000), tensor(1.0000))
    """
    assert len(entropy_nhw.shape) == 3
    assert (
        max_distortion >= 1
    ), "Warp max distortion must be in [1, infty) but is {}".format(max_distortion)
    assert (
        0 <= min_distortion <= 1
    ), "Warp min distortion must be in [0, 1] but is {}".format(min_distortion)

    valid_max_distortion = (
        1 <= max_distortion < np.inf
        and compute_max_distortion(entropy_nhw) > max_distortion
    )
    valid_min_distortion = (
        0 <= min_distortion <= 1
        and compute_min_distortion(entropy_nhw) < min_distortion
    )
    if valid_max_distortion or valid_min_distortion:
        entropy_nhw = restrict_entropy_distortion(
            entropy_nhw, max_distortion=max_distortion, min_distortion=min_distortion
        )

    return entropy_nhw


def restrict_entropy_distortion(entropy_nhw, max_distortion, min_distortion):
    """Restrict possible distortions.

    Restrict max and min distortion for the provided entropy map, for every
    column bin.
    """
    n, h, w = entropy_nhw.shape
    row_nw = torch.sum(entropy_nhw, dim=1)
    total_n = torch.sum(row_nw, dim=1)

    def upper(e_n):
        return (max_distortion * total_n - w * e_n) / (
            (w * (1 - max_distortion)) + 1e-10
        )

    def lower(e_n):
        return (min_distortion * total_n - w * e_n) / (
            (w * (1 - min_distortion)) + 1e-10
        )

    def zero(e_n):
        return e_n * 0

    def values(e_n):
        values = []
        if 1 <= max_distortion < np.inf:
            values.append(upper(e_n)[None])
        if 0 <= min_distortion <= 1:
            values.append(lower(e_n)[None])
        values.append(zero(e_n)[None])
        return tuple(values)

    def const(e_n):
        return torch.max(torch.cat(values(e_n), dim=1), dim=1)[0]

    const_wn = [const(e_n)[None] for e_n in row_nw.t()]
    const_wn = torch.cat(const_wn, dim=0)
    const_n, _ = torch.max(const_wn, dim=0)

    if entropy_nhw.is_cuda:
        const_n = const_n.cuda()

    const_per_cell_n = const_n / h
    new_entropy_nhw = entropy_nhw + const_per_cell_n[:, None, None]
    return new_entropy_nhw


def compute_max_distortion(entropy_nhw):
    return torch.max(compute_inverse_widths(entropy_nhw))


def compute_min_distortion(entropy_nhw):
    return torch.min(compute_inverse_widths(entropy_nhw))


########
# FLOW #
########


def compute_identity_flow(height, width):
    """Compute initial flow, basically a coordinate map.

    :param height int: Height of image to warp
    :param width int: Width of image to warp

    >>> M = compute_identity_flow(2, 2)
    >>> M[0, 1]
    tensor([1., 0.])
    >>> M[1, 0]
    tensor([0., 1.])
    >>> M[0, 0]
    tensor([0., 0.])
    >>> M = compute_identity_flow(3, 2)
    >>> M[2, 0]
    tensor([0., 2.])
    """
    rows = np.array([[row] * width for row in range(height)])
    cols = np.array([[col] * height for col in range(width)]).T

    rows = rows[:, :, None]
    cols = cols[:, :, None]
    coordinates = np.dstack((cols, rows))
    flow_hw2 = torch.Tensor(coordinates)
    return flow_hw2


def compute_entropy_flow(entropy_nhw, bin_size=1):
    """Compute flow based on entropy"""
    row_nw = compute_new_columns(entropy_nhw, bin_size=bin_size)
    col_nh = compute_new_columns(torch.transpose(entropy_nhw, 1, 2), bin_size=bin_size)
    flow_nhw2 = compute_flow_from_rows_cols(row_nw, col_nh)
    return flow_nhw2


def compute_flow_from_rows_cols(row_nw, col_nh):
    """Compute flow map from given rows and cols indices

    >>> row_nw = torch.Tensor([1., 2., 3.])[None]
    >>> col_nh = torch.Tensor([4., 5.])[None]
    >>> product = compute_flow_from_rows_cols(row_nw, col_nh).data.numpy()
    >>> product
    array([[[[1., 4.],
             [2., 4.],
             [3., 4.]],
    <BLANKLINE>
            [[1., 5.],
             [2., 5.],
             [3., 5.]]]], dtype=float32)
    >>> product.shape
    (1, 2, 3, 2)
    """
    n, width = row_nw.shape
    n, height = col_nh.shape
    rows_nhw = torch.transpose(torch.stack([col_nh for _ in range(width)], 1), 1, 2)
    cols_nhw = torch.stack([row_nw for _ in range(height)], 1)

    rows_nhw1 = rows_nhw[:, :, :, None]
    cols_nhw1 = cols_nhw[:, :, :, None]
    flow_nhw2 = torch.cat((cols_nhw1, rows_nhw1), 3)
    return flow_nhw2


def compute_flow(entropy_nhw, bin_size=1):
    """Compute dense flow map for provided entropy"""
    assert len(entropy_nhw.shape) == 3
    flow_nhw2 = compute_entropy_flow(entropy_nhw, bin_size=bin_size)

    assert flow_nhw2.shape[1:3] == entropy_nhw.shape[1:3] and flow_nhw2.shape[3] == 2, (
        flow_nhw2.shape,
        entropy_nhw.shape,
    )
    return flow_nhw2


def grid_sample(image_nhwc, flow_nhw2):
    image_nchw = image_nhwc.permute(0, 3, 1, 2)
    assert (
        image_nchw.shape[2:] == flow_nhw2.shape[1:3]
        and image_nchw.shape[0] == flow_nhw2.shape[0]
    ), (image_nchw.shape, flow_nhw2.shape)

    warped_nchw = F.grid_sample(image_nchw, flow_nhw2)
    return warped_nchw


########
# WARP #
########


def warp_using_entropy(
    image_nchw, entropy_nhw, max_distortion=np.inf, min_distortion=0
):
    """Apply warp to image using entropy.

    >>> image_xhwx = torch.rand(1, 513, 513, 3)
    >>> entropy_xhw = torch.zeros(1, 513, 513) + 1
    >>> warped_nchw = warp_using_entropy(image_xhwx, entropy_xhw)
    >>> warped_nhwc = warped_nchw.permute(0, 2, 3, 1)
    >>> warped_nhwc.shape == image_xhwx.shape
    True
    >>> bool(torch.all(warped_nhwc == image_xhwx))
    True
    >>> warped_nchw = warp_using_entropy(
    ...     image_xhwx, entropy_xhw, max_distortion=1)
    >>> warped_nhwc = warped_nchw.permute(0, 2, 3, 1)
    >>> bool(torch.all(warped_nhwc == image_xhwx))
    True
    >>> warped_nchw = warp_using_entropy(
    ...     image_xhwx, entropy_xhw, max_distortion=2)
    >>> warped_nhwc = warped_nchw.permute(0, 2, 3, 1)
    >>> bool(torch.all(warped_nhwc == image_xhwx))
    True
    """
    image_nhwc = image_nchw.permute(0, 2, 3, 1)
    entropy_nhw = restrict_entropy(entropy_nhw, max_distortion, min_distortion)
    flow_nhw2 = compute_flow(entropy_nhw)
    assert entropy_nhw.shape[:3] == image_nhwc.shape[:3] == flow_nhw2.shape[:3], (
        entropy_nhw.shape,
        image_nhwc.shape,
        flow_nhw2.shape,
    )
    return grid_sample(image_nhwc, flow_nhw2)
