#!/usr/bin/env python3

import copy

import numpy as np
import torch
import torch.nn as nn
from common.utils_pytorch.model_utils import find_module_key
from mobile_cv.arch.fbnet.fbnet_building_blocks import ConvBNRelu, QuantConvBNRelu
from mobile_cv.arch.layers import Conv2d
from silicon.quantization.quantPack import QuantConv2d


def convert_to_torch_model(model):
    if isinstance(model, Conv2d):
        # Convert to nn.Conv2d
        conv_2d = torch.nn.Conv2d(
            in_channels=model.in_channels,
            out_channels=model.out_channels,
            kernel_size=model.kernel_size,
            stride=model.stride,
            padding=model.padding,
            bias=(model.bias is not None),
        )
        conv_2d.weight = model.weight
        conv_2d.eval()
        return conv_2d

    elif isinstance(model, nn.BatchNorm2d):
        return model

    else:
        raise AssertionError("Not supported operator")


def fuse_convbnrelu(model, inplace=False):
    """Wrapper around _fuse_convbnrelu so we can choose whether to create or modify inplace
    Inputs: (nn.Module) model
            (bool) whether to modify the input model or return a new model
    Return: (nn.Module) fused model
    """
    if inplace:
        _fuse_convbnrelu(model)
    else:
        fused_model = copy.deepcopy(model)
        _fuse_convbnrelu(fused_model)
        return fused_model


def _fuse_convbnrelu(model):
    """Remove batchnorm2d from the model and fuse with the previous conv

    Batch norm operators add a weight and bias to the ouptut of conv (here, we
    denote convolution as the @ symbol):
        x -> Conv(w, b) -> y -> BN(mean, std, gamma, beta) -> z

        y = w @ x + b
        z = (y - mean) * gamma  / std + beta
          = (w @ x + b - mean) * gamma / std + beta
          = (w * gamma / std) @ x + (b - mean) * gamma / std + beta

    This operation removes the batch norm and updates the conv parameters so
    that the output is the same:
        x -> Conv(w', b') -> z

        w' = w * gamma / std
        b' = (b - mean) * gamma / std  + beta

        z = w' @ x + b'
          = (w * gamma / std) @ x + (b - mean) * gamma / std + beta

    The approach is to recursively check if a module is convbnrelu. Then, we
    remove bn from the modules (ordereddict) and update the conv values.
    Inputs: (nn.Module) model
            (nn.Module) fused model
    """
    if isinstance(model, (ConvBNRelu, QuantConvBNRelu)):
        # find the batch norm key in the ordered dict
        bn_key = find_module_key(model, nn.BatchNorm2d)
        if bn_key is None:
            return

        # remove the batch norm module
        modules = model._modules
        bn = modules.pop(bn_key)

        # find the conv module
        if isinstance(model, ConvBNRelu):
            conv_type = Conv2d
        elif isinstance(model, QuantConvBNRelu):
            conv_type = QuantConv2d
        conv = modules[find_module_key(model, conv_type)]

        # update the conv values with the batch norm values
        eps = 1e-5
        mean = bn.running_mean.detach().numpy()
        var = bn.running_var.detach().numpy()
        gamma = bn.weight.detach().numpy()
        beta = bn.bias.detach().numpy()
        weight = conv.weight.detach().numpy()
        if conv.bias is not None:
            bias = conv.bias.detach().numpy()
        else:
            bias = np.zeros(conv.out_channels, dtype=np.float32)
        gamma_by_std = gamma / (np.sqrt(var) + eps)

        for channel in range(gamma_by_std.shape[0]):
            weight[channel, :, :, :] *= gamma_by_std[channel]
        bias = (bias - mean) * gamma_by_std + beta

        # update the clipping limits of the conv weight and bias
        # assumption: symmetric quantization (take the max of abs)
        if isinstance(model, QuantConvBNRelu):
            conv._w_scale_factor = nn.Parameter(
                torch.Tensor([np.max(np.abs(weight))]), requires_grad=False
            )
            conv._b_scale_factor = nn.Parameter(
                torch.Tensor([np.max(np.abs(bias))]), requires_grad=False
            )

        conv.weight = nn.Parameter(torch.from_numpy(weight))
        conv.bias = nn.Parameter(torch.from_numpy(bias))

    for child in model.children():
        _fuse_convbnrelu(child)
