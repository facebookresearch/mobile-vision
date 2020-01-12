#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Compute per module flops and number of parameters for a given pytorch model

Usage:
    python3 -m examples.run_flops_estimation

The output will look like:
FBNet(
  input_shapes=[[1, 3, 128, 128]], output_shapes=[1, 1000],
  nparams=5.99297, nflops=56.274952
  (backbone): FBNetBackbone(
    input_shapes=[[1, 3, 128, 128]], output_shapes=[1, 1600, 1, 1],
    nparams=4.39297, nflops=54.673352
    (stages): Sequential(
      input_shapes=[[1, 3, 128, 128]], output_shapes=[1, 1600, 1, 1],
      nparams=4.39297, nflops=54.673352
      (xif0_0): ConvBNRelu(
        input_shapes=[[1, 3, 128, 128]], output_shapes=[1, 8, 64, 64],
        nparams=0.000216, nflops=0.884736
        (conv): Conv2d(
          3, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
          input_shapes=[[1, 3, 128, 128]], output_shapes=[1, 8, 64, 64],
          nparams=0.000216, nflops=0.884736
        )
    ...
"""

import torch

import mobile_cv.lut.lib.pt.flops_utils as flops_utils
from mobile_cv.model_zoo.models.fbnet_v2 import fbnet


def run_flops_estimation():
    # fbnet models, supported models could be found in
    # mobile_cv/model_zoo/models/model_info/fbnet_v2/*.json
    model_name = "dmasking_l2_hs"
    model = fbnet(model_name, pretrained=False)
    model.eval()

    res = model.arch_def.get("input_size", 224)
    input_batch = torch.zeros([1, 3, res, res])
    with torch.no_grad():
        flops_utils.print_model_flops(model, input_batch)


if __name__ == "__main__":
    run_flops_estimation()
