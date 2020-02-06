#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Example code to run fbnet traced int8 model on a given image

Usage:
    python3 -m examples.run_fbnet_v2_jit_int8

"""

import urllib

import torch
from PIL import Image

from mobile_cv.model_zoo.models.model_jit import model_jit
from mobile_cv.model_zoo.models.preprocess import get_preprocess


def _get_input():
    # Download an example image from the pytorch website
    url, filename = (
        "https://github.com/pytorch/hub/raw/master/dog.jpg",
        "dog.jpg",
    )
    local_filename, headers = urllib.request.urlretrieve(url, filename)
    input_image = Image.open(local_filename)
    return input_image


def run_fbnet_v2_jit_int8():
    # int8 model in jit format, supported models could be found in
    # mobile_cv/model_zoo/models/model_info/model_jit/*.json
    model_name = "fbnet_c_i8f_int8_jit"

    # the model is quantized with qnnpack backend
    torch.backends.quantized.engine = "qnnpack"

    # load model
    model = model_jit(model_name, pretrained=True)
    model.eval()
    preprocess = get_preprocess(224)

    # load and process input
    input_image = _get_input()
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # run model
    with torch.no_grad():
        output = model(input_batch)
    output_softmax = torch.nn.functional.softmax(output[0], dim=0)
    print(output_softmax.max(0))


if __name__ == "__main__":
    run_fbnet_v2_jit_int8()
