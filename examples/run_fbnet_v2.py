#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Example code to run fbnet model on a given image

Usage:
    python3 -m examples.run_fbnet_v2

"""

import urllib

import torch
from PIL import Image

from mobile_cv.model_zoo.models.fbnet_v2 import fbnet
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


def run_fbnet_v2():
    # fbnet models, supported models could be found in
    # mobile_cv/model_zoo/models/model_info/fbnet_v2/*.json
    model_name = "dmasking_l3"

    # load model
    model = fbnet(model_name, pretrained=True)
    model.eval()
    preprocess = get_preprocess(model.arch_def.get("input_size", 224))

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
    run_fbnet_v2()
