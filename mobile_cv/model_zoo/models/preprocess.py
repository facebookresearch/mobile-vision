#!/usr/bin/env python3
"""
Preprocess function for pretrained models
"""

import math

from torchvision import transforms


def get_preprocess(crop_size):
    ratio = 256.0 / 224.0
    resize_res = int(math.ceil(ratio * crop_size // 8) * 8.0)
    preprocess = transforms.Compose(
        [
            transforms.Resize(resize_res),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return preprocess
