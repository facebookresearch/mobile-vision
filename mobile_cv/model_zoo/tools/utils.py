#!/usr/bin/env python3

import json
import os
import shutil

import torch


def copy_file(src, dst, skip_exists=True):
    if os.path.exists(src):
        if skip_exists and os.path.exists(dst):
            return
        shutil.copy2(src, dst)
    else:
        print(f"Warning: {src} does not exist!")


def save_json(file, data):
    with open(file, "w") as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True)


def get_model_attributes(model: torch.nn.Module):
    model_attrs = None
    if hasattr(model, "attrs"):
        model_attrs = model.attrs
        if model_attrs is not None:
            assert isinstance(
                model_attrs, dict
            ), f"Invalid model attributes type: {model_attrs}"
    return model_attrs
