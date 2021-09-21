#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os


def is_devserver() -> bool:
    return os.getenv("HOSTNAME") is not None and os.getenv("HOSTNAME").startswith("dev")
