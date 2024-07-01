#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from mobile_cv.common.misc.oss_utils import fb_overwritable


# to register for model_zoo
@fb_overwritable()
def register_models():
    from mobile_cv.model_zoo.models import (  # noqa
        fbnet_v2,  # noqa
        model_jit,  # noqa
        model_torchvision,  # noqa
    )


register_models()
