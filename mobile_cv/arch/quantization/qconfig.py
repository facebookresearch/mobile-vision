#!/usr/bin/env python3

import torch
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import MinMaxObserver
from torch.ao.quantization.qconfig import QConfig

from .observer import (
    FixedMinMaxObserver,
    UpdatableSymmetricMovingAverageMinMaxObserver,
    UpdateableReLUMovingAverageMinMaxObserver,
)


# used in weights
symmetric_minmax_per_tensor_fake_quant = FakeQuantize.with_args(
    observer=MinMaxObserver,
    quant_min=-128,
    quant_max=127,
    dtype=torch.qint8,
    qscheme=torch.per_tensor_symmetric,
    reduce_range=False,
)


# used in conv, convbn
updateable_symmetric_moving_avg_minmax_config = QConfig(
    activation=FakeQuantize.with_args(
        observer=UpdatableSymmetricMovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        reduce_range=False,
        qscheme=torch.per_tensor_affine,
        averaging_constant=0.001,
    ),
    weight=symmetric_minmax_per_tensor_fake_quant,
)


# used in convbnrelu
updateable_relu_moving_avg_minmax_config = QConfig(
    activation=FakeQuantize.with_args(
        observer=UpdateableReLUMovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        reduce_range=False,
        qscheme=torch.per_tensor_affine,
        averaging_constant=0.001,
    ),
    weight=symmetric_minmax_per_tensor_fake_quant,
)


# used in quantstub
fixed_minmax_config = QConfig(
    activation=FakeQuantize.with_args(
        observer=FixedMinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        reduce_range=False,
        qscheme=torch.per_tensor_affine,
        fixed_min_val=0,
        fixed_max_val=1,
    ),
    weight=symmetric_minmax_per_tensor_fake_quant,
)
