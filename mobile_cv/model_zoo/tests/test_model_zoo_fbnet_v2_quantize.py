#!/usr/bin/env python3

import os
import tempfile
import unittest

import mobile_cv.arch.utils.fuse_utils as fuse_utils
import torch
from mobile_cv.model_zoo.models.fbnet_v2 import fbnet
from torch.ao.quantization.observer import MinMaxObserver, default_weight_observer


class TestModelZooFBNetV2Quantize(unittest.TestCase):
    def test_fbnet_v2_quant(self):
        for backend in ["qnnpack", "fbgemm"]:
            print(f"Testing quantized fbnet on backend: {backend}...")

            model = fbnet("fbnet_c", pretrained=False)
            torch.backends.quantized.engine = "fbgemm"

            model = torch.ao.quantization.QuantWrapper(model)
            model.eval()
            model = fuse_utils.fuse_model(model, inplace=False)

            print(f"Fused model {model}")

            model.qconfig = torch.ao.quantization.default_qconfig
            if backend == "qnnpack":
                torch.backends.quantized.engine = "qnnpack"

                # Using MinMaxObserver for qnnpack with reduce_range set to False
                # HistogramObserver with reduce range false is very slow and
                # can cause the test to time-out.
                model.qconfig = torch.ao.quantization.QConfig(
                    activation=MinMaxObserver.with_args(
                        dtype=torch.quint8, reduce_range=False
                    ),
                    weight=default_weight_observer,
                )

            print(model.qconfig)

            torch.ao.quantization.prepare(model, inplace=True)

            # calibration
            for _ in range(5):
                data = torch.rand([1, 3, 224, 224])
                model(data)

            # Convert to quantized model
            torch.ao.quantization.convert(model, inplace=True)
            print(f"Quant model {model}")

            data = torch.randn([1, 3, 224, 224])
            quant_output = model(data)
            self.assertEqual(quant_output.shape, torch.Size([1, 1000]))

            # Make sure model can be traced
            traced_model = torch.jit.trace(model, torch.randn([1, 3, 224, 224]))
            traced_output = traced_model(data)
            self.assertEqual(traced_output.shape, torch.Size([1, 1000]))
            self.assertEqual(quant_output.norm(), traced_output.norm())

            with tempfile.TemporaryDirectory() as tmp_dir:
                fn = os.path.join(tmp_dir, "model.jit")
                torch.jit.save(traced_model, fn)
                self.assertTrue(os.path.isfile(fn))
