#!/usr/bin/env python3

import io
import math
import unittest

import torch
from mobile_cv.arch.quantization.observer import (
    FixedMinMaxObserver,
    UpdatableMovingAverageMaxStatObserver,
    UpdatableSymmetricMovingAverageMinMaxObserver,
    UpdateableReLUMovingAverageMinMaxObserver,
    update_stat,
)
from torch.ao.quantization import QuantStub
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import MinMaxObserver
from torch.ao.quantization.qconfig import QConfig


def _reload_state_dict(state_dict):
    """Helper function to save and load state_dict"""
    b = io.BytesIO()
    torch.save(state_dict, b)
    b.seek(0)
    loaded_dict = torch.load(b)
    return loaded_dict


class UpdateableObserver(UpdatableMovingAverageMaxStatObserver):
    # dummy observer bc we just want to test maxstat but it has abstract method
    def update_stat(self):
        return


class TestObserver(unittest.TestCase):
    def test_update_stat(self):
        """Check update_stat runs update_stat in observer"""
        averaging_constant = 0.01
        x = torch.tensor([-1.0, 2.0, 3.0, 4.0, 5.0])
        y = torch.tensor([2.0, 5.0, 5.0, 7.0, 10.0])
        max_stat = torch.tensor(5.0 + averaging_constant * 5.0)

        model = QuantStub()
        model.qconfig = QConfig(
            activation=FakeQuantize.with_args(
                observer=UpdatableSymmetricMovingAverageMinMaxObserver,
                quant_min=0,
                quant_max=255,
                dtype=torch.quint8,
                reduce_range=False,
                qscheme=torch.per_tensor_affine,
                averaging_constant=averaging_constant,
            ),
            weight=FakeQuantize.with_args(
                observer=MinMaxObserver,
                quant_min=-128,
                quant_max=127,
                dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric,
                reduce_range=False,
            ),
        )
        model.train()
        qat_model = torch.ao.quantization.prepare_qat(model, inplace=False)
        qat_model(x)
        qat_model(y)
        qat_model.apply(update_stat)

        # check the observer has changed
        observer = qat_model.activation_post_process.activation_post_process
        self.assertEqual(observer.min_val, torch.tensor(-max_stat))
        self.assertEqual(observer.max_val, torch.tensor(max_stat))
        self.assertEqual(observer.max_stat, torch.tensor(max_stat))


class TestFixedMinMaxObserver(unittest.TestCase):
    def test_pertensor(self):
        """Checked fixedminmax has specified minmax values regardless of input data"""
        for fixed_min_val, fixed_max_val, gt_scale, gt_zp in [
            [0, 1, 0.0039216, -128],
            [1, 8, 0.0313725, -128],
        ]:
            observer = FixedMinMaxObserver(
                dtype=torch.qint8,
                qscheme=torch.per_tensor_affine,
                reduce_range=False,
                fixed_min_val=fixed_min_val,
                fixed_max_val=fixed_max_val,
            )
            x = torch.tensor([1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            y = torch.tensor([4.0, 5.0, 5.0, 6.0, 7.0, 8.0])

            result = observer(x)
            result = observer(y)
            self.assertTrue(torch.allclose(result, y))
            self.assertEqual(observer.min_val, torch.tensor(fixed_min_val))
            self.assertEqual(observer.max_val, torch.tensor(fixed_max_val))
            qparams = observer.calculate_qparams()
            self.assertTrue(
                math.isclose(qparams[0].item(), gt_scale, abs_tol=1e-5, rel_tol=0)
            )
            self.assertEqual(qparams[1].item(), gt_zp)

    def test_load_state_dict(self):
        """Check that state dict can be loaded"""
        observer = FixedMinMaxObserver(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=False,
            fixed_min_val=-1,
            fixed_max_val=1,
        )
        state_dict = observer.state_dict()
        loaded_dict = _reload_state_dict(state_dict)
        for key in state_dict:
            self.assertEqual(state_dict[key], loaded_dict[key])
        loaded_obs = FixedMinMaxObserver(
            dtype=torch.qint8, qscheme=torch.per_tensor_affine, reduce_range=False
        )
        loaded_obs.load_state_dict(loaded_dict)
        self.assertEqual(observer.min_val, loaded_obs.min_val)
        self.assertEqual(observer.max_val, loaded_obs.max_val)
        self.assertEqual(observer.calculate_qparams(), loaded_obs.calculate_qparams())


class TestUpdatableMovingAverageMaxStatObserver(unittest.TestCase):
    def test_pertensor_noupdate(self):
        """Check moving average observer updates max stat but ignores min max"""
        for obs_type in [
            UpdateableObserver,
            UpdatableSymmetricMovingAverageMinMaxObserver,
            UpdateableReLUMovingAverageMinMaxObserver,
        ]:
            averaging_constant = 0.01
            observer = obs_type(
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
                reduce_range=False,
                averaging_constant=averaging_constant,
            )
            x = torch.tensor([-1.0, 2.0, 3.0, 4.0, 5.0])
            y = torch.tensor([2.0, 5.0, 5.0, 7.0, 10.0])

            result = observer(x)
            result = observer(y)
            self.assertTrue(torch.allclose(result, y))
            self.assertEqual(observer.min_val, float("inf"))
            self.assertEqual(observer.max_val, torch.tensor(float("-inf")))
            self.assertEqual(
                observer.max_stat, torch.tensor(5.0 + averaging_constant * 5.0)
            )

    def test_pertensor_updatestat(self):
        """Check min max values based on observer type"""
        averaging_constant = 0.01
        x = torch.tensor([-1.0, 2.0, 3.0, 4.0, 5.0])
        y = torch.tensor([2.0, 5.0, 5.0, 7.0, 10.0])
        max_stat = torch.tensor(5.0 + averaging_constant * 5.0)

        for obs_type, min_val, max_val, gt_scale, gt_zp in [
            [
                UpdatableSymmetricMovingAverageMinMaxObserver,
                -max_stat,
                max_stat,
                0.0396078,
                128,
            ],
            [UpdateableReLUMovingAverageMinMaxObserver, 0, max_stat, 0.0198039, 0],
        ]:
            observer = obs_type(
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
                reduce_range=False,
                averaging_constant=averaging_constant,
            )
            result = observer(x)
            result = observer(y)
            observer.update_stat()
            self.assertTrue(torch.allclose(result, y))
            self.assertEqual(observer.min_val, torch.tensor(min_val))
            self.assertEqual(observer.max_val, torch.tensor(max_val))
            self.assertEqual(observer.max_stat, torch.tensor(max_stat))
            qparams = observer.calculate_qparams()
            self.assertTrue(
                math.isclose(qparams[0].item(), gt_scale, abs_tol=1e-5, rel_tol=0)
            )
            self.assertTrue(math.isclose(qparams[1].item(), gt_zp))

    def test_stdlimit(self):
        """Check that outliers are prevented"""
        averaging_constant = 0.01
        observer = UpdateableObserver(
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=False,
            averaging_constant=averaging_constant,
        )
        x = torch.zeros(100)
        x[0] = 100
        observer(x)
        self.assertTrue(observer.max_stat < torch.tensor(100))
        self.assertEqual(
            observer.max_stat, torch.tensor(torch.mean(x) + 4 * torch.std(x))
        )

    def test_load_state_dict(self):
        """Check that state dict can be loaded"""
        averaging_constant = 0.01
        observer = UpdateableObserver(
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=False,
            averaging_constant=averaging_constant,
        )
        observer(torch.tensor([-1.0, 2.0, 3.0, 4.0, 5.0]))
        observer(torch.tensor([2.0, 5.0, 5.0, 7.0, 10.0]))

        state_dict = observer.state_dict()
        loaded_dict = _reload_state_dict(state_dict)
        for key in state_dict:
            self.assertEqual(state_dict[key], loaded_dict[key])
        loaded_obs = UpdateableObserver(
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=False,
            averaging_constant=averaging_constant,
        )
        loaded_obs.load_state_dict(loaded_dict)
        self.assertEqual(observer.min_val, loaded_obs.min_val)
        self.assertEqual(observer.max_val, loaded_obs.max_val)
        self.assertEqual(observer.max_stat, loaded_obs.max_stat)

    def test_load_state_dict_without_max_stat(self):
        """Check fallback to max_val if max_stat missing"""
        averaging_constant = 0.01
        observer = UpdateableObserver(
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=False,
            averaging_constant=averaging_constant,
        )
        observer(torch.tensor([-1.0, 2.0, 3.0, 4.0, 5.0]))
        observer(torch.tensor([2.0, 5.0, 5.0, 7.0, 10.0]))

        state_dict = observer.state_dict()
        state_dict.pop("max_stat")
        loaded_dict = _reload_state_dict(state_dict)
        self.assertFalse("max_stat" in loaded_dict)
        loaded_obs = UpdateableObserver(
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=False,
            averaging_constant=averaging_constant,
        )
        loaded_obs.load_state_dict(loaded_dict)
        self.assertEqual(observer.min_val, loaded_obs.min_val)
        self.assertEqual(observer.max_val, loaded_obs.max_val)
        self.assertEqual(observer.max_val, loaded_obs.max_stat)
