#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.


class IdentityPreprocess:
    def __call__(self, x):
        inputs = x
        return inputs


class IdentityPostprocess:
    def __call__(self, x, inputs, outputs):
        return outputs


class BatchPreprocess:
    def __call__(self, x):
        """Batch all tensors recursively"""
        raise NotImplementedError()


class DebatchPostprocess:
    def __call__(self, x, inputs, outputs):
        """Debatch all tensors recursively"""
        raise NotImplementedError()


class NaiveRunFunc:
    def __call__(self, model_or_models, inputs):
        return model_or_models(inputs)


class RunFuncWithListInputs:
    def __call__(self, model_or_models, inputs):
        assert isinstance(inputs, (list, tuple)), "Inputs should be a list/tuple"
        return model_or_models(*inputs)
