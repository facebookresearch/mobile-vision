#!/usr/bin/env python3


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
