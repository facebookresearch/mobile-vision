# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import torch


class FreezeContainer(object):
    def __init__(self, module: torch.nn.Module) -> None:
        self.module = module
        self.module.eval()
        for x in self.module.parameters():
            x.requires_grad = False

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class FreezeModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = FreezeContainer(module)

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
