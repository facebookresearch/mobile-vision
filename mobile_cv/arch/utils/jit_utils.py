import copy

import torch


def get_traceable_model(model: torch.nn.Module):
    """Return a copy of the model and apply `module.to_traceable()` for each
    module in model if `to_traceable()` is implemented.
    Each `module.to_traceable() -> None` function convert the module to make it suitable
    for tracing (like scripting some of the sub modules) inplace.
    Useful to mix tracing and scripting together.
    """
    model = copy.deepcopy(model)

    def _convert(module):
        if hasattr(module, "to_traceable"):
            module.to_traceable()

    model.apply(_convert)
    return model
