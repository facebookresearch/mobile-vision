# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe


class ConverterBase(object):
    """
    Interface for "Converter", users are expected to implement their own converter class
    for each type of model. This interface is agnostic to codebase, each codebase may
    inherit this interface and create their own favored base class.
    """

    # TODO: create a constructor interface suitable for multiple codebases

    # TODO: define the interface for `get_xxx_model`
