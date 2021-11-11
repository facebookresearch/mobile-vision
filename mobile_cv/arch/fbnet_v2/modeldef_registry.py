#!/usr/bin/env python3

import copy


class FBNetV2ModelArch(object):

    _MODEL_ARCH = {}

    @staticmethod
    def add(name, arch):
        assert (
            name not in FBNetV2ModelArch._MODEL_ARCH
        ), "Arch name '{}' is already existed".format(name)
        FBNetV2ModelArch._MODEL_ARCH[name] = arch

    @staticmethod
    def add_archs(archs):
        for name, arch in archs.items():
            FBNetV2ModelArch.add(name, arch)

    @staticmethod
    def get(name):
        return copy.deepcopy(FBNetV2ModelArch._MODEL_ARCH[name])
