#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from mobile_cv.common.misc.test_utils import SubPackageInitFileTestMixin


class TestSubPackageInitFile(SubPackageInitFileTestMixin, unittest.TestCase):
    def get_pacakge_name(self) -> str:
        return "mobile_cv"
