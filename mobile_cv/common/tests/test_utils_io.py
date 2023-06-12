import unittest

from mobile_cv.common.utils_io import get_path_manager


class TestUtilsIo(unittest.TestCase):
    def test_get_path_manager(self):
        pathmanager1 = get_path_manager()
        pathmanager2 = get_path_manager()
        self.assertIs(
            pathmanager1,
            pathmanager2,
            "get_path_manager should return same object because of @lru_cache",
        )
