import unittest

import mobile_cv.torch.utils_pytorch.worker_helper as wh


class TestWorkerHelperChunkIndices(unittest.TestCase):
    def test_evenly_divisible(self):
        start, end = wh.compute_chunk_indices(0, 5, 100)  # 20 elements per rank
        self.assertEqual((start, end), (0, 20))

        start, end = wh.compute_chunk_indices(4, 5, 100)
        self.assertEqual((start, end), (80, 100))

    def test_unevenly_divisible(self):
        # For 103 elements and 5 ranks:
        # First 3 ranks get 21 elements each and last 2 ranks get 20 elements each.
        start, end = wh.compute_chunk_indices(0, 5, 103)
        self.assertEqual((start, end), (0, 21))

        start, end = wh.compute_chunk_indices(2, 5, 103)
        self.assertEqual((start, end), (42, 63))

        start, end = wh.compute_chunk_indices(3, 5, 103)
        self.assertEqual((start, end), (63, 83))

        start, end = wh.compute_chunk_indices(4, 5, 103)
        self.assertEqual((start, end), (83, 103))

    def test_single_rank(self):
        start, end = wh.compute_chunk_indices(
            0, 1, 50
        )  # Single rank, should get all elements
        self.assertEqual((start, end), (0, 50))

    def test_more_ranks_than_elements(self):
        start, end = wh.compute_chunk_indices(0, 10, 5)
        self.assertEqual((start, end), (0, 1))

        start, end = wh.compute_chunk_indices(4, 10, 5)
        self.assertEqual((start, end), (4, 5))

        # For ranks >= total_elements, should return end index same as start index (i.e., no elements).
        start, end = wh.compute_chunk_indices(5, 10, 5)
        self.assertEqual((start, end), (5, 5))

        start, end = wh.compute_chunk_indices(9, 10, 5)
        self.assertEqual((start, end), (5, 5))
