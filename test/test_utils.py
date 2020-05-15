import sys
from unittest import TestCase

from graphtage.utils import SparseMatrix


class TestSparseMatrix(TestCase):
    def test_matrix_bounds(self):
        matrix: SparseMatrix[int] = SparseMatrix(num_rows=10, num_cols=10, default_value=None)
        with self.assertRaises(IndexError):
            _ = matrix[matrix.num_rows]
        with self.assertRaises(IndexError):
            _ = matrix[0][matrix.num_cols]

    def test_matrix_default_value(self):
        matrix: SparseMatrix[int] = SparseMatrix(default_value=10)
        self.assertEqual(matrix[0][0], 10)
        matrix[0][0] = 11
        self.assertEqual(matrix[0][0], 11)

    def test_matrix_getsizeof(self):
        matrix: SparseMatrix[int] = SparseMatrix()
        size_before = matrix.getsizeof()
        dim = 1000
        int_sizes = 0
        for i in range(dim):
            for j in range(dim):
                matrix[i][j] = i * dim + j
                int_sizes += sys.getsizeof(matrix[i][j])
        size_after = matrix.getsizeof()
        self.assertGreaterEqual(size_after - size_before, int_sizes)

    def test_matrix_shape(self):
        matrix: SparseMatrix[int] = SparseMatrix()
        self.assertEqual((0, 0), matrix.shape())
        matrix[10][20] = 1
        self.assertEqual((11, 21), matrix.shape())
        matrix = SparseMatrix(num_rows=10, num_cols=10)
        self.assertEqual((10, 10), matrix.shape())
