from unittest import TestCase

from graphtage.utils import SparseMatrix


class TestSparseMatrix(TestCase):
    def test_matrix_bounds(self):
        matrix: SparseMatrix[int] = SparseMatrix(num_rows=10, num_cols=10, default_value=None)
        with self.assertRaises(ValueError):
            matrix[matrix.num_rows]
        with self.assertRaises(ValueError):
            matrix[0][matrix.num_cols]

    def test_matrix_default_value(self):
        matrix: SparseMatrix[int] = SparseMatrix(default_value=10)
        self.assertEqual(matrix[0][0], 10)
        matrix[0][0] = 11
        self.assertEqual(matrix[0][0], 11)
