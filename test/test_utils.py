import sys
from collections import Counter
from unittest import TestCase

from graphtage.json import build_tree
from graphtage.sequences import SequenceNode
from graphtage.utils import HashableCounter, SparseMatrix, largest, smallest

from .timing import run_with_time_limit


def _count_comparisons(first, second) -> int:
    """Returns the number of :meth:`SequenceNode.__eq__` calls that comparing `first` to `second` makes."""
    original = SequenceNode.__eq__
    calls = 0

    def counted(self, other):
        nonlocal calls
        calls += 1
        return original(self, other)

    SequenceNode.__eq__ = counted
    try:
        if first != second:
            raise AssertionError('the two trees must be equal for the comparison to traverse them')
    finally:
        SequenceNode.__eq__ = original
    return calls


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

    def test_smallest(self):
        for i in smallest(range(1000), n=10):
            self.assertGreater(10, i)
        for i in smallest(*list(range(1000)), n=10):
            self.assertGreater(10, i)

    def test_largest(self):
        for i in largest(range(1000), n=10):
            self.assertLess(1000 - 11, i)
        for i in largest(*list(range(1000)), n=10):
            self.assertLess(1000 - 11, i)


class TestHashableCounter(TestCase):
    def test_equality_matches_counter(self):
        self.assertEqual(HashableCounter('aab'), Counter('aab'))
        self.assertNotEqual(HashableCounter('aab'), Counter('ab'))
        self.assertNotEqual(HashableCounter('aab'), Counter('aabc'))
        self.assertNotEqual(HashableCounter('aab'), Counter('abb'))
        self.assertNotEqual(HashableCounter('aab'), 'aab')
        self.assertEqual(HashableCounter(), Counter())
        # collections.Counter treats a missing count and a zero count as the same:
        self.assertEqual(HashableCounter({'a': 1, 'b': 0}), Counter({'a': 1}))
        self.assertEqual(HashableCounter({'a': 1}), Counter({'a': 1, 'b': 0}))

    def test_nested_equality_is_not_exponential(self):
        """Comparing deeply nested dictionaries must not be exponential in the nesting depth.

        :meth:`collections.Counter.__eq__` looks up every element in both counters, so each level of nesting
        compares its children twice and the whole comparison makes ``2 ** (depth + 1) - 1`` calls. Before
        :class:`graphtage.utils.HashableCounter` overrode it, comparing this document against itself took over
        half an hour.

        The assertion counts comparisons rather than measuring elapsed time, so it does not depend on how fast
        the machine is. The time limit is a backstop that stops a regression from hanging the suite: at this
        depth the exponential version would make more than two billion calls.

        """
        depth = 30
        obj = {'leaf': 'x'}
        for i in range(depth):
            obj = {f'k{i}': obj}
        with run_with_time_limit(seconds=5):
            calls = _count_comparisons(build_tree(obj), build_tree(obj))
        self.assertLess(calls, 10 * depth, f'{calls} comparisons at depth {depth} is not linear')
