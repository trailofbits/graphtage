"""Tests for the parallel execution module."""

import sys
from unittest import TestCase

from graphtage.parallel import (
    Backend,
    ParallelConfig,
    _reset_backend_cache,
    available_backends,
    compute_edge_matrix,
    compute_edge_matrix_sequential,
    compute_edge_matrix_threaded,
    configure,
    get_backend,
    get_config,
    is_free_threading_available,
    is_webgpu_available,
    tighten_diagonal,
    tighten_diagonal_sequential,
    tighten_diagonal_threaded,
)

from .test_bounds import RandomDecreasingRange


class TestBackendDetection(TestCase):
    """Tests for backend detection functions."""

    def setUp(self):
        _reset_backend_cache()

    def tearDown(self):
        _reset_backend_cache()
        configure()  # Reset to defaults

    def test_sequential_always_available(self):
        """SEQUENTIAL backend should always be available."""
        backends = available_backends()
        self.assertIn(Backend.SEQUENTIAL, backends)

    def test_free_threading_detection(self):
        """Test that free-threading detection works."""
        result = is_free_threading_available()
        self.assertIsInstance(result, bool)

        # On Python < 3.13, should always be False
        if sys.version_info < (3, 13):
            self.assertFalse(result)

    def test_webgpu_detection(self):
        """Test that WebGPU detection works without crashing."""
        result = is_webgpu_available()
        self.assertIsInstance(result, bool)


class TestConfiguration(TestCase):
    """Tests for the configuration API."""

    def tearDown(self):
        configure()  # Reset to defaults

    def test_default_config(self):
        """Test default configuration values."""
        config = get_config()
        self.assertIsNone(config.preferred_backend)
        self.assertIsNone(config.max_workers)
        self.assertEqual(10000, config.webgpu_min_size)
        self.assertEqual(100, config.threaded_min_size)
        self.assertTrue(config.enabled)

    def test_configure_updates_config(self):
        """Test that configure() updates the global config."""
        configure(
            preferred_backend=Backend.THREADED,
            max_workers=4,
            webgpu_min_size=5000,
            threaded_min_size=50,
            enabled=False,
        )
        config = get_config()
        self.assertEqual(Backend.THREADED, config.preferred_backend)
        self.assertEqual(4, config.max_workers)
        self.assertEqual(5000, config.webgpu_min_size)
        self.assertEqual(50, config.threaded_min_size)
        self.assertFalse(config.enabled)


class TestBackendSelection(TestCase):
    """Tests for backend selection logic."""

    def tearDown(self):
        _reset_backend_cache()
        configure()  # Reset to defaults

    def test_get_backend_respects_disabled(self):
        """When disabled, should always return SEQUENTIAL."""
        configure(enabled=False)
        backend = get_backend(problem_size=100000)
        self.assertEqual(Backend.SEQUENTIAL, backend)

    def test_get_backend_respects_min_size(self):
        """Should return SEQUENTIAL for small problems."""
        configure(threaded_min_size=100)
        backend = get_backend(problem_size=50)
        self.assertEqual(Backend.SEQUENTIAL, backend)

    def test_get_backend_explicit_preference(self):
        """Explicit preference should override auto-selection."""
        backend = get_backend(preferred=Backend.SEQUENTIAL, problem_size=100000)
        self.assertEqual(Backend.SEQUENTIAL, backend)


class TestEdgeMatrixComputation(TestCase):
    """Tests for edge matrix computation functions."""

    def test_sequential_basic(self):
        """Test basic sequential edge matrix computation."""
        from_nodes = [1, 2, 3]
        to_nodes = ["a", "b"]

        def get_edge(f, t):
            return f"{f}-{t}"

        result = compute_edge_matrix_sequential(from_nodes, to_nodes, get_edge)

        self.assertEqual(3, len(result))
        self.assertEqual(2, len(result[0]))
        self.assertEqual("1-a", result[0][0])
        self.assertEqual("3-b", result[2][1])

    def test_threaded_matches_sequential(self):
        """Test that threaded results match sequential."""
        from_nodes = list(range(10))
        to_nodes = list(range(10))

        def get_edge(f, t):
            return f * 100 + t

        sequential = compute_edge_matrix_sequential(from_nodes, to_nodes, get_edge)
        threaded = compute_edge_matrix_threaded(
            from_nodes, to_nodes, get_edge, max_workers=4
        )

        self.assertEqual(len(sequential), len(threaded))
        for i in range(len(sequential)):
            self.assertEqual(len(sequential[i]), len(threaded[i]))
            for j in range(len(sequential[i])):
                self.assertEqual(sequential[i][j], threaded[i][j])

    def test_empty_inputs(self):
        """Test handling of empty inputs."""
        result = compute_edge_matrix_sequential([], [1, 2], lambda f, t: f + t)
        self.assertEqual([], result)

        result = compute_edge_matrix_threaded([], [1, 2], lambda f, t: f + t)
        self.assertEqual([], result)

    def test_compute_edge_matrix_auto_backend(self):
        """Test that compute_edge_matrix works with auto backend selection."""
        from_nodes = list(range(5))
        to_nodes = list(range(5))

        def get_edge(f, t):
            return f * 10 + t

        result = compute_edge_matrix(from_nodes, to_nodes, get_edge)

        self.assertEqual(5, len(result))
        self.assertEqual(5, len(result[0]))
        self.assertEqual(0, result[0][0])
        self.assertEqual(44, result[4][4])


class TestDiagonalTightening(TestCase):
    """Tests for diagonal tightening functions."""

    def test_sequential_tightening(self):
        """Test sequential diagonal tightening."""
        edits = [[RandomDecreasingRange() for _ in range(5)] for _ in range(5)]

        # Tighten the first diagonal (just cell 0,0)
        cells = [(0, 0)]
        tighten_diagonal_sequential(cells, lambda r, c: edits[r][c])

        # Should be definitive after tightening
        self.assertTrue(edits[0][0].bounds().definitive())

    def test_threaded_tightening(self):
        """Test threaded diagonal tightening."""
        edits = [[RandomDecreasingRange() for _ in range(5)] for _ in range(5)]

        # Tighten a diagonal
        cells = [(0, 1), (1, 0)]
        tighten_diagonal_threaded(cells, lambda r, c: edits[r][c], max_workers=2)

        # All cells should be definitive
        for row, col in cells:
            self.assertTrue(edits[row][col].bounds().definitive())

    def test_threaded_matches_sequential_results(self):
        """Test that threaded tightening produces equivalent results."""
        # Create two identical sets of edits
        import random

        random.seed(42)
        edits_seq = [[RandomDecreasingRange() for _ in range(5)] for _ in range(5)]
        random.seed(42)
        edits_thr = [[RandomDecreasingRange() for _ in range(5)] for _ in range(5)]

        # Tighten the same diagonal with both methods
        cells = [(0, 2), (1, 1), (2, 0)]

        tighten_diagonal_sequential(cells, lambda r, c: edits_seq[r][c])
        tighten_diagonal_threaded(cells, lambda r, c: edits_thr[r][c], max_workers=3)

        # Results should have the same final values
        for row, col in cells:
            self.assertEqual(
                edits_seq[row][col].final_value, edits_thr[row][col].final_value
            )
            self.assertTrue(edits_seq[row][col].bounds().definitive())
            self.assertTrue(edits_thr[row][col].bounds().definitive())

    def test_empty_diagonal(self):
        """Test handling of empty diagonal."""
        result = tighten_diagonal_sequential([], lambda r, c: None)
        self.assertFalse(result)

        result = tighten_diagonal_threaded([], lambda r, c: None)
        self.assertFalse(result)


class TestIntegration(TestCase):
    """Integration tests with actual Graphtage types."""

    def tearDown(self):
        configure()  # Reset to defaults

    def test_weighted_bipartite_matcher_uses_parallel(self):
        """Test that WeightedBipartiteMatcher uses parallel edge computation."""
        from graphtage.matching import WeightedBipartiteMatcher

        from_nodes = list(range(20))
        to_nodes = list(range(20))
        edges = [
            [RandomDecreasingRange() for _ in range(len(to_nodes))]
            for _ in range(len(from_nodes))
        ]

        # Force diagonal to have zero cost
        for i in range(min(len(from_nodes), len(to_nodes))):
            edges[i][i] = RandomDecreasingRange(fixed_lb=0, fixed_ub=100000, final_value=0)

        matcher = WeightedBipartiteMatcher(
            from_nodes=from_nodes,
            to_nodes=to_nodes,
            get_edge=lambda n1, n2: edges[n1][n2],
        )

        # Access edges to trigger computation
        edge_matrix = matcher.edges

        # Verify matrix dimensions
        self.assertEqual(len(from_nodes), len(edge_matrix))
        self.assertEqual(len(to_nodes), len(edge_matrix[0]))

        # Verify matcher still works correctly
        while matcher.tighten_bounds():
            pass

        self.assertTrue(matcher.bounds().definitive())
        self.assertEqual(0, matcher.bounds().upper_bound)
