import itertools
import random
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from tqdm import tqdm, trange

from graphtage.bounds import ConstantBound, Range, make_distinct
from graphtage.matching import (
    MatchingFromNode,
    MatchingToNode,
    WeightedBipartiteMatcher,
    get_dtype,
    min_weight_bipartite_matching,
)

from .test_bounds import RandomDecreasingRange


class TestWeightedBipartiteMatcher(TestCase):
    def test_weighted_bipartite_matching(self):
        for n in trange(1, 25, 3):
            from_nodes = list(range(n))
            to_nodes = list(range(n))
            edges = [
                [RandomDecreasingRange() for _ in range(len(to_nodes))] for _ in range(len(from_nodes))
            ]
            for i in range(min(len(from_nodes), len(to_nodes))):
                edges[i][i] = RandomDecreasingRange(fixed_lb=0, fixed_ub=100000, final_value=0)
            matcher = WeightedBipartiteMatcher(
                from_nodes=from_nodes,
                to_nodes=to_nodes,
                # `edges` is bound as a default so the closure captures this iteration's matrix.
                get_edge=lambda n1, n2, edges=edges: edges[n1][n2]
            )
            initial_bounds = matcher.bounds()
            prev_diff = initial_bounds.upper_bound - initial_bounds.lower_bound
            with tqdm(leave=False, total=prev_diff) as t:
                t.update(0)
                while matcher.tighten_bounds():
                    new_bounds = matcher.bounds()
                    new_diff = new_bounds.upper_bound - new_bounds.lower_bound
                    self.assertLess(new_diff, prev_diff)
                    t.update(prev_diff - new_diff)
                    prev_diff = new_diff
            self.assertTrue(matcher.bounds().definitive())
            self.assertEqual(0, matcher.bounds().upper_bound)

    def test_min_weight_bipartite_matching(self):
        for _ in trange(50):
            num_from = random.randint(1, 500)
            num_to = random.randint(1, 500)
            from_nodes = [f'f{i}' for i in range(num_from)]
            to_nodes = [f't{i}' for i in range(num_to)]
            # Force an optimal, zero-value matching:
            expected_matching = {
                i: (i, 0) for i in range(min(num_from, num_to))
            }
            edges = {
                (from_nodes[i], to_nodes[i]): 0 for i in range(min(num_from, num_to))
            }
            edge_probability = 0.9
            edges.update({
                (i, j): random.randint(1, 2**16) for i, j in itertools.product(from_nodes, to_nodes)
                if (i, j) not in edges and random.random() < edge_probability
            })

            def get_edge(f, t, edges=edges):
                if (f, t) in edges:
                    return edges[(f, t)]
                else:
                    return None

            matching = min_weight_bipartite_matching(from_nodes=from_nodes, to_nodes=to_nodes, get_edges=get_edge)

            self.assertEqual(expected_matching, matching)

    def test_get_dtype(self):
        for min_range, max_range, expected in (
            (0, 255, np.uint8),
            (-1, 127, np.int8),
            (-128, 255, np.int16),
            (0, 2**64 - 1, np.uint64),
            (0, 2**64, int)
        ):
            actual = get_dtype(min_range, max_range)
            self.assertEqual(np.dtype(expected), actual)


class TestMatchingNode(TestCase):
    """MatchingNode caches its edges lazily, and every accessor has to trigger that."""

    class StubMatcher:
        """The smallest thing MatchingNode.construct_edges needs from a matcher."""

        def __init__(self):
            self.from_nodes = []
            self.to_nodes = []

        @staticmethod
        def get_edge(from_node, to_node):
            return Range(0, 1)

    def setUp(self):
        self.matcher = TestMatchingNode.StubMatcher()
        self.from_node = MatchingFromNode(self.matcher, 'f')
        self.to_node = MatchingToNode(self.matcher, 't')
        self.matcher.from_nodes.append(self.from_node)
        self.matcher.to_nodes.append(self.to_node)

    def test_contains_populates_the_cache(self):
        """__contains__ used to reference the `edges` method without calling it, leaving the cache empty."""
        self.assertIn(self.to_node, self.from_node)

    def test_getitem_populates_the_cache(self):
        """__getitem__ had the same no-op, so it raised TypeError on a None cache."""
        self.assertIsNotNone(self.from_node[self.to_node])

    def test_edges_agrees_with_getitem(self):
        self.assertEqual([self.from_node[self.to_node]], list(self.from_node.edges()))


class TestMakeEdgesDistinct(TestCase):
    """Covers the shortcut in :meth:`graphtage.matching.WeightedBipartiteMatcher._make_edges_distinct`.

    Every edge whose weight is a :class:`graphtage.StringEdit` knows its cost the moment it is constructed, so a
    matching over leaves hands :func:`graphtage.bounds.make_distinct` nothing to tighten. It would still build an
    interval tree over every edge before finding that out, which on a matching of a few hundred elements costs
    more than solving the matching.

    """

    WEIGHTS = (
        (7, 2, 9, 4),
        (3, 8, 1, 6),
        (5, 5, 2, 2),
        (9, 1, 4, 8),
    )

    def constant_matcher(self, weights=None) -> WeightedBipartiteMatcher[int]:
        """Builds a matcher whose every edge already has a definitive bound."""
        weights = self.WEIGHTS if weights is None else weights
        return WeightedBipartiteMatcher(
            from_nodes=list(range(len(weights))),
            to_nodes=list(range(len(weights[0]))),
            get_edge=lambda f, t, weights=weights: ConstantBound(weights[f][t]),
        )

    @staticmethod
    def solve(matcher: WeightedBipartiteMatcher[int]) -> dict[int, tuple[int, int]]:
        """Tightens a matcher to completion and returns the pairs it chose with their weights."""
        while matcher.tighten_bounds():
            pass
        return {f: (t, edge.bounds().upper_bound) for f, (t, edge) in matcher.matching.items()}

    def test_definitive_edges_do_not_reach_make_distinct(self):
        """A matching whose edges are all definitive skips the interval tree entirely.

        Prevents the shortcut from being written but never taken, which no comparison of matchings can detect
        because :func:`graphtage.bounds.make_distinct` tightens nothing in this case either.

        """
        with patch('graphtage.matching.make_distinct') as distinct:
            matching = self.solve(self.constant_matcher())
        distinct.assert_not_called()
        self.assertEqual(4, len(matching))

    def test_indefinite_edges_still_reach_make_distinct(self):
        """An edge whose bounds still overlap another's is tightened before the matching is solved.

        Prevents the shortcut from swallowing the case it does not apply to, which would hand
        :func:`scipy.optimize.linear_sum_assignment` a matrix of upper bounds that overstate some pairs.

        """
        random.seed(0x5EED)
        edges = [[RandomDecreasingRange() for _ in range(4)] for _ in range(4)]
        matcher = WeightedBipartiteMatcher(
            from_nodes=list(range(4)),
            to_nodes=list(range(4)),
            get_edge=lambda f, t: edges[f][t],
        )
        with patch('graphtage.matching.make_distinct', wraps=make_distinct) as distinct:
            self.solve(matcher)
        self.assertTrue(distinct.called)

    def test_the_shortcut_does_not_change_the_matching(self):
        """The pairs chosen are the same whether or not :func:`graphtage.bounds.make_distinct` runs.

        Prevents the shortcut from leaving out a step that the matching depends on. The comparison is against the
        implementation the shortcut replaced, run over the same weights.

        """
        def always_make_distinct(matcher):
            if matcher._edges_are_distinct:
                return False
            make_distinct(*itertools.chain(*matcher.edges))
            matcher._edges_are_distinct = True
            return True

        random.seed(0xDEC1DE)
        for _ in range(20):
            rows, columns = random.randint(1, 7), random.randint(1, 7)
            weights = tuple(tuple(random.randint(0, 40) for _ in range(columns)) for _ in range(rows))
            shortcut = self.solve(self.constant_matcher(weights))
            with patch.object(WeightedBipartiteMatcher, '_make_edges_distinct', always_make_distinct):
                forced = self.solve(self.constant_matcher(weights))
            self.assertEqual(forced, shortcut, f"weights={weights!r}")
