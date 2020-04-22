import itertools
import random
from unittest import TestCase

import numpy as np
from tqdm import tqdm, trange

from graphtage.matching import get_dtype, min_weight_bipartite_matching, WeightedBipartiteMatcher

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
                get_edge=lambda n1, n2: edges[n1][n2]
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

            def get_edge(f, t):
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
