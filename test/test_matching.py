import itertools
import random
from unittest import TestCase

import numpy as np
from tqdm import trange

from graphtage.matching import get_dtype, min_weight_bipartite_matching, WeightedBipartiteMatcher

from .test_bounds import RandomDecreasingRange


class TestWeightedBipartiteMatcher(TestCase):
    def test_weighted_bipartite_matching(self):
        from_nodes = list(range(3))
        to_nodes = list(range(3))
        edges = [
            [RandomDecreasingRange() for _ in range(len(to_nodes))] for _ in range(len(from_nodes))
        ]
        edges = [
            [RandomDecreasingRange(n, n) for n in [810177, 20679, 612881]],
            [RandomDecreasingRange(n, n) for n in [679754, 810042, 809299]],
            [RandomDecreasingRange(n, n) for n in [429760, 385568, 982600]]]
        for i in range(len(edges)):
           print([r.final_value for r in edges[i]])
        matcher = WeightedBipartiteMatcher(
            from_nodes=from_nodes,
            to_nodes=to_nodes,
            get_edge=lambda n1, n2: edges[n1][n2]
        )
        matcher.tighten_bounds()

    def test_min_weight_bipartite_matching(self):
        for _ in trange(100):
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
