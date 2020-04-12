from unittest import TestCase

from graphtage.matching import HungarianMethod, WeightedBipartiteMatcher

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
