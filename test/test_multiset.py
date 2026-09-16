import random
import string
from unittest import TestCase

import numpy as np
from scipy.optimize import linear_sum_assignment

import graphtage
from graphtage.builder import BasicBuilder
from graphtage.levenshtein import levenshtein_distance
from graphtage.multiset import MultiSetEdit
from graphtage.pydiff import diff


class TestMultiSetEdit(TestCase):
    """Tests for matching one unordered collection against another.

    :class:`graphtage.DictNode` is itself a :class:`graphtage.MultiSetNode`, so comparing a dictionary against a set
    is routed through :class:`graphtage.multiset.MultiSetEdit`.

    """

    def assertMultiSetEdit(self, from_obj, to_obj, expected_cost: int) -> MultiSetEdit:
        """Asserts that diffing two Python objects yields a multiset edit of the given cost."""
        edit = BasicBuilder().build_tree(from_obj).edits(BasicBuilder().build_tree(to_obj))
        self.assertIsInstance(edit, MultiSetEdit)
        self.assertEqual(expected_cost, edit.bounds().lower_bound)
        self.assertEqual(expected_cost, edit.bounds().upper_bound)
        return edit

    def test_dict_versus_set(self):
        """A dictionary on the `from` side and a set on the `to` side used to raise an :class:`AttributeError`."""
        self.assertMultiSetEdit({"a": 1}, {1, 2, 3}, 9)
        self.assertEqual(9, diff({"a": 1}, {1, 2, 3}).edited_cost())

    def test_dict_versus_set_of_tuples(self):
        """Set members do not have to be leaves for the `to` side to contain no key/value pairs."""
        self.assertMultiSetEdit({"a": 1}, {(1, 2), (3, 4)}, 10)

    def test_set_versus_dict(self):
        """This direction always worked, because the outer loop skips a `from` side that holds no key/value pairs."""
        self.assertMultiSetEdit({1, 2, 3}, {"a": 1}, 9)

    def test_dict_versus_dict_auto_matches_keys(self):
        """Key/value pairs that share a key are matched up front, so neither is left to insert or remove.

        Passing `auto_match_keys=False` leaves both of them in the symmetric difference, which is what makes this
        an assertion about auto-matching rather than about the cost of the edit.

        """
        edit = self.assertMultiSetEdit({"a": 1}, {"a": 2}, 1)
        self.assertEqual([], list(edit.to_insert.elements()))
        self.assertEqual([], list(edit.to_remove.elements()))

        options = graphtage.BuildOptions(auto_match_keys=False)
        unmatched = BasicBuilder(options).build_tree({"a": 1}).edits(BasicBuilder(options).build_tree({"a": 2}))
        self.assertEqual(1, len(list(unmatched.to_insert.elements())))
        self.assertEqual(1, len(list(unmatched.to_remove.elements())))

    def test_set_versus_set(self):
        self.assertMultiSetEdit({1, 2, 3}, {1, 2, 4}, 1)

    def test_dict_with_set_value(self):
        self.assertMultiSetEdit({"a": {1, 2}}, {"a": {1, 3}}, 1)

    @staticmethod
    def kvp(key: str, value: int) -> graphtage.KeyValuePairNode:
        """Builds a key/value pair node outside any dictionary."""
        return graphtage.KeyValuePairNode(graphtage.StringNode(key), graphtage.IntegerNode(value))

    def test_repeated_key_value_pairs_match_as_many_times_as_they_repeat(self):
        """A key that appears several times on each side is matched as many times as the smaller side holds it.

        Prevents the multiplicity from being lost when the loop that decides which pairs share a key is separated
        from the loop that builds their edits. A multiset can hold the same key/value pair more than once, and the
        surplus on the longer side has to be left for the matcher rather than silently matched or dropped.

        """
        from_node = graphtage.MultiSetNode([self.kvp('a', 1) for _ in range(3)])
        to_node = graphtage.MultiSetNode([self.kvp('a', 2) for _ in range(2)])
        edit = from_node.edits(to_node)
        self.assertIsInstance(edit, MultiSetEdit)
        self.assertEqual(2, len(edit._matched_kvp_edits))
        self.assertEqual(1, sum(edit.to_remove.values()))
        self.assertEqual(0, sum(edit.to_insert.values()))
        for kvp_edit in edit._matched_kvp_edits:
            self.assertEqual(1, kvp_edit.bounds().upper_bound)


class TestMatchingOptimality(TestCase):
    """Checks that the matching a diff of two string sets produces is optimal under exact edit costs.

    :class:`graphtage.matching.WeightedBipartiteMatcher` prices its edges with ``bounds().upper_bound`` after
    only partial tightening, so it can hand :func:`scipy.optimize.linear_sum_assignment` a weight matrix that
    over-states some pairs and solve a different problem than the one it meant to. Nothing else in the suite
    checks the weights the matcher actually used, only that the cost it reports is self-consistent.

    The assignment problem is solved here a second time, from a cost matrix built directly with
    :func:`graphtage.levenshtein.levenshtein_distance`, and the two totals must agree. That is an absolute
    statement about the matching rather than a comparison against any particular earlier implementation.

    """

    def assert_matching_is_optimal(self, from_strings: set[str], to_strings: set[str]):
        """Asserts that the matcher's chosen pairs cost what an independent optimal assignment costs.

        Args:
            from_strings: the set to match from.
            to_strings: the set to match to.

        """
        edit = BasicBuilder().build_tree(from_strings).edits(BasicBuilder().build_tree(to_strings))
        self.assertIsInstance(edit, MultiSetEdit)
        while edit.tighten_bounds():
            pass
        matcher = edit._matcher
        from_nodes, to_nodes = list(matcher.from_nodes), list(matcher.to_nodes)
        self.assertTrue(from_nodes)
        self.assertTrue(to_nodes)
        costs = np.array(
            [[levenshtein_distance(f.object, t.object) for t in to_nodes] for f in from_nodes],
            dtype=np.int64
        )
        rows, columns = linear_sum_assignment(costs)
        matching = matcher.matching
        self.assertEqual(min(len(from_nodes), len(to_nodes)), len(matching))
        self.assertEqual(
            int(costs[rows, columns].sum()),
            sum(levenshtein_distance(f.object, t.object) for f, (t, _) in matching.items())
        )

    @staticmethod
    def words(rng: random.Random, count: int, length: int) -> set[str]:
        return {''.join(rng.choices(string.ascii_lowercase, k=length)) for _ in range(count)}

    def test_unrelated_strings(self):
        rng = random.Random(1)
        for _ in range(5):
            self.assert_matching_is_optimal(self.words(rng, 12, 10), self.words(rng, 12, 10))

    def test_near_misses(self):
        """Pairs that differ in two characters make the difference between assignments small and easy to miss."""
        rng = random.Random(2)
        originals = self.words(rng, 14, 12)
        mutated = {w[:5] + ''.join(rng.choices(string.ascii_lowercase, k=2)) + w[7:] for w in originals}
        self.assert_matching_is_optimal(originals, mutated)

    def test_shared_affixes(self):
        """A shared prefix and suffix is what the exact cost helper strips, so it has to stay exact."""
        rng = random.Random(3)
        self.assert_matching_is_optimal(
            {f"/usr/local/lib/{w}/bin" for w in self.words(rng, 10, 6)},
            {f"/usr/local/lib/{w}/bin" for w in self.words(rng, 10, 6)}
        )

    def test_rectangular_matching(self):
        """More elements on one side than the other leaves some unmatched, which scipy also handles."""
        rng = random.Random(4)
        self.assert_matching_is_optimal(self.words(rng, 8, 9), self.words(rng, 15, 9))
        self.assert_matching_is_optimal(self.words(rng, 15, 9), self.words(rng, 8, 9))

    def test_small_alphabet(self):
        """A small alphabet maximizes the number of assignments that tie, which is where a bias shows up."""
        rng = random.Random(5)
        self.assert_matching_is_optimal(
            {''.join(rng.choices('abcd', k=8)) for _ in range(12)},
            {''.join(rng.choices('abcd', k=8)) for _ in range(12)}
        )
