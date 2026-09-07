from unittest import TestCase

import graphtage
from graphtage.builder import BasicBuilder
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
