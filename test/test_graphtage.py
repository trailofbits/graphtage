import random
from io import StringIO
from unittest import TestCase

import graphtage
import graphtage.json
import graphtage.multiset
import graphtage.tree

from graphtage.printer import Printer

from .timing import run_with_time_limit


def unordered(*values) -> graphtage.UnorderedListNode:
    return graphtage.UnorderedListNode([graphtage.IntegerNode(v) for v in values])


def ordered(*values) -> graphtage.ListNode:
    return graphtage.ListNode([graphtage.IntegerNode(v) for v in values])


class TestGraphtage(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.small_from = graphtage.json.build_tree({
            "test": "foo",
            "baz": 1
        })
        cls.small_to = graphtage.json.build_tree({
            "test": "bar",
            "baz": 2
        })
        cls.list_from = graphtage.json.build_tree([0, 1, 2, 3, 4, 5])
        cls.list_to = graphtage.json.build_tree([1, 2, 3, 4, 5])

    def test_string_diff_printing(self):
        s1 = graphtage.StringNode("abcdef")
        s2 = graphtage.StringNode("azced")
        diff = s1.diff(s2)
        out_stream = StringIO()
        p = Printer(ansi_color=True, out_stream=out_stream)
        diff.print(p)
        self.assertEqual(diff.edited_cost(), 5)
        self.assertEqual('\x1b[32m"\x1b[37m\x1b[41m\x1b[1m\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[42m\x1b[1m\x1b[0m\x1b[49m\x1b[32ma\x1b[37m\x1b[41m\x1b[1m\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[42m\x1b[1m\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[41m\x1b[1m\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[42m\x1b[1mz̟\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[41m\x1b[1mb̶\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[42m\x1b[1m\x1b[0m\x1b[49m\x1b[32mc\x1b[37m\x1b[41m\x1b[1m\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[42m\x1b[1m\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[41m\x1b[1md̶\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[42m\x1b[1m\x1b[0m\x1b[49m\x1b[32me\x1b[37m\x1b[41m\x1b[1m\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[42m\x1b[1m\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[41m\x1b[1m\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[42m\x1b[1md̟\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[41m\x1b[1mf̶\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[42m\x1b[1m\x1b[0m\x1b[49m\x1b[32m"\x1b[39m', out_stream.getvalue())

    def test_string_diff_remove_insert_reordering(self):
        s1 = graphtage.StringNode('abcdefg')
        s2 = graphtage.StringNode('abhijfg')
        diff = s1.diff(s2)
        out_stream = StringIO()
        p = Printer(ansi_color=True, out_stream=out_stream)
        diff.print(p)
        self.assertEqual('\x1b[32m"\x1b[37m\x1b[41m\x1b[1m\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[42m\x1b[1m\x1b[0m\x1b[49m\x1b[32ma\x1b[37m\x1b[41m\x1b[1m\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[42m\x1b[1m\x1b[0m\x1b[49m\x1b[32mb\x1b[37m\x1b[41m\x1b[1m\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[42m\x1b[1m\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[41m\x1b[1m\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[42m\x1b[1mh̟\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[41m\x1b[1m\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[42m\x1b[1mi̟\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[41m\x1b[1m\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[42m\x1b[1mj̟\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[41m\x1b[1mc̶\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[42m\x1b[1m\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[41m\x1b[1md̶\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[42m\x1b[1m\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[41m\x1b[1me̶\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[42m\x1b[1m\x1b[0m\x1b[49m\x1b[32mf\x1b[37m\x1b[41m\x1b[1m\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[42m\x1b[1m\x1b[0m\x1b[49m\x1b[32mg\x1b[37m\x1b[41m\x1b[1m\x1b[0m\x1b[49m\x1b[32m\x1b[37m\x1b[42m\x1b[1m\x1b[0m\x1b[49m\x1b[32m"\x1b[39m', out_stream.getvalue())

    def test_small_diff(self):
        diff = self.small_from.diff(self.small_to)
        self.assertIsInstance(diff, graphtage.DictNode)
        self.assertIsInstance(diff, graphtage.tree.EditedTreeNode)
        self.assertEqual(1, len(diff.edit_list))
        self.assertIsInstance(diff.edit_list[0], graphtage.multiset.MultiSetEdit)
        has_test_match = False
        has_baz_match = False
        for edit in diff.edit_list[0].edits():
            if edit.bounds().upper_bound > 0:
                self.assertIsInstance(edit, graphtage.KeyValuePairEdit)
                key_edit = edit.key_edit
                value_edit = edit.value_edit
                if isinstance(value_edit.from_node, graphtage.StringNode):
                    self.assertIsInstance(key_edit.to_node, graphtage.StringNode)
                    self.assertEqual(key_edit.from_node.object, 'test')
                    self.assertEqual(value_edit.from_node.object, 'foo')
                    self.assertEqual(value_edit.to_node.object, 'bar')
                    self.assertEqual(edit.bounds().upper_bound, 6)
                    self.assertFalse(has_test_match)
                    has_test_match = True
                elif isinstance(value_edit.from_node, graphtage.IntegerNode):
                    self.assertIsInstance(value_edit.to_node, graphtage.IntegerNode)
                    self.assertEqual(value_edit.from_node.object, 1)
                    self.assertEqual(value_edit.to_node.object, 2)
                    self.assertEqual(value_edit.bounds().upper_bound, 1)
                    self.assertFalse(has_baz_match)
                    has_baz_match = True
                else:
                    self.fail()
        self.assertTrue(has_test_match)
        self.assertTrue(has_baz_match)

    def test_list_diff(self):
        diff = self.list_from.diff(self.list_to)
        self.assertIsInstance(diff, graphtage.ListNode)
        self.assertIsInstance(diff, graphtage.tree.EditedTreeNode)
        self.assertEqual(1, len(diff.edit_list))
        self.assertIsInstance(diff.edit_list[0], graphtage.EditDistance)
        for edit in diff.edit_list[0].edits():
            if edit.bounds().upper_bound > 0:
                self.assertIsInstance(edit, graphtage.Remove)
                self.assertIsInstance(edit.from_node, graphtage.IntegerNode)
                self.assertEqual(edit.from_node.object, 0)
                self.assertIsInstance(edit.to_node, graphtage.ListNode)
                self.assertEqual(edit.to_node, self.list_from)
            else:
                self.assertIsInstance(edit, graphtage.Match)

    def test_single_element_list(self):
        diff = graphtage.json.build_tree([1]).diff(graphtage.json.build_tree([2]))
        self.assertIsInstance(diff, graphtage.ListNode)
        self.assertIsInstance(diff, graphtage.tree.EditedTreeNode)
        self.assertEqual(1, len(diff.edit_list))
        self.assertIsInstance(diff.edit_list[0], graphtage.FixedLengthSequenceEdit)

    def test_empty_list(self):
        diff = graphtage.ListNode(()).diff(graphtage.ListNode(()))
        self.assertEqual(1, len(diff.edit_list))
        self.assertIsInstance(diff.edit_list[0], graphtage.Match)
        self.assertEqual(0, diff.edit_list[0].bounds().upper_bound)

    def test_null_json(self):
        diff = graphtage.json.build_tree([None]).diff(graphtage.json.build_tree([1]))
        self.assertIsInstance(diff, graphtage.ListNode)
        self.assertIsInstance(diff, graphtage.tree.EditedTreeNode)
        self.assertEqual(1, len(diff.edit_list))
        self.assertIsInstance(diff.edit_list[0], graphtage.FixedLengthSequenceEdit)


class TestUnorderedListNode(TestCase):
    def test_ordered_list_reorder_costs(self):
        """Reordering an ordinary list is an edit; this is what :class:`graphtage.UnorderedListNode` changes."""
        edit = ordered(1, 2, 3).edits(ordered(3, 2, 1))
        self.assertIsInstance(edit, graphtage.EditDistance)
        self.assertEqual(4, ordered(1, 2, 3).diff(ordered(3, 2, 1)).edited_cost())

    def test_reorder_is_free(self):
        edit = unordered(1, 2, 3).edits(unordered(3, 2, 1))
        self.assertIsInstance(edit, graphtage.Match)
        self.assertEqual(0, edit.bounds().upper_bound)
        self.assertEqual(0, unordered(1, 2, 3).diff(unordered(3, 2, 1)).edited_cost())

    def test_reorder_against_an_ordered_list(self):
        """Only one side has to ignore order, which is what makes cross-format comparisons work."""
        self.assertIsInstance(unordered(1, 2, 3).edits(ordered(3, 2, 1)), graphtage.Match)
        self.assertEqual(0, unordered(1, 2, 3).edits(ordered(3, 2, 1)).bounds().upper_bound)

    def test_reorder_with_a_change(self):
        edit = unordered(1, 2, 3).edits(unordered(3, 9, 1))
        self.assertIsInstance(edit, graphtage.multiset.MultiSetEdit)
        self.assertEqual(1, edit.bounds().upper_bound)

    def test_duplicates_are_counted(self):
        self.assertIsInstance(unordered(1, 1, 2).edits(unordered(2, 1, 1)), graphtage.Match)
        self.assertEqual(0, unordered(1, 1, 2).edits(unordered(2, 1, 1)).bounds().upper_bound)
        self.assertGreater(unordered(1, 1, 2).diff(unordered(1, 2, 2)).edited_cost(), 0)

    def test_nested_lists(self):
        from_node = graphtage.UnorderedListNode([unordered(1, 2), unordered(3, 4)])
        to_node = graphtage.UnorderedListNode([unordered(4, 3), unordered(2, 1)])
        edit = from_node.edits(to_node)
        self.assertIsInstance(edit, graphtage.Match)
        self.assertEqual(0, edit.bounds().upper_bound)

    def test_versus_dict(self):
        """A list is never matched against a dictionary, whether or not its order is ignored."""
        dict_node = graphtage.DictNode.from_dict({graphtage.StringNode("a"): graphtage.IntegerNode(1)})
        self.assertIsInstance(unordered(1, 2, 3).edits(dict_node), graphtage.Replace)
        self.assertIsInstance(dict_node.edits(unordered(1, 2, 3)), graphtage.Replace)

    def test_to_obj_of_unhashable_children(self):
        """:meth:`graphtage.MultiSetNode.to_obj` counts its children, which fails for a list of dictionaries."""
        def children():
            return [
                graphtage.DictNode.from_dict({graphtage.StringNode("a"): graphtage.IntegerNode(1)}),
                graphtage.DictNode.from_dict({graphtage.StringNode("b"): graphtage.IntegerNode(2)}),
            ]

        self.assertEqual([{"a": 1}, {"b": 2}], graphtage.UnorderedListNode(children()).to_obj())
        with self.assertRaises(TypeError):
            graphtage.MultiSetNode(children()).to_obj()

    def test_large_shuffle_is_fast(self):
        """Equal multisets short-circuit to a match, so a big shuffle must not reach the bipartite matcher."""
        values = list(range(1000))
        shuffled = list(values)
        random.Random(56).shuffle(shuffled)
        with run_with_time_limit(seconds=30):
            edit = unordered(*values).edits(unordered(*shuffled))
        self.assertIsInstance(edit, graphtage.Match)
        self.assertEqual(0, edit.bounds().upper_bound)
