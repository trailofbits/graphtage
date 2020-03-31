from unittest import TestCase

import graphtage


class TestGraphtage(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.small_from = graphtage.build_tree({
            "test": "foo",
            "baz": 1
        })
        cls.small_to = graphtage.build_tree({
            "test": "bar",
            "baz": 2
        })
        cls.list_from = graphtage.build_tree([0, 1, 2, 3, 4, 5])
        cls.list_to = graphtage.build_tree([1, 2, 3, 4, 5])

    def test_small_diff(self):
        diff = graphtage.diff(self.small_from, self.small_to)
        has_test_match = False
        has_baz_match = False
        for edit in diff.edits:
            if edit.cost().upper_bound > 0:
                self.assertIsInstance(edit, graphtage.Match)
                if isinstance(edit.from_node, graphtage.StringNode):
                    self.assertIsInstance(edit.to_node, graphtage.StringNode)
                    self.assertEqual(edit.from_node.object, 'foo')
                    self.assertEqual(edit.to_node.object, 'bar')
                    self.assertEqual(edit.cost().upper_bound, 3)
                    self.assertFalse(has_test_match)
                    has_test_match = True
                elif isinstance(edit.from_node, graphtage.IntegerNode):
                    self.assertIsInstance(edit.to_node, graphtage.IntegerNode)
                    self.assertEqual(edit.from_node.object, 1)
                    self.assertEqual(edit.to_node.object, 2)
                    self.assertEqual(edit.cost().upper_bound, 1)
                    self.assertFalse(has_baz_match)
                    has_baz_match = True
                else:
                    self.fail()
        self.assertTrue(has_test_match)
        self.assertTrue(has_baz_match)

    def test_list_diff(self):
        diff = graphtage.diff(self.list_from, self.list_to)
        for edit in diff.edits:
            if edit.cost().upper_bound > 0:
                self.assertIsInstance(edit, graphtage.Remove)
                self.assertIsInstance(edit.from_node, graphtage.IntegerNode)
                self.assertEqual(edit.from_node.object, 0)
                self.assertIsInstance(edit.to_node, graphtage.ListNode)
                self.assertIs(edit.to_node, self.list_from)
            else:
                self.assertIsInstance(edit, graphtage.Match)
