from typing import List
from unittest import TestCase

from graphtage import IntegerNode, ListNode, StringNode, TreeNode
from graphtage.builder import BasicBuilder, Builder


class TestVisitor(TestCase):
    def test_visitor(self):
        class Tester(BasicBuilder):
            pass

        result = Tester().build_tree([1, "a", (2, "b"), {1, 2}, {"a": "b"}, None])
        self.assertIsInstance(result, ListNode)
        self.assertEqual(6, len(result.children()))
