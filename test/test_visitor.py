from typing import List
from unittest import TestCase

from graphtage import IntegerNode, ListNode, StringNode, TreeNode
from graphtage.visitor import AbstractVisitor, Visitor


class TestVisitor(TestCase):
    def test_visitor(self):
        class Tester(AbstractVisitor):
            pass

        result = Tester().build_tree([1, 2, "a"])
        self.assertIsInstance(result, ListNode)
        self.assertEqual(3, len(result.children()))
