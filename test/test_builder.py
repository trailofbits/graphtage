from typing import List
from unittest import TestCase

from graphtage import IntegerNode, ListNode, TreeNode
from graphtage.builder import BasicBuilder, Builder


class TestBuilder(TestCase):
    def test_basic_builder(self):
        result = BasicBuilder().build_tree([1, "a", (2, "b"), {1, 2}, {"a": "b"}, None])
        self.assertIsInstance(result, ListNode)
        self.assertEqual(6, len(result.children()))

    def test_custom_builder(self):
        test = self

        class Foo:
            def __init__(self, bar):
                self.bar = bar

        class Tester(BasicBuilder):
            @Builder.expander(Foo)
            def expand_foo(self, obj: Foo):
                yield obj.bar

            @Builder.builder(Foo)
            def build_foo(self, obj: Foo, children: List[TreeNode]):
                test.assertEqual(1, len(children))
                return children[0]

        tree = Tester().build_tree(Foo(10))
        self.assertIsInstance(tree, IntegerNode)
        self.assertEqual(10, tree.object)
