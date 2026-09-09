import ast
import dataclasses
from io import StringIO
from unittest import TestCase

import graphtage
from graphtage.ast import Subscript
from graphtage.printer import Printer
from graphtage.pydiff import PyDiffFormatter, ast_to_tree, build_tree, print_diff

from .timing import run_with_time_limit


class TestPyDiff(TestCase):
    def test_build_tree(self):
        self.assertIsInstance(build_tree([1, 2, 3, 4]), graphtage.ListNode)
        self.assertIsInstance(build_tree({1: 2, 'a': 'b'}), graphtage.DictNode)

    def test_python_list_literals_stay_ordered(self):
        """Elements of a Python list literal are positional, so `ignore_list_order` must not apply to them."""
        options = graphtage.BuildOptions(ignore_list_order=True)
        tree = ast_to_tree(ast.parse("x = [1, 2, 3]"), options)
        lists = [n for n in tree.dfs() if isinstance(n, graphtage.ListNode)]
        self.assertTrue(lists)
        for node in tree.dfs():
            self.assertNotIsInstance(node, graphtage.UnorderedListNode)

    def test_diff(self):
        t1 = [1, 2, {3: "three"}, 4]
        t2 = [1, 2, {3: 3}, "four"]
        printer = graphtage.printer.Printer(ansi_color=True)
        print_diff(t1, t2, printer=printer)

    def test_custom_class(self):
        class Foo:
            def __init__(self, bar, baz):
                self.bar = bar
                self.baz = baz

        printer = graphtage.printer.Printer(ansi_color=True)
        print_diff(Foo("bar", "baz"), Foo("bar", "bak"), printer=printer)

    def test_nested_tuple_diff(self):
        tree = build_tree({"a": (1, 2)})
        self.assertIsInstance(tree, graphtage.DictNode)
        children = tree.children()
        self.assertEqual(1, len(children))
        kvp = children[0]
        self.assertIsInstance(kvp, graphtage.KeyValuePairNode)
        self.assertIsInstance(kvp.key, graphtage.StringNode)
        self.assertIsInstance(kvp.value, graphtage.ListNode)

    def _only_subscript(self, source: str) -> Subscript:
        subscripts = [node for node in ast_to_tree(ast.parse(source)).dfs() if isinstance(node, Subscript)]
        self.assertEqual(1, len(subscripts))
        return subscripts[0]

    def test_subscript_node_print(self):
        """Reproduces the ``TreeNode.write`` half of https://github.com/trailofbits/graphtage/issues/154

        ``Subscript.print`` is the fallback that :meth:`graphtage.tree.GraphtageFormatter.print` uses when no
        formatter resolves the node type, so this calls it directly rather than through a formatter.

        """
        stream = StringIO()
        self._only_subscript("a[1]").print(Printer(out_stream=stream, ansi_color=False))
        self.assertEqual("a[1]", stream.getvalue())

    def test_subscript_formatter_print(self):
        """Reproduces the unbalanced bracket half of https://github.com/trailofbits/graphtage/issues/154"""
        stream = StringIO()
        node = self._only_subscript("a[1]")
        PyDiffFormatter.DEFAULT_INSTANCE.print(Printer(out_stream=stream, ansi_color=False), node)
        self.assertEqual("a[1]", stream.getvalue())

    def test_infinite_loop(self):
        """Reproduces https://github.com/trailofbits/graphtage/issues/82"""

        @dataclasses.dataclass
        class Thing:
            foo: str

        with run_with_time_limit(60):
            _ = graphtage.pydiff.diff([Thing("ok")], [Thing("bad")])
