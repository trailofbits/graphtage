import dataclasses
from unittest import TestCase

import graphtage
from graphtage.pydiff import build_tree, print_diff, PyDiffFormatter

from .timing import run_with_time_limit


class TestPyDiff(TestCase):
    def test_build_tree(self):
        self.assertIsInstance(build_tree([1, 2, 3, 4]), graphtage.ListNode)
        self.assertIsInstance(build_tree({1: 2, 'a': 'b'}), graphtage.DictNode)

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

    def test_infinite_loop(self):
        """Reproduces https://github.com/trailofbits/graphtage/issues/82"""

        @dataclasses.dataclass
        class Thing:
            foo: str

        with run_with_time_limit(60):
            _ = graphtage.pydiff.diff([Thing("ok")], [Thing("bad")])
