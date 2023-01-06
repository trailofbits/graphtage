from unittest import TestCase

import graphtage
from graphtage.pydiff import build_tree, print_diff, PyDiffFormatter


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
