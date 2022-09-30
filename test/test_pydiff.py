from io import StringIO
from unittest import TestCase

import graphtage
from graphtage.pydiff import build_tree, PyDiffFormatter

from graphtage.printer import Printer


class TestPyDiff(TestCase):
    def test_build_tree(self):
        self.assertIsInstance(build_tree([1, 2, 3, 4]), graphtage.ListNode)
        self.assertIsInstance(build_tree({1: 2, 'a': 'b'}), graphtage.DictNode)

    def test_diff(self):
        diff = build_tree([1, 2, 3, 4]).diff(build_tree([1, 2, 3, '4']))
        printer = graphtage.printer.Printer(ansi_color=True)
        diff.print(printer)

    def test_custom_class(self):
        class Foo:
            def __init__(self, bar, baz):
                self.bar = bar
                self.baz = baz

        diff = build_tree(Foo("bar", "baz")).diff(build_tree(Foo("bar", "bak")))
        printer = graphtage.printer.Printer(ansi_color=True)
        diff.print(printer)
