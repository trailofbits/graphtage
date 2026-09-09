import ast
import dataclasses
from io import StringIO
from unittest import TestCase

import graphtage
from graphtage.ast import Import
from graphtage.pydiff import PyAlias, PyDiffFormatter, ast_to_tree, build_tree, print_diff

from .timing import run_with_time_limit


def format_tree(tree: graphtage.TreeNode) -> str:
    stream = StringIO()
    printer = graphtage.printer.Printer(out_stream=stream, ansi_color=False)
    with printer:
        PyDiffFormatter.DEFAULT_INSTANCE.print(printer, tree)
    return stream.getvalue()


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

    def test_plain_import_builds(self):
        """Reproduces https://github.com/trailofbits/graphtage/issues/151.

        `ASTBuilder` had no builder for `ast.Import`, so every module containing a plain `import` statement raised
        `NotImplementedError`.

        """
        expected = {
            "import os": [("os", "")],
            "import os.path": [("os.path", "")],
            "import os as o": [("os", "o")],
            "import os, sys": [("os", ""), ("sys", "")],
        }
        for source, aliases in expected.items():
            with self.subTest(source=source):
                imports = [node for node in ast_to_tree(ast.parse(source)).dfs() if isinstance(node, Import)]
                self.assertEqual(1, len(imports))
                self.assertEqual("", imports[0].from_name.object)
                names = imports[0].names.children()
                for name, (expected_name, expected_as_name) in zip(names, aliases, strict=True):
                    self.assertIsInstance(name, PyAlias)
                    self.assertEqual(expected_name, name.name.object)
                    self.assertEqual(expected_as_name, name.as_name.object)

    def test_plain_import_printing(self):
        """A plain `import` must not render the `from` clause that `from x import y` gets."""
        for source in ("import os", "import os.path", "import os, sys", "from os import path"):
            with self.subTest(source=source):
                self.assertEqual(f"{source}\n", format_tree(ast_to_tree(ast.parse(source))))

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
