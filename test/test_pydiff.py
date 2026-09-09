import ast
import dataclasses
from io import StringIO
from unittest import TestCase

import graphtage
from graphtage.ast import Import, Subscript
from graphtage.printer import Printer
from graphtage.pydiff import PyAlias, PyDiffFormatter, PyObjAttribute, ast_to_tree, build_tree, print_diff

from .timing import run_with_time_limit


def format_tree(tree: graphtage.TreeNode) -> str:
    stream = StringIO()
    printer = Printer(out_stream=stream, ansi_color=False)
    with printer:
        PyDiffFormatter.DEFAULT_INSTANCE.print(printer, tree)
    return stream.getvalue()


def render_diff(from_source: str, to_source: str) -> str:
    """Diffs two Python sources and returns the rendering produced by :class:`PyDiffFormatter`."""
    from_tree = ast_to_tree(ast.parse(from_source))
    to_tree = ast_to_tree(ast.parse(to_source))
    stream = StringIO()
    printer = Printer(out_stream=stream, ansi_color=False)
    with printer:
        PyDiffFormatter.DEFAULT_INSTANCE.print(printer, from_tree.diff(to_tree))
    return stream.getvalue().strip()


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
        for source in (
            "import os",
            "import os.path",
            "import os, sys",
            "from os import path",
            "from os import path as p",
        ):
            with self.subTest(source=source):
                self.assertEqual(f"{source}\n", format_tree(ast_to_tree(ast.parse(source))))
    def test_attribute_receiver_is_not_quoted(self):
        stream = StringIO()
        attribute = PyObjAttribute(graphtage.StringNode("package"), graphtage.StringNode("member"))

        PyDiffFormatter.DEFAULT_INSTANCE.print(
            graphtage.Printer(out_stream=stream, ansi_color=False),
            attribute,
        )

        self.assertEqual("package.member", stream.getvalue())

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
    def test_dataclass_edit_preserves_syntax(self):
        """Reproduces https://github.com/trailofbits/graphtage/issues/150

        Without `DataClassEdit.print`, an edited `DataClassNode` falls back to `AbstractCompoundEdit.print`, which
        prints the slot edits back to back and drops the syntax the node formatter writes between them. The first
        case below rendered as `[x]foo[1,2,++3++]`.
        """
        self.assertEqual("x = foo(1, 2, ++3++)", render_diff("x = foo(1, 2)", "x = foo(1, 2, 3)"))
        self.assertEqual("x -> y = foo(1, 2)", render_diff("x = foo(1, 2)", "y = foo(1, 2)"))
        self.assertEqual("x = ~~foo~~++bar++(1, 2)", render_diff("x = foo(1, 2)", "x = bar(1, 2)"))

    def test_attribute_edit_preserves_separators(self):
        """An edited `PyObjAttribute` keeps the dots between its slots; it used to render as `[x]abc -> d`."""
        self.assertEqual("x = a.b.c -> d", render_diff("x = a.b.c", "x = a.b.d"))

    def test_infinite_loop(self):
        """Reproduces https://github.com/trailofbits/graphtage/issues/82"""

        @dataclasses.dataclass
        class Thing:
            foo: str

        with run_with_time_limit(60):
            _ = graphtage.pydiff.diff([Thing("ok")], [Thing("bad")])
