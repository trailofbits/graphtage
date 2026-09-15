import pickle
from io import StringIO
from unittest import TestCase

import graphtage
from graphtage.printer import Printer
from graphtage.utils import Tempfile


def build(obj) -> graphtage.TreeNode:
    with Tempfile(pickle.dumps(obj)) as path:
        return graphtage.FILETYPES_BY_TYPENAME["pickle"].build_tree(path)


def render_diff(from_obj, to_obj) -> str:
    diff = build(from_obj).diff(build(to_obj))
    stream = StringIO()
    printer = Printer(out_stream=stream, ansi_color=False)
    graphtage.FILETYPES_BY_TYPENAME["pickle"].get_default_formatter().print(printer, diff)
    printer.flush(final=True)
    return stream.getvalue().strip()


class TestPickleDiff(TestCase):
    """Covers the pickle filetype, which is the shortest route from a file to a ``bytes`` leaf node."""

    def test_changed_bytes_value(self):
        """Diffing a changed ``bytes`` value used to raise ``TypeError: object of type 'int' has no len()``."""
        diff = build({"payload": b"hello"}).diff(build({"payload": b"hellp"}))
        self.assertEqual(1, diff.edited_cost())

    def test_changed_bytes_value_renders_the_edit(self):
        self.assertIn("hell~~o~~++p++", render_diff({"payload": b"hello"}, {"payload": b"hellp"}))

    def test_unchanged_bytes_value(self):
        self.assertEqual(0, build({"payload": b"hello"}).diff(build({"payload": b"hello"})).edited_cost())
