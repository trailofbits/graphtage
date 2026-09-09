from io import StringIO
from unittest import TestCase

import graphtage
from graphtage.printer import Printer
from graphtage.utils import Tempfile

TABLE = b"""[k]
a = 1
"""

EDITED_TABLE = b"""[k]
a = 2
"""

LIST_VALUE = b"""k = [2, 3]
"""

SCALAR_ONE = b"""k = 1
"""

SCALAR_TWO = b"""k = 2
"""

NESTED_TABLE = b"""[outer.k]
a = 1
"""

NESTED_LIST_VALUE = b"""[outer]
k = [2, 3]
"""


def build(content: bytes) -> graphtage.TreeNode:
    with Tempfile(content) as path:
        return graphtage.FILETYPES_BY_TYPENAME["toml"].build_tree(path)


def render(node: graphtage.TreeNode) -> str:
    stream = StringIO()
    printer = Printer(out_stream=stream, ansi_color=False)
    graphtage.FILETYPES_BY_TYPENAME["toml"].get_default_formatter().print(printer, node)
    printer.flush(final=True)
    return stream.getvalue()


def render_diff(from_content: bytes, to_content: bytes) -> str:
    return render(build(from_content).diff(build(to_content)))


def diff_lines(from_content: bytes, to_content: bytes) -> list[str]:
    """Returns the rendered diff as a list of lines, dropping the blank lines that separate tables."""
    return [line for line in render_diff(from_content, to_content).splitlines() if line.strip()]


class TestTOMLDiff(TestCase):
    """Covers the diff path, which ``@filetype_test`` cannot reach because it only round-trips unedited trees."""

    def test_table_replaced_by_a_list_is_marked(self):
        """A table replaced by a value of another type used to render byte-identically to no change at all."""
        self.assertNotEqual(diff_lines(TABLE, TABLE), diff_lines(TABLE, LIST_VALUE))
        self.assertEqual(["k = {a = 1} -> [2,3]"], diff_lines(TABLE, LIST_VALUE))

    def test_table_replacing_a_list_keeps_its_key(self):
        """The replacement used to be written as a headerless table body, orphaning ``a = 1`` from its ``k``."""
        self.assertEqual(["k = [2,3] -> {a = 1}"], diff_lines(LIST_VALUE, TABLE))

    def test_nested_table_replaced_by_a_list_is_marked(self):
        """A replaced table below the document root is written inline under the table that contains it."""
        self.assertEqual(["[outer]", "k = {a = 1} -> [2,3]"], diff_lines(NESTED_TABLE, NESTED_LIST_VALUE))
        self.assertEqual(["[outer]", "k = [2,3] -> {a = 1}"], diff_lines(NESTED_LIST_VALUE, NESTED_TABLE))

    def test_replaced_scalar_is_marked(self):
        """The control: a scalar replacement was never broken and must keep rendering as one line."""
        self.assertEqual(["k = 1 -> 2"], diff_lines(SCALAR_ONE, SCALAR_TWO))

    def test_table_that_is_not_replaced_keeps_its_header(self):
        """A table whose value stays a table is still written as a ``[table]`` section rather than inline."""
        self.assertEqual(["[k]", "a = 1 -> 2"], diff_lines(TABLE, EDITED_TABLE))

    def test_unchanged_document_has_no_edit_markers(self):
        unchanged = render_diff(TABLE, TABLE)
        self.assertNotIn("->", unchanged)
        self.assertNotIn("~~", unchanged)
        self.assertNotIn("++", unchanged)
        self.assertEqual(["[k]", "a = 1"], diff_lines(TABLE, TABLE))
