from io import StringIO
from unittest import TestCase

import graphtage
import graphtage.ini
from graphtage.printer import Printer
from graphtage.utils import Tempfile

ONE_SECTION = b"""[alpha]
x = 1
"""

TWO_SECTIONS = b"""[alpha]
x = 1

[beta]
y = 2
"""

TWO_OPTIONS = b"""[alpha]
x = 1
z = 3
"""

WITH_DEFAULTS = b"""[DEFAULT]
shared = 1

[alpha]
x = 2
"""


def build(content: bytes) -> graphtage.TreeNode:
    with Tempfile(content) as path:
        return graphtage.FILETYPES_BY_TYPENAME["ini"].build_tree(path)


def render(node: graphtage.TreeNode) -> str:
    stream = StringIO()
    printer = Printer(out_stream=stream, ansi_color=False)
    graphtage.FILETYPES_BY_TYPENAME["ini"].get_default_formatter().print(printer, node)
    printer.flush(final=True)
    return stream.getvalue()


def render_diff(from_content: bytes, to_content: bytes) -> str:
    return render(build(from_content).diff(build(to_content)))


class TestINIDiff(TestCase):
    """Covers the diff path, which ``@filetype_test`` cannot reach because it only round-trips unedited trees."""

    def test_removed_section_is_marked(self):
        """A removed section used to render byte-identically to no change at all."""
        unchanged = render_diff(TWO_SECTIONS, TWO_SECTIONS)
        removed = render_diff(TWO_SECTIONS, ONE_SECTION)
        self.assertNotEqual(unchanged, removed)
        self.assertIn("~~", removed)
        self.assertIn("beta", removed)

    def test_added_section_is_visible(self):
        added = render_diff(ONE_SECTION, TWO_SECTIONS)
        self.assertIn("++", added)
        self.assertIn("beta", added)

    def test_added_option_is_visible(self):
        added = render_diff(ONE_SECTION, TWO_OPTIONS)
        self.assertIn("++", added)
        self.assertIn("z", added)

    def test_unchanged_document_has_no_edit_markers(self):
        unchanged = render_diff(TWO_SECTIONS, TWO_SECTIONS)
        self.assertNotIn("~~", unchanged)
        self.assertNotIn("++", unchanged)
        self.assertEqual(0, build(TWO_SECTIONS).diff(build(TWO_SECTIONS)).edited_cost())

    def test_default_section_is_not_copied_into_other_sections(self):
        """``configparser`` serves DEFAULT options from every section; the tree must match the file as written."""
        self.assertEqual({"DEFAULT": {"shared": "1"}, "alpha": {"x": "2"}}, build(WITH_DEFAULTS).to_obj())

    def test_replaced_value_is_unquoted_on_both_sides(self):
        """INI has no string delimiters, so neither side of a replacement may be quoted."""
        self.assertNotIn('"', render_diff(ONE_SECTION, b"[alpha]\nx = 2\n"))
