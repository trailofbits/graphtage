from io import StringIO
from unittest import TestCase

import graphtage
import graphtage.flamegraph
from graphtage.printer import Printer
from graphtage.utils import Tempfile

BASELINE = b"""main;work 100
main;work;parse 40
main;work;emit 30
"""

MORE_SAMPLES = b"""main;work 100
main;work;parse 55
main;work;emit 30
"""

FRAME_REMOVED = b"""main;work 100
main;parse 40
main;work;emit 30
"""

STACK_ADDED = b"""main;work 100
main;work;parse 40
main;work;emit 30
main;work;flush 7
"""


def build(content: bytes) -> graphtage.TreeNode:
    with Tempfile(content) as path:
        return graphtage.FILETYPES_BY_TYPENAME["flamegraph"].build_tree(path)


def render(node: graphtage.TreeNode) -> str:
    stream = StringIO()
    printer = Printer(out_stream=stream, ansi_color=False, quiet=True)
    graphtage.FILETYPES_BY_TYPENAME["flamegraph"].get_default_formatter().print(printer, node)
    printer.flush(final=True)
    return stream.getvalue()


def render_diff(from_content: bytes, to_content: bytes) -> str:
    return render(build(from_content).diff(build(to_content)))


class TestFlameGraphDiff(TestCase):
    """Covers the diff path, which ``@filetype_test`` cannot reach because it only round-trips unedited trees."""

    def test_sample_count_change_is_reported(self):
        """PR #50 hashed a stack trace on its frames alone, so a changed sample count produced an empty diff."""
        unchanged = render_diff(BASELINE, BASELINE)
        changed = render_diff(BASELINE, MORE_SAMPLES)
        self.assertNotEqual(unchanged, changed)
        self.assertIn("40 -> 55", changed)

    def test_identical_graphs_produce_no_edits(self):
        unchanged = render_diff(BASELINE, BASELINE)
        self.assertEqual(unchanged, render(build(BASELINE)))
        for marker in ("~~", "++", "->"):
            self.assertNotIn(marker, unchanged)

    def test_removed_frame_is_marked(self):
        removed = render_diff(BASELINE, FRAME_REMOVED)
        self.assertIn("~~", removed)
        self.assertIn("main;~~work~~;parse 40", removed)

    def test_added_stack_is_marked(self):
        added = render_diff(BASELINE, STACK_ADDED)
        self.assertIn("++", added)
        self.assertIn("flush", added)

    def test_removed_stack_is_marked(self):
        removed = render_diff(STACK_ADDED, BASELINE)
        self.assertIn("~~", removed)
        self.assertIn("flush", removed)


class TestFlameGraphParsing(TestCase):
    def test_frame_names_may_contain_spaces(self):
        """Only the final space separates the stack trace from its sample count."""
        content = b"main;std::vector<int, std::allocator<int> >::push_back 12\n"
        self.assertEqual(render(build(content)), content.decode("utf-8").rstrip("\n"))

    def test_blank_lines_are_ignored(self):
        self.assertEqual(build(b"\n\nmain 1\n\n"), build(b"main 1\n"))

    def test_to_obj_maps_folded_stacks_to_sample_counts(self):
        """The frames are a list, so keying the mapping on them directly raises TypeError."""
        self.assertEqual(
            {"main;work": 100, "main;work;parse": 40, "main;work;emit": 30},
            build(BASELINE).to_obj()
        )

    def test_a_line_without_a_sample_count_is_an_error(self):
        with self.assertRaises(graphtage.flamegraph.FlameGraphParseError):
            build(b"main;work\n")

    def test_a_non_integer_sample_count_is_an_error(self):
        with self.assertRaises(graphtage.flamegraph.FlameGraphParseError):
            build(b"main;work several\n")

    def test_a_negative_sample_count_is_an_error(self):
        with self.assertRaises(graphtage.flamegraph.FlameGraphParseError):
            build(b"main;work -1\n")

    def test_parse_errors_are_reported_by_the_filetype(self):
        filetype = graphtage.FILETYPES_BY_TYPENAME["flamegraph"]
        with Tempfile(b"main;work\n") as path:
            result = filetype.build_tree_handling_errors(path)
        self.assertIsInstance(result, str)
        self.assertIn("Error parsing", result)
