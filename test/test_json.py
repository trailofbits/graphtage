from io import StringIO
from unittest import TestCase

import graphtage
from graphtage.printer import Printer
from graphtage.utils import Tempfile

NESTED = b'{"foo": [1, 2, 3], "bar": "baz"}'
DEEP = b'{"x": [{"y": [1, 2]}, 3]}'


def build(content: bytes) -> graphtage.TreeNode:
    with Tempfile(content) as path:
        return graphtage.FILETYPES_BY_TYPENAME["json"].build_tree(path)


def render(from_content: bytes, to_content: bytes | None = None, *,
           join_lists: bool = False, join_dict_items: bool = False) -> str:
    """Renders the diff of two JSON documents with the given join options.

    Args:
        from_content: The JSON source of the "from" document.
        to_content: The JSON source of the "to" document, or :const:`None` to diff ``from_content`` against itself.
        join_lists: Whether to enable the ``--join-lists`` option.
        join_dict_items: Whether to enable the ``--join-dict-items`` option.

    Returns:
        str: The rendered diff.

    """
    if to_content is None:
        to_content = from_content
    stream = StringIO()
    printer = Printer(
        out_stream=stream,
        ansi_color=False,
        quiet=True,
        options={"join_lists": join_lists, "join_dict_items": join_dict_items},
    )
    diff = build(from_content).diff(build(to_content))
    graphtage.FILETYPES_BY_TYPENAME["json"].get_default_formatter().print(printer, diff)
    printer.flush(final=True)
    return stream.getvalue()


class TestJSONJoinOptions(TestCase):
    """Covers ``--join-lists`` and ``--join-dict-items``, which previously had no test coverage at all."""

    def test_default_output_is_pretty_printed(self):
        self.assertEqual(
            '{\n    "bar": "baz",\n    "foo": [\n        1,\n        2,\n        3\n    ]\n}',
            render(NESTED)
        )

    def test_join_lists_keeps_the_dict_pretty_printed(self):
        self.assertEqual('{\n    "bar": "baz",\n    "foo": [1, 2, 3]\n}', render(NESTED, join_lists=True))

    def test_join_dict_items_does_not_indent_a_nested_list(self):
        """A joined dict used to leave an unused indent on the printer, which the nested list then doubled.

        The list items came out eight spaces deep and the closing bracket four, as though the collapsed dict were
        still occupying a level of indentation.

        """
        self.assertEqual('{"bar": "baz", "foo": [\n    1,\n    2,\n    3\n]}', render(NESTED, join_dict_items=True))

    def test_condensed_joins_both_lists_and_dicts(self):
        self.assertEqual(
            '{"bar": "baz", "foo": [1, 2, 3]}',
            render(NESTED, join_lists=True, join_dict_items=True)
        )

    def test_join_dict_items_indents_by_the_depth_that_breaks_lines(self):
        """Only the sequences that actually emit newlines contribute a level of indentation."""
        self.assertEqual(
            '{"x": [\n    {"y": [\n        1,\n        2\n    ]},\n    3\n]}',
            render(DEEP, join_dict_items=True)
        )

    def test_join_lists_indents_by_the_depth_that_breaks_lines(self):
        self.assertEqual('{\n    "x": [{\n        "y": [1, 2]\n    }, 3]\n}', render(DEEP, join_lists=True))

    def test_joined_items_are_separated_by_a_space(self):
        """Joined output used to run the items together as ``[1,2,3]`` and ``{"bar": "baz","foo": …}``."""
        self.assertIn('[1, 2, 3]', render(NESTED, join_lists=True))
        self.assertIn('"baz", "foo"', render(NESTED, join_dict_items=True))

    def test_joined_output_separates_inserted_and_removed_items(self):
        self.assertEqual(
            '{"a": [1, ~~2~~, 3], ++"b": 4++}',
            render(b'{"a": [1, 2, 3]}', b'{"a": [1, 3], "b": 4}', join_lists=True, join_dict_items=True)
        )
