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


class TestRenderedDiffs(TestCase):
    """Pins the rendered diff of documents whose output the string edit script decides.

    Which characters come out marked removed and inserted, and in which order, is decided by the candidate
    ordering in :meth:`graphtage.levenshtein.EditDistance._best_match` and by the shared-prefix strip in its
    constructor. ``test/test_levenshtein.py`` pins both of those edit by edit; these snapshots pin what a user
    sees, so a change to either shows up in a normal test run.

    """

    def test_dict_with_every_key_and_value_changed(self):
        """Pins the diff of a dict in which no key and no value survives unchanged."""
        self.assertEqual(
            '{\n    "source++s++": "/usr/l~~ocal~~++ib++",\n    "target++s++": "/opt/l~~ocal~~++ib++"\n}',
            render(
                b'{"source": "/usr/local", "target": "/opt/local"}',
                b'{"sources": "/usr/lib", "targets": "/opt/lib"}'
            )
        )

    def test_list_of_similar_strings(self):
        """Pins the diff of a list whose items are matched to each other by string edit distance."""
        self.assertEqual(
            '[\n    "re++a++d",\n    "gre~~e~~n",\n    "blue++s++"\n]',
            render(b'["red", "green", "blue"]', b'["read", "gren", "blues"]')
        )

    def test_nested_dict_with_a_changed_string_and_number(self):
        """Pins the diff of a string nested two levels deep alongside a replaced number."""
        self.assertEqual(
            '{\n    "server": {\n        "path": "/usr/l~~ocal~~++ib++/bin",\n        "retries": 3 -> 4\n    }\n}',
            render(
                b'{"server": {"path": "/usr/local/bin", "retries": 3}}',
                b'{"server": {"path": "/usr/lib/bin", "retries": 4}}'
            )
        )

    def test_nested_dict_with_an_appended_list_item(self):
        """Pins the indentation and the insertion marker of a list nested inside two dicts."""
        self.assertEqual(
            '{\n    "outer": {\n        "inner": {\n            "leaf": 1 -> 2\n        },\n'
            '        "sibling": [\n            1,\n            2,\n            ++3++\n        ]\n    }\n}',
            render(
                b'{"outer": {"inner": {"leaf": 1}, "sibling": [1, 2]}}',
                b'{"outer": {"inner": {"leaf": 2}, "sibling": [1, 2, 3]}}'
            )
        )
