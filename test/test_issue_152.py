from io import StringIO
from typing import Any
from unittest import TestCase

import graphtage
from graphtage import json as graphtage_json
from graphtage.printer import Printer
from graphtage.pydiff import print_diff

REPLACEMENT_IN_A_LIST: tuple[Any, Any] = ([1, {"a": 1}], [1, [2, 3]])
"""A dict nested in a list, replaced by a list.

Rendering the replaced dict needs a formatter that the enclosing list formatter does not provide, so the list
formatter hands the node back to the format's root formatter. That hand-off is what printed the replacement twice.

"""

REPLACEMENT_IN_A_DICT: tuple[Any, Any] = ({"k": {"a": 1}}, {"k": [2, 3]})
"""The same replacement as a dict value, which never took the delegating path and always rendered correctly."""

DELEGATING_TYPENAMES = ("json", "json5", "yaml", "toml", "ini")
"""The filetypes whose list and dict formatters delegate to each other through ``self.parent.print(...)``."""

TYPENAMES_THAT_RENDER_A_REPLACED_DICT_VALUE = ("json", "json5", "yaml", "ini")
""":data:`DELEGATING_TYPENAMES` minus TOML, which writes a replaced table value as the original table and no edit."""


class TestIssue152(TestCase):
    """Reproduces https://github.com/trailofbits/graphtage/issues/152"""

    @staticmethod
    def render(from_obj: Any, to_obj: Any, typename: str) -> str:
        """Returns the diff of two Python objects as rendered by the default formatter for ``typename``."""
        from_tree = graphtage_json.build_tree(from_obj)
        to_tree = graphtage_json.build_tree(to_obj)
        formatter = graphtage.FILETYPES_BY_TYPENAME[typename].get_default_formatter()
        stream = StringIO()
        printer = Printer(out_stream=stream, ansi_color=False, quiet=True)
        with printer:
            formatter.print(printer, from_tree.diff(to_tree))
        return stream.getvalue()

    def test_json_prints_a_replacement_in_a_list_once(self):
        self.assertEqual(
            '[\n    1,\n    {\n        "a": 1\n    } -> [\n        2,\n        3\n    ]\n]',
            self.render(*REPLACEMENT_IN_A_LIST, "json"),
        )

    def test_pydiff_prints_a_replacement_in_a_list_once(self):
        """Renders the exact reproducer from the issue."""
        stream = StringIO()
        printer = Printer(out_stream=stream, ansi_color=False, quiet=True)
        with printer:
            print_diff(*REPLACEMENT_IN_A_LIST, printer=printer)
        self.assertEqual('[1,{"a": 1} -> [2,3]]', stream.getvalue())

    def test_delegating_formats_print_a_replacement_in_a_list_once(self):
        """Every format whose list formatter delegates to a sibling formatter shared the same defect."""
        for typename in DELEGATING_TYPENAMES:
            with self.subTest(typename=typename):
                rendered = self.render(*REPLACEMENT_IN_A_LIST, typename)
                self.assertEqual(1, rendered.count(" -> "), f"printed more than one replacement: {rendered!r}")

    def test_delegating_formats_print_a_replacement_in_a_dict_once(self):
        """The control case from the issue: a replaced dict value was always rendered correctly."""
        for typename in TYPENAMES_THAT_RENDER_A_REPLACED_DICT_VALUE:
            with self.subTest(typename=typename):
                rendered = self.render(*REPLACEMENT_IN_A_DICT, typename)
                self.assertEqual(1, rendered.count(" -> "), f"printed more than one replacement: {rendered!r}")
