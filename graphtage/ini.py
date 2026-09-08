"""A :class:`graphtage.Filetype` for parsing, diffing, and rendering INI files.

The parser is implemented atop :mod:`configparser`. Option names are compared case-sensitively, which differs from
:mod:`configparser`'s default of lowercasing them, so that a diff reflects the file as it was written. Interpolation is
disabled for the same reason, and ``[DEFAULT]`` is treated as an ordinary section rather than supplying values to every
other section. Comments are not represented in the tree, so they are absent from the output.

When another format is rendered as INI, a top-level value that is not itself a mapping has no section to live under.
It is written without a section header so that the diff is still visible, which means the output is not necessarily a
file that :mod:`configparser` can read back.

"""

import configparser
import os

from . import json
from .graphtage import BuildOptions, Filetype, KeyValuePairNode, MappingNode, StringFormatter, StringNode
from .printer import Printer
from .sequences import SequenceFormatter
from .tree import GraphtageFormatter, TreeNode

DEFAULT_SECTION_SENTINEL = "\x00graphtage-no-default-section"
"""A ``default_section`` name that cannot occur in a real file.

:class:`configparser.ConfigParser` copies the options of its ``default_section`` into every other section. Pointing it
at a name no file can contain keeps ``[DEFAULT]`` an ordinary section, so the tree matches the file as written.

"""


def _parser() -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None, default_section=DEFAULT_SECTION_SENTINEL)
    parser.optionxform = str
    return parser


def build_tree(path: str, options: BuildOptions | None = None) -> TreeNode:
    parser = _parser()
    with open(path, encoding="utf-8") as f:
        parser.read_file(f)
    data = {section: dict(parser[section]) for section in parser.sections()}
    return json.build_tree(data, options)


def _unquote(node: TreeNode | None):
    """Clears a string node's quoting, since INI has no string delimiters."""
    if isinstance(node, StringNode):
        node.quoted = False
    elif isinstance(node, KeyValuePairNode):
        _unquote(node.key)
        _unquote(node.value)


class INIStringFormatter(StringFormatter):
    """A string formatter for INI keys and values."""
    is_partial = True

    def escape(self, c: str) -> str:
        # INI represents a multi-line value as indented continuation lines:
        return c.replace("\n", "\n\t")


class INIOptionFormatter(GraphtageFormatter):
    """A formatter for a single INI key/value pair, or for a section header and its body."""
    is_partial = True

    def print_KeyValuePairNode(self, printer: Printer, node: KeyValuePairNode):
        if isinstance(node.value, MappingNode):
            printer.write("[")
            self.print(printer, node.key)
            printer.write("]")
            printer.newline()
            self.parent.print(printer, node.value)
            printer.newline()
        else:
            self.print(printer, node.key)
            printer.write(" = ")
            self.print(printer, node.value)
            printer.newline()


class INIListFormatter(SequenceFormatter):
    """A formatter for sequences.

    INI has no syntax for a list, so one is written as a comma-separated value. This only arises when rendering
    another format as INI; reading the result back yields a single string rather than a list.

    """
    is_partial = True

    def __init__(self):
        super().__init__("", "", ",")

    def print_ListNode(self, *args, **kwargs):
        super().print_SequenceNode(*args, **kwargs)

    print_UnorderedListNode = print_ListNode

    def print_SequenceNode(self, *args, **kwargs):
        self.parent.print(*args, **kwargs)

    def item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        pass

    def items_indent(self, printer: Printer):
        return printer


class INIMappingFormatter(SequenceFormatter):
    """A formatter for an INI document and for each of its section bodies."""
    is_partial = True
    sub_format_types = [INIOptionFormatter]

    def __init__(self):
        super().__init__("", "", "")

    def print_MappingNode(self, *args, **kwargs):
        super().print_SequenceNode(*args, **kwargs)

    print_MultiSetNode = print_MappingNode

    def print_SequenceNode(self, *args, **kwargs):
        self.parent.print(*args, **kwargs)

    def item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        # INI's line structure is written by INIOptionFormatter, which knows whether it is emitting a section
        # header or an option; a delimiter here would double the newlines.
        pass

    def items_indent(self, printer: Printer):
        return printer


class INIFormatter(GraphtageFormatter):
    """A formatter for INI files."""
    sub_format_types = [INIMappingFormatter, INIListFormatter, INIStringFormatter]

    def print(self, printer: Printer, *args, **kwargs):
        if args and isinstance(args[0], TreeNode) and args[0].parent is None:
            # INI has no string delimiters. Sweeping the whole tree covers both sides of every edit, and also
            # covers trees built by another filetype when rendering it as INI.
            for node in args[0].dfs():
                _unquote(node)
                _unquote(getattr(node, "matched_to", None))
                for inserted in getattr(node, "inserted", ()):
                    for descendant in inserted.dfs():
                        _unquote(descendant)
        super().print(printer, *args, **kwargs)


class INI(Filetype):
    """The INI filetype."""

    def __init__(self):
        """Initializes the INI filetype."""
        super().__init__(
            "ini",
            "text/ini",
            "application/ini",
            "text/x-ini",
        )

    def build_tree(self, path: str, options: BuildOptions | None = None) -> TreeNode:
        """Equivalent to :func:`build_tree`"""
        return build_tree(path, options=options)

    def build_tree_handling_errors(self, path: str, options: BuildOptions | None = None) -> str | TreeNode:
        try:
            return self.build_tree(path=path, options=options)
        except (configparser.Error, OSError, ValueError) as e:
            return f"Error parsing {os.path.basename(path)}: {e}"

    def get_default_formatter(self) -> INIFormatter:
        return INIFormatter.DEFAULT_INSTANCE
