"""A :class:`graphtage.Filetype` for parsing, diffing, and rendering INI files."""

import configparser
import os
from typing import ClassVar, Optional, Type, Union

from . import json
from .graphtage import BuildOptions, Filetype, KeyValuePairNode, MappingNode, StringFormatter, StringNode
from .printer import Printer
from .tree import GraphtageFormatter, TreeNode


def _parser() -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    return parser


def build_tree(path: str, options: Optional[BuildOptions]) -> TreeNode:
    parser = _parser()
    with open(path) as f:
        parser.read_file(f)

    data = {}
    if parser.defaults():
        data[parser.default_section] = dict(parser.defaults())
    for section in parser.sections():
        data[section] = dict(parser[section])
    return json.build_tree(data, options)


class INIStringFormatter(StringFormatter):
    """A string formatter for INI keys and values."""
    is_partial = True

    def escape(self, c: str) -> str:
        return c.replace("\n", "\\n")


class INIFormatter(GraphtageFormatter):
    """A formatter for INI files."""
    sub_format_types: ClassVar[list[Type[GraphtageFormatter]]] = [INIStringFormatter]

    def print_KeyValuePairNode(self, printer: Printer, node: KeyValuePairNode):
        if isinstance(node.key, StringNode):
            node.key.quoted = False
        self.print(printer, node.key)
        printer.write(" = ")
        if isinstance(node.value, StringNode):
            node.value.quoted = False
        self.print(printer, node.value)
        printer.newline()

    def print_MappingNode(self, printer: Printer, node: MappingNode):
        for section in node:
            if not isinstance(section.value, MappingNode):
                continue
            printer.write("[")
            if isinstance(section.key, StringNode):
                section.key.quoted = False
            self.print(printer, section.key)
            printer.write("]")
            printer.newline()
            for item in section.value:
                self.print(printer, item)
            printer.newline()


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

    def build_tree(self, path: str, options: Optional[BuildOptions] = None) -> TreeNode:
        """Equivalent to :func:`build_tree`"""
        return build_tree(path, options=options)

    def build_tree_handling_errors(self, path: str, options: Optional[BuildOptions] = None) -> Union[str, TreeNode]:
        try:
            return self.build_tree(path=path, options=options)
        except (configparser.Error, OSError) as e:
            return f"Error parsing {os.path.basename(path)}: {e}"

    def get_default_formatter(self) -> INIFormatter:
        return INIFormatter.DEFAULT_INSTANCE
