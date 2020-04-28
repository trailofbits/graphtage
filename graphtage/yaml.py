import os
import sys

from yaml import load, YAMLError
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from . import json
from .formatter import Formatter
from .graphtage import Filetype, KeyValuePairNode, LeafNode
from .printer import Printer
from .sequences import SequenceFormatter
from .tree import TreeNode


def build_tree(path: str, allow_key_edits=True, *args, **kwargs) -> TreeNode:
    with open(path, 'rb') as stream:
        data = load(stream, Loader=Loader)
        return json.build_tree(data, allow_key_edits=allow_key_edits, *args, **kwargs)


class YAMLListFormatter(SequenceFormatter):
    is_partial = True

    def __init__(self):
        super().__init__('', '', '', lambda p: p.newline())

    def print_ListNode(self, *args, **kwargs):
        self.print_SequenceNode(*args, **kwargs)

    def item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        printer.newline()
        printer.write('- ')

    def items_indent(self, printer: Printer):
        return printer


class YAMLDictFormatter(SequenceFormatter):
    is_partial = True

    def __init__(self):
        super().__init__('', '', '', lambda p: p.newline())

    def print_MultiSetNode(self, *args, **kwargs):
        self.print_SequenceNode(*args, **kwargs)

    def print_FixedKeyDictNode(self, *args, **kwargs):
        self.print_SequenceNode(*args, **kwargs)

    def item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        printer.newline()


class YAMLFormatter(Formatter):
    sub_format_types = [YAMLDictFormatter, YAMLListFormatter]

    def print_LeafNode(self, printer: Printer, node: LeafNode):
        node.print(printer)

    def print_KeyValuePairNode(self, printer: Printer, node: KeyValuePairNode):
        self.print(printer, node.key)
        with printer.bright():
            printer.write(": ")
        self.print(printer, node.value)


class YAML(Filetype):
    def __init__(self):
        super().__init__(
            'yaml',
            'application/x-yaml',
            'application/yaml',
            'text/yaml',
            'text/x-yaml',
            'text/vnd.yaml'
        )

    def build_tree(self, path: str, allow_key_edits: bool = True) -> TreeNode:
        return build_tree(path=path, allow_key_edits=allow_key_edits)

    def build_tree_handling_errors(self, path: str, allow_key_edits: bool = True) -> TreeNode:
        try:
            return self.build_tree(path=path, allow_key_edits=allow_key_edits)
        except YAMLError as ye:
            sys.stderr.write(f'Error parsing {os.path.basename(path)}: {ye})\n\n')
            sys.exit(1)

    def get_default_formatter(self) -> YAMLFormatter:
        return YAMLFormatter.DEFAULT_INSTANCE
