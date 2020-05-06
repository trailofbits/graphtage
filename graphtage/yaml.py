import os
import sys

from yaml import load, YAMLError
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from . import json
from .formatter import Formatter
from .graphtage import Filetype, KeyValuePairNode, LeafNode, MappingNode, StringNode
from .printer import Printer
from .sequences import SequenceFormatter, SequenceNode
from .tree import Edit, TreeNode


def build_tree(path: str, allow_key_edits=True, *args, **kwargs) -> TreeNode:
    with open(path, 'rb') as stream:
        data = load(stream, Loader=Loader)
        return json.build_tree(data, allow_key_edits=allow_key_edits, *args, **kwargs)


class YAMLListFormatter(SequenceFormatter):
    is_partial = True

    def __init__(self):
        super().__init__('', '', '')

    def print_SequenceNode(self, printer: Printer, node):
        self.parent.print(printer, node)

    def print_ListNode(self, printer: Printer, *args, **kwargs):
        printer.newline()
        super().print_SequenceNode(printer, *args, **kwargs)

    def edit_print(self, printer: Printer, edit: Edit):
        printer.indents += 1
        self.print(printer, edit)
        printer.indents -= 1

    def item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        if not is_last:
            if not is_first:
                printer.newline()
            printer.write('- ')

    def items_indent(self, printer: Printer):
        return printer


class YAMLKeyValuePairFormatter(Formatter):
    is_partial = True

    def print_KeyValuePairNode(self, printer: Printer, node: KeyValuePairNode):
        self.print(printer, node.key)
        with printer.bright():
            printer.write(": ")
        if isinstance(node.value, MappingNode):
            printer.newline()
            printer.indents += 1
            self.parent.print(printer, node.value)
            printer.indents -= 1
        elif isinstance(node.value, SequenceNode):
            self.parent.parent.print(printer, node.value)
        else:
            self.print(printer, node.value)


class YAMLDictFormatter(SequenceFormatter):
    is_partial = True
    sub_format_types = [YAMLKeyValuePairFormatter]

    def __init__(self):
        super().__init__('', '', '')

    def print_MultiSetNode(self, *args, **kwargs):
        super().print_SequenceNode(*args, **kwargs)

    def print_MappingNode(self, *args, **kwargs):
        super().print_SequenceNode(*args, **kwargs)

    def print_SequenceNode(self, *args, **kwargs):
        self.parent.print(*args, **kwargs)

    def item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        if not is_first and not is_last:
            printer.newline()

    def items_indent(self, printer: Printer):
        return printer


class YAMLFormatter(Formatter):
    sub_format_types = [YAMLDictFormatter, YAMLListFormatter]

    def print_LeafNode(self, printer: Printer, node: LeafNode):
        node.print(printer)

    def print_StringNode(self, printer: Printer, node: StringNode):
        node.quoted = False
        if (node.edited and node.edit is not None) or '\n' not in node.object:
            node.print(printer)
        else:
            printer.write('|')
            with printer.indent():
                for line in node.object.split('\n'):
                    printer.newline()
                    printer.write(line)


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
