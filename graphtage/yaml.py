"""A :class:`graphtage.Filetype` for parsing, diffing, and rendering YAML files."""
import os
from io import StringIO
from typing import Optional, Union

from yaml import dump, load_all, YAMLError
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from . import json
from .edits import Insert, Match
from .graphtage import BuildOptions, Filetype, KeyValuePairNode, LeafNode, ListNode, MappingNode, StringNode, \
    StringEdit, StringFormatter
from .printer import Fore, Printer
from .sequences import SequenceFormatter, SequenceNode
from .tree import ContainerNode, Edit, GraphtageFormatter, TreeNode


def build_tree(path: str, options: Optional[BuildOptions] = None, *args, **kwargs) -> TreeNode:
    """Constructs a YAML tree from an YAML file."""
    with open(path, 'rb') as stream:
        document_stream = load_all(stream, Loader=Loader)
        documents = list(document_stream)
        if len(documents) == 0:
            return json.build_tree(None, options=options, *args, **kwargs)
        elif len(documents) > 1:
            return json.build_tree(documents, options=options, *args, **kwargs)
        else:
            singleton = documents[0]
            return json.build_tree(singleton, options=options, *args, **kwargs)


class YAMLListFormatter(SequenceFormatter):
    is_partial = True

    def __init__(self):
        super().__init__('', '', '')

    def print_SequenceNode(self, printer: Printer, node: SequenceNode):
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
            with printer.bright().color(Fore.WHITE):
                printer.write('- ')

    def items_indent(self, printer: Printer):
        return printer


class YAMLKeyValuePairFormatter(GraphtageFormatter):
    is_partial = True

    def print_KeyValuePairNode(self, printer: Printer, node: KeyValuePairNode):
        if printer.context().fore is None:
            with printer.color(Fore.BLUE) as p:
                self.print(p, node.key)
        else:
            self.print(printer, node.key)
        with printer.bright().color(Fore.CYAN):
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


class YAMLStringFormatter(StringFormatter):
    is_partial = True
    has_newline = False

    def write_start_quote(self, printer: Printer, edit: StringEdit):
        for sub_edit in edit.edit_distance.edits():
            if isinstance(sub_edit, Match) and '\n' in sub_edit.from_node.object:
                self.has_newline = True
                break
            elif isinstance(sub_edit, Insert) and '\n' in sub_edit.from_node.object:
                self.has_newline = True
                break
        else:
            self.has_newline = False
        if self.has_newline:
            printer.write('|')
            printer.indents += 1
            printer.newline()

    def context(self, printer: Printer):
        return printer

    def write_end_quote(self, printer: Printer, edit: StringEdit):
        if self.has_newline:
            printer.indents -= 1

    def print_StringNode(self, printer: Printer, node: 'StringNode'):
        s = node.object
        if '\n' in s:
            if printer.context().fore is None:
                context = printer.color(Fore.CYAN)
            else:
                context = printer
            with context as c:
                c.write('|')
                with c.indent():
                    lines = s.split('\n')
                    if lines[-1] == '':
                        # Remove trailing newline
                        lines = lines[:-1]
                    for line in lines:
                        c.newline()
                        self.parent.write_obj(c, line)
        else:
            self.parent.write_obj(printer, s)

    def write_char(self, printer: Printer, c: str, index: int, num_edits: int, removed=False, inserted=False):
        if c == '\n':
            if removed or inserted:
                super().write_char(printer, '\u23CE', index, num_edits, removed, inserted)
            if not removed and index < num_edits - 1:
                # Do not print a trailing newline
                printer.newline()
        else:
            super().write_char(printer, c, index, num_edits, removed, inserted)


class YAMLFormatter(GraphtageFormatter):
    sub_format_types = [YAMLStringFormatter, YAMLDictFormatter, YAMLListFormatter]

    def print(self, printer: Printer, *args, **kwargs):
        # YAML only gets a two-space indent
        printer.indent_str = '  '
        super().print(printer, *args, **kwargs)

    @staticmethod
    def write_obj(printer: Printer, obj):
        if obj == '':
            return
        s = StringIO()
        dump(obj, stream=s, Dumper=Dumper)
        ret = s.getvalue()
        if isinstance(obj, str) and obj.strip().startswith('#'):
            if ret.startswith("'"):
                ret = ret[1:]
            if ret.endswith("\n"):
                ret = ret[:-1]
            if ret.endswith("'"):
                ret = ret[:-1]
        if ret.endswith('\n...\n'):
            ret = ret[:-len('\n...\n')]
        elif ret.endswith('\n'):
            ret = ret[:-1]
        printer.write(ret)

    def print_LeafNode(self, printer: Printer, node: LeafNode):
        self.write_obj(printer, node.object)

    def print_ContainerNode(self, printer: Printer, node: ContainerNode):
        """Prints a :class:`graphtage.ContainerNode`.

        This is a fallback to permit the printing of custom containers, like :class:`graphtage.xml.XMLElement`.

        """
        # Treat the container like a list
        list_node = ListNode(node.children())
        self.print(printer, list_node)


class YAML(Filetype):
    """The YAML filetype."""
    def __init__(self):
        """Initializes the YAML file type.

        By default, YAML associates itself with the "yaml", "application/x-yaml", "application/yaml", "text/yaml",
        "text/x-yaml", and "text/vnd.yaml" MIME types.

        """
        super().__init__(
            'yaml',
            'application/x-yaml',
            'application/yaml',
            'text/yaml',
            'text/x-yaml',
            'text/vnd.yaml'
        )

    def build_tree(self, path: str, options: Optional[BuildOptions] = None) -> TreeNode:
        tree = build_tree(path=path, options=options)
        for node in tree.dfs():
            if isinstance(node, StringNode):
                node.quoted = False
        return tree

    def build_tree_handling_errors(self, path: str, options: Optional[BuildOptions] = None) -> Union[str, TreeNode]:
        try:
            return self.build_tree(path=path, options=options)
        except YAMLError as ye:
            return f'Error parsing {os.path.basename(path)}: {ye})'

    def get_default_formatter(self) -> YAMLFormatter:
        return YAMLFormatter.DEFAULT_INSTANCE
