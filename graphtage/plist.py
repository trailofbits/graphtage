"""A :class:`graphtage.Filetype` for parsing, diffing, and rendering Apple plist files."""
import os
from xml.parsers.expat import ExpatError
from typing import Optional, Tuple, Union

from plistlib import dumps, load

from . import json
from .edits import Edit, EditCollection, Match
from .graphtage import BoolNode, BuildOptions, Filetype, FloatNode, KeyValuePairNode, IntegerNode, LeafNode, StringNode
from .printer import Printer
from .sequences import SequenceFormatter, SequenceNode
from .tree import ContainerNode, GraphtageFormatter, TreeNode


class PLISTNode(ContainerNode):
    def __init__(self, root: TreeNode):
        self.root: TreeNode = root

    def to_obj(self):
        return self.root.to_obj()

    def edits(self, node: 'TreeNode') -> Edit:
        if isinstance(node, PLISTNode):
            return EditCollection(
                from_node=self,
                to_node=node,
                edits=iter((
                    Match(self, node, 0),
                    self.root.edits(node.root)
                )),
                collection=list,
                add_to_collection=list.append,
                explode_edits=False
            )
        return self.root.edits(node)

    def calculate_total_size(self) -> int:
        return self.root.calculate_total_size()

    def print(self, printer: Printer):
        printer.write(PLIST_HEADER)
        self.root.print(printer)
        printer.write(PLIST_FOOTER)

    def __iter__(self):
        yield self.root

    def __len__(self) -> int:
        return 1


def build_tree(path: str, options: Optional[BuildOptions] = None, *args, **kwargs) -> PLISTNode:
    """Constructs a PLIST tree from an PLIST file."""
    with open(path, "rb") as stream:
        data = load(stream)
        return PLISTNode(json.build_tree(data, options=options, *args, **kwargs))


class PLISTSequenceFormatter(SequenceFormatter):
    is_partial = True

    def __init__(self):
        super().__init__('', '', '')

    def print_SequenceNode(self, printer: Printer, node: SequenceNode):
        self.parent.print(printer, node)

    def print_ListNode(self, printer: Printer, *args, **kwargs):
        printer.write("<array>")
        super().print_SequenceNode(printer, *args, **kwargs)
        printer.write("</array>")

    def print_MultiSetNode(self, printer: Printer, *args, **kwargs):
        printer.write("<dict>")
        super().print_SequenceNode(printer, *args, **kwargs)
        printer.write("</dict>")

    def print_KeyValuePairNode(self, printer: Printer, node: KeyValuePairNode):
        printer.write("<key>")
        if isinstance(node.key, StringNode):
            printer.write(node.key.object)
        else:
            self.print(printer, node.key)
        printer.write("</key>")
        printer.newline()
        self.print(printer, node.value)

    print_MappingNode = print_MultiSetNode


def _plist_header_footer() -> Tuple[str, str]:
    string = "1234567890"
    encoded = dumps(string).decode("utf-8")
    expected = f"<string>{string}</string>"
    body_offset = encoded.find(expected)
    if body_offset <= 0:
        raise ValueError("Unexpected plist encoding!")
    return encoded[:body_offset], encoded[body_offset+len(expected):]


PLIST_HEADER: str
PLIST_FOOTER: str
PLIST_HEADER, PLIST_FOOTER = _plist_header_footer()


class PLISTFormatter(GraphtageFormatter):
    sub_format_types = [PLISTSequenceFormatter]

    def print(self, printer: Printer, *args, **kwargs):
        # PLIST uses an eight-space indent
        printer.indent_str = " " * 8
        super().print(printer, *args, **kwargs)

    @staticmethod
    def write_obj(printer: Printer, obj):
        encoded = dumps(obj).decode("utf-8")
        printer.write(encoded[len(PLIST_HEADER):-len(PLIST_FOOTER)])

    def print_StringNode(self, printer: Printer, node: StringNode):
        printer.write(f"<string>{node.object}</string>")

    def print_IntegerNode(self, printer: Printer, node: IntegerNode):
        printer.write(f"<integer>{node.object}</integer>")

    def print_FloatNode(self, printer: Printer, node: FloatNode):
        printer.write(f"<real>{node.object}</real>")

    def print_BoolNode(self, printer, node: BoolNode):
        if node.object:
            printer.write("<true />")
        else:
            printer.write("<false />")

    def print_LeafNode(self, printer: Printer, node: LeafNode):
        self.write_obj(printer, node.object)

    def print_PLISTNode(self, printer: Printer, node: PLISTNode):
        printer.write(PLIST_HEADER)
        self.print(printer, node.root)
        printer.write(PLIST_FOOTER)


class PLIST(Filetype):
    """The Apple PLIST filetype."""
    def __init__(self):
        """Initializes the PLIST file type.

        By default, PLIST associates itself with the "plist" and "application/x-plist" MIME types.

        """
        super().__init__(
            'plist',
            'application/x-plist'
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
        except ExpatError as ee:
            return f'Error parsing {os.path.basename(path)}: {ee})'

    def get_default_formatter(self) -> PLISTFormatter:
        return PLISTFormatter.DEFAULT_INSTANCE
