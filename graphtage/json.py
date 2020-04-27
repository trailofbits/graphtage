import json
import os
import sys

from .formatter import Formatter
from .graphtage import BoolNode, DictNode, Filetype, FixedKeyDictNode, \
    FloatNode, IntegerNode, LeafNode, ListNode, StringNode
from .printer import Printer
from .sequences import SequenceFormatter, SequenceNode
from .tree import TreeNode


def build_tree(python_obj, allow_key_edits=True, force_leaf_node=False) -> TreeNode:
    if isinstance(python_obj, int):
        return IntegerNode(python_obj)
    elif isinstance(python_obj, float):
        return FloatNode(python_obj)
    elif isinstance(python_obj, bool):
        return BoolNode(python_obj)
    elif isinstance(python_obj, str):
        return StringNode(python_obj)
    elif isinstance(python_obj, bytes):
        return StringNode(python_obj.decode('utf-8'))
    elif force_leaf_node:
        raise ValueError(f"{python_obj!r} was expected to be an int or string, but was instead a {type(python_obj)}")
    elif isinstance(python_obj, list) or isinstance(python_obj, tuple):
        return ListNode([build_tree(n, allow_key_edits=allow_key_edits) for n in python_obj])
    elif isinstance(python_obj, dict):
        dict_items = {
            build_tree(k, allow_key_edits=allow_key_edits, force_leaf_node=True):
                build_tree(v, allow_key_edits=allow_key_edits) for k, v in python_obj.items()
        }
        if allow_key_edits:
            return DictNode(dict_items)
        else:
            return FixedKeyDictNode(dict_items)
    else:
        raise ValueError(f"Unsupported Python object {python_obj!r} of type {type(python_obj)}")


class JSON(Filetype):
    def __init__(self):
        super().__init__(
            'json',
            'application/json',
            'application/x-javascript',
            'text/javascript',
            'text/x-javascript',
            'text/x-json'
        )

    def build_tree(self, path: str, allow_key_edits: bool = True) -> TreeNode:
        with open(path) as f:
            return build_tree(json.load(f), allow_key_edits=allow_key_edits)

    def build_tree_handling_errors(self, path: str, allow_key_edits: bool = True) -> TreeNode:
        try:
            return self.build_tree(path=path, allow_key_edits=allow_key_edits)
        except json.decoder.JSONDecodeError as de:
            sys.stderr.write(
                f'Error parsing {os.path.basename(path)}: {de.msg}: line {de.lineno}, column {de.colno} (char {de.pos})\n\n')
            sys.exit(1)


class JSONListFormatter(SequenceFormatter):
    def __init__(self):
        super().__init__('[', ']', ',')

    def print_ListNode(self, printer: Printer, node: ListNode):
        self.print_SequenceNode(printer=printer, node=node)


class JSONDictFormatter(SequenceFormatter):
    def __init__(self):
        super().__init__('{', '}', ',')

    def print_MultiSetNode(self, printer: Printer, node: SequenceNode):
        self.print_SequenceNode(printer=printer, node=node)

    def print_FixedKeyDictNode(self, printer: Printer, node: SequenceNode):
        self.print_SequenceNode(printer=printer, node=node)


class JSONFormatter(Formatter):
    def __init__(self):
        self.sub_formatters = [JSONListFormatter(), JSONDictFormatter()]

    def print_LeafNode(self, printer: Printer, node: LeafNode):
        node.print(printer)
