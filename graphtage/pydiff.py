"""A :class:`graphtage.Filetype` for parsing, diffing, and rendering Apple plist files."""
import os
from xml.parsers.expat import ExpatError
from typing import Any, Dict, List, Optional, Tuple, Union

from plistlib import dumps, load

from .edits import Edit, EditCollection, Match
from .graphtage import (
    BoolNode, BuildOptions, DictNode, FixedKeyDictNode, FloatNode, IntegerNode, KeyValuePairNode, LeafNode, ListNode,
    NullNode, StringNode
)
from .printer import Printer
from .sequences import SequenceFormatter, SequenceNode
from .tree import ContainerNode, GraphtageFormatter, TreeNode


class _PyObj:
    pass


class FixedKeyPyObj(FixedKeyDictNode, _PyObj):
    pass


class PyObj(DictNode, _PyObj):
    pass


def build_tree(python_obj: Any, options: Optional[BuildOptions] = None) -> TreeNode:
    """Builds a Graphtage tree from an arbitrary Python object, even complex custom classes.

    Args:
        python_obj: The object from which to build the tree.
        options: An optional set of options for building the tree.

    Returns:
        TreeNode: The resulting tree.

    Raises:
        ValueError: If the object is of an unsupported type.

    """
    class IncompletePyObj(dict):
        pass

    class DictValue:
        def __init__(self, key, value):
            self.key = key
            self.value = value
            self.key_node: Optional[TreeNode] = None
            self.value_node: Optional[TreeNode] = None

    stack: List[Tuple[Any, List[TreeNode], List[Any]]] = [
        (None, [], [python_obj])
    ]

    if options is None:
        options = BuildOptions()

    while True:
        parent, children, work = stack.pop()

        new_node: Optional[Union[TreeNode, DictValue]] = None

        while not work:
            if parent is None:
                assert len(children) == 1
                return children[0]

            assert len(stack) > 0

            if isinstance(parent, DictValue):
                assert parent.key_node is None
                assert parent.value_node is None
                assert len(children) == 2
                parent.key_node, parent.value_node = children
                new_node = parent
            elif isinstance(parent, list):
                new_node = ListNode(
                    children,
                    allow_list_edits=options.allow_list_edits,
                    allow_list_edits_when_same_length=options.allow_list_edits_when_same_length
                )
            elif isinstance(parent, dict):
                assert all(
                    isinstance(c, DictValue) and c.key_node is not None and c.value_node is not None
                    for c in children
                )
                dict_items = {
                    c.key_node: c.value_node for c in children
                }
                if options.allow_key_edits:
                    dict_node = DictNode.from_dict(dict_items)
                    dict_node.auto_match_keys = options.auto_match_keys
                    new_node = dict_node
                else:
                    new_node = FixedKeyDictNode.from_dict(dict_items)
            else:
                # this should never happen
                raise NotImplementedError(f"Unexpected parent: {parent!r}")

            parent, children, work = stack.pop()
            children.append(new_node)
            new_node = None

        obj = work.pop()
        stack.append((parent, children, work))

        if isinstance(obj, bool):
            new_node = BoolNode(obj)
        elif isinstance(obj, int):
            new_node = IntegerNode(obj)
        elif isinstance(obj, float):
            new_node = FloatNode(obj)
        elif isinstance(obj, str):
            new_node = StringNode(obj)
        elif isinstance(obj, bytes):
            new_node = StringNode(obj.decode('utf-8'))
        elif isinstance(obj, DictValue):
            stack.append((obj, [], [obj.value, obj.key]))
        elif isinstance(obj, dict):
            stack.append(({}, [], [DictValue(key=k, value=v) for k, v in reversed(obj.items())]))
        elif isinstance(python_obj, (list, tuple)):
            stack.append(([], [], list(reversed(python_obj))))
        elif python_obj is None:
            new_node = NullNode()
        else:
            raise ValueError(f"Unsupported Python object {python_obj!r} of type {type(python_obj)}")

        if new_node is not None:
            parent, children, work = stack.pop()
            children.append(new_node)
            stack.append((parent, children, work))
