"""Functions to diff in-memory Python objects.

See :doc:`the documentation on using Graphtage programmatically <library>` for some examples.
"""
from typing import Any, List, Optional, Tuple, Union, Iterator


from .edits import AbstractCompoundEdit, Edit, Replace
from .graphtage import (
    BoolNode, BuildOptions, DictNode, FixedKeyDictNode, FloatNode, IntegerNode, KeyValuePairNode, LeafNode, ListNode,
    NullNode, StringNode
)
from .json import JSONDictFormatter, JSONListFormatter
from .printer import Fore, Printer
from .sequences import SequenceFormatter
from .tree import ContainerNode, GraphtageFormatter, TreeNode


class PyObjEdit(AbstractCompoundEdit):
    def __init__(self, from_obj: "PyObj", to_obj: "PyObj"):
        super().__init__(from_obj, to_obj)
        self.from_obj: PyObj = from_obj
        self.to_obj: PyObj = to_obj
        self.name_edit: Edit = self.from_obj.class_name.edits(self.to_obj.class_name)
        self.attrs_edit: Edit = from_obj.attrs.edits(to_obj.attrs)

    def edits(self) -> Iterator[Edit]:
        yield self.name_edit
        yield self.attrs_edit

    def tighten_bounds(self) -> bool:
        if self.name_edit.tighten_bounds():
            return True
        else:
            return self.attrs_edit.tighten_bounds()

    def print(self, formatter: GraphtageFormatter, printer: Printer):
        raise NotImplementedError()


class PyObjAttribute(KeyValuePairNode):
    pass


class PyObjAttributes(DictNode):
    @classmethod
    def make_key_value_pair_node(cls, key: LeafNode, value: TreeNode, allow_key_edits: bool = True) -> KeyValuePairNode:
        return PyObjAttribute(key=key, value=value, allow_key_edits=allow_key_edits)


class PyObjFixedAttributes(FixedKeyDictNode):
    @classmethod
    def make_key_value_pair_node(cls, key: LeafNode, value: TreeNode, allow_key_edits: bool = True) -> KeyValuePairNode:
        return PyObjAttribute(key=key, value=value, allow_key_edits=allow_key_edits)


PyObjAttributeMapping = Union[PyObjAttributes, PyObjFixedAttributes]


class PyObj(ContainerNode):
    def __init__(self, class_name: StringNode, attrs: Optional[PyObjAttributeMapping]):
        self.class_name: StringNode = class_name
        if attrs is None:
            attrs = PyObjAttributes.from_dict({})
        self.attrs: PyObjAttributeMapping = attrs

    def to_obj(self):
        return {
            self.class_name: self.attrs.to_obj()
        }

    def edits(self, node: TreeNode) -> Edit:
        if not isinstance(node, PyObj) or node.class_name != self.class_name:
            return Replace(self, node)
        else:
            return PyObjEdit(self, node)

    def calculate_total_size(self) -> int:
        return self.attrs.calculate_total_size()

    def print(self, printer: Printer):
        PyDiffFormatter.DEFAULT_INSTANCE.print(printer, self)

    def __iter__(self) -> Iterator[TreeNode]:
        yield self.class_name
        yield self.attrs

    def __len__(self) -> int:
        return 2


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
    class PyObjMember:
        def __init__(self, attr: str, value):
            self.attr: str = attr
            self.value = value
            self.value_node: Optional[TreeNode] = None

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

        new_node: Optional[Union[TreeNode, DictValue, PyObjMember]] = None

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
            elif isinstance(parent, PyObjMember):
                assert parent.value_node is None
                assert len(children) == 1
                parent.value_node = children[0]
                new_node = parent
            elif isinstance(parent, PyObj):
                assert all(
                    isinstance(c, PyObjMember) and c.value_node is not None
                    for c in children
                )
                members = {
                    StringNode(c.attr, quoted=False): c.value_node for c in children
                }
                if options.allow_key_edits:
                    dict_node = PyObjAttributes.from_dict(members)
                    dict_node.auto_match_keys = options.auto_match_keys
                else:
                    dict_node = PyObjFixedAttributes.from_dict(members)
                parent.attrs = dict_node
                dict_node.parent = parent
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
        elif isinstance(obj, PyObjMember):
            stack.append((obj, [], [obj.value]))
        elif isinstance(obj, dict):
            stack.append(({}, [], [DictValue(key=k, value=v) for k, v in reversed(list(obj.items()))]))
        elif isinstance(python_obj, (list, tuple)):
            stack.append(([], [], list(reversed(obj))))
        elif python_obj is None:
            new_node = NullNode()
        else:
            pyobj = PyObj(class_name=StringNode(obj.__class__.__name__, quoted=False), attrs=None)  # type: ignore
            stack.append((pyobj, [], [
                PyObjMember(attr=attr, value=getattr(obj, attr))
                for attr in reversed(dir(obj))
                if not attr.startswith("__")
            ]))

        if new_node is not None:
            parent, children, work = stack.pop()
            children.append(new_node)
            stack.append((parent, children, work))


class PyListFormatter(JSONListFormatter):
    is_partial = True

    def item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        pass


class PyDictFormatter(JSONDictFormatter):
    is_partial = True

    def item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        pass


class PyObjFormatter(SequenceFormatter):
    is_partial = True

    sub_format_types = [PyListFormatter, PyDictFormatter]

    def __init__(self):
        super().__init__('(', ')', ', ')

    def item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        pass

    def print_PyObj(self, printer: Printer, node: PyObj):
        with printer.color(Fore.YELLOW):
            self.print(printer, node.class_name)
        self.print(printer, node.attrs)

    def print_PyObjAttributes(self, *args, **kwargs):
        super().print_SequenceNode(*args, **kwargs)

    def print_PyObjFixedAttributes(self, *args, **kwargs):
        super().print_SequenceNode(*args, **kwargs)

    def print_PyObjAttribute(self, printer: Printer, node: KeyValuePairNode):
        """Prints a :class:`graphtage.PyObjAttribute` key/value pair.

        By default, the key is printed in red, followed by "=", followed by the value in light blue.

        """
        with printer.color(Fore.RED):
            self.print(printer, node.key)
        printer.write("=")
        with printer.color(Fore.LIGHTBLUE_EX):
            self.print(printer, node.value)


class PyDiffFormatter(GraphtageFormatter):
    sub_format_types = [PyObjFormatter, PyListFormatter, PyDictFormatter]


def diff(from_py_obj, to_py_obj):
    return build_tree(from_py_obj).diff(build_tree(to_py_obj))


def print_diff(from_py_obj, to_py_obj, printer: Optional[Printer] = None):
    if printer is None:
        printer = Printer()
    d = diff(from_py_obj, to_py_obj)
    with printer:
        PyDiffFormatter.DEFAULT_INSTANCE.print(printer, d)
