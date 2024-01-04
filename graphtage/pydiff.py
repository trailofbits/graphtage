"""Functions to diff in-memory Python objects.

See :doc:`the documentation on using Graphtage programmatically <library>` for some examples.
"""
import ast
from typing import Any, List, Optional, Tuple, Union, Iterator

from . import Range
from .dataclasses import DataClassNode
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
        self.name_edit: Edit = from_obj.class_name.edits(to_obj.class_name)
        self.attrs_edit: Edit = from_obj.attrs.edits(to_obj.attrs)
        super().__init__(from_obj, to_obj)
        self.from_obj: PyObj = from_obj
        self.to_obj: PyObj = to_obj

    def edits(self) -> Iterator[Edit]:
        yield self.name_edit
        yield self.attrs_edit

    def bounds(self) -> Range:
        return self.name_edit.bounds() + self.attrs_edit.bounds()

    def tighten_bounds(self) -> bool:
        return self.name_edit.tighten_bounds() or self.attrs_edit.tighten_bounds()

    def print(self, formatter: GraphtageFormatter, printer: Printer):
        raise NotImplementedError()

    def __repr__(self):
        return f"{self.__class__.__name__}(from_obj={self.from_obj!r}, to_obj={self.to_obj!r})"


class PyKeywordArgument(KeyValuePairNode):
    pass


class PyObjAttribute(DataClassNode):
    object: TreeNode
    attr: StringNode

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(object, StringNode):
            object.quoted = False
        self.attr.quoted = False


class PyObjAttributes(DictNode):
    @classmethod
    def make_key_value_pair_node(cls, key: LeafNode, value: TreeNode, allow_key_edits: bool = True) -> KeyValuePairNode:
        return PyKeywordArgument(key=key, value=value, allow_key_edits=allow_key_edits)


class PyObjFixedAttributes(FixedKeyDictNode):
    @classmethod
    def make_key_value_pair_node(cls, key: LeafNode, value: TreeNode, allow_key_edits: bool = True) -> KeyValuePairNode:
        return PyKeywordArgument(key=key, value=value, allow_key_edits=allow_key_edits)


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

    def __repr__(self):
        return f"{self.__class__.__name__}(class_name={self.class_name!r}, attrs={self.attrs!r})"


ASTNode = Union[ast.AST, ast.stmt, ast.expr, ast.alias]


class PyModule(ListNode):
    def print(self, printer: Printer):
        SequenceFormatter('', '', '\n').print(printer, self)


class PyAssignment(DataClassNode):
    """A node representing a Python assignment."""

    targets: ListNode
    value: TreeNode

    def print(self, printer: Printer):
        """Prints this node."""
        SequenceFormatter('', '', ', ').print(printer, self.targets)
        with printer.bright():
            printer.write(" = ")
        self.value.print(printer)

    def __str__(self):
        return f"{', '.join(map(str, self.targets.children()))} = {self.value!s}"


class PyCallArguments(ListNode):
    pass


class PyCallKeywords(DictNode):
    pass


class PyCall(DataClassNode):
    """A node representing a Python function call."""

    func: TreeNode
    args: PyCallArguments
    kwargs: PyCallKeywords

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.func, StringNode):
            self.func.quoted = False

    def print(self, printer: Printer):
        with printer.color(Fore.YELLOW):
            self.func.print(printer)
        printer.write("(")
        SequenceFormatter('', '', ', ').print(printer, self.args)
        if self.args and len(self.kwargs) > 0:
            printer.write(", ")
        for kvp in self.kwargs:
            with printer.color(Fore.RED):
                kvp.key.print(printer)
            with printer.bright():
                printer.write("=")
            kvp.value.print(printer)
        printer.write(")")

    def __str__(self):
        args = ", ".join([str(a) for a in self.args] + [
            f"{kvp.key!s}={kvp.value!s}"
            for kvp in self.kwargs
        ])
        return f"{self.func!s}({args})"


class PyAlias(DataClassNode):
    name: StringNode
    as_name: StringNode

    def print(self, printer: Printer):
        self.name.print(printer)
        if self.as_name.object:
            printer.write(" as ")
            self.as_name.print(printer)


class PyImport(DataClassNode):
    names: ListNode
    from_name: StringNode

    def __init__(self, names: ListNode, from_name: StringNode):
        super().__init__(names=names, from_name=from_name)
        self.from_name.quoted = False
        for child in self.names:
            if isinstance(child, StringNode):
                child.quoted = False

    def print(self, printer: Printer):
        if self.from_name.object:
            with printer.color(Fore.YELLOW):
                printer.write("from ")
            self.from_name.print(printer)
            printer.write(" ")
        with printer.color(Fore.YELLOW):
            printer.write("import ")
        SequenceFormatter('', '', ', ').print(printer, self.names)


def ast_to_tree(tree: ast.AST, options: Optional[BuildOptions] = None) -> TreeNode:
    """Builds a Graphtage tree from a Python Abstract Syntax Tree.

    Args:
        tree: The abstract syntax tree from which to build the Graphtage tree.
        options: An optional set of options for building the tree.

    Returns:
        TreeNode: The resulting tree.
    """
    work: List[Tuple[ASTNode, List[TreeNode], List[ASTNode]]] = [(tree, [], list(reversed(tree.body)))]
    while work:
        node, processed_children, remaining_children = work.pop()
        if remaining_children:
            child = remaining_children.pop()
            new_children: List[ASTNode]
            if isinstance(child, ast.Module):
                new_children = child.body
            elif isinstance(child, ast.List):
                new_children = child.elts
            elif isinstance(child, ast.Assign):
                new_children = child.targets + [child.value]
            elif isinstance(child, (ast.Name, ast.Constant)):
                new_children = []
            elif isinstance(child, ast.Call):
                new_children = [child.func] + child.args
            elif isinstance(child, ast.ImportFrom):
                new_children = child.names
            elif isinstance(child, ast.alias):
                new_children = []
            elif isinstance(child, ast.Expr):
                work.append((node, processed_children, remaining_children + [child.value]))
                continue
            elif isinstance(child, ast.Attribute):
                new_children = [child.value]
            elif isinstance(child, ast.Dict):
                new_children = child.keys + child.values
            else:
                raise NotImplementedError(str(child.__class__))
            work.append((node, processed_children, remaining_children))
            work.append((child, [], list(reversed(new_children))))  # type: ignore
            continue
        result: Optional[Any] = None
        if not remaining_children:
            if isinstance(node, ast.Module):
                result = PyModule(tuple(processed_children))
            elif isinstance(node, ast.List):
                result = ListNode(
                    processed_children,
                    allow_list_edits=options.allow_list_edits,
                    allow_list_edits_when_same_length=options.allow_list_edits_when_same_length
                )
            elif isinstance(node, ast.Name):
                result = StringNode(node.id, quoted=False)
            elif isinstance(node, ast.Constant):
                result = build_tree(node.value, options=options)
            elif isinstance(node, ast.Assign):
                result = PyAssignment(targets=ListNode(processed_children[:-1]), value=processed_children[-1])
            elif isinstance(node, ast.Call):
                func_name = processed_children[0]
                if isinstance(func_name, StringNode):
                    func_name.quoted = False
                result = PyCall(
                    func_name,
                    PyCallArguments(processed_children[1:]),  # type: ignore
                    PyCallKeywords(())
                )
            elif isinstance(node, ast.alias):
                if not node.asname:
                    as_name = StringNode("")
                else:
                    as_name = StringNode(node.asname)
                result = PyAlias(StringNode(node.name, quoted=False), as_name)
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    from_name = StringNode("", quoted=False)
                else:
                    from_name = StringNode(node.module, quoted=False)
                result = PyImport(names=ListNode(processed_children), from_name=from_name)
            elif isinstance(node, ast.Attribute):
                result = PyObjAttribute(processed_children[0], StringNode(node.attr, quoted=False))
            elif isinstance(node, ast.Dict):
                n = len(processed_children) // 2
                keys = processed_children[:n]
                values = processed_children[n:]
                dict_items = {
                    k: v
                    for k, v in zip(keys, values)
                }
                if options.allow_key_edits:
                    dict_node = DictNode.from_dict(dict_items)
                    dict_node.auto_match_keys = options.auto_match_keys
                    result = dict_node
                else:
                    result = FixedKeyDictNode.from_dict(dict_items)
            else:
                raise NotImplementedError(str(node.__class__))
        if not work:
            return result
        else:
            work[-1][1].append(result)
    return build_tree(None, options=options)


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
            try:
                new_node = StringNode(obj.decode('utf-8'))
            except UnicodeDecodeError:
                new_node = ListNode(map(IntegerNode, obj))
        elif isinstance(obj, DictValue):
            stack.append((obj, [], [obj.value, obj.key]))
        elif isinstance(obj, PyObjMember):
            stack.append((obj, [], [obj.value]))
        elif isinstance(obj, dict):
            stack.append(({}, [], [DictValue(key=k, value=v) for k, v in reversed(list(obj.items()))]))
        elif isinstance(obj, (list, tuple)):
            stack.append(([], [], list(reversed(obj))))
        elif obj is None:
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


class PyImportFormatter(SequenceFormatter):
    is_partial = True

    sub_format_types = [PyListFormatter]

    def __init__(self):
        super().__init__('', '', ', ')

    def item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        pass

    def print_PyAssignment(self, printer: Printer, node: PyAssignment):
        super().print_SequenceNode(printer, node.targets)
        printer.write(" = ")
        self.print(printer, node.value)

    def print_PyImport(self, printer: Printer, node: PyImport):
        if node.from_name.object:
            with printer.color(Fore.BLUE):
                printer.write("from ")
            self.print(printer, node.from_name)
            printer.write(" ")
        with printer.color(Fore.BLUE):
            printer.write("import ")
        self.print_SequenceNode(printer, node.names)


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

    def print_PyCall(self, printer: Printer, node: PyCall):
        with printer.color(Fore.YELLOW):
            self.print(printer, node.func)
        self.print(printer, node.args)
        if node.kwargs.children():
            raise NotImplementedError("TODO: Implement full support for keword arguments")

    def print_PyCallArguments(self, *args, **kwargs):
        super().print_SequenceNode(*args, **kwargs)

    def print_PyObjAttributes(self, *args, **kwargs):
        super().print_SequenceNode(*args, **kwargs)

    def print_PyObjFixedAttributes(self, *args, **kwargs):
        super().print_SequenceNode(*args, **kwargs)

    def print_PyObjAttribute(self, printer: Printer, node: PyObjAttribute):
        self.print(printer, node.object)
        printer.write(".")
        self.print(printer, node.attr)

    def print_PyKeywordArgument(self, printer: Printer, node: KeyValuePairNode):
        """Prints a :class:`graphtage.PyKeywordArgument` key/value pair.

        By default, the key is printed in red, followed by "=", followed by the value in light blue.

        """
        with printer.color(Fore.RED):
            self.print(printer, node.key)
        printer.write("=")
        with printer.color(Fore.LIGHTBLUE_EX):
            self.print(printer, node.value)


class PyModuleFormatter(SequenceFormatter):
    is_partial = True

    sub_format_types = [PyListFormatter]

    def __init__(self):
        super().__init__('', '', '')

    def items_indent(self, printer: Printer) -> Printer:
        return printer
    
    def item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        if not is_first:
            printer.newline()

    def print_PyModule(self, printer: Printer, node: PyModule):
        super().print_SequenceNode(printer, node)


class PyDiffFormatter(GraphtageFormatter):
    sub_format_types = [PyObjFormatter, PyImportFormatter, PyModuleFormatter, PyListFormatter, PyDictFormatter]

    def print_PyAlias(self, printer: Printer, node: PyAlias):
        self.print(printer, node.name)
        if node.as_name.object:
            with printer.color(Fore.BLUE):
                printer.write(" as ")
                self.print(printer, node.as_name)


def diff(from_py_obj, to_py_obj):
    return build_tree(from_py_obj).diff(build_tree(to_py_obj))


def print_diff(from_py_obj, to_py_obj, printer: Optional[Printer] = None):
    if printer is None:
        printer = Printer()
    d = diff(from_py_obj, to_py_obj)
    with printer:
        PyDiffFormatter.DEFAULT_INSTANCE.print(printer, d)
