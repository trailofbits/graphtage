"""Functions to diff in-memory Python objects.

See :doc:`the documentation on using Graphtage programmatically <library>` for some examples.
"""
import ast
import logging
from typing import Any, List, Optional, Tuple, Union, Iterator

from . import Range
from .builder import BasicBuilder, Builder
from .dataclasses import DataClassNode
from .edits import AbstractCompoundEdit, Edit, Replace
from .graphtage import (
    BoolNode, BuildOptions, DictNode, FixedKeyDictNode, FloatNode, IntegerNode, KeyValuePairNode, LeafNode, ListNode,
    MultiSetNode, NullNode, StringNode
)
from .json import JSONDictFormatter, JSONListFormatter
from .object_set import ObjectSet
from .printer import Fore, Printer
from .sequences import SequenceFormatter
from .tree import ContainerNode, GraphtageFormatter, TreeNode


log = logging.getLogger(__name__)


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


class PySubscript(DataClassNode):
    value: TreeNode
    slice: TreeNode

    def print(self, printer: Printer):
        self.value.print(printer)
        with printer.color(Fore.LIGHTBLUE_EX):
            printer.write("[")
        self.slice.write(printer)
        with printer.color(Fore.LIGHTBLUE_EX):
            printer.write("]")


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


class ASTBuilder(BasicBuilder):
    """Builds a Graphtage tree from a Python Abstract Syntax Tree"""

    @Builder.expander(ast.Module)
    def expand_module(self, node: ast.Module):
        yield from node.body

    @Builder.builder(ast.Module)
    def build_module(self, _, children: List[TreeNode]):
        return PyModule(tuple(children))

    @Builder.expander(ast.List)
    @Builder.expander(ast.Tuple)
    @Builder.expander(ast.Set)
    def expand_collection(self, node: Union[ast.List, ast.Tuple, ast.Set]):
        yield from node.elts

    @Builder.builder(ast.Set)
    def build_set(self, _, children: List[TreeNode]):
        return MultiSetNode(items=children, auto_match_keys=self.options.auto_match_keys)

    @Builder.builder(ast.List)
    @Builder.builder(ast.Tuple)
    def build_ast_list(self, node: ast.List, children):
        return self.build_list(node, children)

    @Builder.expander(ast.Assign)
    def expand_assign(self, node: ast.Assign):
        return node.targets + [node.value]

    @Builder.builder(ast.Assign)
    def build_assign(self, _, children):
        return PyAssignment(targets=ListNode(children[:-1]), value=children[-1])

    @Builder.builder(ast.Name)
    def build_name(self, node: ast.Name, _):
        return StringNode(node.id, quoted=False)

    @Builder.expander(ast.Constant)
    @Builder.expander(ast.Expr)
    @Builder.expander(ast.Attribute)
    def expand_constant(self, node: ast.Constant):
        yield node.value

    @Builder.builder(ast.Constant)
    @Builder.builder(ast.Expr)
    def build_constant(self, _, children: List[TreeNode]):
        assert len(children) == 1
        return children[0]

    @Builder.expander(ast.Call)
    def expand_call(self, node: ast.Call):
        return [node.func] + node.args

    @Builder.builder(ast.Call)
    def build_call(self, _, children: List[TreeNode]):
        func_name = children[0]
        if isinstance(func_name, StringNode):
            func_name.quoted = False
        return PyCall(
            func_name,
            PyCallArguments(children[1:]),  # type: ignore
            PyCallKeywords(())
        )

    @Builder.expander(ast.ImportFrom)
    def expand_import_from(self, node: ast.ImportFrom):
        return node.names

    @Builder.builder(ast.ImportFrom)
    def build_import_from(self, node: ast.ImportFrom, children: List[TreeNode]):
        if node.module is None:
            from_name = StringNode("", quoted=False)
        else:
            from_name = StringNode(node.module, quoted=False)
        return PyImport(names=ListNode(children), from_name=from_name)

    @Builder.builder(ast.alias)
    def build_alias(self, node: ast.alias, _):
        if not node.asname:
            as_name = StringNode("")
        else:
            as_name = StringNode(node.asname)
        return PyAlias(StringNode(node.name, quoted=False), as_name)

    @Builder.builder(ast.Attribute)
    def build_attribute(self, node: ast.Attribute, children: List[TreeNode]):
        assert len(children) == 1
        return PyObjAttribute(children[0], StringNode(node.attr, quoted=False))

    @Builder.expander(ast.Dict)
    def expand_ast_dict(self, node: ast.Dict):
        yield from node.keys
        yield from node.values

    @Builder.builder(ast.Dict)
    def build_ast_dict(self, node: ast.Dict, children: List[TreeNode]):
        return self.build_dict(node, children)

    @Builder.expander(ast.Subscript)
    def expand_subscript(self, node: ast.Subscript):
        yield node.value
        yield node.slice

    @Builder.builder(ast.Subscript)
    def build_subscript(self, _, children: List[TreeNode]):
        return PySubscript(*children)


def ast_to_tree(tree: ast.AST, options: Optional[BuildOptions] = None) -> TreeNode:
    """Builds a Graphtage tree from a Python Abstract Syntax Tree.

    Args:
        tree: The abstract syntax tree from which to build the Graphtage tree.
        options: An optional set of options for building the tree.

    Returns:
        TreeNode: The resulting tree.
    """
    return ASTBuilder(options).build_tree(tree)


def build_tree(python_obj: Any, options: Optional[BuildOptions] = None) -> TreeNode:
    """Builds a Graphtage tree from an arbitrary Python object, even complex custom classes.

    Args:
        python_obj: The object from which to build the tree.
        options: An optional set of options for building the tree.

    Returns:
        TreeNode: The resulting tree.

    Raises:
        ValueError: If the object is of an unsupported type, or if a cycle is detected and not ignored.

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
    history = ObjectSet()

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
                if len(children) != 2 and options.check_for_cycles and options.ignore_cycles:
                    # one of our children induced a cycle, so discard the parent
                    parent, children, work = stack.pop()
                    continue
                assert len(children) == 2
                parent.key_node, parent.value_node = children
                new_node = parent
            elif isinstance(parent, PyObjMember):
                assert parent.value_node is None
                if len(children) != 1 and options.check_for_cycles and options.ignore_cycles:
                    # one of our children induced a cycle, so discard the parent
                    parent, children, work = stack.pop()
                    continue
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

        if options.check_for_cycles:
            if obj in history:
                if options.ignore_cycles:
                    log.debug(f"Detected a cycle in {python_obj!r} at member {obj!r}; ignoring…")
                    continue
                else:
                    raise ValueError(f"Detected a cycle in {python_obj!r} at member {obj!r}")
            history.add(obj)

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

    def print_PySubscript(self, printer: Printer, node: PySubscript):
        self.print(printer, node.value)
        with printer.color(Fore.BLUE):
            printer.write("[")
        self.print(printer, node.slice)
        with printer.color(Fore.BLUE):
            printer.write("[")


def diff(from_py_obj, to_py_obj, options: Optional[BuildOptions] = None):
    return build_tree(from_py_obj, options=options).diff(build_tree(to_py_obj, options=options))


def print_diff(from_py_obj, to_py_obj, printer: Optional[Printer] = None, options: Optional[BuildOptions] = None):
    if printer is None:
        printer = Printer()
    d = diff(from_py_obj, to_py_obj, options=options)
    with printer:
        PyDiffFormatter.DEFAULT_INSTANCE.print(printer, d)
