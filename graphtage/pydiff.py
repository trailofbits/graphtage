"""Functions to diff in-memory Python objects.

See :doc:`the documentation on using Graphtage programmatically <library>` for some examples.
"""
import ast
import logging
from typing import Any, List, Optional, Union, Iterator, Iterable

from . import Range
from .ast import Assignment, Call, CallArguments, CallKeywords, Import, KeywordArgument, Module, Subscript
from .builder import BasicBuilder, Builder
from .dataclasses import DataClassNode
from .edits import AbstractCompoundEdit, Edit, Replace
from .graphtage import (
    BuildOptions, DictNode, FixedKeyDictNode, KeyValuePairNode, LeafNode, ListNode, MultiSetNode, StringNode
)
from .json import JSONDictFormatter, JSONListFormatter
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
        return KeywordArgument(key=key, value=value, allow_key_edits=allow_key_edits)


class PyObjFixedAttributes(FixedKeyDictNode):
    @classmethod
    def make_key_value_pair_node(cls, key: LeafNode, value: TreeNode, allow_key_edits: bool = True) -> KeyValuePairNode:
        return KeywordArgument(key=key, value=value, allow_key_edits=allow_key_edits)


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


class PyAlias(DataClassNode):
    name: StringNode
    as_name: StringNode

    def print(self, printer: Printer):
        self.name.print(printer)
        if self.as_name.object:
            printer.write(" as ")
            self.as_name.print(printer)


class ASTBuilder(BasicBuilder):
    """Builds a Graphtage tree from a Python Abstract Syntax Tree"""

    @Builder.expander(ast.Module)
    def expand_module(self, node: ast.Module):
        yield from node.body

    @Builder.builder(ast.Module)
    def build_module(self, _, children: List[TreeNode]):
        return Module(tuple(children))

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
        return Assignment(targets=ListNode(children[:-1]), value=children[-1])

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
        return Call(
            func_name,
            CallArguments(children[1:]),  # type: ignore
            CallKeywords(())
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
        return Import(names=ListNode(children), from_name=from_name)

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
        return Subscript(*children)


def ast_to_tree(tree: ast.AST, options: Optional[BuildOptions] = None) -> TreeNode:
    """Builds a Graphtage tree from a Python Abstract Syntax Tree.

    Args:
        tree: The abstract syntax tree from which to build the Graphtage tree.
        options: An optional set of options for building the tree.

    Returns:
        TreeNode: The resulting tree.
    """
    return ASTBuilder(options).build_tree(tree)


class PyObjBuilder(BasicBuilder):
    def default_expander(self, node: Any) -> Iterable[Any]:
        if self.resolve_builder(node.__class__) is not None:
            # we have a builder for this node type but no expander, which means it is a basic type like int or str,
            # so default to epanding to nothing
            return ()
        yield node.__class__.__name__
        for attr in dir(node):
            if not attr.startswith("__"):
                yield attr
                yield getattr(node, attr)

    def default_builder(self, node: Any, children: List[TreeNode]):
        name = children[0]
        assert isinstance(name, StringNode)
        name.quoted = False
        assert (len(children) - 1) % 2 == 0
        members = {
            attr: value
            for attr, value in zip(children[1::2], children[2::2])
        }
        for attr in members.keys():
            assert isinstance(attr, StringNode)
            attr.quoted = False
        if self.options.allow_key_edits:
            dict_node = PyObjAttributes.from_dict(members)
            dict_node.auto_match_keys = self.options.auto_match_keys
        else:
            dict_node = PyObjFixedAttributes.from_dict(members)
        return PyObj(name, dict_node)


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
    return PyObjBuilder(options).build_tree(python_obj)


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

    def print_Assignment(self, printer: Printer, node: Assignment):
        super().print_SequenceNode(printer, node.targets)
        printer.write(" = ")
        self.print(printer, node.value)

    def print_Import(self, printer: Printer, node: Import):
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

    def print_Call(self, printer: Printer, node: Call):
        with printer.color(Fore.YELLOW):
            self.print(printer, node.func)
        self.print(printer, node.args)
        if node.kwargs.children():
            raise NotImplementedError("TODO: Implement full support for keword arguments")

    def print_CallArguments(self, *args, **kwargs):
        super().print_SequenceNode(*args, **kwargs)

    def print_PyObjAttributes(self, *args, **kwargs):
        super().print_SequenceNode(*args, **kwargs)

    def print_PyObjFixedAttributes(self, *args, **kwargs):
        super().print_SequenceNode(*args, **kwargs)

    def print_PyObjAttribute(self, printer: Printer, node: PyObjAttribute):
        self.print(printer, node.object)
        printer.write(".")
        self.print(printer, node.attr)

    def print_KeywordArgument(self, printer: Printer, node: KeyValuePairNode):
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

    def print_Module(self, printer: Printer, node: Module):
        super().print_SequenceNode(printer, node)


class PyDiffFormatter(GraphtageFormatter):
    sub_format_types = [PyObjFormatter, PyImportFormatter, PyModuleFormatter, PyListFormatter, PyDictFormatter]

    def print_PyAlias(self, printer: Printer, node: PyAlias):
        self.print(printer, node.name)
        if node.as_name.object:
            with printer.color(Fore.BLUE):
                printer.write(" as ")
                self.print(printer, node.as_name)

    def print_Subscript(self, printer: Printer, node: Subscript):
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
