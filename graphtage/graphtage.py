import mimetypes
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Collection, Dict, Generic, Iterable, Iterator, List, Optional, Tuple, Type, TypeVar

from .bounds import Range
from .edits import AbstractEdit, EditCollection, EditSequence
from .edits import Insert, Match, Remove, Replace, AbstractCompoundEdit
from .levenshtein import EditDistance, levenshtein_distance
from .multiset import MultiSetEdit
from .printer import Back, Fore, NullANSIContext, Printer
from .sequences import SequenceEdit, SequenceNode
from .tree import ContainerNode, Edit, GraphtageFormatter, TreeNode
from .utils import HashableCounter


class LeafNode(TreeNode):
    def __init__(self, obj):
        self.object = obj

    def to_obj(self):
        return self.object

    def children(self) -> Collection[TreeNode]:
        return ()

    def calculate_total_size(self):
        return len(str(self.object))

    def edits(self, node: TreeNode) -> Edit:
        if isinstance(node, LeafNode):
            return Match(self, node, levenshtein_distance(str(self.object), str(node.object)))
        elif isinstance(node, ContainerNode):
            return Replace(self, node)

    def print(self, printer: Printer):
        printer.write(repr(self.object))

    def __lt__(self, other):
        if isinstance(other, LeafNode):
            try:
                return self.object < other.object
            except TypeError:
                return str(self.object) < str(other.object)
        else:
            try:
                return self.object < other
            except TypeError:
                return str(self.object) < str(other)

    def __eq__(self, other):
        if isinstance(other, LeafNode):
            return self.object == other.object
        else:
            return self.object == other

    def __hash__(self):
        return hash(self.object)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.object!r})"

    def __str__(self):
        return str(self.object)


class KeyValuePairEdit(AbstractCompoundEdit):
    def __init__(
            self,
            from_kvp: 'KeyValuePairNode',
            to_kvp: 'KeyValuePairNode'
    ):
        if from_kvp.key == to_kvp.key:
            self.key_edit: Edit = Match(from_kvp.key, to_kvp.key, 0)
        elif from_kvp.allow_key_edits:
            self.key_edit: Edit = from_kvp.key.edits(to_kvp.key)
        else:
            raise ValueError("Keys must match!")
        if from_kvp.value == to_kvp.value:
            self.value_edit: Edit = Match(from_kvp.value, to_kvp.value, 0)
        else:
            self.value_edit: Edit = from_kvp.value.edits(to_kvp.value)
        super().__init__(
            from_node=from_kvp,
            to_node=to_kvp
        )

    def print(self, formatter: GraphtageFormatter, printer: Printer):
        # Raise NotImplementedError() to cause the formatter to fall back on its own implementations
        raise NotImplementedError()

    def bounds(self) -> Range:
        return self.key_edit.bounds() + self.value_edit.bounds()

    def tighten_bounds(self) -> bool:
        return self.key_edit.tighten_bounds() or self.value_edit.tighten_bounds()

    def edits(self) -> Iterator[Edit]:
        yield self.key_edit
        yield self.value_edit


class KeyValuePairNode(ContainerNode):
    def __init__(self, key: LeafNode, value: TreeNode, allow_key_edits: bool = True):
        self.key: LeafNode = key
        self.value: TreeNode = value
        self.allow_key_edits: bool = allow_key_edits

    def to_obj(self):
        return self.key, self.value

    def editable_dict(self) -> Dict[str, Any]:
        ret = dict(self.__dict__)
        ret['key'] = self.key.make_edited()
        ret['value'] = self.value.make_edited()
        return ret

    def children(self) -> Tuple[LeafNode, TreeNode]:
        return self.key, self.value

    def edits(self, node: TreeNode) -> Edit:
        if not isinstance(node, KeyValuePairNode):
            raise RuntimeError("KeyValuePairNode.edits() should only ever be called with another KeyValuePair object!")
        if self.allow_key_edits or self.key == node.key:
            return KeyValuePairEdit(self, node)
        else:
            return Replace(self, node)

    def print(self, printer: Printer):
        if isinstance(self.key, LeafNode):
            with printer.color(Fore.BLUE):
                self.key.print(printer)
        else:
            self.key.print(printer)
        with printer.bright():
            printer.write(": ")
        self.value.print(printer)

    def calculate_total_size(self):
        return self.key.total_size + self.value.total_size

    def __lt__(self, other):
        if not isinstance(other, KeyValuePairNode):
            return self.key < other
        return (self.key < other.key) or (self.key == other.key and self.value < other.value)

    def __eq__(self, other):
        if not isinstance(other, KeyValuePairNode):
            return False
        return self.key == other.key and self.value == other.value

    def __hash__(self):
        return hash((self.key, self.value))

    def __len__(self):
        return 2

    def __iter__(self):
        yield self.key
        yield self.value

    def __repr__(self):
        return f"{self.__class__.__name__}(key={self.key!r}, value={self.value!r})"

    def __str__(self):
        return f"{self.key!s}: {self.value!s}"


T = TypeVar('T', bound=TreeNode)


class ListNode(SequenceNode[Tuple[T, ...]], Generic[T]):
    def __init__(self, nodes: Iterable[T]):
        super().__init__(tuple(nodes))

    def to_obj(self):
        return [n.to_obj() for n in self]

    @property
    def container_type(self) -> Type[Tuple[T, ...]]:
        return tuple

    def edits(self, node: TreeNode) -> Edit:
        if isinstance(node, ListNode):
            if len(self._children) == len(node._children) == 0:
                return Match(self, node, 0)
            elif len(self._children) == len(node._children) == 1:
                return EditSequence(from_node=self, to_node=node, edits=iter((
                    Match(self, node, 0),
                    self._children[0].edits(node._children[0])
                )))
            elif self._children == node._children:
                return Match(self, node, 0)
            else:
                if self.all_children_are_leaves() and node.all_children_are_leaves():
                    insert_remove_penalty = 0
                else:
                    insert_remove_penalty = 1
                return EditDistance(
                    self,
                    node,
                    self._children,
                    node._children,
                    insert_remove_penalty=insert_remove_penalty
                )
        else:
            return Replace(self, node)


class MultiSetNode(SequenceNode[HashableCounter[T]], Generic[T]):
    def __init__(self, items: Iterable[T]):
        if not isinstance(items, HashableCounter):
            items = HashableCounter(items)
        super().__init__(items)

    def to_obj(self):
        return HashableCounter(n.to_obj() for n in self)

    @property
    def container_type(self) -> Type[HashableCounter[T]]:
        return HashableCounter

    def edits(self, node: TreeNode) -> Edit:
        if isinstance(node, MultiSetNode):
            if len(self._children) == len(node._children) == 0:
                return Match(self, node, 0)
            elif self._children == node._children:
                return Match(self, node, 0)
            else:
                return MultiSetEdit(self, node, self._children, node._children)
        else:
            return Replace(self, node)

    def calculate_total_size(self):
        return sum(c.total_size * count for c, count in self._children.items())

    def __len__(self):
        return sum(self._children.values())

    def __iter__(self) -> Iterator[TreeNode]:
        return self._children.elements()

    def __repr__(self):
        return f"{self.__class__.__name__}({list(self)!r})"


class MappingNode(ContainerNode, ABC):
    def to_obj(self) -> Dict[Any, Any]:
        return {
            k.to_obj(): v.to_obj() for k, v in self.items()
        }

    def items(self) -> Iterator[Tuple[TreeNode, TreeNode]]:
        for kvp in self:
            yield kvp.key, kvp.value

    def __contains__(self, item: TreeNode):
        return any(k == item for k, _ in self.items())

    def __getitem__(self, item: TreeNode) -> KeyValuePairNode:
        for kvp in self:
            if kvp.key == item:
                return kvp
        raise KeyError(item)

    @abstractmethod
    def __iter__(self) -> Iterator[KeyValuePairNode]:
        raise NotImplementedError()


class DictNode(MappingNode, MultiSetNode[KeyValuePairNode]):
    @staticmethod
    def from_dict(source_dict: Dict[LeafNode, TreeNode]) -> 'DictNode':
        return DictNode(
            sorted(KeyValuePairNode(key, value, allow_key_edits=True) for key, value in source_dict.items())
        )

    def edits(self, node: TreeNode) -> Edit:
        if isinstance(node, MultiSetNode):
            return super().edits(node)
        else:
            return Replace(self, node)

    def __iter__(self) -> Iterator[KeyValuePairNode]:
        yield from self._children.elements()


class FixedKeyDictNodeEdit(SequenceEdit, EditCollection[List]):
    def __init__(
            self,
            from_node: 'FixedKeyDictNode',
            to_node: TreeNode,
            edits: Iterator[Edit]
    ):
        super().__init__(
            from_node=from_node,
            to_node=to_node,
            edits=edits,
            collection=list,
            add_to_collection=list.append,
            explode_edits=False
        )


class FixedKeyDictNode(MappingNode, SequenceNode[Dict[LeafNode, KeyValuePairNode]]):
    """A dictionary that only matches KeyValuePairs if they share the same key
    NOTE: This implementation does not currently support duplicate keys!
    """
    @property
    def container_type(self) -> Type[Dict[LeafNode, KeyValuePairNode]]:
        return dict

    @staticmethod
    def from_dict(source_dict: Dict[LeafNode, TreeNode]) -> 'FixedKeyDictNode':
        return FixedKeyDictNode({
            kvp.key: kvp
            for kvp in (KeyValuePairNode(key, value, allow_key_edits=False) for key, value in source_dict.items())
        })

    def __getitem__(self, item: TreeNode):
        return self._children[item]

    def __contains__(self, item: TreeNode):
        return item in self._children

    def _child_edits(self, node: MappingNode) -> Iterator[Edit]:
        unshared_kvps = set()
        for key, kvp in self._children.items():
            if key in node:
                other_kvp = node[key]
                if kvp == other_kvp:
                    yield Match(kvp, other_kvp, 0)
                else:
                    yield KeyValuePairEdit(kvp, other_kvp)
            else:
                unshared_kvps.add(kvp)
        for kvp in unshared_kvps:
            yield Remove(to_remove=kvp, remove_from=self)
        for kvp in node:
            if kvp.key not in self._children:
                yield Insert(to_insert=kvp, insert_into=self)

    def edits(self, node: TreeNode) -> Edit:
        if isinstance(node, MappingNode):
            if len(self._children) == len(node) == 0:
                return Match(self, node, 0)
            elif frozenset(self) == frozenset(node):
                return Match(self, node, 0)
            else:
                return FixedKeyDictNodeEdit(from_node=self, to_node=node, edits=self._child_edits(node))
        else:
            return Replace(self, node)

    def items(self) -> Iterator[Tuple[LeafNode, TreeNode]]:
        for k, v in self._children.items():
            yield k.to_obj(), v.to_obj()

    def editable_dict(self) -> Dict[str, Any]:
        ret = dict(self.__dict__)
        ret['_children'] = {e.key: e for e in (kvp.make_edited() for kvp in self)}
        return ret

    def __hash__(self):
        return hash(frozenset(self._children.values()))

    def __iter__(self) -> Iterator[KeyValuePairNode]:
        return iter(self._children.values())


class StringEdit(AbstractEdit):
    def __init__(
            self,
            from_node: 'StringNode',
            to_node: 'StringNode'
    ):
        self.edit_distance = string_edit_distance(from_node.object, to_node.object)
        super().__init__(
            from_node=from_node,
            to_node=to_node
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(from_node={self.from_node!r}, to_node={self.to_node!r})"

    def bounds(self) -> Range:
        return self.edit_distance.bounds()

    def tighten_bounds(self) -> bool:
        return self.edit_distance.tighten_bounds()

    def print(self, formatter: GraphtageFormatter, printer: Printer):
        raise NotImplementedError()


class StringFormatter(GraphtageFormatter):
    is_quoted = False

    def write_start_quote(self, printer: Printer, edit: StringEdit):
        if edit.from_node.quoted:
            self.is_quoted = True
            printer.write('"')
        else:
            self.is_quoted = False

    def write_end_quote(self, printer: Printer, edit: StringEdit):
        if edit.from_node.quoted:
            self.is_quoted = True
            printer.write('"')
        else:
            self.is_quoted = False

    def escape(self, c: str):
        if self.is_quoted:
            return c.replace('"', '\\"')
        else:
            return c

    def write_char(self, printer: Printer, c: str, index: int, num_edits: int, removed=False, inserted=False):
        escaped = self.escape(c)
        if escaped != c and not isinstance(printer, NullANSIContext):
            with printer.color(Fore.YELLOW):
                printer.write(escaped)
        else:
            printer.write(c)

    def context(self, printer: Printer):
        if printer.context().fore is None:
            return printer.color(Fore.GREEN)
        else:
            return printer

    def print_StringNode(self, printer: Printer, node: 'StringNode'):
        with self.context(printer) as p:
            self.write_start_quote(p, StringEdit(node, node))
            num_edits = len(node.object)
            for i, c in enumerate(node.object):
                self.write_char(p, c, i, num_edits)
            self.write_end_quote(p, StringEdit(node, node))

    def print_StringEdit(self, printer: Printer, edit: StringEdit):
        with self.context(printer) as p:
            self.write_start_quote(p, edit)
            remove_seq = []
            add_seq = []
            index = 0
            edits = list(edit.edit_distance.edits())
            num_edits = len(edits)
            for sub_edit in edits:
                to_remove = None
                to_add = None
                matched = None
                if isinstance(sub_edit, Match):
                    if sub_edit.from_node.object == sub_edit.to_node.object:
                        matched = sub_edit.from_node.object
                    else:
                        to_remove = sub_edit.from_node.object
                        to_add = sub_edit.to_node.object
                elif isinstance(sub_edit, Remove):
                    to_remove = sub_edit.from_node.object
                else:
                    assert isinstance(sub_edit, Insert)
                    to_add = sub_edit.to_insert.object
                if to_remove is not None and to_add is not None:
                    assert matched is None
                    remove_seq.append(to_remove)
                    add_seq.append(to_add)
                else:
                    with printer.color(Fore.WHITE).background(Back.RED).bright():
                        with printer.strike():
                            for rm in remove_seq:
                                self.write_char(p, rm, index, num_edits, removed=True)
                                index += 1
                    remove_seq = []
                    with printer.color(Fore.WHITE).background(Back.GREEN).bright():
                        with printer.under_plus():
                            for add in add_seq:
                                self.write_char(p, add, index, num_edits, inserted=True)
                                index += 1
                    add_seq = []
                    if to_remove is not None:
                        remove_seq.append(to_remove)
                    if to_add is not None:
                        add_seq.append(to_add)
                    if matched is not None:
                        self.write_char(p, matched, index, num_edits)
                        index += 1
            with printer.color(Fore.WHITE).background(Back.RED).bright():
                with printer.strike():
                    for j, rm in enumerate(remove_seq):
                        self.write_char(p, rm, index, num_edits, removed=True)
                        index += 1
            with printer.color(Fore.WHITE).background(Back.GREEN).bright():
                with printer.under_plus():
                    for add in add_seq:
                        self.write_char(p, add, index, num_edits, inserted=True)
                        index += 1
            self.write_end_quote(p, edit)


class StringNode(LeafNode):
    def __init__(self, string_like: str, quoted=True):
        super().__init__(string_like)
        self.quoted = quoted

    def edits(self, node: TreeNode) -> Edit:
        if isinstance(node, StringNode):
            if self.object == node.object:
                return Match(self, node, 0)
            elif len(self.object) == 1 and len(node.object) == 1:
                return Match(self, node, 1)
            return StringEdit(self, node)
        else:
            return super().edits(node)

    def print(self, printer: Printer):
        StringFormatter.DEFAULT_INSTANCE.print(printer, self)

    def init_args(self) -> Dict[str, Any]:
        return {
            'string_like': self.object
        }


class IntegerNode(LeafNode):
    def __init__(self, int_like: int):
        super().__init__(int_like)

    def init_args(self) -> Dict[str, Any]:
        return {
            'int_like': self.object
        }


class FloatNode(LeafNode):
    def __init__(self, float_like: float):
        super().__init__(float_like)

    def init_args(self) -> Dict[str, Any]:
        return {
            'float_like': self.object
        }


class BoolNode(LeafNode):
    def __init__(self, bool_like: bool):
        super().__init__(bool_like)

    def init_args(self) -> Dict[str, Any]:
        return {
            'bool_like': self.object
        }


def string_edit_distance(s1: str, s2: str) -> EditDistance:
    list1 = ListNode([StringNode(c) for c in s1])
    list2 = ListNode([StringNode(c) for c in s2])
    return EditDistance(list1, list2, list1.children(), list2.children(), insert_remove_penalty=0)


FILETYPES_BY_MIME: Dict[str, 'Filetype'] = {}
FILETYPES_BY_TYPENAME: Dict[str, 'Filetype'] = {}


class FiletypeWatcher(ABCMeta):
    def __init__(cls, name, bases, clsdict):
        if len(cls.mro()) > 2:
            # Instantiate a version of the filetype to add it to our global dicts:
            instance = cls()
            assert instance.name in FILETYPES_BY_TYPENAME
            assert instance.default_mimetype in FILETYPES_BY_MIME
        super().__init__(name, bases, clsdict)


class Filetype(metaclass=FiletypeWatcher):
    def __init__(self, type_name: str, default_mimetype: str, *mimetypes: str):
        self.name = type_name
        self.default_mimetype: str = default_mimetype
        self.mimetypes: Tuple[str, ...] = (default_mimetype,) + tuple(mimetypes)
        for mime_type in self.mimetypes:
            if mime_type in FILETYPES_BY_MIME:
                raise ValueError(f"MIME type {mime_type} is already assigned to {FILETYPES_BY_MIME[mime_type]}")
            FILETYPES_BY_MIME[mime_type] = self
        FILETYPES_BY_MIME[default_mimetype] = self
        if type_name in FILETYPES_BY_TYPENAME:
            raise ValueError(
                f'Type {type_name} is already associated with Filetype {FILETYPES_BY_TYPENAME[type_name]}')
        FILETYPES_BY_TYPENAME[self.name] = self

    @abstractmethod
    def build_tree(self, path: str, allow_key_edits: bool = True) -> TreeNode:
        raise NotImplementedError()

    @abstractmethod
    def build_tree_handling_errors(self, path: str, allow_key_edits: bool = True) -> TreeNode:
        raise NotImplementedError()

    @abstractmethod
    def get_default_formatter(self) -> GraphtageFormatter:
        raise NotImplementedError()


def get_filetype(path: Optional[str] = None, mime_type: Optional[str] = None) -> Filetype:
    if path is None and mime_type is None:
        raise ValueError("get_filetype requires a path or a MIME type")
    elif mime_type is None:
        mime_type = mimetypes.guess_type(path)[0]
    if mime_type is None:
        raise ValueError(f"Could not determine the filetype for {path}")
    elif mime_type not in FILETYPES_BY_MIME:
        raise ValueError(f"Unsupported MIME type {mime_type} for {path}")
    else:
        return FILETYPES_BY_MIME[mime_type]
