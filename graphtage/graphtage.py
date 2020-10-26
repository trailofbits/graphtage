__docformat__ = "google"

import mimetypes
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Collection, Dict, Generic, Iterable, Iterator, List, Optional, Tuple, Type, TypeVar, Union

from .bounds import Range
from .edits import AbstractEdit, EditCollection
from .edits import Insert, Match, Remove, Replace, AbstractCompoundEdit
from .levenshtein import EditDistance, levenshtein_distance
from .multiset import MultiSetEdit
from .printer import Back, Fore, NullANSIContext, Printer
from .sequences import FixedLengthSequenceEdit, SequenceEdit, SequenceNode
from .tree import ContainerNode, Edit, GraphtageFormatter, TreeNode
from .utils import HashableCounter


class LeafNode(TreeNode):
    """Abstract class for nodes that have no children."""

    def __init__(self, obj):
        """Creates a node with the given object.

        Args:
            obj: The underlying Python object wrapped by the node.

        """
        self.object = obj

    def to_obj(self):
        """Returns the object wrapped by this node.

        This is equivalent to::

            return self.object

        """
        return self.object

    def children(self) -> Collection[TreeNode]:
        """Leaf nodes have no children, so this always returns an empty tuple.

        Returns:
            tuple: An empty tuple.

        """
        return ()

    def calculate_total_size(self) -> int:
        """By default, leaf nodes' sizes are equal to the length of their wrapped object's string representation.

        This is equivalent to::

            return len(str(self.object))

        However, subclasses may override this function to return whatever size is required.

        Returns:
            int: The length of the string representation of :obj:`self.object`.

        """
        return len(str(self.object))

    def edits(self, node: TreeNode) -> Edit:
        if isinstance(node, LeafNode):
            return Match(self, node, levenshtein_distance(str(self.object), str(node.object)))
        elif isinstance(node, ContainerNode):
            return Replace(self, node)

    def print(self, printer: Printer):
        """Prints this leaf node.

        By default, leaf nodes print the :func:`repr` of their wrapped object::

            printer.write(repr(self.object))

        Args:
            printer: The printer to which to write this node.

        """
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
    """An edit type for two key/value pairs"""

    def __init__(
            self,
            from_kvp: 'KeyValuePairNode',
            to_kvp: 'KeyValuePairNode'
    ):
        """Creates a key/value pair edit.

        Args:
            from_kvp: The key/value pair from which to match.
            to_kvp: The key/value pair to which to match.

        Raises:
            ValueError: If :meth:`from_kvp.allow_key_edits<KeyValuePairNode.__init__>` and the keys do not match.

        """
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
    """A node containing a key/value pair.

    This is used by nodes subclassing :class:`MappingNode`.

    """

    def __init__(self, key: LeafNode, value: TreeNode, allow_key_edits: bool = True):
        """Creates a new key/value pair node.

        Args:
            key: The key of the pair.
            value: The value of the pair.
            allow_key_edits: If :const:`False`, only consider matching against another key/value pair node if it has
                the same key.

        """
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
        """Prints this node.

        This default implementation prints the key in blue, followed by a bright white ": ", followed by the value.

        """
        if isinstance(self.key, LeafNode):
            with printer.color(Fore.BLUE):
                self.key.print(printer)
        else:
            self.key.print(printer)
        with printer.bright():
            printer.write(": ")
        self.value.print(printer)

    def calculate_total_size(self):
        return self.key.total_size + self.value.total_size + 2

    def __lt__(self, other):
        """ Compares this key/value pair to another.

        If :obj:`other` is also an instance of :class:`KeyValuePairNode`, return::

            (self.key < other.key) or (self.key == other.key and self.value < other.value)

        otherwise, return::

            self.key < other

        Args:
            other: The object to which to compare.

        Returns:
            bool: :const:`True` if this key/value pair is smaller than :obj:`other`.

        """
        if not isinstance(other, KeyValuePairNode):
            return self.key < other
        return (self.key < other.key) or (self.key == other.key and self.value < other.value)

    def __eq__(self, other):
        """Tests whether this key/value pair equals another.

        Equivalent to::

            isinstance(other, KeyValuePair) and self.key == other.key and self.value == other.value

        Args:
            other: The object to test.

        Returns:
            bool: :const:`True` if this key/value pair is equal to :obj:`other`.

        """
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
    """A node containing an ordered sequence of nodes."""

    def __init__(
            self, nodes: Iterable[T],
            allow_list_edits: bool = True,
            allow_list_edits_when_same_length: bool = True
    ):
        """Initializes a List node.

        Args:
            nodes: The set of nodes in this list.
            allow_list_edits: Whether to consider removal and insertion when editing this list.
            allow_list_edits_when_same_length: Whether to consider removal and insertion when comparing this list to
                another list of the same length.
        """
        super().__init__(tuple(nodes))
        self.allow_list_edits: bool = allow_list_edits
        self.allow_list_edits_when_same_length: bool = allow_list_edits_when_same_length

    def to_obj(self):
        return [n.to_obj() for n in self]

    @property
    def container_type(self) -> Type[Tuple[T, ...]]:
        """The container type required by :class:`graphtage.sequences.SequenceNode`

        Returns:
            Type[Tuple[T, ...]]: :class:`tuple`

        """
        return tuple

    def edits(self, node: TreeNode) -> Edit:
        if isinstance(node, ListNode):
            if self._children == node._children:
                return Match(self, node, 0)
            elif not self.allow_list_edits or (len(self._children) == len(node._children) and (
                not self.allow_list_edits_when_same_length or len(self._children) == 1
            )):
                return FixedLengthSequenceEdit(
                    from_node=self,
                    to_node=node
                )
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
    """A node representing a set that can contain duplicate items."""

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
        return sum((c.total_size + 1) * count for c, count in self._children.items())

    def __len__(self):
        return sum(self._children.values())

    def __iter__(self) -> Iterator[TreeNode]:
        return self._children.elements()

    def __repr__(self):
        return f"{self.__class__.__name__}({list(self)!r})"


class MappingNode(ContainerNode, ABC):
    """An abstract base class for nodes that represent mappings."""

    def to_obj(self) -> Dict[Any, Any]:
        return {
            k.to_obj(): v.to_obj() for k, v in self.items()
        }

    def items(self) -> Iterator[Tuple[TreeNode, TreeNode]]:
        """Iterates over the key/value pairs in this mapping, similar to :meth:`dict.items`.

        The implementation is equivalent to::

            for kvp in self:
                yield kvp.key, kvp.value

        since :meth:`MappingNode.__iter__` returns an iterator over :class:`graphtage.KeyValuePairNode`.

        """
        for kvp in self:
            yield kvp.key, kvp.value

    def __contains__(self, item: TreeNode):
        """Tests whether the given item is a key in this mapping.

        The implementation is equivalent to::

            return any(k == item for k, _ in self.items())

        Note:
            This implementation runs in worst-case linear time in the size of the mapping.

        Args:
            item: The key of the item sought.

        Returns:
            bool: :const:`True` if the key exists in this mapping.

        """
        return any(k == item for k, _ in self.items())

    def __getitem__(self, item: TreeNode) -> KeyValuePairNode:
        """Looks up a key/value pair item from this mapping by its key.

        The implementation is equivalent to::

            for kvp in self:
                if kvp.key == item:
                    return kvp
            raise KeyError(item)

        Note:
            This implementation runs in worst-case linear time in the size of the mapping.

        Args:
            item: The key of the key/value pair that is sought.

        Returns:
            KeyValuePairNode: The first key/value pair found with key :obj:`item`.

        Raises:
            KeyError: If the key is not found.

        """
        for kvp in self:
            if kvp.key == item:
                return kvp
        raise KeyError(item)

    @abstractmethod
    def __iter__(self) -> Iterator[KeyValuePairNode]:
        """Mappings should return an iterator over :class:`graphtage.KeyValuePairNode`."""
        raise NotImplementedError()


class DictNode(MappingNode, MultiSetNode[KeyValuePairNode]):
    """A dictionary node implemented as a multiset of key/value pairs.

    This is the default dictionary type used by Graphtage. Unlike its more efficient alternative,
    :class:`FixedKeyDictNode`, this class supports matching dictionaries with duplicate keys.

    """

    @staticmethod
    def from_dict(source_dict: Dict[LeafNode, TreeNode]) -> 'DictNode':
        """Constructs a :class:`DictNode` from a mapping of :class:`LeafNode` to :class:`TreeNode`.

        Args:
            source_dict: The source mapping.

        Returns:
            DictNode: The resulting :class:`DictNode`.

        """
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
    """The edit type returned by :class:`FixedKeyDictNode`."""
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
    """A dictionary that only attempts to match two :class:`KeyValuePairNode` objects if they share the same key.

    This is the most efficient dictionary matching node type, and is what is used with the
    :obj:`--no-key-edits`/:obj:`-k` command line argument.

    Note:
        This implementation does not currently support duplicate keys.
    """
    @property
    def container_type(self) -> Type[Dict[LeafNode, KeyValuePairNode]]:
        """The container type required by :class:`graphtage.sequences.SequenceNode`

        Returns:
            Type[Dict[LeafNode, KeyValuePairNode]]: :class:`dict`

        """
        return dict

    @staticmethod
    def from_dict(source_dict: Dict[LeafNode, TreeNode]) -> 'FixedKeyDictNode':
        """Constructs a :class:`FixedKeyDictNode` from a mapping of :class:`LeafNode` to :class:`TreeNode`.

        Args:
            source_dict: The source mapping.

        Note:
            This implementation does not currently check for duplicate keys. Only the first key returned from
            `source_dict.items()` will be included in the output.

        Returns:
            FixedKeyDictNode: The resulting :class:`FixedKeyDictNode`

        """
        return FixedKeyDictNode({
            kvp.key: kvp
            for kvp in (KeyValuePairNode(key, value, allow_key_edits=False) for key, value in source_dict.items())
        })

    def __getitem__(self, item: LeafNode):
        return self._children[item]

    def __contains__(self, item: LeafNode):
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
        yield from iter(self._children.items())

    def editable_dict(self) -> Dict[str, Any]:
        ret = dict(self.__dict__)
        ret['_children'] = {e.key: e for e in (kvp.make_edited() for kvp in self)}
        return ret

    def __hash__(self):
        return hash(frozenset(self._children.values()))

    def __iter__(self) -> Iterator[KeyValuePairNode]:
        return iter(self._children.values())


class StringEdit(AbstractEdit):
    """An edit returned from a :class:`StringNode`"""

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
        """`StringEdit` does not implement :meth:`graphtage.tree.Edit.print`.

        Instead, it raises :exc:`NotImplementedError` so that the formatting protocol will default to using a
        formatter like :class:`StringFormatter`.

        """
        raise NotImplementedError()


class StringFormatter(GraphtageFormatter):
    """A default string formatter"""
    is_quoted: bool = False
    _last_was_inserted: bool = False
    _last_was_removed: bool = False

    def write_start_quote(self, printer: Printer, edit: StringEdit):
        """Prints a starting quote for the string, if necessary"""
        if edit.from_node.quoted:
            self.is_quoted = True
            printer.write('"')
        else:
            self.is_quoted = False

    def write_end_quote(self, printer: Printer, edit: StringEdit):
        """Prints an ending quote for the string, if necessary"""
        if edit.from_node.quoted:
            self.is_quoted = True
            printer.write('"')
        else:
            self.is_quoted = False

    def escape(self, c: str) -> str:
        """String escape.

        This function is called once for each character in the string.

        Returns:
            str: The escaped version of `c`, or `c` itself if no escaping is required.

        """
        if self.is_quoted:
            return c.replace('"', '\\"')
        else:
            return c

    def write_char(self, printer: Printer, c: str, index: int, num_edits: int, removed=False, inserted=False):
        """Writes a character to the printer.

        Note:
            This function calls :meth:`graphtage.StringFormatter.escape`; classes extending
            :class:`graphtage.StringFormatter` should also call :meth:`graphtage.StringFormatter.escape` when
            reimplementing this function.

        Note:
            There is no need to specially format characters that have been removed or inserted; the printer will have
            already automatically been configured to format them prior to the call to
            :meth:`StringFormatter.write_char`.

        Args:
            printer: The printer to which to write the character.
            c: The character to write.
            index: The index of the character in the string.
            num_edits: The total number of characters that will be printed.
            removed: Whether this character was removed from the source string.
            inserted: Whether this character is inserted into the source string.

        """
        if not printer.ansi_color:
            if self._last_was_inserted and not inserted:
                printer.write(Insert.INSERT_STRING)
                if removed:
                    printer.write(Remove.REMOVE_STRING)
            elif self._last_was_removed and not removed:
                printer.write(Remove.REMOVE_STRING)
                if inserted:
                    printer.write(Insert.INSERT_STRING)
            elif removed and not self._last_was_removed:
                printer.write(Remove.REMOVE_STRING)
            elif inserted and not self._last_was_inserted:
                printer.write(Insert.INSERT_STRING)
            self._last_was_removed = removed
            self._last_was_inserted = inserted
        escaped = self.escape(c)
        if escaped != c and not isinstance(printer, NullANSIContext):
            with printer.color(Fore.YELLOW):
                printer.write(escaped)
        else:
            printer.write(escaped)

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
        self._last_was_inserted = False
        self._last_was_removed = False
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
            if self._last_was_inserted:
                printer.write(Insert.INSERT_STRING)
                self._last_was_inserted = False
            elif self._last_was_removed:
                printer.write(Remove.REMOVE_STRING)
                self._last_was_removed = False
            self.write_end_quote(p, edit)


class StringNode(LeafNode):
    """A node containing a string"""

    def __init__(self, string_like: str, quoted=True):
        """Initializes a string node.

        Args:
            string_like: the string contained by the node
            quoted: whether or not the string should be quoted when being formatted

        """
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


class IntegerNode(LeafNode):
    """A node containing an int"""

    def __init__(self, int_like: int):
        super().__init__(int_like)


class FloatNode(LeafNode):
    """A node containing a float"""

    def __init__(self, float_like: float):
        super().__init__(float_like)


class BoolNode(LeafNode):
    """A node containing either :const:`True` or :const:`False`."""

    def __init__(self, bool_like: bool):
        super().__init__(bool_like)


class NullNode(LeafNode):
    """A node representing a null or :const:`None` type."""

    def __init__(self):
        super().__init__(None)

    def calculate_total_size(self) -> int:
        return 0

    def edits(self, node: TreeNode) -> Edit:
        if isinstance(node, NullNode):
            return Match(self, node, 0)
        else:
            return Replace(self, node)

    def __lt__(self, other):
        if isinstance(other, NullNode):
            return False
        else:
            return True

    def __eq__(self, other):
        return isinstance(other, NullNode)

    def __hash__(self):
        return hash(self.object)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


def string_edit_distance(s1: str, s2: str) -> EditDistance:
    """A convenience function for computing the edit distance between two strings.

    This is equivalent to::

        list1 = ListNode([StringNode(c) for c in s1])
        list2 = ListNode([StringNode(c) for c in s2])
        return EditDistance(list1, list2, list1.children(), list2.children(), insert_remove_penalty=0)

    Args:
        s1: the string to compare from
        s2: the string to compare to

    Returns:
        EditDistance: The :class:`graphtage.levenshtein.EditDistance` edit object for the strings.

    """
    list1 = ListNode([StringNode(c) for c in s1])
    list2 = ListNode([StringNode(c) for c in s2])
    return EditDistance(list1, list2, list1.children(), list2.children(), insert_remove_penalty=0)


FILETYPES_BY_MIME: Dict[str, 'Filetype'] = {}
FILETYPES_BY_TYPENAME: Dict[str, 'Filetype'] = {}


class FiletypeWatcher(ABCMeta):
    """Abstract metaclass for the :class:`Filetype` class.

    This registers any subclasses of :class:`Filetype`, automatically adding them to the `graphtage` command line
    arguments and mimetype lookup in :func:`get_filetype`.

    """
    def __init__(cls, name, bases, clsdict):
        if len(cls.mro()) > 2:
            # Instantiate a version of the filetype to add it to our global dicts:
            instance = cls()
            assert instance.name in FILETYPES_BY_TYPENAME
            assert instance.default_mimetype in FILETYPES_BY_MIME
        super().__init__(name, bases, clsdict)


class BuildOptions:
    """A class for passing options to tree building functions in :class:`Filetype`"""

    def __init__(self, *,
                 allow_key_edits=True,
                 allow_list_edits=True,
                 allow_list_edits_when_same_length=True,
                 **kwargs
                 ):
        """Initializes the options. All keyword values will be set as attributes of this class.

        Options not specified will default to :const:`False`.

        """
        self.allow_key_edits = allow_key_edits
        """Whether to consider editing keys when matching :class:`KeyValuePairNode` objects"""
        self.allow_list_edits = allow_list_edits
        """Whether to consider insert and remove edits to lists"""
        self.allow_list_edits_when_same_length = allow_list_edits_when_same_length
        """Whether to consider insert and remove edits on lists that are the same length"""
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    def __getattr__(self, item):
        """Default all undefined options to :const:`False`"""
        return False


class Filetype(metaclass=FiletypeWatcher):
    """Abstract base class from which all Graphtage file formats should extend.

    When this class is subclassed, the subclass will automatically be added to Graphtage's filetype registry.
    This includes automatic inclusion in :mod:`graphtage`'s command line arguments and mime type auto-detection in
    :func:`get_filetype`.

    """

    def __init__(self, type_name: str, default_mimetype: str, *mimetypes: str):
        """Initializes a new Graphtage file format type

        Args:
            type_name: A short name for the :class:`Filetype`. This will be used for specifying this :class:`Filetype`
                via command line arguments.
            default_mimetype: The default mimetype to be assigned to this :class:`Filetype`.
            *mimetypes: Zero or more additional mimetypes that should be associated with this :class:`Filetype`.

        Raises:
            ValueError: The :obj:`type_name` and/or one of the mimetypes of this :class:`Filetype` conflicts with that
                of a preexisting :class:`Filetype`.

        """
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
    def build_tree(self, path: str, options: Optional[BuildOptions] = None) -> TreeNode:
        """Builds an intermediate representation tree from a file of this :class:`Filetype`.

        Args:
            path: Path to the file to parse
            options: An optional set of options for building the tree

        Returns:
            TreeNode: The root tree node of the provided file

        """
        raise NotImplementedError()

    @abstractmethod
    def build_tree_handling_errors(self, path: str, options: Optional[BuildOptions] = None) -> Union[str, TreeNode]:
        """Same as :meth:`Filetype.build_tree`, but it should return a human-readable error string on failure.

        This function should never throw an exception.

        Args:
            path: Path to the file to parse
            options: An optional set of options for building the tree

        Returns:
            Union[str, TreeNode]: On success, the root tree node, or a string containing the error message on failure.

        """
        raise NotImplementedError()

    @abstractmethod
    def get_default_formatter(self) -> GraphtageFormatter:
        """Returns the default formatter for printing files of this type."""
        raise NotImplementedError()


def get_filetype(path: Optional[str] = None, mime_type: Optional[str] = None) -> Filetype:
    """Looks up the filetype for the given path.

    At least one of :obj:`path` or :obj:`mime_type` must be not :const:`None`. If both are provided, only
    :obj:`mime_type` will be used. If only :obj:`path` is provided, its mimetype will be guessed using
    :func:`mimetypes.guess_type`.

    Args:
        path: An optional path to a file.
        mime_type: An optional MIME type string.

    Returns:
        Filetype: The filetype object associated with the given path and/or MIME type.

    Raises:
        ValueError: If both :obj:`path` and :obj:`mime_type` are :const:`None`.
        ValueError: If :obj:`mime_type` was not provided and :func:`mimetypes.guess_type` could not identify the file at
            :obj:`path`.
        ValueError: If either the provided or guessed mimetype is not supported by any registered
            :class:`Filetype`.
    """
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
