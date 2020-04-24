from typing import Any, cast, Dict, Iterable, Iterator, List, Sequence, Tuple, Union

from .bounds import Range
from .edits import AbstractEdit, EditCollection, EditSequence
from .edits import Insert, Match, Remove, Replace, AbstractCompoundEdit
from .levenshtein import EditDistance, levenshtein_distance
from .multiset import MultiSetEdit
from .printer import Back, Fore, NullANSIContext, Printer
from .sequences import SequenceEdit, SequenceNode
from .tree import ContainerNode, Edit, EditedTreeNode, TreeNode
from .utils import HashableCounter


class LeafNode(TreeNode):
    def __init__(self, obj):
        self.object = obj

    def calculate_total_size(self):
        return len(str(self.object))

    def edits(self, node: TreeNode) -> Edit:
        if isinstance(node, LeafNode):
            return Match(self, node, levenshtein_distance(str(self.object), str(node.object)))
        elif isinstance(node, ContainerNode):
            return Replace(self, node)

    def init_args(self) -> Dict[str, Any]:
        return {
            'obj': self.object
        }

    def print(self, printer: Printer):
        printer.write(repr(self.object))

    def __lt__(self, other):
        if isinstance(other, LeafNode):
            return self.object < other.object
        else:
            return self.object < other

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

    def bounds(self) -> Range:
        return self.key_edit.bounds() + self.value_edit.bounds()

    def tighten_bounds(self) -> bool:
        return self.key_edit.tighten_bounds() or self.value_edit.tighten_bounds()

    def edits(self) -> Iterator[Edit]:
        yield self.key_edit
        yield self.value_edit

    def print(self, printer: Printer):
        with printer.color(Fore.BLUE):
            self.key_edit.print(printer)
        with printer.bright():
            printer.write(": ")
        self.value_edit.print(printer)


class KeyValuePairNode(ContainerNode):
    def __init__(self, key: LeafNode, value: TreeNode, allow_key_edits: bool = True):
        self.key: LeafNode = key
        self.value: TreeNode = value
        self.allow_key_edits: bool = allow_key_edits

    def edits(self, node: TreeNode) -> Edit:
        if not isinstance(node, KeyValuePairNode):
            raise RuntimeError("KeyValuePairNode.edits() should only ever be called with another KeyValuePair object!")
        if self.allow_key_edits or self.key == node.key:
            return KeyValuePairEdit(self, node)
        else:
            return Replace(self, node)

    def make_edited(self) -> Union[EditedTreeNode, 'KeyValuePairNode']:
        return self.edited_type()(
            key=self.key.make_edited(),
            value=self.value.make_edited(),
            allow_key_edits=self.allow_key_edits
        )

    def init_args(self) -> Dict[str, Any]:
        return {
            'key': self.key,
            'value': self.value,
            'allow_key_edits': self.allow_key_edits
        }

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


class ListNode(SequenceNode):
    def __init__(self, list_like: Sequence[TreeNode]):
        super().__init__()
        self.children: Tuple[TreeNode] = tuple(list_like)

    def all_children_are_leaves(self) -> bool:
        return all(isinstance(c, LeafNode) for c in self.children)

    def edits(self, node: TreeNode) -> Edit:
        if isinstance(node, ListNode):
            if len(self.children) == len(node.children) == 0:
                return Match(self, node, 0)
            elif len(self.children) == len(node.children) == 1:
                return EditSequence(from_node=self, to_node=node, edits=iter((
                    Match(self, node, 0),
                    self.children[0].edits(node.children[0])
                )))
            elif self.children == node.children:
                return Match(self, node, 0)
            else:
                if self.all_children_are_leaves() and node.all_children_are_leaves():
                    insert_remove_penalty = 0
                else:
                    insert_remove_penalty = 1
                return EditDistance(
                    self,
                    node,
                    self.children,
                    node.children,
                    insert_remove_penalty=insert_remove_penalty
                )
        else:
            return Replace(self, node)

    def init_args(self) -> Dict[str, Any]:
        return {
            'list_like': self.children
        }

    def print_item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        if hasattr(printer, 'join_lists') and printer.join_lists:
            if not is_first and not is_last:
                printer.write(' ')
        else:
            printer.newline()

    def calculate_total_size(self):
        return sum(c.total_size for c in self.children)

    def __eq__(self, other):
        return isinstance(other, ListNode) and self.children == other.children

    def __hash__(self):
        return hash(self.children)

    def __len__(self):
        return len(self.children)

    def __iter__(self) -> Iterator[TreeNode]:
        return iter(self.children)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.children!r})"

    def __str__(self):
        return str(self.children)


class MultiSetNode(SequenceNode):
    def __init__(self, items: Iterable[TreeNode]):
        super().__init__()
        self.children: HashableCounter[TreeNode] = HashableCounter(items)

    def edits(self, node: TreeNode) -> Edit:
        if isinstance(node, MultiSetNode):
            if len(self.children) == len(node.children) == 0:
                return Match(self, node, 0)
            elif self.children == node.children:
                return Match(self, node, 0)
            else:
                return MultiSetEdit(self, node, self.children, node.children)
        else:
            return Replace(self, node)

    def init_args(self) -> Dict[str, Any]:
        return {
            'items': self.children.elements()
        }

    def calculate_total_size(self):
        return sum(c.total_size * count for c, count in self.children.items())

    def __eq__(self, other):
        return isinstance(other, MultiSetNode) and other.children == self.children

    def __hash__(self):
        return hash(self.children)

    def __len__(self):
        return sum(self.children.values())

    def __iter__(self) -> Iterator[TreeNode]:
        return self.children.elements()

    def __repr__(self):
        return f"{self.__class__.__name__}({list(self)!r})"

    def __str__(self):
        return str(self.children)


class DictNode(MultiSetNode):
    def __init__(self, dict_like_or_kvp_list: Union[Dict[LeafNode, TreeNode], Sequence[KeyValuePairNode]]):
        if hasattr(dict_like_or_kvp_list, 'items'):
            super().__init__(
                sorted(KeyValuePairNode(key, value, allow_key_edits=True)
                       for key, value in dict_like_or_kvp_list.items())
            )
        else:
            super().__init__(dict_like_or_kvp_list)
        self.start_symbol = '{'
        self.end_symbol = '}'

    def items(self) -> Iterator[Tuple[LeafNode, TreeNode]]:
        yield from self.children.elements()

    def print_item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        if hasattr(printer, 'join_dict_items') and printer.join_dict_items:
            if not is_first and not is_last:
                printer.write(' ')
        else:
            printer.newline()

    # def make_edited(self) -> Union[EditedTreeNode, 'DictNode']:
    #     return self.edited_type()({
    #         kvp.key.make_edited(): kvp.value.make_edited() for kvp in cast(Iterator[KeyValuePairNode], self)
    #     })

    def init_args(self) -> Dict[str, Any]:
        return {
            'dict_like': {
                kvp.key: kvp.value for kvp in self
            }
        }

    def edits(self, node: TreeNode) -> Edit:
        if isinstance(node, MultiSetNode):
            return super().edits(node)
        else:
            return Replace(self, node)


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


class FixedKeyDictNode(SequenceNode):
    """A dictionary that only matches KeyValuePairs if they share the same key
    NOTE: This implementation does not currently support duplicate keys!
    """
    def __init__(self, dict_like: Dict[LeafNode, TreeNode]):
        is_edited = isinstance(self, EditedTreeNode)

        def kvp_type(key, value):
            ret = KeyValuePairNode(key, value, allow_key_edits=False)
            if is_edited:
                return ret.make_edited()
            else:
                return ret

        self.children: Dict[LeafNode, KeyValuePairNode] = {
            kvp.key: kvp for kvp in (kvp_type(key, value) for key, value in dict_like.items())
        }
        super().__init__(start_symbol='{', end_symbol='}')

    def print_item_newline(self, printer: Printer, is_first: bool = False, is_last: bool = False):
        if hasattr(printer, 'join_dict_items') and printer.join_dict_items:
            if not is_first and not is_last:
                printer.write(' ')
        else:
            printer.newline()

    def _child_edits(self, node: 'FixedKeyDictNode') -> Iterator[Edit]:
        unshared_kvps = set()
        for key, kvp in self.children.items():
            if key in node.children:
                other_kvp = node.children[key]
                if kvp == other_kvp:
                    yield Match(kvp, other_kvp, 0)
                else:
                    yield KeyValuePairEdit(kvp, other_kvp)
            else:
                unshared_kvps.add(kvp)
        for kvp in unshared_kvps:
            yield Remove(to_remove=kvp, remove_from=self)
        for kvp in node.children.values():
            if kvp.key not in self.children:
                yield Insert(to_insert=kvp, insert_into=self)

    def edits(self, node: TreeNode) -> Edit:
        if isinstance(node, FixedKeyDictNode):
            if len(self.children) == len(node.children) == 0:
                return Match(self, node, 0)
            elif self.children == node.children:
                return Match(self, node, 0)
            else:
                return FixedKeyDictNodeEdit(from_node=self, to_node=node, edits=self._child_edits(node))
        else:
            return Replace(self, node)

    def init_args(self) -> Dict[str, Any]:
        return {
            'dict_like': {
                kvp.key: kvp.value for kvp in self.children.values()
            }
        }

    def make_edited(self) -> Union[EditedTreeNode, 'FixedKeyDictNode']:
        return self.edited_type()({
            kvp.key: kvp.value for kvp in self.children.values()
        })

    def calculate_total_size(self):
        return sum(c.total_size for c in self.children.values())

    def __eq__(self, other):
        return isinstance(other, FixedKeyDictNode) and other.children == self.children

    def __hash__(self):
        return hash(frozenset(self.children.values()))

    def __len__(self):
        return len(self.children)

    def __iter__(self) -> Iterator[TreeNode]:
        return iter(self.children.values())

    def __repr__(self):
        return f"{self.__class__.__name__}({list(self)!r})"

    def __str__(self):
        return str(self.children)


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

    def print(self, printer: Printer):
        if self.from_node.quoted:
            printer.write('"')
        remove_seq = []
        add_seq = []
        for sub_edit in self.edit_distance.edits():
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
                            printer.write(rm)
                remove_seq = []
                with printer.color(Fore.WHITE).background(Back.GREEN).bright():
                    with printer.under_plus():
                        for add in add_seq:
                            printer.write(add)
                add_seq = []
                if to_remove is not None:
                    remove_seq.append(to_remove)
                if to_add is not None:
                    add_seq.append(to_add)
                if matched is not None:
                    printer.write(matched)
        with printer.color(Fore.WHITE).background(Back.RED).bright():
            with printer.strike():
                for rm in remove_seq:
                    printer.write(rm)
        with printer.color(Fore.WHITE).background(Back.GREEN).bright():
            with printer.under_plus():
                for add in add_seq:
                    printer.write(add)
        if self.from_node.quoted:
            printer.write('"')


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
        if printer.context().fore is None:
            context = printer.color(Fore.GREEN)
            null_context = False
        else:
            context = NullANSIContext()
            null_context = True
        with context:
            if self.quoted:
                printer.write('"')
            for c in self.object:
                if c == '"' and self.quoted:
                    if not null_context:
                        with printer.color(Fore.YELLOW):
                            printer.write('\\"')
                    else:
                        printer.write('\\"')
                else:
                    printer.write(c)
            if self.quoted:
                printer.write('"')

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


def build_tree(python_obj, force_leaf_node=False, allow_key_edits=True) -> TreeNode:
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
            build_tree(k, force_leaf_node=True, allow_key_edits=allow_key_edits):
                build_tree(v, allow_key_edits=allow_key_edits) for k, v in python_obj.items()
        }
        if allow_key_edits:
            return DictNode(dict_items)
        else:
            return FixedKeyDictNode(dict_items)
    else:
        raise ValueError(f"Unsupported Python object {python_obj!r} of type {type(python_obj)}")


class EditedMatch(AbstractEdit):
    def __init__(self, match_from: LeafNode, match_to: LeafNode):
        self.edit_distance: EditDistance = string_edit_distance(str(match_from.object), str(match_to.object))
        super().__init__(
            from_node=match_from,
            to_node=match_to
        )

    def print(self, printer: Printer):
        for edit in self.edit_distance.edits():
            edit.print(printer)

    def bounds(self) -> Range:
        return self.edit_distance.bounds()

    def tighten_bounds(self) -> bool:
        return self.edit_distance.tighten_bounds()

    def __repr__(self):
        return f"{self.__class__.__name__}(to_replace={self.from_node!r}, replace_with={self.to_node!r})"


def string_edit_distance(s1: str, s2: str) -> EditDistance:
    list1 = ListNode([StringNode(c) for c in s1])
    list2 = ListNode([StringNode(c) for c in s2])
    return EditDistance(list1, list2, list1.children, list2.children, insert_remove_penalty=0)
