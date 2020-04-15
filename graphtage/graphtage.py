import itertools
from abc import ABC
from collections import defaultdict, Sized
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, TextIO, Tuple, Union

from .edits import EditSequence, CompoundEdit, Edit, Insert, Match, Remove, Replace
from .levenshtein import EditDistance, levenshtein_distance
from .multiset import MultiSetEdit
from .printer import Back, DEFAULT_PRINTER, Fore, NullANSIContext, Printer
from graphtage.bounds import Range
from .tree import ContainerNode, TreeNode
from .utils import HashableCounter


class Diff:
    def __init__(self, from_root, to_root, edits: Iterable[Edit]):
        self.from_root: TreeNode = from_root
        self.to_root: TreeNode = to_root
        edit_list: List[Edit] = []
        self.edits_by_node: Dict[TreeNode, List[Edit]] = defaultdict(list)
        edit_stack = []
        for edit in edits:
            edit_list.append(edit)
            edit_stack.append(edit)
            while edit_stack:
                e = edit_stack.pop()
                self.edits_by_node[e.from_node].append(e)
                if isinstance(e, CompoundEdit):
                    edit_stack.extend(e.edits())
        self.edits: Tuple[Edit, ...] = tuple(edit_list)

    def __iter__(self) -> Iterator['AtomicEdit']:
        return itertools.chain(*[explode_edits(edit) for edit in self.edits])

    def __contains__(self, node):
        return node in self.edits_by_node

    def __getitem__(self, node) -> Sequence[Edit]:
        return self.edits_by_node[node]

    def print(self, out_stream: Optional[Union[TextIO, Printer]] = None):
        if out_stream is None:
            out_stream = DEFAULT_PRINTER
        elif isinstance(out_stream, TextIO):
            out_stream = Printer(out_stream)
        self.from_root.print(out_stream, self)
        out_stream.newline()

    def cost(self) -> int:
        return sum(e.cost().upper_bound for e in self.edits)

    def __repr__(self):
        return f"{self.__class__.__name__}(from_root={self.from_root!r}, to_root={self.to_root!r}, edits={self.edits!r})"


class LeafNode(TreeNode):
    def __init__(self, obj):
        self.object = obj

    def calculate_total_size(self):
        return len(str(self.object))

    def edits(self, node: TreeNode) -> Edit:
        if isinstance(node, LeafNode):
            # if self.object == node.object:
            #     return Match(self, node, 0)
            # elif len(str(self.object)) == len(str(node.object)) == 1:
            #     return Match(self, node, 1)
            # else:
            #     return EditedMatch(self, node)
            return Match(self, node, levenshtein_distance(str(self.object), str(node.object)))
        elif isinstance(node, ContainerNode):
            return Replace(self, node)

    def print(self, printer: Printer, diff: Optional[Diff] = None):
        if diff is None or self not in diff:
            printer.write(repr(self.object))
        else:
            for edit in diff[self]:
                edit.print(printer)

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


class KeyValuePairNode(ContainerNode):
    def __init__(self, key: LeafNode, value: TreeNode, allow_key_edits: bool = True):
        self.key: LeafNode = key
        self.value: TreeNode = value
        self.allow_key_edits: bool = allow_key_edits

    def edits(self, node: TreeNode) -> Edit:
        if not isinstance(node, KeyValuePairNode):
            raise RuntimeError("KeyValuePairNode.edits() should only ever be called with another KeyValuePair object!")
        if self.allow_key_edits:
            return EditSequence(
                from_node=self,
                to_node=node,
                edits=iter((self.key.edits(node.key), self.value.edits(node.value)))
            )
        elif self.key == node.key:
            return EditSequence(
                from_node=self,
                to_node=node,
                edits=iter((Match(self.key, node.key, 0), self.value.edits(node.value)))
            )
        else:
            return Replace(self, node)

    def print(self, printer: Printer, diff: Optional[Diff] = None):
        if diff is None or self not in diff or (len(diff[self]) == 1 and isinstance(diff[self][0], CompoundEdit)):
            if isinstance(self.key, LeafNode):
                with printer.color(Fore.BLUE):
                    self.key.print(printer, diff)
            else:
                self.key.print(printer, diff)
            with printer.bright():
                printer.write(": ")
            self.value.print(printer, diff)
        else:
            for edit in diff[self]:
                edit.print(printer)

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


class SequenceNode(ContainerNode, Sized, ABC):
    def __init__(self):
        self.start_symbol: str = '['
        self.end_symbol: str = ']'

    def print(self, printer: Printer, diff: Optional[Diff] = None):
        if diff is not None and diff[self] and isinstance(diff[self][0], CompoundEdit):
            assert len(diff[self]) == 1
            edits: List[Edit] = list(diff[self][0].edits())
        else:
            edits: List[Edit] = [Match(child, child, 0) for child in self]
        with printer.bright():
            printer.write(self.start_symbol)
        with printer.indent() as p:
            to_remove: int = 0
            to_insert: int = 0
            for i, edit in enumerate(edits):
                if isinstance(edit, Remove):
                    to_remove += 1
                elif isinstance(edit, Insert):
                    to_insert += 1
                while to_remove > 0 and to_insert > 0:
                    to_remove -= 1
                    to_insert -= 1
                if i > 0:
                    with printer.bright():
                        if to_remove:
                            to_remove -= 1
                            with p.strike():
                                p.write(',')
                        elif to_insert:
                            to_insert -= 1
                            with p.under_plus():
                                p.write(',')
                        else:
                            p.write(',')
                p.newline()
                if isinstance(edit, Match):
                    edit.from_node.print(p, diff)
                elif isinstance(edit, CompoundEdit):
                    edit.from_node.print(p, diff)
                else:
                    edit.print(p)
        if len(self) > 0:
            printer.newline()
        with printer.bright():
            printer.write(self.end_symbol)


class ListNode(SequenceNode):
    def __init__(self, list_like: Sequence[TreeNode]):
        super().__init__()
        self.children: Tuple[TreeNode] = tuple(list_like)

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
                return EditDistance(self, node, self.children, node.children)
        else:
            return Replace(self, node)

    def calculate_total_size(self):
        return sum(c.total_size for c in self.children)

    def __eq__(self, other):
        return self.children == other.children

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
    def __init__(self, dict_like: Dict[LeafNode, TreeNode]):
        super().__init__(sorted(KeyValuePairNode(key, value, allow_key_edits=True) for key, value in dict_like.items()))
        self.start_symbol = '{'
        self.end_symbol = '}'

    def edits(self, node: TreeNode) -> Edit:
        if isinstance(node, MultiSetNode):
            return super().edits(node)
        else:
            return Replace(self, node)


class FixedKeyDictNode(SequenceNode):
    """A dictionary that only matches KeyValuePairs if they share the same key
    NOTE: This implementation does not currently support duplicate keys!
    """
    def __init__(self, dict_like: Dict[LeafNode, TreeNode]):
        super().__init__()
        self.start_symbol = '{'
        self.end_symbol = '}'
        self.children: Dict[LeafNode, KeyValuePairNode] = {
            key: KeyValuePairNode(key, value, allow_key_edits=False) for key, value in dict_like.items()
        }

    def _child_edits(self, node: 'FixedKeyDictNode') -> Iterator[Edit]:
        unshared_kvps = set()
        for key, kvp in self.children.items():
            if key in node.children:
                other_kvp = node.children[key]
                if kvp == other_kvp:
                    yield Match(kvp, other_kvp, 0)
                else:
                    yield kvp.edits(node.children[key])
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
                return EditSequence(from_node=self, to_node=node, edits=self._child_edits(node))
        else:
            return Replace(self, node)

    def calculate_total_size(self):
        return sum(c.total_size for c in self.children)

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


class StringNode(LeafNode):
    def __init__(self, string_like: str):
        super().__init__(string_like)

    def print(self, printer: Printer, diff: Optional[Diff] = None):
        if diff is None or self not in diff:
            if printer.context().fore is None:
                context = printer.color(Fore.GREEN)
                null_context = False
            else:
                context = NullANSIContext()
                null_context = True
            with context:
                printer.write('"')
                for c in self.object:
                    if c == '"':
                        if not null_context:
                            with printer.color(Fore.YELLOW):
                                printer.write('\\"')
                        else:
                            printer.write('\\"')
                    else:
                        printer.write(c)
                printer.write('"')
        else:
            for edit in diff[self]:
                if isinstance(edit, Match) and isinstance(edit.to_node, StringNode):
                    printer.write('"')
                    sub_edits = string_edit_distance(self.object, edit.to_node.object).edits()
                    remove_seq = []
                    add_seq = []
                    for sub_edit in sub_edits:
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
                    printer.write('"')
                else:
                    edit.print(printer)


class IntegerNode(LeafNode):
    def __init__(self, int_like: int):
        super().__init__(int_like)


def diff(tree1_root: TreeNode, tree2_root: TreeNode, callback: Optional[Callable[[Range], Any]] = None) -> Diff:
    root_edit = tree1_root.edits(tree2_root)
    if callback is not None:
        prev_bounds = root_edit.bounds()
        callback(prev_bounds)
    while root_edit.valid and not root_edit.bounds().definitive():
        if not root_edit.tighten_bounds():
            break
        if callback is not None:
            if root_edit.bounds().lower_bound != prev_bounds.lower_bound \
                    or root_edit.bounds().upper_bound != prev_bounds.upper_bound:
                prev_bounds = root_edit.bounds()
                callback(prev_bounds)
    if callback is not None:
        callback(root_edit.bounds())
    return Diff(tree1_root, tree2_root, (root_edit,))


def build_tree(python_obj, force_leaf_node=False, allow_key_edits=True) -> TreeNode:
    if isinstance(python_obj, int):
        return IntegerNode(python_obj)
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


class EditedMatch(Edit):
    def __init__(self, match_from: LeafNode, match_to: LeafNode):
        self.edit_distance: EditDistance = string_edit_distance(str(match_from.object), str(match_to.object))
        super().__init__(
            from_node=match_from,
            to_node=match_to
        )

    def print(self, printer: Printer):
        for edit in self.edit_distance.edits():
            edit.print(printer)

    def cost(self) -> Range:
        return self.edit_distance.bounds()

    def tighten_bounds(self) -> bool:
        return self.edit_distance.tighten_bounds()

    def __repr__(self):
        return f"{self.__class__.__name__}(to_replace={self.from_node!r}, replace_with={self.to_node!r})"


def string_edit_distance(s1: str, s2: str) -> EditDistance:
    list1 = ListNode([StringNode(c) for c in s1])
    list2 = ListNode([StringNode(c) for c in s2])
    return EditDistance(list1, list2, list1.children, list2.children)


AtomicEdit = Union[Insert, Remove, Replace, Match, EditedMatch]


def explode_edits(edit: Union[AtomicEdit, CompoundEdit]) -> Iterator[AtomicEdit]:
    if isinstance(edit, CompoundEdit):
        while not edit.cost().definitive() and edit.tighten_bounds():
            pass
        return itertools.chain(*map(explode_edits, edit.edits()))
    else:
        return iter((edit,))


if __name__ == '__main__':
    obj1 = build_tree({
        "test": "foo",
        "baz": 1
    })
    obj2 = build_tree({
        "test": "bar",
        "baz": 2
    })

    obj_diff = diff(obj1, obj2)
    print(obj_diff.cost())
    print(obj_diff)
    obj_diff.print()
