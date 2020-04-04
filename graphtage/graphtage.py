import itertools
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, TextIO, Tuple, Union

from .edits import EditSequence, CompoundEdit, Edit, Insert, Match, PossibleEdits, Remove, Replace
from .levenshtein import EditDistance, levenshtein_distance
from .printer import Back, DEFAULT_PRINTER, Fore, NullANSIContext, Printer
from .search import Range
from .tree import ContainerNode, TreeNode


class Diff:
    def __init__(self, from_root, to_root, edits: Iterable[Edit]):
        self.from_root: TreeNode = from_root
        self.to_root: TreeNode = to_root
        edit_list = []
        for edit in edits:
            edit_list.extend(explode_edits(edit))   # type: ignore
        self.edits: Tuple[AtomicEdit] = tuple(edit_list)
        self.edits_by_node: Dict[TreeNode, List[Edit]] = defaultdict(list)
        for e in self.edits:
            self.edits_by_node[e.from_node].append(e)

    def __contains__(self, node):
        return node in self.edits_by_node

    def __getitem__(self, node) -> Iterable[Edit]:
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
    def __init__(self, key: LeafNode, value: TreeNode):
        self.key: LeafNode = key
        self.value: TreeNode = value

    def edits(self, node: TreeNode) -> Edit:
        if not isinstance(node, KeyValuePairNode):
            raise RuntimeError("KeyValuePairNode.edits() should only ever be called with another KeyValuePair object!")
        return EditSequence(
            from_node=self,
            to_node=node,
            edits=iter((self.key.edits(node.key), self.value.edits(node.value)))
        )

    def print(self, printer: Printer, diff: Optional[Diff] = None):
        if diff is None or self not in diff:
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
        return (self.key < other.key) or (self.key == other.key and self.value < other.value)

    def __eq__(self, other):
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


class ListNode(ContainerNode):
    def __init__(self, list_like: Sequence[TreeNode]):
        self.children: Tuple[TreeNode] = tuple(list_like)

    def print(self, printer: Printer, diff: Optional[Diff] = None):
        if diff is None or self not in diff:
            with printer.bright():
                printer.write("[")
            with printer.indent() as p:
                for i, child in enumerate(self.children):
                    if i > 0:
                        with printer.bright():
                            removed = False
                            if diff is not None and self.children[i-1] in diff:
                                for edit in diff[self.children[i-1]]:
                                    if isinstance(edit, Remove) and edit.from_node is self.children[i-1]:
                                        removed = True
                                        break
                            if removed:
                                with p.strike():
                                    p.write(',')
                            else:
                                p.write(',')
                    p.newline()
                    child.print(p, diff)
            if self.children:
                printer.newline()
            with printer.bright():
                printer.write("]")
        else:
            for edit in diff[self]:
                edit.print(printer)

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

    def _match(self, node: TreeNode, l1: Tuple[TreeNode], l2: Tuple[TreeNode]) -> Iterator[Edit]:
        if not l1 and not l2:
            return
        elif l1 and not l2:
            yield EditSequence(from_node=self, to_node=None, edits=(Remove(n, remove_from=self) for n in l1))
        elif l2 and not l1:
            yield EditSequence(from_node=self, to_node=node, edits=(Insert(n, insert_into=self) for n in l2))
        else:
            leading_matches: List[Edit] = []
            match: Optional[Edit] = None
            matched_node_from: Optional[TreeNode] = None
            matched_node_to: Optional[TreeNode] = None
            # Pop off as many perfect matches as possible
            while l1 and l2 and isinstance(l1[0], LeafNode) and isinstance(l2[0], LeafNode):
                matched_node_from = l1[0]
                matched_node_to = l2[0]
                match = l1[0].edits(l2[0])
                l1 = l1[1:]
                l2 = l2[1:]
                if match.cost().upper_bound == 0:
                    leading_matches.append(match)
                    match = None
                else:
                    break

            if match is None and l1 and l2:
                matched_node_from = l1[0]
                matched_node_to = l2[0]
                match = l1[0].edits(l2[0])
                l1 = l1[1:]
                l2 = l2[1:]

            if match is not None:
                leading_matches.append(match)

            if not l1 or not l2:
                yield EditSequence(
                    from_node=self,
                    to_node=node,
                    edits=itertools.chain(
                        iter(leading_matches),
                        self._match(node, l1, l2)
                    )
                )
            elif match is None:
                assert not l1 and not l2
                if not leading_matches:
                    return
                elif len(leading_matches) == 1:
                    yield leading_matches[0]
                else:
                    yield EditSequence(
                        from_node=self,
                        to_node=node,
                        edits=iter(leading_matches)
                    )
            else:
                assert matched_node_from is not None
                possibilities = [
                    EditSequence(
                        from_node=self,
                        to_node=node,
                        edits=itertools.chain(
                            iter(leading_matches),
                            self._match(node, l1, l2)
                        )
                    ),
                    EditSequence(
                        from_node=self,
                        to_node=node,
                        edits=itertools.chain(
                            iter(leading_matches[:-1]),
                            iter((Remove(matched_node_from, remove_from=self),)),
                            self._match(node, l1, (matched_node_to,) + l2)
                        )
                    ),
                    EditSequence(
                        from_node=self,
                        to_node=node,
                        edits=itertools.chain(
                            iter(leading_matches[:-1]),
                            iter((Insert(matched_node_to, insert_into=self),)),
                            self._match(node, (matched_node_from,) + l1, l2)
                        )
                    )
                ]
                cost_lower_bound: int = min(
                    match.cost().lower_bound,
                    matched_node_from.total_size + 1,
                    matched_node_to.total_size + 1
                )

                yield PossibleEdits(
                    from_node=self,
                    to_node=node,
                    initial_cost=Range(
                        cost_lower_bound,
                        sum(n.total_size for n in l1) + sum(n.total_size for n in l2) + 1
                    ),
                    edits=iter(possibilities)
                )

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


class DictNode(ListNode):
    def __init__(self, dict_like: Dict[LeafNode, TreeNode]):
        super().__init__(sorted(KeyValuePairNode(key, value) for key, value in dict_like.items()))

    def edits(self, node: TreeNode) -> Edit:
        if isinstance(node, DictNode):
            return super().edits(node)
        else:
            return Replace(self, node)


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
                    for sub_edit in sub_edits:
                        to_remove = None
                        to_add = None
                        if isinstance(sub_edit, Match):
                            if sub_edit.from_node.object == sub_edit.to_node.object:
                                printer.write(sub_edit.from_node.object)
                            else:
                                to_remove = sub_edit.from_node.object
                                to_add = sub_edit.to_node.object
                        elif isinstance(sub_edit, Remove):
                            to_remove = sub_edit.from_node.object
                        else:
                            assert isinstance(sub_edit, Insert)
                            to_add = sub_edit.to_node.object
                        if to_remove is not None:
                            with printer.color(Fore.WHITE).background(Back.RED).bright():
                                with printer.strike():
                                    printer.write(to_remove)
                        if to_add is not None:
                            with printer.color(Fore.WHITE).background(Back.GREEN).bright():
                                with printer.under_plus():
                                    printer.write(to_add)
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


def build_tree(python_obj, force_leaf_node=False) -> TreeNode:
    if isinstance(python_obj, int):
        return IntegerNode(python_obj)
    elif isinstance(python_obj, str):
        return StringNode(python_obj)
    elif isinstance(python_obj, bytes):
        return StringNode(python_obj.decode('utf-8'))
    elif force_leaf_node:
        raise ValueError(f"{python_obj!r} was expected to be an int or string, but was instead a {type(python_obj)}")
    elif isinstance(python_obj, list) or isinstance(python_obj, tuple):
        return ListNode([build_tree(n) for n in python_obj])
    elif isinstance(python_obj, dict):
        return DictNode({
            build_tree(k, force_leaf_node=True): build_tree(v) for k, v in python_obj.items()
        })
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
