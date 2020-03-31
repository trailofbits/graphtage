import itertools

from abc import abstractmethod, ABCMeta
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, TextIO, Tuple, Union

from .search import Bounded, IterativeTighteningSearch, Range


def levenshtein_distance(s: str, t: str) -> int:
    rows = len(s) + 1
    cols = len(t) + 1
    dist: List[List[int]] = [[0] * cols for _ in range(rows)]

    for i in range(1, rows):
        dist[i][0] = i

    for i in range(1, cols):
        dist[0][i] = i

    col = row = 0
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row - 1] == t[col - 1]:
                cost = 0
            else:
                cost = 1
            dist[row][col] = min(dist[row - 1][col] + 1,
                                 dist[row][col - 1] + 1,
                                 dist[row - 1][col - 1] + cost)

    return dist[row][col]


class Printer:
    def __init__(self, out_stream: TextIO):
        self.out_stream = out_stream
        self.indents = 0

    def write(self, s: str):
        self.out_stream.write(s)

    def newline(self):
        self.write('\n')
        self.write(' ' * (4 * self.indents))

    def indent(self):
        class Indent:
            def __init__(self, printer):
                self.printer = printer

            def __enter__(self):
                self.printer.indents += 1
                return self.printer

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.printer.indents -= 1

        return Indent(self)


class Edit(Bounded):
    def __init__(self,
                 from_node,
                 to_node=None,
                 constant_cost: Optional[int] = 0,
                 cost_upper_bound: Optional[int] = None):
        self.from_node: TreeNode = from_node
        self.to_node: TreeNode = to_node
        self._constant_cost = constant_cost
        self._cost_upper_bound = cost_upper_bound
        self.initial_cost = self.cost()

    def tighten_bounds(self) -> bool:
        return False

    @abstractmethod
    def print(self, printer: Printer):
        pass

    def __lt__(self, other):
        return self.cost() < other.cost()

    def cost(self) -> Range:
        lb = self._constant_cost
        if self._cost_upper_bound is None:
            if self.to_node is None:
                ub = self.initial_cost.upper_bound
            else:
                ub = self.from_node.total_size + self.to_node.total_size + 1
        else:
            ub = self._cost_upper_bound
        return Range(lb, ub)

    def bounds(self):
        return self.cost()


class Diff:
    def __init__(self, from_root, to_root, edits: Iterable[Edit]):
        self.from_root: TreeNode = from_root
        self.to_root: TreeNode = to_root
        self.edits: Tuple[AtomicEdit] = tuple(itertools.chain(*(explode_edits(edit) for edit in edits)))
        self.edits_by_node: Dict[TreeNode, List[Edit]] = defaultdict(list)
        for e in self.edits:
            self.edits_by_node[e.from_node].append(e)

    def __contains__(self, node):
        return node in self.edits_by_node

    def __getitem__(self, node) -> Iterable[Edit]:
        return self.edits_by_node[node]

    def print(self, out_stream: TextIO):
        self.from_root.print(out_stream, self)

    def cost(self) -> int:
        return sum(e.cost().upper_bound for e in self.edits)

    def __repr__(self):
        return f"{self.__class__.__name__}(from_root={self.from_root!r}, to_root={self.to_root!r}, edits={self.edits!r})"


class TreeNode(metaclass=ABCMeta):
    _total_size = None

    @abstractmethod
    def edits(self, node) -> Edit:
        return None

    @property
    def total_size(self) -> int:
        if self._total_size is None:
            self._total_size = self.calculate_total_size()
        return self._total_size

    @abstractmethod
    def calculate_total_size(self) -> int:
        return 0

    @abstractmethod
    def print(self, printer: Printer, diff: Optional[Diff] = None):
        pass


class PossibleEdits(Edit):
    def __init__(
            self,
            from_node: TreeNode,
            to_node: TreeNode,
            edits: Iterator[Edit] = (),
            initial_cost: Optional[Range] = None
    ):
        if initial_cost is not None:
            self.initial_cost = initial_cost
        self._search: IterativeTighteningSearch[Edit] = IterativeTighteningSearch(
            possibilities=edits,
            initial_bounds=initial_cost
        )
        while not self._search.bounds().finite:
            self._search.tighten_bounds()
        super().__init__(from_node=from_node, to_node=to_node)

    @property
    def best_possibility(self) -> Edit:
        return self._search.best_match

    def print(self, printer: Printer):
        self.best_possibility.print(printer, diff=diff)

    def tighten_bounds(self) -> bool:
        tightened = self._search.tighten_bounds()
        #assert tightened or not self._search or self.cost().definitive()
        return tightened

    def cost(self) -> Range:
        bounds = self._search.bounds()
        # if not (bounds.lower_bound >= self.initial_cost.lower_bound \
        #        and bounds.upper_bound <= self.initial_cost.upper_bound):
        #     breakpoint()
        #assert bounds.lower_bound >= self.initial_cost.lower_bound \
        #       and bounds.upper_bound <= self.initial_cost.upper_bound
        return bounds


class ContainerNode(TreeNode, metaclass=ABCMeta):
    pass


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
        return CompoundEdit(
            from_node=self,
            to_node=node,
            edits=iter((self.key.edits(node.key), self.value.edits(node.value)))
        )

    def print(self, printer: Printer, diff: Optional[Diff] = None):
        if diff is None or self not in diff:
            self.key.print(printer, diff)
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


class CompoundEdit(Edit):
    def __init__(self, from_node: TreeNode, to_node: Optional[TreeNode], edits: Iterator[Edit]):
        self._edit_iter: Iterator[Edit] = edits
        self._sub_edits: List[Edit] = []
        cost_upper_bound = from_node.total_size + 1
        if to_node is not None:
            cost_upper_bound += to_node.total_size
        self._cost = None
        super().__init__(from_node=from_node,
                         to_node=to_node,
                         cost_upper_bound=cost_upper_bound)

    def print(self, printer: Printer):
        for sub_edit in self.sub_edits:
            sub_edit.print(printer)

    @property
    def sub_edits(self):
        while self._edit_iter is not None and self.tighten_bounds():
            pass
        return self._sub_edits

    def tighten_bounds(self) -> bool:
        starting_bounds: Range = self.cost()
        while True:
            if self._edit_iter is not None:
                try:
                    next_edit: Edit = next(self._edit_iter)
                    if isinstance(next_edit, CompoundEdit):
                        self._sub_edits.extend(next_edit.sub_edits)
                    else:
                        self._sub_edits.append(next_edit)
                except StopIteration:
                    self._edit_iter = None
            tightened = False
            for child in self._sub_edits:
                if child.tighten_bounds():
                    self._cost = None
                    tightened = True
                    new_cost = self.cost()
                    #assert new_cost.lower_bound >= starting_bounds.lower_bound
                    #assert new_cost.upper_bound <= starting_bounds.upper_bound
                    if new_cost.lower_bound > starting_bounds.lower_bound or \
                            new_cost.upper_bound < starting_bounds.upper_bound:
                        return True
            if not tightened and self._edit_iter is None:
                return self.cost().lower_bound > starting_bounds.lower_bound or \
                    self.cost().upper_bound < starting_bounds.upper_bound

    def cost(self) -> Range:
        if self._cost is not None:
            return self._cost
        elif self._edit_iter is None:
            # We've expanded all of the sub-edits, so calculate the bounds explicitly:
            total_cost = sum(e.cost() for e in self._sub_edits)
            if total_cost.definitive():
                self._cost = total_cost
        else:
            # We have not yet expanded all of the sub-edits
            total_cost = Range(0, self._cost_upper_bound)
            for e in self._sub_edits:
                total_cost.lower_bound += e.cost().lower_bound
                total_cost.upper_bound -= e.initial_cost.upper_bound - e.cost().upper_bound
        return total_cost

    def __len__(self):
        return len(self.sub_edits)

    def __iter__(self) -> Iterator[Edit]:
        return iter(self.sub_edits)

    def __repr__(self):
        return f"{self.__class__.__name__}(*{self.sub_edits!r})"


class ListNode(ContainerNode):
    def __init__(self, list_like: Sequence[TreeNode]):
        self.children: Tuple[TreeNode] = tuple(list_like)

    def print(self, printer: Printer, diff: Optional[Diff] = None):
        if diff is None or self not in diff:
            printer.write("[")
            with printer.indent() as p:
                for i, child in enumerate(self.children):
                    if i > 0:
                        p.write(',')
                    p.newline()
                    child.print(p, diff)
            if self.children:
                printer.newline()
            printer.write("]")
        else:
            for edit in diff[self]:
                edit.print(printer)

    def edits(self, node: TreeNode) -> Edit:
        if isinstance(node, ListNode):
            return next(self._match(node, self.children, node.children))
        else:
            return Replace(self, node)

    def _match(self, node: TreeNode, l1: Tuple[TreeNode], l2: Tuple[TreeNode]) -> Iterator[Edit]:
        if not l1 and not l2:
            yield Match(self, node, 0)
        elif l1 and not l2:
            yield CompoundEdit(from_node=self, to_node=None, edits=(Remove(n, remove_from=self) for n in l1))
        elif l2 and not l1:
            yield CompoundEdit(from_node=self, to_node=node, edits=(Insert(n, insert_into=self) for n in l2))
        else:
            match = l1[0].edits(l2[0])
            if len(l1) == 1 and len(l2) == 1:
                yield match
            else:
                yield PossibleEdits(
                    from_node=self,
                    to_node=node,
                    initial_cost=Range(0, sum(n.total_size for n in l1) + sum(n.total_size for n in l2) + 1),
                    edits=iter((
                        CompoundEdit(
                            from_node=self,
                            to_node=node,
                            edits=itertools.chain(
                                iter((match,)),
                                self._match(node, l1[1:], l2[1:])
                            )
                        ),
                        CompoundEdit(
                            from_node=self,
                            to_node=node,
                            edits=itertools.chain(
                                iter((Remove(l1[0], remove_from=self),)),
                                self._match(node, l1[1:], l2)
                            )
                        ),
                        CompoundEdit(
                            from_node=self,
                            to_node=node,
                            edits=itertools.chain(
                                iter((Insert(l2[0], insert_into=self),)),
                                self._match(node, l1, l2[1:])
                            )
                        )
                    ))
                )

    def calculate_total_size(self):
        return sum(c.total_size for c in self.children)

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


class StringNode(LeafNode):
    def __init__(self, string_like: str):
        super().__init__(string_like)


class IntegerNode(LeafNode):
    def __init__(self, int_like: int):
        super().__init__(int_like)


class InitialState(Edit):
    def __init__(self, tree1_root: TreeNode, tree2_root: TreeNode):
        super(tree1_root, tree2_root)

    def print(self, printer: Printer):
        pass

    def cost(self):
        return Range(0, 0)


class Match(Edit):
    def __init__(self, match_from: TreeNode, match_to: TreeNode, cost: int):
        super().__init__(
            from_node=match_from,
            to_node=match_to,
            constant_cost=cost,
            cost_upper_bound=cost
        )

    def print(self, printer: Printer):
        self.from_node.print(printer)
        if self.cost() > Range(0, 0):
            printer.write(' -> ')
            self.to_node.print(printer)

    def __repr__(self):
        return f"{self.__class__.__name__}(match_from={self.from_node!r}, match_to={self.to_node!r}, cost={self.cost().lower_bound!r})"


class Replace(Edit):
    def __init__(self, to_replace: TreeNode, replace_with: TreeNode):
        cost = max(to_replace.total_size, replace_with.total_size) + 1
        super().__init__(
            from_node=to_replace,
            to_node=replace_with,
            constant_cost=cost,
            cost_upper_bound=cost
        )

    def print(self, printer: Printer):
        self.from_node.print(printer)
        if self.cost() > 0:
            printer.write(' -> ')
            self.to_node.print(printer)

    def __repr__(self):
        return f"{self.__class__.__name__}(to_replace={self.from_node!r}, replace_with={self.to_node!r})"


class Remove(Edit):
    def __init__(self, to_remove: TreeNode, remove_from: TreeNode):
        super().__init__(
            from_node=to_remove,
            to_node=remove_from,
            constant_cost=to_remove.total_size + 1,
            cost_upper_bound=to_remove.total_size + 1
        )

    def print(self, printer: Printer):
        printer.write('~~~~')
        self.from_node.print(printer)
        printer.write('~~~~')

    def __repr__(self):
        return f"{self.__class__.__name__}({self.from_node!r}, remove_from={self.to_node!r})"


class Insert(Edit):
    def __init__(self, to_insert: TreeNode, insert_into: TreeNode):
        super().__init__(
            from_node=to_insert,
            to_node=insert_into,
            constant_cost=to_insert.total_size + 1,
            cost_upper_bound=to_insert.total_size + 1
        )

    def print(self, printer: Printer):
        printer.write('++++')
        self.from_node.print(printer)
        printer.write('++++')

    def __repr__(self):
        return f"{self.__class__.__name__}(to_insert={self.from_node!r}, insert_into={self.to_node!r})"


AtomicEdit = Union[Insert, Remove, Replace, Match]


def explode_edits(edit: Edit) -> Iterator[AtomicEdit]:
    if isinstance(edit, CompoundEdit):
        for sub_edit in edit.sub_edits:
            yield from explode_edits(sub_edit)
    elif isinstance(edit, PossibleEdits):
        while not edit.cost().definitive():
            if not edit.tighten_bounds():
                break
        if edit.best_possibility is None:
            yield edit
        else:
            yield from explode_edits(edit.best_possibility)
    else:
        yield edit


def diff(tree1_root: TreeNode, tree2_root: TreeNode) -> Diff:
    return Diff(tree1_root, tree2_root, (tree1_root.edits(tree2_root),))


def build_tree(python_obj, force_leaf_node=False) -> TreeNode:
    if isinstance(python_obj, int):
        return IntegerNode(python_obj)
    elif isinstance(python_obj, str) or isinstance(python_obj, bytes):
        return StringNode(python_obj)
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


if __name__ == '__main__':
    import sys

    obj1 = build_tree({
        "test": "foo",
        "baz": 1
    })
    obj2 = build_tree({
        "test": "bar",
        "baz": 2
    })

    obj1 = build_tree(list(range(5)))
    obj2 = build_tree(list(range(1, 5)))

    obj_diff = diff(obj1, obj2)
    print(obj_diff.cost())
    print(obj_diff)
    obj_diff.print(Printer(sys.stdout))
