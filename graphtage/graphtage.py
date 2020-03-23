import heapq
import itertools

from abc import abstractmethod, ABCMeta
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union


def levenshtein_distance(s: str, t: str) -> int:
    rows = len(s) + 1
    cols = len(t) + 1
    dist: List[List[int]] = [[0] * cols for _ in range(rows)]

    for i in range(1, rows):
        dist[i][0] = i

    for i in range(1, cols):
        dist[0][i] = i

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


class Range:
    def __init__(self, lower_bound: int = None, upper_bound: int = None):
        self.lower_bound: int = lower_bound
        self.upper_bound: int = upper_bound

    def __lt__(self, other):
        return self.upper_bound is not None and other.lower_bound is not None and self.upper_bound < other.lower_bound

    def __bool__(self):
        return self.lower_bound is not None and self.upper_bound is not None

    def definitive(self) -> bool:
        return bool(self) and self.lower_bound == self.upper_bound

    def intersect(self, other):
        if not self or not other or self < other or other < self:
            return Range()
        elif self.lower_bound < other.lower_bound:
            if self.upper_bound < other.upper_bound:
                return Range(other.lower_bound, self.upper_bound)
            else:
                return other
        elif self.upper_bound < other.upper_bound:
            return self
        else:
            return Range(self.lower_bound, other.upper_bound)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.lower_bound!r}, {self.upper_bound!r})"

    def __str__(self):
        return f"[{self.lower_bound}, {self.upper_bound}]"


class Edit(metaclass=ABCMeta):
    def __init__(self,
                 constant_cost: Optional[int] = 0,
                 cost_upper_bound: Optional[int] = None,
                 sub_edits: Sequence = ()):
        self._constant_cost = constant_cost
        self._cost_upper_bound = cost_upper_bound
        self.sub_edits: Tuple[Edit] = tuple(sub_edits)

    def tighten_bounds(self) -> bool:
        if self.cost().definitive():
            return False
        for child in self.sub_edits:
            if child.tighten_bounds():
                return True
        return False

    def __lt__(self, other):
        while True:
            if self.cost() < other.cost():
                return True
            elif self.cost().definitive() and other.cost().definitive():
                return False
            self.tighten_bounds()
            other.tighten_bounds()

    def __eq__(self, other):
        while True:
            if self.cost().definitive() and other.cost().definitive():
                return self.cost().lower_bound == other.cost().lower_bound
            self.tighten_bounds()
            other.tighten_bounds()

    def cost(self) -> Range:
        lb = self._constant_cost
        if self._cost_upper_bound is None:
            ub = 0
        else:
            ub = self._cost_upper_bound
        for e in self.sub_edits:
            cost = e.cost()
            lb += cost.lower_bound
            ub += cost.upper_bound
        return Range(lb, ub)


class PossibleEdits(Edit):
    def __init__(self, edits: Iterable[Edit] = ()):
        super().__init__()
        self.possibilities: Tuple[Edit] = tuple(edits)

    @property
    def best_possibility(self) -> Edit:
        best: Edit = None
        for e in self.possibilities:
            if best is None or e.cost().upper_bound < best.cost().upper_bound:
                best = e
        return best

    def tighten_bounds(self) -> bool:
        if self.cost().definitive():
            return False
        for child in self.possibilities:
            if child.tighten_bounds():
                return True
        return False

    def cost(self) -> Range:
        lb = None
        ub = 0
        for e in self.possibilities:
            cost = e.cost()
            if lb is None:
                lb = cost.lower_bound
            else:
                lb = min(lb, cost.lower_bound)
            ub = max(ub, cost.upper_bound)
        return Range(lb, ub)


class TreeNode(metaclass=ABCMeta):
    @abstractmethod
    def edits(self, node) -> Edit:
        return None

    @abstractmethod
    def total_size(self) -> int:
        return 0


class ContainerNode(TreeNode, metaclass=ABCMeta):
    pass


class LeafNode(TreeNode):
    def __init__(self, object):
        self.object = object

    def total_size(self):
        return len(str(self.object))

    def edits(self, node: TreeNode) -> Edit:
        if isinstance(node, LeafNode):
            return Match(self, node, levenshtein_distance(str(self.object), str(node.object)))
        elif isinstance(node, ContainerNode):
            return Replace(self, node)

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
        return CompoundEdit(Match(self, node, 0), self.key.edits(node.key), self.value.edits(node.value))

    def total_size(self):
        return self.key.total_size() + self.value.total_size()

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
    def __init__(self, *edits: Edit):
        sub_edits = []
        for edit in edits:
            if isinstance(edit, CompoundEdit):
                sub_edits.extend(edit.edits)
            else:
                sub_edits.append(edit)
        super().__init__(sub_edits=sub_edits)
        self.edits = edits

    def __len__(self):
        return len(self.edits)

    def __iter__(self) -> Iterator[Edit]:
        return iter(self.edits)

    def __repr__(self):
        return f"{self.__class__.__name__}(*{self.edits!r})"


class ListNode(ContainerNode):
    def __init__(self, list_like: Sequence[TreeNode]):
        self.children: Tuple[TreeNode] = tuple(list_like)

    def edits(self, node: TreeNode) -> Edit:
        if isinstance(node, ListNode):
            return PossibleEdits(self._match(self.children, node.children))
        else:
            return Replace(self, node)

    def _match(self, l1: Tuple[TreeNode], l2: Tuple[TreeNode]) -> Iterator[Edit]:
        if not l1 and not l2:
            return
        elif l1 and not l2:
            yield CompoundEdit(*(Remove(n, remove_from=self) for n in l1))
        elif l2 and not l1:
            yield CompoundEdit(*(Insert(n, insert_into=self) for n in l2))
        else:
            for possibility in self._match(l1[1:], l2):
                yield CompoundEdit(Remove(l1[0], remove_from=self), possibility)
            for possibility in self._match(l1, l2[1:]):
                yield CompoundEdit(Insert(l2[0], insert_into=self), possibility)
            matches: List[Edit] = [Replace(l1[0], l2[0]), l1[0].edits(l2[0])]
            if len(l1) == 1 and len(l2) == 1:
                yield from iter(matches)
            else:
                possibilities = self._match(l1[1:], l2[1:])
                for m, p in itertools.product(matches, possibilities):
                    yield CompoundEdit(m, p)

    def total_size(self):
        return sum(c.total_size() for c in self.children)

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
        self.tree1: TreeNode = tree1_root
        self.tree2: TreeNode = tree2_root

    def cost(self):
        return 0


class Match(Edit):
    def __init__(self, match_from: TreeNode, match_to: TreeNode, cost: int):
        super().__init__(constant_cost=cost, cost_upper_bound=cost)
        self.match_from = match_from
        self.match_to = match_to

    def __repr__(self):
        return f"{self.__class__.__name__}(match_from={self.match_from!r}, match_to={self.match_to!r}, cost={self.cost().lower_bound!r})"


class Replace(Edit):
    def __init__(self, to_replace: TreeNode, replace_with: TreeNode):
        cost = max(to_replace.total_size(), replace_with.total_size()) + 1
        super().__init__(constant_cost=cost, cost_upper_bound=cost)
        self.to_replace: TreeNode = to_replace
        self.replace_with: TreeNode = replace_with

    def __repr__(self):
        return f"{self.__class__.__name__}(to_replace={self.to_replace!r}, replace_with={self.replace_with!r})"


class Remove(Edit):
    def __init__(self, to_remove: TreeNode, remove_from: TreeNode):
        super().__init__(constant_cost=to_remove.total_size() + 1, cost_upper_bound=to_remove.total_size() + 1)
        self.removed: TreeNode = to_remove
        self.remove_from: TreeNode = remove_from

    def __repr__(self):
        return f"{self.__class__.__name__}({self.removed!r}, remove_from={self.remove_from!r})"


class Insert(Edit):
    def __init__(self, to_insert: TreeNode, insert_into: TreeNode):
        super().__init__(constant_cost=to_insert.total_size() + 1, cost_upper_bound=to_insert.total_size() + 1)
        self.inserted: TreeNode = to_insert
        self.into: TreeNode = insert_into

    def __repr__(self):
        return f"{self.__class__.__name__}(to_insert={self.inserted!r}, insert_into={self.into!r})"


AtomicEdit = Union[Insert, Remove, Replace, Match]


def explode_edits(edit: Edit) -> Iterator[AtomicEdit]:
    if isinstance(edit, CompoundEdit):
        for sub_edit in edit.edits:
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


class Diff:
    def __init__(self, from_root: TreeNode, to_root: TreeNode, edits: Iterable[Edit]):
        self.from_root = from_root
        self.to_root = to_root
        self.edits: Tuple[AtomicEdit] = tuple(itertools.chain(*(explode_edits(edit) for edit in edits)))

    def cost(self) -> int:
        return sum(e.cost().upper_bound for e in self.edits)

    def __repr__(self):
        return f"{self.__class__.__name__}(from_root={self.from_root!r}, to_root={self.to_root!r}, edits={self.edits!r})"


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
    obj1 = build_tree({
        "test": "foo"
    })
    obj2 = build_tree({
        "test": "bar",
        "baz": 1337
    })

    edits = diff(obj1, obj2)
    print(edits.cost())
    print(edits)
