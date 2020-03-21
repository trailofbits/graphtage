import heapq
import itertools

from abc import abstractmethod, ABCMeta
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple


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


class Fringe:
    def __init__(self, *fringe):
        self._pairs: List[Tuple[TreeNode,TreeNode]] = list(fringe)

    def __len__(self):
        return len(self._pairs)

    def __iter__(self):
        return iter(self._pairs)

    def __getitem__(self, index):
        return self._pairs[index]


class Edit(metaclass=ABCMeta):
    @abstractmethod
    def cost(self):
        return 0

    @abstractmethod
    def update_fringe(self, fringe: Fringe) -> Fringe:
        raise NotImplementedError()


class TreeNode(metaclass=ABCMeta):
    @abstractmethod
    def edits(self, node) -> Iterator[Edit]:
        raise StopIteration

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

    def edits(self, node: TreeNode) -> Iterator[Edit]:
        yield Remove(self)
        if isinstance(node, LeafNode):
            yield Match(self, node, levenshtein_distance(str(self.object), str(node.object)))
        elif isinstance(node, ContainerNode):
            yield Replace(self, node)

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

    def edits(self, node: TreeNode) -> Iterator[Edit]:
        if not isinstance(node, KeyValuePairNode):
            raise RuntimeError("KeyValuePairNode.edits() should only ever be called with another KeyValuePair object!")
        for kedit, vedit in itertools.product(self.key.edits(node.key), self.value.edits(node.value)):
            yield CompoundEdit(Match(self, node, 0), kedit, vedit)

    def total_size(self):
        return self.key.total_size() + self.value.total_size()

    def __len__(self):
        return 2

    def __iter__(self):
        yield self.key
        yield self.value

    def __repr__(self):
        return f"{self.__class__.__name__}(key={self.key!r}, value={self.value!r})"

    def __str__(self):
        return f"{self.key!s}: {self.value!s}"


def list_match(l1: Tuple[TreeNode], l2: Tuple[TreeNode]) -> Iterator[Edit]:
    if not l1 and not l2:
        return
    elif l1 and not l2:
        yield CompoundEdit(*(Remove(n) for n in l1))
    elif l2 and not l1:
        yield CompoundEdit(*(Insert(n) for n in l2))
    else:
        for possibility in list_match(l1[1:], l2):
            yield CompoundEdit(Remove(l1[0]), *possibility)
        for possibility in list_match(l1, l2[1:]):
            yield CompoundEdit(Insert(l2[0]), possibility)
        matches: List[Edit] = [Replace(l1[0], l2[0])] + list(l1[0].edits(l2[0]))
        possibilities = list_match(l1[1:], l2[1:])
        yield from itertools.product(matches, possibilities)


class ListNode(ContainerNode):
    def __init__(self, list_like: Sequence[TreeNode]):
        self.children: Tuple[TreeNode] = tuple(list_like)

    def edits(self, node: TreeNode) -> Iterator[Edit]:
        if isinstance(node, ListNode):
            yield from list_match(self.children, node.children)
        else:
            yield Replace(self, node)

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

    def update_fringe(self, fringe: Fringe) -> Fringe:
        return fringe

    def cost(self):
        return 0


class CompoundEdit(Edit):
    def __init__(self, *edits: Edit):
        self.edits = edits

    def __len__(self):
        return len(self.edits)

    def __iter__(self) -> Iterator[Edit]:
        return iter(self.edits)

    def update_fringe(self, fringe: Fringe) -> Fringe:
        for edit in self.edits:
            fringe = edit.update_fringe(fringe)
        return fringe

    def cost(self) -> int:
        return sum(e.cost() for e in self.edits)


class Match(Edit):
    def __init__(self, match_from: TreeNode, match_to: TreeNode, cost: int):
        self.match_from = match_from
        self.match_to = match_to
        self._cost = cost

    def update_fringe(self, fringe: Fringe) -> Fringe:
        return Fringe(*[(n1, n2) for n1, n2 in fringe if n1 != self.match_from and n2 != self.match_to])

    def cost(self) -> int:
        return self._cost


class Replace(Edit):
    def __init__(self, to_replace: TreeNode, replace_with: TreeNode):
        self.to_replace: TreeNode = to_replace
        self.replace_with: TreeNode = replace_with

    def update_fringe(self, fringe: Fringe) -> Fringe:
        return Fringe(*[(n1, n2) for n1, n2 in fringe if n1 != self.match_from and n2 != self.match_to])

    def cost(self) -> int:
        return self.to_replace.total_size() + self.replace_with.total_size()


class Remove(Edit):
    def __init__(self, to_remove: TreeNode):
        self.removed: TreeNode = to_remove

    def update_fringe(self, fringe: Fringe) -> Fringe:
        return Fringe(*[(n1, n2) for n1, n2 in fringe if n1 != self.removed])

    def cost(self):
        return self.removed.total_size()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.removed!r})"


class Insert(Edit):
    def __init__(self, to_insert: TreeNode):
        self.inserted: TreeNode = to_insert

    def update_fringe(self, fringe: Fringe) -> Fringe:
        return Fringe(*[(n1, n2) for n1, n2 in fringe if n2 != self.inserted])

    def cost(self):
        return self.inserted.total_size()


class SearchState:
    def __init__(self, edit: Edit, parent=None):
        self.parent: SearchState = parent
        self.edit: Edit = edit
        self._fringe: Fringe = None
        self._depth = None

    @property
    def depth(self):
        if self._depth is None:
            if self.parent is not None:
                self._depth = self.parent.depth + 1
            else:
                self._depth = 0
        return self._depth

    @property
    def path_cost(self):
        # TODO: Add the heuristic cost
        return self.edit.cost()

    def __lt__(self, other):
        return self.path_cost < other.path_cost or (self.path_cost == other.path_cost and self.depth < other.depth)

    def __eq__(self, other):
        return self.path_cost == other.path_cost and self.depth == other.depth

    def __hash__(self):
        return self.path_cost * self.depth

    @property
    def edits(self):
        if self.parent is None:
            return ()
        else:
            return self.parent.edits + (self.edit,)

    def goal_state(self) -> bool:
        return len(self.fringe) == 0

    def successors(self):
        if self.goal_state():
            return
        n1, n2 = self.fringe[-1]
        for edit in n1.edits(n2):
            yield SearchState(edit, self)

    @property
    def fringe(self) -> Fringe:
        if self._fringe is None:
            if self.parent is None:
                self._fringe = Fringe((self.edit.tree1, self.edit.tree2))
            else:
                self._fringe = self.edit.update_fringe(self.parent.fringe)
        return self._fringe


def diff(tree1_root: TreeNode, tree2_root: TreeNode) -> SearchState:
    states: List[SearchState] = [SearchState(InitialState(tree1_root, tree2_root))]
    while states:
        next_state: SearchState = heapq.heappop(states)
        if next_state.goal_state():
            return next_state
        for succ in next_state.successors():
            heapq.heappush(states, succ)
    return None


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
        "test": "bar"
    })

    print(diff(obj1, obj2).edits)

