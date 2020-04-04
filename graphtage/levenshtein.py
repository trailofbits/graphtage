from enum import Enum
from itertools import combinations
from typing import Iterator, List, Optional, Sequence, Union

from tqdm import tqdm

from .edits import CompoundEdit, Edit, EditSequence, Insert, Remove
from .search import Bounded, POSITIVE_INFINITY, Range
from .tree import TreeNode


def levenshtein_distance(s: str, t: str) -> int:
    """Canonical implementation of the Levenshtein distance metric"""
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


class AbstractNode(Bounded):
    def __init__(self):
        self.neighbors: List[SearchNode] = []

    def __lt__(self, other):
        return self.bounds() < other.bounds()


class EditType(Enum):
    MATCH = 0
    REMOVE = 1
    INSERT = 2


class FringeEdit:
    def __init__(self, from_node: 'SearchNode', to_node: 'SearchNode', edit_type: EditType):
        self.from_node: SearchNode = from_node
        self.to_node: SearchNode = to_node
        assert from_node not in to_node.neighbors
        to_node.neighbors.append(from_node)
        self.edit_type: EditType = edit_type

    def __hash__(self):
        return id(self)

    def __lt__(self, other):
        return (self.to_node.bounds() < other.to_node.bounds()) or (
                self.to_node.bounds() == other.to_node.bounds() and self.edit_type.value < other.edit_type.value
        )


class SearchNode(AbstractNode):
    def __init__(
            self,
            node_from: TreeNode,
            node_to: TreeNode,
            **kwargs
    ):
        super().__init__()
        self.node_from: TreeNode = node_from
        self.node_to: TreeNode = node_to
        self._fringe: List[FringeEdit] = []
        for edit_type, fringe_node in kwargs.items():
            if edit_type.upper() not in EditType.__members__.keys():
                raise ValueError(f"edit type must be one of {EditType.__members__.keys()}, not {edit_type}")
            edit = FringeEdit(
                from_node=self,
                to_node=fringe_node,
                edit_type=EditType.__members__.get(edit_type.upper())
            )
            self._fringe.append(edit)
        # changed = True
        # while changed:
        #     to_remove = set()
        #     changed = False
        #     for n1, n2 in combinations(self._fringe, 2):
        #         if n1.to_node.bounds().dominates(n2.to_node.bounds()):
        #             if n1.to_node.bounds().lower_bound == n2.to_node.bounds().upper_bound:
        #                 # there is a tie; try and save a direct match
        #                 if n2.edit_type == EditType.MATCH:
        #                     to_remove.add(n1)
        #                 else:
        #                     to_remove.add(n2)
        #             else:
        #                 to_remove.add(n2)
        #             changed = True
        #             break
        #     for node in to_remove:
        #         self._fringe.remove(node)
        #         assert self in node.to_node.neighbors
        #         node.to_node.neighbors.remove(self)
        self._match: Optional[Edit] = None
        self._bounds: Optional[Range] = Range(0, POSITIVE_INFINITY)

    @property
    def match(self) -> Edit:
        if self._match is None:
            self._match = self.node_from.edits(self.node_to)
        return self._match

    def _invalidate_neighbors(self):
        for node in self.neighbors:
            node._bounds = None

    def tighten_bounds(self) -> bool:
        initial_bounds = self.bounds()
        if initial_bounds.definitive() or not self._fringe:
            return False
        self._fringe = sorted(self._fringe)
        while True:
            tightened = False
            for node in self._fringe:
                if node.to_node.tighten_bounds():
                    # see if this node dominates any of the others
                    for other_node in self._fringe:
                        if other_node is node:
                            continue
                        if node.to_node.bounds().dominates(other_node.to_node.bounds()):
                            self._fringe.remove(other_node)
                            assert self in other_node.to_node.neighbors
                            other_node.to_node.neighbors.remove(self)
                    self._bounds = None
                    if self.bounds().lower_bound > initial_bounds.lower_bound \
                            or self.bounds().upper_bound < initial_bounds.upper_bound:
                        self._invalidate_neighbors()
                        return True
                    else:
                        tightened = True
                        break
            if not tightened:
                return False

    def best_predecessor(self) -> FringeEdit:
        return min(self._fringe)

    def ancestors(self) -> Iterator['SearchNode']:
        stack: List[SearchNode] = [f.to_node for f in self._fringe]
        result: List[SearchNode] = list(stack)
        history = set(stack)
        while stack:
            node = stack.pop()
            if isinstance(node, ConstantNode):
                continue
            fringe = (
                a for a in node._fringe if a.to_node not in history
            )
            match: Optional[SearchNode] = None
            for a in fringe:
                if a.edit_type == EditType.MATCH:
                    match = a.to_node
                else:
                    result.append(a.to_node)
                    history.add(a.to_node)
                    stack.append(a.to_node)
            if match is not None:
                result.append(match)
                history.add(match)
                stack.append(match)
        return reversed(result)

    def _bounds_fold_iterative(self):
        for node in self.ancestors():
            node.bounds()

    def bounds(self) -> Range:
        if self._bounds is None:
            bounds = self.match.bounds()
            lb, ub = bounds.lower_bound, bounds.upper_bound
            if sum(f.to_node._bounds is not None for f in self._fringe) < len(self._fringe):
                # This means at least one of our fringe nodes hasn't been bounded yet.
                # self.bounds() is potentially recursive if our ancestors haven't been bounded yet,
                # which can sometimes exhaust Python's stack, so do this iteratively.
                self._bounds_fold_iterative()
            bounds = sorted(f.to_node.bounds() for f in self._fringe)
            assert bounds
            if len(bounds) == 1 or (
                bounds[0].dominates(bounds[1]) and (len(bounds) < 3 or bounds[0].dominates(bounds[2]))
            ):
                self._bounds = Range(lb + bounds[0].lower_bound, ub + bounds[0].upper_bound)
            else:
                lb += min(b.lower_bound for b in bounds)
                ub += max(b.upper_bound for b in bounds)
                self._bounds = Range(lb, ub)
        return self._bounds

    def __repr__(self):
        ret = f"{self.__class__.__name__}(node_from={self.node_from!r}, node_to={self.node_to!r}"
        for node in self._fringe:
            ret += f", {node.edit_type.name.lower()}={node.to_node!r}"
        return ret


class ConstantNode(AbstractNode):
    def __init__(
        self,
        node: Optional[TreeNode] = None,
        is_from: Optional[bool] = None,
        predecessor: Optional['ConstantNode'] = None
    ):
        super().__init__()
        if node is None:
            cost = 0
            assert predecessor is None
        else:
            assert predecessor is not None
            cost = node.total_size + predecessor._cost.upper_bound
        self._cost: Range = Range(cost, cost)
        self.node = node
        self.predecessor: Optional[ConstantNode] = predecessor
        self.is_from: Optional[bool] = is_from

    def tighten_bounds(self) -> bool:
        return False

    def bounds(self) -> Range:
        return self._cost

    def __repr__(self):
        return f"{self.__class__.__name__}(node={self.node!r}, is_from={self.is_from!r})"


class EditDistance(CompoundEdit):
    def __init__(
            self,
            from_node: TreeNode,
            to_node: TreeNode,
            from_seq: Sequence[TreeNode],
            to_seq: Sequence[TreeNode]
    ):
        from_seq: Sequence[TreeNode] = from_seq
        to_seq: Sequence[TreeNode] = to_seq
        matrix: List[List[Union[ConstantNode, SearchNode]]] = []
        for i in range(len(to_seq) + 1):
            matrix.append([])
            for j in range(len(from_seq) + 1):
                if i == 0:
                    if j == 0:
                        matrix[i].append(ConstantNode())
                    else:
                        matrix[i].append(ConstantNode(
                            node=from_seq[j-1],
                            is_from=True,
                            predecessor=matrix[i][j-1]
                        ))
                elif j == 0:
                    matrix[i].append(ConstantNode(
                        node=to_seq[i-1],
                        is_from=False,
                        predecessor=matrix[i-1][0]
                    ))
                else:
                    matrix[i].append(SearchNode(
                        node_from=from_seq[j-1],
                        node_to=to_seq[i-1],
                        insert=matrix[i-1][j],
                        remove=matrix[i][j-1],
                        match=matrix[i-1][j-1]
                    ))
        self._goal = matrix[len(to_seq)][len(from_seq)]
        super().__init__(
            from_node=from_node,
            to_node=to_node
        )

    def tighten_bounds(self) -> bool:
        return self._goal.tighten_bounds()

    def bounds(self) -> Range:
        return self._goal.bounds()

    def edits(self) -> Iterator[Edit]:
        if not self.bounds().definitive():
            with tqdm(leave=False) as t:
                t.total = self.bounds().upper_bound - self.bounds().lower_bound
                while not self.bounds().definitive() and self.tighten_bounds():
                    t.update(t.total - (self.bounds().upper_bound - self.bounds().lower_bound))
        edits: List[Edit] = []
        node = self._goal
        while not isinstance(node, ConstantNode):
            best = node.best_predecessor()
            if best.edit_type == EditType.REMOVE:
                edits.append(Remove(to_remove=node.node_from, remove_from=self.from_node))
            elif best.edit_type == EditType.INSERT:
                edits.append(Insert(to_insert=node.node_to, insert_into=self.from_node))
            else:
                assert best.edit_type == EditType.MATCH
                edits.append(node.match)
            node = best.to_node
        while node.predecessor is not None:
            if node.is_from:
                edits.append(Remove(to_remove=node.node, remove_from=self.from_node))
            else:
                edits.append(Insert(to_insert=node.node, insert_into=self.from_node))
            node = node.predecessor

        return reversed(edits)
