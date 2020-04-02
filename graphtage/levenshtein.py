from typing import List, Optional, Sequence, Union

from itertools import combinations

from .graphtage import Edit, StringNode, TreeNode
from .search import Bounded, POSITIVE_INFINITY, Range


class AbstractNode(Bounded):
    def __init__(self):
        self.down: Optional[SearchNode] = None
        self.right: Optional[SearchNode] = None
        self.lower_right: Optional[SearchNode] = None

    def __lt__(self, other):
        return self.bounds() < other.bounds()


class SearchNode(AbstractNode):
    def __init__(
            self,
            node_from: TreeNode,
            node_to: TreeNode,
            up: 'SearchNode',
            left: 'SearchNode',
            diag: 'SearchNode'
    ):
        super().__init__()
        self.node_from: TreeNode = node_from
        self.node_to: TreeNode = node_to
        self.up: SearchNode = up
        self.left: SearchNode = left
        self.diag: SearchNode = diag
        assert up.down is None
        up.down = self
        assert left.right is None
        left.right = self
        assert diag.lower_right is None
        diag.lower_right = self
        self._fringe: List[SearchNode] = [up, left, diag]
        to_remove = set()
        for n1, n2 in combinations(self._fringe, 2):
            if n1.bounds().dominates(n2.bounds()):
                to_remove.add(n2)
        for node in to_remove:
            self._fringe.remove(node)
        self._match: Optional[Edit] = None
        self._bounds: Range = None

    @property
    def match(self) -> Edit:
        if self._match is None:
            self._match = self.node_from.edits(self.node_to)
        return self._match

    def _invalidate_neighbors(self):
        if self.down is not None:
            self.down._bounds = None
        if self.right is not None:
            self.right._bounds = None
        if self.lower_right is not None:
            self.lower_right._bounds = None

    def tighten_bounds(self) -> bool:
        initial_bounds = self.bounds()
        if initial_bounds.definitive() or not self._fringe:
            return False
        self._fringe = sorted(self._fringe)
        while True:
            tightened = False
            for node in self._fringe:
                if node.tighten_bounds():
                    # see if this node dominates any of the others
                    for other_node in self._fringe:
                        if other_node is node:
                            continue
                        if node.bounds().dominates(other_node.bounds()):
                            self._fringe.remove(other_node)
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

    def best_predecessor(self) -> 'SearchNode':
        return min(self._fringe)

    def bounds(self) -> Range:
        if self._bounds is None:
            bounds = self.match.bounds()
            lb, ub = bounds.lower_bound, bounds.upper_bound
            bounds = sorted(f.bounds() for f in self._fringe)
            assert bounds
            if len(bounds) == 1 or \
                    (bounds[0].dominates(bounds[1]) and \
                    (len(bounds) < 3 or bounds[0].dominates(bounds[2]))):
                self._bounds = Range(lb + bounds[0].lower_bound, ub + bounds[0].upper_bound)
            else:
                lb += min(b.lower_bound for b in bounds)
                ub += max(b.upper_bound for b in bounds)
                self._bounds = Range(lb, ub)
        return self._bounds

    def __repr__(self):
        return f"{self.__class__.__name__}(node_from={self.node_from!r}, node_to={self.node_to!r}, up={self.up!r}, left={self.left!r}, diag={self.diag!r})"


class ConstantNode(AbstractNode):
    def __init__(
        self,
        cost: int
    ):
        super().__init__()
        self._cost: Range = Range(cost, cost)

    def tighten_bounds(self) -> bool:
        return False

    def bounds(self) -> Range:
        return self._cost


class EditDistance(Bounded):
    def __init__(self, from_seq: Sequence[TreeNode], to_seq: Sequence[TreeNode]):
        self.from_seq: Sequence[TreeNode] = from_seq
        self.to_seq: Sequence[TreeNode] = to_seq
        self._bounds = Range(0, POSITIVE_INFINITY)
        matrix: List[List[Union[ConstantNode, SearchNode]]] = []
        for i in range(len(self.to_seq) + 1):
            matrix.append([])
            for j in range(len(self.from_seq) + 1):
                if i == 0:
                    if j == 0:
                        matrix[i].append(ConstantNode(0))
                    else:
                        matrix[i].append(ConstantNode(self.from_seq[j-1].total_size + 1))
                elif j == 0:
                    matrix[i].append(ConstantNode(self.to_seq[i-1].total_size + 1))
                else:
                    matrix[i].append(SearchNode(
                        node_from=self.from_seq[j-1],
                        node_to=self.to_seq[i-1],
                        up=matrix[i-1][j],
                        left=matrix[i][j-1],
                        diag=matrix[i-1][j-1]
                    ))
        self._goal = matrix[len(self.to_seq)][len(self.from_seq)]

    def tighten_bounds(self) -> bool:
        return self._goal.tighten_bounds()

    def bounds(self) -> Range:
        return self._goal.bounds()


def string_edit_distance(s1: str, s2: str) -> EditDistance:
    list1 = [StringNode(c) for c in s1]
    list2 = [StringNode(c) for c in s2]
    return EditDistance(list1, list2)
