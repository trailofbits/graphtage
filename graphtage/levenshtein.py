from typing import List, Optional, Sequence, Union

from itertools import combinations

from .edits import Edit, CompoundEdit, Insert, Remove
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
        changed = True
        while changed:
            to_remove = set()
            changed = False
            for n1, n2 in combinations(self._fringe, 2):
                if n1.bounds().dominates(n2.bounds()):
                    if n1.bounds().lower_bound == n2.bounds().upper_bound:
                        # there is a tie; try and save a direct match
                        if n2 is self.diag:
                            to_remove.add(n1)
                        else:
                            to_remove.add(n2)
                    else:
                        to_remove.add(n2)
                    changed = True
                    break
            for node in to_remove:
                self._fringe.remove(node)
        self._match: Optional[Edit] = None
        self._bounds: Optional[Range] = None

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
        best = min(self._fringe)
        if best is self.diag and best.bounds().lower_bound > 0:
            # is there a tie? if so, give preference to either a remove or an insert:
            for other in self._fringe:
                if other is not best and other < best:
                    return other
        return best

    def bounds(self) -> Range:
        if self._bounds is None:
            bounds = self.match.bounds()
            lb, ub = bounds.lower_bound, bounds.upper_bound
            bounds = sorted(f.bounds() for f in self._fringe)
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
        return f"{self.__class__.__name__}(node_from={self.node_from!r}, node_to={self.node_to!r}, up={self.up!r}, left={self.left!r}, diag={self.diag!r})"


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
        else:
            cost = node.total_size + 1
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


class EditDistance(Bounded):
    def __init__(
            self,
            from_node: TreeNode,
            to_node: TreeNode,
            from_seq: Sequence[TreeNode],
            to_seq: Sequence[TreeNode]
    ):
        self.from_node: TreeNode = from_node
        self.to_node: TreeNode = to_node
        from_seq: Sequence[TreeNode] = from_seq
        to_seq: Sequence[TreeNode] = to_seq
        self._bounds = Range(0, POSITIVE_INFINITY)
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
                        up=matrix[i-1][j],
                        left=matrix[i][j-1],
                        diag=matrix[i-1][j-1]
                    ))
        self._goal = matrix[len(to_seq)][len(from_seq)]

    def tighten_bounds(self) -> bool:
        return self._goal.tighten_bounds()

    def bounds(self) -> Range:
        return self._goal.bounds()

    def edits(self) -> CompoundEdit:
        while not self.bounds().definitive() and self.tighten_bounds():
            pass
        edits: List[Edit] = []
        node = self._goal
        while not isinstance(node, ConstantNode):
            best = node.best_predecessor()
            if best is node.left:
                edits.append(Remove(to_remove=node.node_from, remove_from=self.from_node))
            elif best is node.up:
                edits.append(Insert(to_insert=node.node_to, insert_into=self.from_node))
            else:
                assert best is node.diag
                edits.append(node.match)
            node = best
        while node.predecessor is not None:
            if node.is_from:
                edits.append(Remove(to_remove=node.node, remove_from=self.from_node))
            else:
                edits.append(Insert(to_insert=node.node, insert_into=self.from_node))
            node = node.predecessor

        return CompoundEdit(self.from_node, self.to_node, reversed(edits))

