import itertools
import logging
from typing import Iterator, List, Optional, Sequence, Tuple, Union

from .bounds import Bounded, Range
from .edits import Insert, Match, Remove
from .fibonacci import FibonacciHeap
from .printer import DEFAULT_PRINTER
from .sequences import SequenceEdit
from .tree import Edit, TreeNode
from .utils import SparseMatrix


log = logging.getLogger(__name__)


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


class SearchNode(Bounded):
    def __init__(self, cost: int, edit: Optional[Edit] = None):
        self.cost = cost
        self.edit: Optional[Edit] = edit

    def bounds(self) -> Range:
        if self.edit is not None:
            return self.edit.bounds() + Range(self.cost, self.cost)
        else:
            return Range(self.cost, self.cost)

    def tighten_bounds(self) -> bool:
        if self.edit is not None:
            return self.edit.tighten_bounds()
        else:
            return False

    def __lt__(self, other):
        return self.bounds() < other.bounds() or (
                self.bounds() == other.bounds() and self.edit.bounds() < other.edit.bounds()
        )

    def __le__(self, other):
        return self.bounds() <= other.bounds()

    def __repr__(self):
        return f"{self.__class__.__name__}(cost={self.cost}, edit={self.edit!r})"

    def __str__(self):
        return str(self.bounds())


class AStarNode:
    def __init__(self, row: int, col: int, parent: Union['EditDistance', 'AStarNode']):
        if isinstance(parent, EditDistance):
            self.parent: Optional[AStarNode] = None
            self.matrix: SparseMatrix[SearchNode] = parent.matrix
            self.path_cost = 0
        else:
            self.parent: Optional[AStarNode] = parent
            self.path_cost: int = parent.path_cost + 1
            self.matrix = self.parent.matrix
        self.row = row
        self.col = col

    def successors(self) -> Iterator['AStarNode']:
        if self.col == 0 and self.row == 0:
            return
        if self.col == 0:
            left_cell = None
        else:
            left_cell = self.matrix[self.row][self.col - 1]
        if self.row == 0:
            up_cell = None
        else:
            up_cell = self.matrix[self.row - 1][self.col]
        if self.col == 0 and self.row == 0:
            diag_cell = None
        else:
            diag_cell = self.matrix[self.row - 1][self.col - 1]
        if diag_cell is not None:
            assert left_cell is not None
            assert up_cell is not None
            d_bound = diag_cell.bounds().upper_bound
            l_bound = left_cell.bounds().upper_bound
            u_bound = up_cell.bounds().upper_bound
            min_cost = min(d_bound, l_bound, u_bound)
            if d_bound == min_cost:
                node = AStarNode(self.row - 1, self.col - 1, self)
                if l_bound > min_cost and u_bound > min_cost:
                    yield node
                elif d_bound < self.matrix[self.row][self.col].edit.from_node.total_size:
                    if l_bound == min_cost and u_bound == min_cost:
                        assert min_cost > 0
                        # This match is equivalent to removing and inserting, so increase the path cost to make it fair
                        node.path_cost += 1
                    yield node
            if l_bound == min_cost:
                yield AStarNode(self.row, self.col - 1, self)
            if u_bound == min_cost:
                yield AStarNode(self.row - 1, self.col, self)
        elif left_cell is not None:
            assert up_cell is None
            yield AStarNode(self.row, self.col - 1, self)
        else:
            assert up_cell is not None
            yield AStarNode(self.row - 1, self.col, self)

    def heuristic(self) -> int:
        return max(self.row, self.col)

    def __lt__(self, other: 'AStarNode'):
        my_f = self.path_cost + self.heuristic()
        other_f = other.path_cost + other.heuristic()
        return my_f < other_f

    def is_goal(self) -> bool:
        return self.row == 0 and self.col == 0

    def edits(self) -> Iterator[Edit]:
        node = self
        while node.parent is not None:
            if node.parent.row == node.row:
                yield self.matrix[0][node.parent.col].edit
            elif node.parent.col == node.col:
                yield self.matrix[node.parent.row][0].edit
            else:
                yield self.matrix[node.parent.row][node.parent.col].edit
            node = node.parent


class EditDistance(SequenceEdit):
    def __init__(
            self,
            from_node: TreeNode,
            to_node: TreeNode,
            from_seq: Sequence[TreeNode],
            to_seq: Sequence[TreeNode]
    ):
        # Optimization: See if the sequences trivially share a common prefix or suffix.
        # If so, this will quadratically reduce the size of the Levenshtein matrix
        self.shared_prefix: List[Tuple[TreeNode, TreeNode]] = []
        for fn, tn in zip(from_seq, to_seq):
            if fn == tn:
                self.shared_prefix.append((fn, tn))
            else:
                break
        self.reversed_shared_suffix: List[Tuple[TreeNode, TreeNode]] = []
        for fn, tn in zip(
                reversed(from_seq[len(self.shared_prefix):]),
                reversed(to_seq[len(self.shared_prefix):])
        ):
            if fn == tn:
                self.reversed_shared_suffix.append((fn, tn))
            else:
                break
        self.reversed_shared_suffix = self.reversed_shared_suffix
        self.from_seq: Sequence[TreeNode] = from_seq[
                                                len(self.shared_prefix):len(from_seq)-len(self.reversed_shared_suffix)
                                            ]
        self.to_seq: Sequence[TreeNode] = to_seq[
                                            len(self.shared_prefix):len(to_seq)-len(self.reversed_shared_suffix)
                                          ]
        log.debug(f"Levenshtein len(shared prefix)={len(self.shared_prefix)}, len(shared suffix)={len(self.reversed_shared_suffix)}, len(from_seq)={len(self.from_seq)}, len(to_seq)={len(self.to_seq)}")
        constant_cost = 0
        if len(from_seq) != len(to_seq):
            sizes: FibonacciHeap[TreeNode, int] = FibonacciHeap(key=lambda node: node.total_size)
            if len(from_seq) < len(to_seq):
                smaller, larger = from_seq, to_seq
            else:
                larger, smaller = from_seq, to_seq
            for node in larger:
                sizes.push(node)
            for _ in range(len(larger) - len(smaller)):
                constant_cost += sizes.pop().total_size
        cost_upper_bound = sum(node.total_size for node in from_seq) + sum(node.total_size for node in to_seq)
        self.matrix: SparseMatrix[SearchNode] = SparseMatrix(
            num_rows=len(self.to_seq) + 1,
            num_cols=len(self.from_seq) + 1,
            default_value=None
        )
        self._fringe_row: int = -1
        self._fringe_col: int = 0
        self.__goal: Optional[SearchNode] = None
        super().__init__(
            from_node=from_node,
            to_node=to_node,
            constant_cost=constant_cost,
            cost_upper_bound=cost_upper_bound
        )
        self.__edits: Optional[List[Edit]] = None

    def _add_node(self, row: int, col: int) -> bool:
        if self.matrix[row][col] is not None or col > len(self.from_seq) or row > len(self.to_seq):
            return False
        if row == 0 and col == 0:
            self.matrix[row][col] = SearchNode(cost=row + col)
            return True
        elif row == 0:
            edit = Remove(to_remove=self.from_seq[col - 1], remove_from=self.from_node, penalty=0)
            fringe_cost = self.matrix[0][col - 1].bounds().upper_bound
        elif col == 0:
            edit = Insert(to_insert=self.to_seq[row - 1], insert_into=self.from_node, penalty=0)
            fringe_cost = self.matrix[row - 1][0].bounds().upper_bound
        else:
            edit = self.from_seq[col-1].edits(self.to_seq[row-1])
            fringe_cost = min(
                self.matrix[row - 1][col - 1].bounds().upper_bound,
                self.matrix[row - 1][col].bounds().upper_bound,
                self.matrix[row][col - 1].bounds().upper_bound
            )
        self.matrix[row][col] = SearchNode(cost=fringe_cost, edit=edit)
        return True

    @property
    def _goal(self) -> Optional[SearchNode]:
        if self.__goal is None:
            self.__goal = self.matrix[len(self.to_seq)][len(self.from_seq)]
        return self.__goal

    def _fringe_diagonal(self) -> Iterator[Tuple[int, int]]:
        row, col = self._fringe_row, self._fringe_col
        while row >= 0 and col < self.matrix.num_cols:
            yield row, col
            row -= 1
            col += 1

    def _next_fringe(self) -> bool:
        if self._goal is not None:
            return False
        self._fringe_row += 1
        if self._fringe_row >= self.matrix.num_rows:
            self._fringe_row = self.matrix.num_rows - 1
            self._fringe_col += 1
        for row, col in self._fringe_diagonal():
            self._add_node(row, col)
        if self._fringe_col >= self.matrix.num_cols - 1:
            if self._fringe_row < self.matrix.num_rows - 1:
                # This is an edge case when the string we are matching from is shorter than the one we are matching to
                assert self._goal is None
                return True
            return False
        else:
            return True

    def is_complete(self) -> bool:
        return self._goal is not None

    def tighten_bounds(self) -> bool:
        if self._goal is not None:
            return self._goal.tighten_bounds()
        # We are still building the matrix
        initial_bounds: Range = self.bounds()
        while True:
            # Tighten the entire fringe diagonal until every node in it is definitive
            if not self._next_fringe():
                assert self._goal is not None
                return self.tighten_bounds()
            if DEFAULT_PRINTER.quiet:
                fringe_ranges = {}
                fringe_total = 0
            else:
                fringe_ranges = {
                    (row, col): self.matrix[row][col].bounds().upper_bound - self.matrix[row][col].bounds().lower_bound
                    for row, col in self._fringe_diagonal()
                }
                fringe_total = sum(fringe_ranges.values())

            with DEFAULT_PRINTER.tqdm(
                    total=fringe_total,
                    initial=fringe_total,
                    desc=f"Tightening Fringe Diagonal {self._fringe_row + self._fringe_col}",
                    disable=fringe_total <= 0,
                    leave=False
            ) as t:
                for row, col in self._fringe_diagonal():
                    while self.matrix[row][col].tighten_bounds():
                        if fringe_total:
                            new_bounds = self.matrix[row][col].bounds()
                            new_range = new_bounds.upper_bound - new_bounds.lower_bound
                            t.update(fringe_ranges[(row, col)] - new_range)
                            fringe_ranges[(row, col)] = new_range
                    assert self.matrix[row][col].bounds().definitive()
            if self.bounds().upper_bound < initial_bounds.upper_bound or \
                    self.bounds().lower_bound > initial_bounds.lower_bound:
                return True

    def bounds(self) -> Range:
        if self._goal is not None:
            return self._goal.bounds()
        else:
            base_bounds: Range = super().bounds()
            if self._fringe_row < 0:
                return base_bounds
            return Range(
                max(base_bounds.lower_bound, min(
                    self.matrix[row][col].bounds().lower_bound for row, col in self._fringe_diagonal()
                )),
                base_bounds.upper_bound
            )

    def edits(self) -> Iterator[Edit]:
        if self.__edits is None:
            while self._goal is None and self.tighten_bounds():
                pass
            assert self._goal is not None
            assert self.matrix.num_rows == len(self.to_seq) + 1
            assert self.matrix.num_cols == len(self.from_seq) + 1
            # Minimize the number of edits in the minimum-cost path by using A*
            queue: FibonacciHeap[AStarNode, AStarNode] = FibonacciHeap()
            queue.push(AStarNode(self.matrix.num_rows - 1, self.matrix.num_cols - 1, self))
            while queue:
                node = queue.pop()
                if node.is_goal():
                    self.__edits = list(node.edits())
                    break
                for successor in node.successors():
                    queue.push(successor)
            else:
                raise RuntimeError("Could not find a path from the goal to the origin! This should never happen!")
            # we only need the goal cell in the matrix, so save memory by wiping out the rest:
            if log.isEnabledFor(logging.DEBUG):
                size_before = self.matrix.getsizeof()
            new_matrix: SparseMatrix[SearchNode] = SparseMatrix(
                num_rows=len(self.to_seq) + 1,
                num_cols=len(self.from_seq) + 1,
                default_value=None
            )
            new_matrix[len(self.to_seq)][len(self.from_seq)] = self._goal
            if log.isEnabledFor(logging.DEBUG):
                size_after = new_matrix.getsizeof()
                log.debug(f"Cleaned up {size_before - size_after} bytes")
            self.matrix = new_matrix
            self.__edits.extend(
                [Match(from_node, to_node, 0) for from_node, to_node in reversed(self.reversed_shared_suffix)]
            )
        return itertools.chain(
            (Match(from_node, to_node, 0) for from_node, to_node in self.shared_prefix),
            self.__edits
        )
