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
    def __init__(
            self,
            up: Optional['SearchEdge'] = None,
            left: Optional['SearchEdge'] = None,
            diag: Optional['SearchEdge'] = None
    ):
        self._path_cost: Optional[int] = None
        self.up: Optional[SearchEdge] = up
        self.left: Optional[SearchEdge] = left
        self.diag: Optional[SearchEdge] = diag
        self._best_predecessor: Optional[SearchEdge] = None

    @property
    def path_cost(self) -> int:
        if self._path_cost is None:
            bp = self.best_predecessor
            if bp is None:
                self._path_cost = 0
            else:
                self._path_cost = bp.to_node.path_cost + 1
        return self._path_cost

    @property
    def best_predecessor(self) -> Optional['SearchEdge']:
        if self.diag is None:
            if self.left is not None:
                return self.left
            elif self.up is not None:
                return self.up
            else:
                return None
        elif self._best_predecessor is None:
            if not self.diag.edit.bounds().definitive():
                raise RuntimeError("best_predecessor can only be called once a SearchNode has been fully tightened!")
            if (self.diag <= self.up and self.diag <= self.left) \
                    and self.diag.edit.bounds() < self.up.edit.bounds() \
                    and self.diag.edit.bounds() < self.left.edit.bounds():
                self._best_predecessor = self.diag
            elif self.left < self.up:
                self._best_predecessor = self.left
            else:
                self._best_predecessor = self.up
        return self._best_predecessor

    def bounds(self) -> Range:
        if self.diag is None and self.left is None and self.up is None:
            return Range(0, 0)
        elif self.diag is None or self.diag.edit.bounds().definitive():
            return self.best_predecessor.bounds()
        else:
            return Range(
                min(
                    self.left.bounds().lower_bound,
                    self.up.bounds().lower_bound,
                    self.diag.bounds().lower_bound
                ),
                max(
                    self.left.bounds().upper_bound,
                    self.up.bounds().upper_bound,
                    self.diag.bounds().upper_bound
                )
            )

    def tighten_bounds(self) -> bool:
        if self.diag is not None:
            return self.diag.edit.tighten_bounds()
        else:
            return False

    def __lt__(self, other):
        our_bounds = self.bounds()
        other_bounds = other.bounds()
        return our_bounds < other_bounds or (
            (our_bounds == other_bounds and self.path_cost < other.path_cost) or (
                (our_bounds == other_bounds and self.path_cost == other.path_cost
                    and self.bounds() < other.bounds())))

    def __le__(self, other):
        return self < other or self == other

    def __eq__(self, other):
        return self.cost == other.cost and self.bounds() == other.bounds() and self.path_cost == other.path_cost

    def __repr__(self):
        return f"{self.__class__.__name__}(up={self.up!r}, left={self.left!r}, diag={self.diag!r})"

    def __str__(self):
        return str(self.bounds())


class SearchEdge(Bounded):
    def __init__(self, to_node: SearchNode, edit: Edit):
        self.to_node: SearchNode = to_node
        self.edit: Edit = edit

    def tighten_bounds(self) -> bool:
        return self.edit.tighten_bounds()

    def bounds(self) -> Range:
        return self.edit.bounds() + self.to_node.bounds()

    def __lt__(self, other):
        return self.bounds() < other.bounds() or (
                self.bounds() == other.bounds() and self.to_node.path_cost < other.to_node.path_cost
        )

    def __le__(self, other):
        return self < other or self == other

    def __eq__(self, other):
        return self.bounds() == other.bounds() and self.to_node.path_cost == other.to_node.path_cost


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
            self.matrix[row][col] = SearchNode()
            return True
        if col > 0:
            if row == 0:
                edit = Remove(to_remove=self.from_seq[col - 1], remove_from=self.from_node, penalty=0)
            else:
                edit = self.matrix[0][col].left.edit
            left = SearchEdge(
                to_node=self.matrix[row][col - 1],
                edit=edit
            )
        else:
            left = None
        if row > 0:
            if col == 0:
                edit = Insert(to_insert=self.to_seq[row - 1], insert_into=self.from_node, penalty=0)
            else:
                edit = self.matrix[row][0].up.edit
            up = SearchEdge(
                to_node=self.matrix[row - 1][col],
                edit=edit
            )
        else:
            up = None
        if col > 0 and row > 0:
            diag = SearchEdge(
                to_node=self.matrix[row - 1][col - 1],
                edit=self.from_seq[col-1].edits(self.to_seq[row-1])
            )
        else:
            diag = None
        self.matrix[row][col] = SearchNode(
            up=up,
            left=left,
            diag=diag
        )
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

            if len(fringe_ranges) == 0 or fringe_total > 0:
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
            self.__edits: List[Edit] = [
                Match(from_node, to_node, 0) for from_node, to_node in self.reversed_shared_suffix
            ]
            node = self._goal
            while node.best_predecessor is not None:
                self.__edits.append(node.best_predecessor.edit)
                node = node.best_predecessor.to_node
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
        return itertools.chain(
            (Match(from_node, to_node, 0) for from_node, to_node in self.shared_prefix),
            reversed(self.__edits)
        )
