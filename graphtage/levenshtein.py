"""An “`online`_”, “`constructive`_” implementation of the `Levenshtein distance metric`_.

The algorithm starts with an unbounded mapping and iteratively improves it until the bounds converge, at which point the
optimal edit sequence is discovered.

.. _online:
    https://en.wikipedia.org/wiki/Online_algorithm

.. _constructive:
    https://en.wikipedia.org/wiki/Constructive_proof

.. _Levenshtein distance metric:
    https://en.wikipedia.org/wiki/Levenshtein_distance

"""

import itertools
import logging
from collections.abc import Iterator, Sequence

import numpy as np

from .bounds import Range
from .edits import Insert, Match, Remove
from .fibonacci import FibonacciHeap
from .printer import get_default_printer
from .sequences import SequenceEdit
from .tree import Edit, TreeNode

log = logging.getLogger(__name__)


def levenshtein_distance(s: str, t: str) -> int:
    """Canonical implementation of the Levenshtein distance metric.

    Args:
        s: the string from which to match
        t: the string to which to match

    Returns:
        int: The Levenshtein edit distance metric between the two strings.

    """
    rows = len(s) + 1
    cols = len(t) + 1
    dist: list[list[int]] = [[0] * cols for _ in range(rows)]

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

    return dist[rows - 1][cols - 1]


class EditDistance(SequenceEdit):
    """An edit that computes the minimum sequence of sub-edits necessary to transform one node to another.

    The edits used to transform the source sequence to the target sequence are :class:`graphtage.Match`,
    :class:`graphtage.Remove`, and :class:`graphtage.Insert`.

    The algorithm works by iteratively constructing the Levenshtein matrix one diagonal at a time, starting from the
    upper left cell and ending at the lower right cell. Each successive call to
    :meth:`EditDistance.tighten_bounds` constructs a new diagonal of the matrix and fully tightens the bounds of its
    edits. Once the lower right cell is expanded, the matrix is complete and the optimal sequence of edits can be
    reconstructed.

    Bounds of this edit are updated after each diagonal is added by observing that the final cost is bounded above
    by the minimum cost of an edit in the last-expanded diagonal. This results in a monotonically decreasing upper
    bound.

    """

    __slots__ = (
        '_EditDistance__edits',
        '_fringe_col',
        '_fringe_row',
        '_last_fringe',
        'costs',
        'edit_matrix',
        'from_seq',
        'path_costs',
        'penalty',
        'reversed_shared_suffix',
        'shared_prefix',
        'to_seq'
    )

    def __init__(
            self,
            from_node: TreeNode,
            to_node: TreeNode,
            from_seq: Sequence[TreeNode],
            to_seq: Sequence[TreeNode],
            insert_remove_penalty: int = 1,
    ):
        """Initializes the edit distance edit.

        Args:
            from_node: The node that will be transformed.
            to_node: The node into which :obj:`from_node` will be transformed.
            from_seq: A sequence of nodes that comprise :obj:`from_node`.
            to_seq: A sequence of nodes that comprise :obj:`to_node`.
            insert_remove_penalty: The penalty for inserting or removing a node (default is 1).

        """
        self.penalty: int = insert_remove_penalty
        # Optimization: See if the sequences trivially share a common prefix or suffix.
        # If so, this will quadratically reduce the size of the Levenshtein matrix
        self.shared_prefix: list[tuple[TreeNode, TreeNode]] = []
        for fn, tn in zip(from_seq, to_seq, strict=False):
            if fn == tn:
                self.shared_prefix.append((fn, tn))
            else:
                break
        self.reversed_shared_suffix: list[tuple[TreeNode, TreeNode]] = []
        for fn, tn in zip(
                reversed(from_seq[len(self.shared_prefix):]),
                reversed(to_seq[len(self.shared_prefix):]),
                strict=False
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
                constant_cost += sizes.pop().total_size + self.penalty
        cost_upper_bound = (
            sum(node.total_size + self.penalty for node in from_seq) +
            sum(node.total_size + self.penalty for node in to_seq)
        )
        self.edit_matrix: list[list[Edit | None]] = [
            [None] * (len(self.from_seq) + 1) for _ in range(len(self.to_seq) + 1)
        ]
        self.path_costs = np.full((len(self.to_seq) + 1, len(self.from_seq) + 1), 0, dtype=np.uint16)
        self.costs = np.full((len(self.to_seq) + 1, len(self.from_seq) + 1), 0, dtype=np.uint64)
        self._fringe_row: int = -1
        self._fringe_col: int = 0
        self._last_fringe: list[tuple[int, int]] = []
        super().__init__(
            from_node=from_node,
            to_node=to_node,
            constant_cost=constant_cost,
            cost_upper_bound=cost_upper_bound
        )
        self.__edits: list[Edit] | None = None

    def _add_node(self, row: int, col: int) -> bool:
        if self.edit_matrix[row][col] is not None or col > len(self.from_seq) or row > len(self.to_seq):
            return False
        if row == 0 and col == 0:
            edit = None
        elif row == 0:
            edit = Remove(to_remove=self.from_seq[col - 1], remove_from=self.from_node, penalty=self.penalty)
            self.costs[0][col] = self.costs[0][col-1] + edit.bounds().upper_bound
            self.path_costs[0][col] = self.path_costs[0][col - 1] + 1
        elif col == 0:
            edit = Insert(to_insert=self.to_seq[row - 1], insert_into=self.from_node, penalty=self.penalty)
            self.costs[row][0] = self.costs[row - 1][0] + edit.bounds().upper_bound
            self.path_costs[row][0] = self.path_costs[row - 1][0] + 1
        else:
            edit = self.from_seq[col-1].edits(self.to_seq[row-1])
        self.edit_matrix[row][col] = edit
        return True

    def _fringe_diagonal(self) -> Iterator[tuple[int, int]]:
        row, col = self._fringe_row, self._fringe_col
        while row >= 0 and col <= len(self.from_seq):
            yield row, col
            row -= 1
            col += 1

    def _next_fringe(self) -> bool:
        if self.is_complete():
            return False
        self._last_fringe = list(self._fringe_diagonal())
        self._fringe_row += 1
        if self._fringe_row >= len(self.to_seq) + 1:
            self._fringe_row = len(self.to_seq)
            self._fringe_col += 1
        for row, col in self._fringe_diagonal():
            self._add_node(row, col)
        if self._fringe_col >= len(self.from_seq):
            # This is an edge case when the string we are matching from is shorter than the one we are matching to
            return self._fringe_row < len(self.to_seq)
        return True

    def is_complete(self) -> bool:
        """An edit distance edit is only complete once its Levenshtein edit matrix has been fully constructed."""
        return self.edit_matrix is None or self.edit_matrix[-1][-1] is not None

    @staticmethod
    def _exact_cost(edit: Edit) -> int:
        """Tightens an edit until its bounds are definitive and returns its exact cost.

        Args:
            edit: The edit to price.

        Returns:
            int: The exact cost of the edit.

        Raises:
            ValueError: If the edit cannot be tightened to a definitive bound.

        """
        while not edit.bounds().definitive() and edit.tighten_bounds():
            pass
        bounds = edit.bounds()
        if not bounds.definitive():
            raise ValueError(f"Could not tighten {edit!r} to a definitive bound; got {bounds!r}")
        return bounds.upper_bound

    def _best_match(self, row: int, col: int) -> tuple[int, int, Edit]:
        """Selects the predecessor cell that reaches this cell of the Levenshtein matrix most cheaply.

        Each candidate is scored by the accumulated cost of its predecessor plus the cost of the edit that
        transitions from that predecessor to this cell. The number of edits along the path is the secondary
        key, which prefers a single substitution over an insertion paired with a removal of equal total cost.

        Ties on both keys are broken by direction, in this fixed order: the diagonal (a substitution) wins over
        both borders, and the border insertion wins over the border removal. Reconstruction walks the matrix
        backwards, so preferring the insertion here places the removal earlier in the forward edit sequence,
        matching the convention of listing deletions before additions. This order is part of the output
        contract: changing it changes the edit sequence for inputs that have several optimal alignments.

        Args:
            row: The row of the cell, indexing :attr:`EditDistance.to_seq`.
            col: The column of the cell, indexing :attr:`EditDistance.from_seq`.

        Returns:
            Tuple[int, int, Edit]: The row and column of the chosen predecessor, and the transition edit.

        """
        if row == 0:
            assert col > 0
            return 0, col - 1, self.edit_matrix[0][col]
        elif col == 0:
            assert row > 0
            return row - 1, col, self.edit_matrix[row][0]
        best_key: tuple[int, int] | None = None
        best: tuple[int, int, Edit] | None = None
        for prev_row, prev_col, edit in (
                (row - 1, col - 1, self.edit_matrix[row][col]),
                (row - 1, col, self.edit_matrix[row][0]),
                (row, col - 1, self.edit_matrix[0][col]),
        ):
            key = (
                int(self.costs[prev_row][prev_col]) + self._exact_cost(edit),
                int(self.path_costs[prev_row][prev_col]) + 1,
            )
            if best_key is None or key < best_key:
                best_key, best = key, (prev_row, prev_col, edit)
        self.costs[row][col], self.path_costs[row][col] = best_key
        return best

    def tighten_bounds(self) -> bool:
        """Tightens the bounds of this edit, if possible.

        If the Levenshtein matrix is not yet complete, construct and fully tighten the next diagonal of the matrix.

        """
        if not self.from_seq and not self.to_seq:
            return False
        elif self.edit_matrix is None:
            # This means we are already fully tightened and deleted the interstitial datastructures to save memory
            return False
        elif self.is_complete() and not self.edit_matrix[-1][-1].bounds().definitive():
            return self.edit_matrix[-1][-1].tighten_bounds()
        # We are still building the matrix
        initial_bounds: Range = self.bounds()
        while True:
            first_fringe = self._fringe_row < 0

            # Tighten the entire fringe diagonal until every node in it is definitive
            if not self._next_fringe():
                assert self.is_complete()
                if not self.edit_matrix[-1][-1].bounds().definitive():
                    ret = self.tighten_bounds()
                else:
                    ret = False
                if not ret:
                    self._cleanup()
                return ret

            if not first_fringe:
                fringe_ranges = {}
                fringe_total = 0
                num_diagonals = 0
                printer = get_default_printer()

                if not printer.quiet:
                    fringe_ranges = {
                        (row, col): (
                            self.edit_matrix[row][col].bounds().upper_bound
                            - self.edit_matrix[row][col].bounds().lower_bound
                        )
                        for row, col in self._fringe_diagonal()
                    }
                    fringe_total = sum(fringe_ranges.values())
                    num_diagonals = len(self.from_seq) + len(self.to_seq)

                with printer.tqdm(
                        total=fringe_total,
                        initial=0,
                        desc=f"Tightening Fringe Diagonal {self._fringe_row + self._fringe_col} of {num_diagonals}",
                        disable=fringe_total <= 0,
                        leave=False
                ) as t:
                    for row, col in self._fringe_diagonal():
                        while self.edit_matrix[row][col].tighten_bounds():
                            if fringe_total:
                                new_bounds = self.edit_matrix[row][col].bounds()
                                new_range = new_bounds.upper_bound - new_bounds.lower_bound
                                t.update(fringe_ranges[(row, col)] - new_range)
                                fringe_ranges[(row, col)] = new_range
                        assert self.edit_matrix[row][col].bounds().definitive()
                        # Call self._best_match because it sets self.path_costs and self.costs for this cell
                        _, _, _ = self._best_match(row, col)

            if self.bounds().upper_bound < initial_bounds.upper_bound or \
                    self.bounds().lower_bound > initial_bounds.lower_bound:
                return True

    def bounds(self) -> Range:
        """Calculates bounds on the cost of this edit.

        If the Levenshtein matrix has been fully constructed, return the cost of the lower right cell.

        If the matrix is incomplete, then use
        :meth:`super().bounds().lower_bound <graphtage.sequences.SequenceEdit.bounds>` as the lower bound and the
        minimum cost in the last completed matrix diagonal as the upper bound.

        Returns:
            Range: The bounds on the cost of this edit.

        """
        if not self.from_seq and not self.to_seq:
            # The shared prefix and suffix consumed both sequences, so every edit is a zero-cost match
            return Range(0, 0)
        base_bounds: Range = super().bounds()
        if self.is_complete():
            if self.__edits is None:
                # We need to construct the edits to finalize the cost matrix:
                _ = self.edits()
            cost = int(self.costs[len(self.to_seq)][len(self.from_seq)])
            return Range(cost, cost)
        else:
            if self._fringe_row <= 0:
                return base_bounds
            return Range(
                max(base_bounds.lower_bound, min(min(
                    int(self.costs[row][col]) for row, col in self._fringe_diagonal()
                ), min(
                    int(self.costs[row][col]) for row, col in self._last_fringe
                ))),
                base_bounds.upper_bound
            )

    def _cleanup(self):
        if self.bounds().definitive() and self.edit_matrix is not None:
            if self.__edits is None:
                self.edits()
            assert self.__edits is not None
            # we don't need the matrix anymore, so save memory by wiping it out
            self.edit_matrix = None
            self.path_costs = None
            # We only need the last cell in the costs matrix, so switch to using a dict to clean up the others:
            self.costs = {len(self.to_seq): {len(self.from_seq): self.costs[len(self.to_seq)][len(self.from_seq)]}}

    def edits(self) -> Iterator[Edit]:
        if self.__edits is None:
            reversed_suffix: list[Edit] = [
                Match(from_node, to_node, 0) for from_node, to_node in self.reversed_shared_suffix
            ]
            if self.to_seq or self.from_seq:
                while not self.is_complete() and self.tighten_bounds():
                    pass
                assert self.is_complete()
                if self.__edits is None:
                    assert len(self.edit_matrix) == len(self.to_seq) + 1
                    assert len(self.edit_matrix[0]) == len(self.from_seq) + 1
                    row, col = len(self.to_seq), len(self.from_seq)
                    while row > 0 or col > 0:
                        prev_row, prev_col, edit = self._best_match(row, col)
                        reversed_suffix.append(edit)
                        row, col = prev_row, prev_col
                    self.__edits = reversed_suffix
            else:
                self.__edits = reversed_suffix
            self._cleanup()
        return itertools.chain(
            (Match(from_node, to_node, 0) for from_node, to_node in self.shared_prefix),
            reversed(self.__edits)
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}<from_seq={list(map(str, self.from_seq))!r}, to_seq={list(map(str, self.to_seq))!r}, insert_remove_penalty={self.penalty}>"
