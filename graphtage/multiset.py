import itertools
from typing import Iterator, List, Tuple

from .edits import CompoundEdit, Edit, Insert, Match, Remove
from .fibonacci import FibonacciHeap
from .search import Range
from .tree import TreeNode
from .utils import HashableCounter


class EditComparator:
    def __init__(self, edit: Edit):
        self.edit = edit

    def __lt__(self, other):
        while not (
                self.edit.bounds().dominates(other.edit.bounds())
                or
                other.edit.bounds().dominates(self.edit.bounds())
        ) and (
                self.edit.tighten_bounds() or other.edit.tighten_bounds()
        ):
            pass
        return self.edit.bounds().dominates(other.edit.bounds()) or (
                self.edit.bounds() == other.edit.bounds() and id(self) < id(other)
        )

    def __le__(self, other):
        if self < other:
            return True
        while self.edit.tighten_bounds() or other.edit.tighten_bounds():
            pass
        return self.edit.bounds() == other.edit.bounds()


class MultiSetEdit(CompoundEdit):
    def __init__(
            self,
            from_node: TreeNode,
            to_node: TreeNode,
            from_set: HashableCounter[TreeNode],
            to_set: HashableCounter[TreeNode]
    ):
        self.from_set: HashableCounter[TreeNode] = from_set
        self.to_set: HashableCounter[TreeNode] = to_set
        self.to_insert = to_set - from_set
        self.to_remove = from_set - to_set
        self.to_match = from_set & to_set
        self._edits: List[Edit] = [Match(n, n, 0) for n in self.to_match.elements()]
        self.best_matches: FibonacciHeap[Edit, EditComparator] = FibonacciHeap(key=EditComparator)
        self._node_combos: Iterator[Tuple[TreeNode, TreeNode]] = \
            itertools.product(self.to_remove.keys(), self.to_insert.keys())
        self.removed = set()
        super().__init__(
            from_node=from_node,
            to_node=to_node
        )

    def edits(self) -> Iterator[Edit]:
        while self.best_matches and self.tighten_bounds():
            pass
        return iter(self._edits)

    def tighten_bounds(self) -> bool:
        starting_bounds: Range = self.cost()
        while self._node_combos is not None:
            try:
                node_from, node_to = next(self._node_combos)
                edit = node_from.edits(node_to)
                if edit.bounds().lower_bound >= node_from.total_size + node_to.total_size + 2:
                    # This edit is no better than replacing/inserting the nodes
                    pass
                else:
                    self.best_matches.push(edit)
            except StopIteration:
                self._node_combos = None
        while self.best_matches:
            best_edit: Edit = self.best_matches.pop()
            if best_edit.from_node in self.removed or best_edit.to_node in self.removed:
                continue
            assert best_edit.from_node in self.to_remove
            assert best_edit.to_node in self.to_insert
            while best_edit.cost().upper_bound > best_edit.from_node.total_size + best_edit.to_node.total_size \
                    and best_edit.tighten_bounds():
                pass
            if best_edit.cost().upper_bound <= best_edit.from_node.total_size + best_edit.to_node.total_size:
                # it is better to match these nodes than to replace/insert them
                self.removed.add(best_edit.from_node)
                self.removed.add(best_edit.to_node)
                # for heap_node in itertools.chain(
                #         heap_nodes[best_edit.from_node], heap_nodes[best_edit.to_node]
                # ):
                #     if not heap_node.deleted:
                #         pass
                #         # best_matches.remove(heap_node)
                num_to_match = min(self.to_remove[best_edit.from_node], self.to_insert[best_edit.to_node])
                assert num_to_match > 0
                for _ in range(num_to_match):
                    self._edits.append(best_edit)
                self.to_insert[best_edit.to_node] -= num_to_match
                self.to_remove[best_edit.from_node] -= num_to_match
                cost = self.cost()
                assert cost.lower_bound >= starting_bounds.lower_bound
                assert cost.upper_bound <= starting_bounds.upper_bound
                if cost.lower_bound > starting_bounds.lower_bound or cost.upper_bound < starting_bounds.upper_bound:
                    return True
        if self.to_remove:
            self._edits.extend(Remove(remove_from=self.from_node, to_remove=node) for node in self.to_remove.elements())
            self.to_remove.clear()
        if self.to_insert:
            self._edits.extend(Insert(insert_into=self.from_node, to_insert=node) for node in self.to_insert.elements())
            self.to_insert.clear()
        cost = self.cost()
        if not cost.definitive():
            for edit in self.edits():
                if edit.tighten_bounds():
                    return True
        return cost.lower_bound > starting_bounds.lower_bound or cost.upper_bound < starting_bounds.upper_bound

    def cost(self) -> Range:
        lower_bound = sum(e.bounds().lower_bound for e in self._edits)
        upper_bound = sum((i.total_size + 1) * c for i, c in self.to_remove.items()) + \
            sum((i.total_size + 1) * c for i, c in self.to_insert.items()) + \
            sum(e.bounds().upper_bound for e in self._edits)
        return Range(lower_bound, upper_bound)
