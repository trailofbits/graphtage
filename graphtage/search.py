from typing import Generic, Iterator, Optional, TypeVar

from .bounds import Bounded, NEGATIVE_INFINITY, POSITIVE_INFINITY, Range
from .fibonacci import FibonacciHeap, HeapNode

B = TypeVar('B', bound=Bounded)


class IterativeTighteningSearch(Generic[B]):
    def __init__(self,
                 possibilities: Iterator[B],
                 initial_bounds: Optional[Range] = None):
        def get_range(bounded: Bounded) -> Range:
            return bounded.bounds()

        self._unprocessed: Iterator[B] = possibilities

        # Heap to track the ranges with the lowest upper bound
        self._untightened: FibonacciHeap[B, Range] = FibonacciHeap(key=get_range)

        # Fully tightened (`definitive`) ranges, sorted by increasing bound
        self._tightened: FibonacciHeap[B, Range] = FibonacciHeap(key=get_range)

        if initial_bounds is None:
            self.initial_bounds = Range(NEGATIVE_INFINITY, POSITIVE_INFINITY)
        else:
            self.initial_bounds = initial_bounds

    def __bool__(self):
        return bool(self._unprocessed or ((self._untightened or self._tightened) and not self.bounds().definitive()))

    @property
    def best_match(self) -> B:
        if self._unprocessed is not None or not (self._untightened or self._tightened):
            return None
        elif self._tightened and self._untightened:
            if self._untightened.peek().bounds() < self._tightened.peek().bounds():
                return self._untightened.peek()
            else:
                return self._tightened.peek()
        elif self._tightened:
            return self._tightened.peek()
        else:
            return self._untightened.peek()

    def remove_best(self) -> B:
        heap: Optional[FibonacciHeap[B, Range]] = None
        if self._unprocessed is not None or not (self._untightened or self._tightened):
            return None
        elif self._tightened and self._untightened:
            if self._untightened.peek().bounds() < self._tightened.peek().bounds():
                heap = self._untightened
            else:
                heap = self._tightened
        elif self._tightened:
            heap = self._tightened
        else:
            heap = self._untightened
        return heap.pop()

    def search(self) -> B:
        while self.tighten_bounds():
            pass
        return self.best_match

    def _nodes(self) -> Iterator[HeapNode[B, Range]]:
        yield from self._untightened.nodes()
        yield from self._tightened.nodes()

    def bounds(self) -> Range:
        if self.best_match is None:
            return self.initial_bounds
        else:
            if self._unprocessed is None and (self._untightened or self._tightened):
                lb = POSITIVE_INFINITY
                for node in self._nodes():
                    if not node.deleted:
                        lb = min(node.key.lower_bound, lb)
                if lb == POSITIVE_INFINITY or lb < self.initial_bounds.lower_bound:
                    lb = self.initial_bounds.lower_bound
            else:
                lb = self.initial_bounds.lower_bound
            return Range(min(lb, self.best_match.bounds().upper_bound), self.best_match.bounds().upper_bound)

    def _delete_node(self, node: HeapNode[B, Range]):
        self._untightened.decrease_key(node, Range(NEGATIVE_INFINITY, NEGATIVE_INFINITY))
        self._untightened.pop()
        node.deleted = True

    def _update_bounds(self, node: HeapNode[B, Range]):
        if self.best_match is not None \
                and self.best_match != node.item \
                and self.best_match.bounds().dominates(node.item.bounds()):
            self._delete_node(node)
            return
        elif self.initial_bounds.dominates(node.item.bounds()):
            self._delete_node(node)
            return
        bounds: Range = node.item.bounds()
        if bounds.definitive():
            self._delete_node(node)
            self._tightened.push(node.item)
        elif bounds.lower_bound > node.key.lower_bound:
            # The lower bound increased, so we need to remove and re-add the node
            # because the Fibonacci heap only permits making keys smaller
            self._untightened.decrease_key(node, Range(NEGATIVE_INFINITY, NEGATIVE_INFINITY))
            self._untightened.pop()
            self._untightened.push(node.item)

    def goal_test(self) -> bool:
        if self._unprocessed is not None:
            return False
        best = self.best_match
        return best is not None and best.bounds().dominates(self.bounds())

    def tighten_bounds(self) -> bool:
        starting_bounds = self.bounds()
        while True:
            if self._unprocessed is not None:
                try:
                    next_best: B = next(self._unprocessed)
                    if self.initial_bounds.lower_bound > NEGATIVE_INFINITY and \
                            self.initial_bounds.lower_bound >= next_best.bounds().upper_bound:
                        # We can't do any better than this choice!
                        self._unprocessed = None
                        self._untightened.clear()
                        self._tightened.clear()
                        if next_best.bounds().definitive():
                            self._tightened.push(next_best)
                        else:
                            self._untightened.push(next_best)
                        return True
                    if starting_bounds.dominates(next_best.bounds()) or \
                            (self.best_match is not None
                             and self.best_match.bounds().dominates(next_best.bounds())) or \
                            self.initial_bounds.dominates(next_best.bounds()):
                        # No need to add this new edit if it is strictly worse than the current best!
                        pass
                    if next_best.bounds().definitive():
                        self._tightened.push(next_best)
                    else:
                        self._untightened.push(next_best)
                except StopIteration:
                    self._unprocessed = None
            tightened = False
            if self._untightened:
                if self._unprocessed is None:
                    if len(self._untightened) == 1:
                        untightened = self._untightened.peek()
                        if untightened.tighten_bounds() and untightened.bounds().definitive():
                            self._untightened.clear()
                            self._tightened.push(untightened)
                    if self.goal_test():
                        best = self.best_match
                        self._untightened.clear()
                        self._tightened.clear()
                        ret = best.tighten_bounds()
                        if best.bounds().definitive():
                            self._tightened.push(best)
                        else:
                            self._untightened.push(best)
                        assert self.best_match == best
                        return ret
                for node in list(self._untightened.min_node):
                    if node.deleted:
                        continue
                    tightened = node.item.tighten_bounds()
                    if tightened:
                        self._update_bounds(node)
                        break
            if starting_bounds.lower_bound < self.bounds().lower_bound \
                    or starting_bounds.upper_bound > self.bounds().upper_bound:
                return True
            elif self._unprocessed is None and not tightened:
                return False
