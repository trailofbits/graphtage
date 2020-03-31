from typing import Callable, Dict, Generic, Iterator, Optional, TypeVar, Union
from typing_extensions import Protocol

from .fibonacci import FibonacciHeap, HeapNode


class Infinity:
    def __init__(self, positive=True):
        self._positive = positive

    @property
    def positive(self):
        return self._positive

    def __eq__(self, other):
        if isinstance(other, Infinity):
            return self._positive == other._positive
        else:
            return False

    def __lt__(self, other):
        if isinstance(other, int):
            return not self._positive
        else:
            return not self._positive and other.positive

    def __gt__(self, other):
        if isinstance(other, int):
            return self._positive
        else:
            return self._positive and not other.positive

    def __ge__(self, other):
        return self > other or self == other

    def __le__(self, other):
        return self < other or self == other

    def __add__(self, other):
        if isinstance(other, Infinity) and other._positive != self._positive:
            raise ValueError("-∞ + ∞ is undefined")
        else:
            return self

    def __neg__(self):
        return Infinity(positive=not self._positive)

    def __abs__(self):
        return POSITIVE_INFINITY

    def __sub__(self, other):
        if isinstance(other, Infinity) and other._positive != self._positive:
            raise ValueError("-∞ + ∞ is undefined")
        else:
            return self

    def __rsub__(self, _):
        return -self

    def __hash__(self):
        return hash(self._positive)

    def __repr__(self):
        return f"{self.__class__.__name__}(positive={self._positive!r})"

    def __str__(self):
        if self._positive:
            return '∞'
        else:
            return '-∞'


NEGATIVE_INFINITY: Infinity = Infinity(positive=False)
POSITIVE_INFINITY = Infinity(positive=True)

RangeValue = Union[int, Infinity]


class Range:
    def __init__(self, lower_bound: RangeValue = NEGATIVE_INFINITY, upper_bound: RangeValue = POSITIVE_INFINITY):
        if upper_bound < lower_bound:
            raise ValueError(f"Upper bound ({upper_bound!s}) must be less than lower bound ({lower_bound!s})")
        self.lower_bound: RangeValue = lower_bound
        self.upper_bound: RangeValue = upper_bound

    def __eq__(self, other):
        return self.lower_bound == other.lower_bound and self.upper_bound == other.upper_bound

    def __lt__(self, other):
        return self.upper_bound < other.upper_bound or \
            (self.upper_bound == other.upper_bound and self.lower_bound < other.lower_bound)

    def __le__(self, other):
        return self < other or self == other

    def dominates(self, other) -> bool:
        return self.upper_bound <= other.lower_bound

    def hash(self):
        return hash((self.lower_bound, self.upper_bound))

    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, Infinity):
            return Range(self.lower_bound + other, self.upper_bound + other)
        else:
            return Range(self.lower_bound + other.lower_bound, self.upper_bound + other.upper_bound)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, int) or isinstance(other, Infinity):
            return Range(self.lower_bound - other, self.upper_bound - other)
        else:
            return Range(self.lower_bound - other.lower_bound, self.upper_bound - other.upper_bound)

    @property
    def finite(self) -> bool:
        return not isinstance(self.lower_bound, Infinity) and not isinstance(self.upper_bound, Infinity)

    def definitive(self) -> bool:
        return self.lower_bound == self.upper_bound and not isinstance(self.lower_bound, Infinity)

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
        return f"[{self.lower_bound!s}, {self.upper_bound!s}]"


class Bounded(Protocol):
    tighten_bounds: Callable[[], bool]
    bounds: Callable[[], Range]


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
                lb = min(node.item.bounds().lower_bound for node in self._nodes())
                assert lb >= self.initial_bounds.lower_bound
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
