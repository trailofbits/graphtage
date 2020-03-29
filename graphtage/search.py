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

    def __sub__(self, other):
        if isinstance(other, Infinity) and other._positive != self._positive:
            raise ValueError("-∞ + ∞ is undefined")
        else:
            return self

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


class InvertedRange(Range):
    """A Range that orders itself based off of decreasing lower bounds
       The default Range object orders itself based off of increasing upper bounds
    """
    def __init__(self, range: Range):
        super().__init__(range.lower_bound, range.upper_bound)
        self._range = range

    def __lt__(self, other):
        assert isinstance(other, InvertedRange)
        return self.lower_bound > other.lower_bound or \
            (self.lower_bound == other.lower_bound and self.upper_bound > other.upper_bound)


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

        def negated_range(bounded: Bounded) -> Range:
            bounds: Bounded = bounded.bounds()
            #return Range(-1 * bounds.upper_bound, -1 * bounds.lower_bound)
            return InvertedRange(bounds)

        self._unprocessed: Iterator[B] = possibilities
        self._untightened: FibonacciHeap[B, Range] = FibonacciHeap(key=get_range)
        self._tightened: FibonacciHeap[B, Range] = FibonacciHeap(key=get_range)

        # Heap to track the ranges with the lowest upper bound
        self._best: FibonacciHeap[B, Range] = FibonacciHeap(key=get_range)

        # Heap to track the ranges with the highest lower bound
        self._worst: FibonacciHeap[B, Range] = FibonacciHeap(key=negated_range)

        # Mapping of nodes in the `_best` heap to nodes in the `_worst` heap
        self._worst_nodes: Dict[HeapNode[B, Range], HeapNode[B, Range]] = {}

        if initial_bounds is None:
            self.initial_bounds = Range(NEGATIVE_INFINITY, POSITIVE_INFINITY)
        else:
            self.initial_bounds = initial_bounds

    @property
    def best_match(self) -> B:
        if not self._best or self._unprocessed is not None:
            return None
        return self._best.peek()

    @property
    def worst_match(self) -> B:
        if not self._worst or self._unprocessed is not None:
            return None
        return self._worst.peek()

    def search(self) -> B:
        while self.tighten_bounds():
            pass
        return self.best_match

    def bounds(self) -> Range:
        if self.best_match is None:
            return self.initial_bounds
        else:
            if self._unprocessed is None and self._best:
                lb = min(node.item.bounds().lower_bound for node in self._best.nodes())
                assert lb >= self.initial_bounds.lower_bound
            else:
                lb = self.initial_bounds.lower_bound
            return Range(min(lb, self.best_match.bounds().upper_bound), self.best_match.bounds().upper_bound)

    def _delete_node(self, node: HeapNode[B, Range]):
        self._best.decrease_key(node, Range(NEGATIVE_INFINITY, NEGATIVE_INFINITY))
        node.deleted = True
        self._best.pop()
        worst_node = self._worst_nodes[node]
        self._worst.decrease_key(worst_node, InvertedRange(Range(POSITIVE_INFINITY, POSITIVE_INFINITY)))
        worst_node.deleted = True
        self._worst.pop()
        del self._worst_nodes[node]

    def _update_bounds(self, node: HeapNode[B, Range]):
        if self.best_match is not None \
                and self.best_match != node.item \
                and self.best_match.bounds().dominates(node.item.bounds()):
            self._delete_node(node)
            return
        bounds = node.item.bounds()
        if bounds.lower_bound > node.key.lower_bound:
            # The lower bound increased, so we need to remove and re-add the node
            # because the Fibonacci heap only permits making keys smaller
            self._best.decrease_key(node, Range(NEGATIVE_INFINITY, NEGATIVE_INFINITY))
            self._best.pop()
            best = self._best.push(node.item)
            self._worst_nodes[best] = self._worst_nodes[node]
            del self._worst_nodes[node]
            node = best
        if bounds.upper_bound < node.key.upper_bound:
            worst = self._worst_nodes[node]
            self._worst.decrease_key(worst, InvertedRange(Range(POSITIVE_INFINITY, POSITIVE_INFINITY)))
            self._worst.pop()
            worst = self._worst.push(node.item)
            self._worst_nodes[node] = worst

    def tighten_bounds(self) -> bool:
        starting_bounds = self.bounds()
        while True:
            if self._unprocessed is not None:
                try:
                    next_best: B = next(self._unprocessed)
                    if self.initial_bounds.lower_bound > NEGATIVE_INFINITY and \
                            self.initial_bounds.lower_bound >= next_best.bounds().upper_bound:
                        # We can't do any better than this choice!
                        self.best_match = next_best
                        self._untightened.clear()
                        self._tightened.clear()
                        self._unprocessed = None
                        return True
                    if starting_bounds.dominates(next_best.bounds()) or \
                            (self.best_match is not None
                             and self.best_match.bounds().dominates(next_best.bounds())):
                        # No need to add this new edit if it is strictly worse than the current best!
                        pass
                    else:
                        best = self._best.push(next_best)
                        worst = self._worst.push(next_best)
                        self._worst_nodes[best] = worst
                except StopIteration:
                    self._unprocessed = None
            tightened = False
            for node in list(self._best.nodes()):
                if node.deleted:
                    continue
                bounds_before: Range = node.key
                if not node.item.tighten_bounds():
                    if self.best_match is not None \
                            and self.best_match != node.item \
                            and self.best_match.bounds().dominates(bounds_before):
                        self._delete_node(node)
                    continue
                assert node.item.bounds().lower_bound >= bounds_before.lower_bound
                assert node.item.bounds().upper_bound <= bounds_before.upper_bound
                self._update_bounds(node)
                tightened = True
            if self._unprocessed is None and not tightened:
                return False
            elif starting_bounds.lower_bound < self.bounds().lower_bound \
                    or starting_bounds.upper_bound > self.bounds().upper_bound:
                return True
