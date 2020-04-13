from typing import Callable, Iterable, Iterator, Optional, TypeVar, Union
from typing_extensions import Protocol

from intervaltree import Interval, IntervalTree

from .fibonacci import FibonacciHeap


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

    def to_inverval(self) -> Interval:
        return Interval(self.lower_bound, self.upper_bound + 1, self)

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


class ConstantBound(Bounded):
    def __init__(self, value: RangeValue):
        self._range = Range(value, value)

    def bounds(self) -> Range:
        return self._range

    def tighten_bounds(self) -> bool:
        return False


class BoundedComparator:
    def __init__(self, bounded: Bounded):
        self.bounded = bounded

    def __lt__(self, other):
        while not (
                self.bounded.bounds().dominates(other.bounded.bounds())
                or
                other.bounded.bounds().dominates(self.bounded.bounds())
        ) and (
                self.bounded.tighten_bounds() or other.bounded.tighten_bounds()
        ):
            pass
        return self.bounded.bounds().dominates(other.bounded.bounds()) or (
                self.bounded.bounds() == other.bounded.bounds() and id(self) < id(other)
        )

    def __le__(self, other):
        if self < other:
            return True
        while self.bounded.tighten_bounds() or other.bounded.tighten_bounds():
            pass
        return self.bounded.bounds() == other.bounded.bounds()


B = TypeVar('B', bound=Bounded)


def sort(items: Iterable[B]) -> Iterator[B]:
    heap: FibonacciHeap[B, BoundedComparator] = FibonacciHeap(key=BoundedComparator)
    for item in items:
        heap.push(item)
    while heap:
        yield heap.pop()


def min_bounded(bounds: Iterator[B]) -> B:
    best_item: Optional[B] = None
    best: Optional[BoundedComparator] = None
    for b in map(BoundedComparator, bounds):
        if best_item is None or b < best:
            best_item = b.bounded
            best = b
    return best_item


def make_distinct(*bounded: Bounded):
    """Ensures that all of the provided bounded arguments are tightened until they finite and
    either definitive or non-overlapping with any of the other arguments"""
    tree: IntervalTree = IntervalTree()
    for b in bounded:
        if not b.bounds().finite:
            b.tighten_bounds()
            if not b.bounds().finite:
                raise ValueError(f"Could not tighten {b!r} to a finite bound")
        tree.add(Interval(b.bounds().lower_bound, b.bounds().upper_bound + 1, b))
    while tree:
        interval = next(iter(tree))
        matching = tree[interval.begin:interval.end]
        if len(matching) > 1:
            for i in matching:
                if i.data.tighten_bounds():
                    tree.remove(i)
                    tree.add(Interval(i.data.bounds().lower_bound, i.data.bounds().upper_bound + 1, i.data))
                    break
            else:
                # There is nothing more we can do to shrink this interval
                for i in matching:
                    tree.remove(i)
        else:
            # There is not any overlap on this interval, so we are done with it
            tree.remove(interval)