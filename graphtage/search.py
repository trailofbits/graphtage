from abc import abstractmethod, ABC
from typing import Callable, Generic, Iterator, Optional, TypeVar, Union
from typing_extensions import Protocol

from .fibonacci import FibonacciHeap


class Infinity:
    def __init__(self, positive=True):
        self.positive = positive

    def __eq__(self, other):
        if isinstance(other, Infinity):
            return self.positive == other.positive
        else:
            return False

    def __lt__(self, other):
        if isinstance(other, int):
            return not self.positive
        else:
            return not self.positive and other.positive

    def __gt__(self, other):
        if isinstance(other, int):
            return self.positive
        else:
            return self.positive and not other.positive

    def __ge__(self, other):
        return self > other or self == other

    def __le__(self, other):
        return self < other or self == other

    def __add__(self, other):
        if isinstance(other, Infinity) and other.positive != self.positive:
            raise ValueError("-∞ + ∞ is undefined")
        else:
            return self

    def __sub__(self, other):
        if isinstance(other, Infinity) and other.positive != self.positive:
            raise ValueError("-∞ + ∞ is undefined")
        else:
            return self

    def __hash__(self):
        return hash(self.positive)

    def __repr__(self):
        return f"{self.__class__.__name__}(positive={self.positive!r})"

    def __str__(self):
        if self.positive:
            return '∞'
        else:
            return '-∞'


NegativeInfinity = Infinity(positive=False)
PositiveInfinity = Infinity(positive=True)

RangeValue = Union[int, Infinity]


class Range:
    def __init__(self, lower_bound: Optional[RangeValue] = None, upper_bound: Optional[RangeValue] = None):
        assert lower_bound is None or upper_bound is None or upper_bound >= lower_bound
        self.lower_bound: Optional[RangeValue] = lower_bound
        self.upper_bound: Optional[RangeValue] = upper_bound

    def __eq__(self, other):
        return self.lower_bound == other.lower_bound and self.upper_bound == other.upper_bound

    def __lt__(self, other):
        if not self or not other:
            slb, sub = self.lower_bound is not None, self.upper_bound is not None
            olb, oub = other.lower_bound is not None, other.upper_bound is not None
            if slb and olb:
                return self.lower_bound < other.lower_bound
            elif sub and olb:
                return self.upper_bound < other.lower_bound
            else:
                return False
        return self.lower_bound < other.lower_bound or \
            (self.lower_bound == other.lower_bound and self.upper_bound < other.upper_bound)

    def __le__(self, other):
        return self < other or self == other

    def hash(self):
        return hash((self.lower_bound, self.upper_bound))

    def __bool__(self):
        return self.lower_bound is not None and self.upper_bound is not None

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
        return bool(self) and self.lower_bound == self.upper_bound

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
        return f"[{self.lower_bound}, {self.upper_bound}]"


class Bounded(Protocol):
    tighten_bounds: Callable[[], bool]
    bounds: Callable[[], Range]


class IterativeTighteningSearch(Bounded):
    def __init__(self,
                 possibilities: Iterator[Bounded],
                 initial_bounds: Optional[Range] = None):
        def get_range(bounded: Bounded) -> Range:
            return bounded.bounds()

        self._unprocessed: Iterator[Bounded] = possibilities
        self._untightened: FibonacciHeap[Bounded, Range] = FibonacciHeap(key=get_range)
        self._tightened: FibonacciHeap[Bounded, Range] = FibonacciHeap(key=get_range)
        self.best_match: Bounded = None
        if initial_bounds is None:
            self.initial_bounds = Range(NegativeInfinity, PositiveInfinity)
        else:
            self.initial_bounds = initial_bounds

    def search(self) -> Bounded:
        while self.tighten_bounds():
            pass
        return self.best_match

    def bounds(self) -> Range:
        if self.best_match is None:
            return self.initial_bounds
        else:
            return self.best_match.bounds()

    def tighten_bounds(self) -> bool:
        starting_bounds = self.bounds()
        while True:
            if self._unprocessed is not None:
                try:
                    next_best: Bounded = next(self._unprocessed)
                    if self.initial_bounds.lower_bound > NegativeInfinity and \
                            self.initial_bounds.lower_bound > next_best.bounds().upper_bound:
                        # We can't do any better than this choice!
                        self.best_match = next_best
                        self._untightened.clear()
                        self._tightened.clear()
                        self._unprocessed = None
                        return True
                    if starting_bounds.upper_bound <= next_best.bounds().lower_bound:
                        # No need to add this new edit if it is strictly worse than the current best!
                        pass
                    else:
                        self._untightened.push(next_best)
                except StopIteration:
                    self._unprocessed = None
            if self._untightened:
                while self._untightened.peek().tighten_bounds():
                    pass
                self._tightened.push(self._untightened.pop())
            if self._tightened:
                assert self.best_match is None or self.best_match.bounds() >= self._tightened.peek().bounds()
                self.best_match = self._tightened.peek()
                if self._unprocessed is None:
                    if not self._untightened:
                        self._tightened.clear()
                        return True
                    elif len(self._tightened) >= 2:
                        # does the min node dominate the second smallest? if so, we are done!
                        self._tightened.pop()
                        second_smallest = self._tightened.peek()
                        if self.best_match.bounds().upper_bound < second_smallest.bounds().lower_bound <= PositiveInfinity:
                            self._unprocessed = None
                            self._untightened.clear()
                            self._tightened.clear()
                            return True
                        else:
                            self._tightened.push(self.best_match)

            if starting_bounds != self.bounds():
                return True
            elif self._unprocessed is None and not self._untightened and not self._tightened:
                return False
