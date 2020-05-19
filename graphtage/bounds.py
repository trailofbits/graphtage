"""A module for representing bounded ranges.

Examples:

    >>> from graphtage import bounds
    >>> p = bounds.Infinity(positive=True)
    >>> str(p)
    '∞'
    >>> str(p + p)
    '∞'
    >>> bounds.Range(0, p)
    Range(0, Infinity(positive=True))
    >>> bounds.Range(0, 10) < bounds.Range(20, 30)
    True

This module provides a variety of data structures and algorithms for both representing bounds as well as operating on
bounded ranges (*e.g.*, sorting).

Attributes:
    NEGATIVE_INFINITY (Infinity): Negative infinity.
    POSITIVE_INFINITY (Infinity): Positive infinity.

"""

import logging
from functools import wraps
from typing import Iterable, Iterator, Optional, TypeVar, Union
from typing_extensions import Protocol

from intervaltree import Interval, IntervalTree

from .fibonacci import FibonacciHeap


log = logging.getLogger(__name__)


class Infinity:
    """A class for representing infinite values. This is primarily used for unbounded ranges."""
    def __init__(self, positive=True):
        self._positive = positive

    @property
    def positive(self) -> bool:
        """Returns whether or not this represents positive infinity."""
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

    def __radd__(self, _):
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
    """An integer range."""
    def __init__(self, lower_bound: RangeValue = NEGATIVE_INFINITY, upper_bound: RangeValue = POSITIVE_INFINITY):
        """Constructs a range.

        Args:
            lower_bound: The lower bound of the range (inclusive).
            upper_bound: The upper bound of the range (inclusive).

        Raises:
            ValueError: If the upper bound is less than the lower bound.

        """
        if upper_bound < lower_bound:
            raise ValueError(f"Upper bound ({upper_bound!s}) must be less than lower bound ({lower_bound!s})")
        self.lower_bound: RangeValue = lower_bound
        """The lower bound of this range."""
        self.upper_bound: RangeValue = upper_bound
        """The upper bound of this range."""

    def __eq__(self, other):
        return self.lower_bound == other.lower_bound and self.upper_bound == other.upper_bound

    def __lt__(self, other):
        return self.upper_bound < other.upper_bound or \
            (self.upper_bound == other.upper_bound and self.lower_bound < other.lower_bound)

    def __le__(self, other):
        return self < other or self == other

    def to_interval(self) -> Interval:
        """Converts this range to an :class:`intervaltree.Interval` for use with the `Interval Tree`_ package.

        .. _Interval Tree:
            https://github.com/chaimleib/intervaltree

        """
        return Interval(self.lower_bound, self.upper_bound + 1, self)

    def dominates(self, other) -> bool:
        """Checks whether this range dominates another.

        One range dominates another if its upper bound is less than or equal to the lower bound of the other.

        This is equivalent to::

            return self.upper_bound <= other.lower_bound

        """
        return self.upper_bound <= other.lower_bound

    def __hash__(self):
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
        """Returns whether this range is finite.

        A range is finite if neither of its bounds is infinite.

        """
        return not isinstance(self.lower_bound, Infinity) and not isinstance(self.upper_bound, Infinity)

    def definitive(self) -> bool:
        """Checks whether this range is definitive.

        A range is definitive if both of its bounds are finite and equal to each other.

        """
        return self.lower_bound == self.upper_bound and not isinstance(self.lower_bound, Infinity)

    def intersect(self, other) -> 'Range':
        """Intersects this range with another."""
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
    """A protocol for objects that have bounds that can be tightened."""

    def tighten_bounds(self) -> bool:
        """Attempts to shrink the bounds of this object.

        Returns:
            bool: :const:`True` if the bounds were tightened.

        """
        raise NotImplementedError(f"Class {self.__class__.__name__} must implement tighten_bounds")

    def bounds(self) -> Range:
        """Returns the bounds of this object."""
        raise NotImplementedError(f"Class {self.__class__.__name__} must implement bounds")


def repeat_until_tightened(func):
    """A decorator that will repeatedly call the function until its class's bounds are tightened.

    Intended for :meth:`Bounded.tighten_bounds`. The value returned by the decorated function is ignored.

    """
    @wraps(func)
    def wrapper(self: Bounded, *args, **kwargs):
        starting_bounds = self.bounds()
        if starting_bounds.definitive():
            return False
        while True:
            func(self, *args, **kwargs)
            new_bounds = self.bounds()
            if new_bounds.lower_bound < starting_bounds.lower_bound \
                    or new_bounds.upper_bound > starting_bounds.upper_bound:
                log.warning(f"The most recent call to {func} on {self} returned bounds {new_bounds} when the previous bounds were {starting_bounds}")
            elif new_bounds.definitive() or new_bounds.lower_bound > starting_bounds.lower_bound \
                    or new_bounds.upper_bound < starting_bounds.upper_bound:
                return True

    return wrapper


class ConstantBound(Bounded):
    """An object with constant bounds."""
    def __init__(self, value: RangeValue):
        """Initializes the constant bounded object.

        Args:
            value: The constant value of the object, which will constitute both its lower and upper bound.

        """
        self._range = Range(value, value)

    def bounds(self) -> Range:
        """Returns a :class:`Range` where both the lower and upper bounds are equal to this object's constant value."""
        return self._range

    def tighten_bounds(self) -> bool:
        """Since the bounds are already definitive, this always returns :const:`False`."""
        return False


class BoundedComparator:
    """A comparator for :class:`Bounded` objects.

    This comparator will automatically tighten the bounds of the :class:`Bounded` object it wraps until they are either
    definitive or sufficiently distinct to differentiate them from another object to which it is being compared.

    """
    def __init__(self, bounded: Bounded):
        """Initializes this bounded comparator.

        Args:
            bounded: The object to wrap.

        """
        self.bounded = bounded
        """The wrapped bounded object."""

    def __lt__(self, other):
        """Compares the wrapped object to :obj:`other`, auto-tightening their bounds if necessary.

        The auto-tightening is equivalent to::

            while not (
                self.bounded.bounds().dominates(other.bounded.bounds())
                or
                other.bounded.bounds().dominates(self.bounded.bounds())
            ) and (
                self.bounded.tighten_bounds() or other.bounded.tighten_bounds()
            ):
                pass

        In the event that :obj:`self.bounded` and :obj:`other` have identical bounds after fully tightening, the object
        with the smaller :func:`id` is returned.

        """
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
    """Sorts a sequence of bounded items.

    Args:
        items: Zero or more bounded objects.

    Returns:
        Iterator[B]: An iterator over the sorted sequence of items.

    This is equivalent to::

        heap: FibonacciHeap[B, BoundedComparator] = FibonacciHeap(key=BoundedComparator)
        for item in items:
            heap.push(item)
        while heap:
            yield heap.pop()


    """
    heap: FibonacciHeap[B, BoundedComparator] = FibonacciHeap(key=BoundedComparator)
    for item in items:
        heap.push(item)
    while heap:
        yield heap.pop()


def min_bounded(bounds: Iterator[B]) -> B:
    """Returns the smallest bounded object.

    The objects are auto-tightened in the event that their ranges overlap and a definitive minimum does not exist.

    """
    best_item: Optional[B] = None
    best: Optional[BoundedComparator] = None
    for b in map(BoundedComparator, bounds):
        if best_item is None or b < best:
            best_item = b.bounded
            best = b
    return best_item


def make_distinct(*bounded: Bounded):
    """Ensures that all of the provided bounded arguments are tightened until they are finite and
    either definitive or non-overlapping with any of the other arguments."""
    tree: IntervalTree = IntervalTree()
    for b in bounded:
        if not b.bounds().finite:
            b.tighten_bounds()
            if not b.bounds().finite:
                raise ValueError(f"Could not tighten {b!r} to a finite bound")
        tree.add(Interval(b.bounds().lower_bound, b.bounds().upper_bound + 1, b))
    while len(tree) > 1:
        # find the biggest interval in the tree
        biggest: Optional[Interval] = None
        for m in tree:
            m_size = m.end - m.begin
            if biggest is None or m_size > biggest.end - biggest.begin:
                biggest = m
        assert biggest is not None
        if biggest.data.bounds().definitive():
            # This means that all intervals are points, so we are done!
            break
        tree.remove(biggest)
        matching = tree[biggest.begin:biggest.end]
        if len(matching) < 1:
            # This interval does not intersect any others, so it is distinct
            continue
        # now find the biggest other interval that intersects with biggest:
        second_biggest: Optional[Interval] = None
        for m in matching:
            m_size = m.end - m.begin
            if second_biggest is None or m_size > second_biggest.end - second_biggest.begin:
                second_biggest = m
        assert second_biggest is not None
        tree.remove(second_biggest)
        # Shrink the two biggest intervals until they are distinct
        while True:
            biggest_bound: Range = biggest.data.bounds()
            second_biggest_bound: Range = second_biggest.data.bounds()
            if (biggest_bound.definitive() and second_biggest_bound.definitive()) or \
                    biggest_bound.upper_bound < second_biggest_bound.lower_bound or \
                    second_biggest_bound.upper_bound < biggest_bound.lower_bound:
                break
            biggest.data.tighten_bounds()
            second_biggest.data.tighten_bounds()
        new_interval = Interval(
            begin=biggest.data.bounds().lower_bound,
            end=biggest.data.bounds().upper_bound + 1,
            data=biggest.data
        )
        if tree.overlaps(new_interval.begin, new_interval.end):
            tree.add(new_interval)
        new_interval = Interval(
            begin=second_biggest.data.bounds().lower_bound,
            end=second_biggest.data.bounds().upper_bound + 1,
            data=second_biggest.data
        )
        if tree.overlaps(new_interval.begin, new_interval.end):
            tree.add(new_interval)
