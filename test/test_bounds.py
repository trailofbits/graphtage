import random
from typing import Optional
from unittest import TestCase

from tqdm import trange

from graphtage.bounds import Bounded, IdentityInterval, Range, make_distinct, sort


class CollidingRange(Bounded):
    """A bounded range whose initial bounds are identical to those of every other instance.

    This is the shape :func:`graphtage.bounds.make_distinct` sees at the start of a bipartite match,
    where every edge reports near-identical initial bounds and therefore produces intervals that share
    a span.

    Equality is by identity, which is what Graphtage's edits use, but every comparison is counted in
    :attr:`comparisons` so that a test can tell an O(n) workload from an O(n^2) one.
    """

    comparisons: int = 0

    def __init__(self, final_value: int, width: int = 1024):
        self.final_value = final_value
        self._lb = 0
        self._ub = width

    def bounds(self) -> Range:
        return Range(self._lb, self._ub)

    def tighten_bounds(self) -> bool:
        if self._lb == self._ub:
            return False
        if self._lb < self.final_value:
            self._lb += max((self.final_value - self._lb) // 2, 1)
        if self._ub > self.final_value:
            self._ub -= max((self._ub - self.final_value) // 2, 1)
        return True

    def __eq__(self, other):
        CollidingRange.comparisons += 1
        return self is other

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.final_value!r})"


class RandomDecreasingRange(Bounded):
    def __init__(self, fixed_lb: int = 0, fixed_ub: int = 2000000, final_value: int | None = None):
        if final_value is None:
            self.final_value = random.randint(fixed_lb, fixed_lb + (fixed_ub - fixed_lb) // 2)
        elif final_value < fixed_lb:
            raise ValueError(f"final_value of {final_value} < fixed lower bound of {fixed_lb}")
        elif final_value > fixed_ub:
            raise ValueError(f"final_value of {final_value} > fixed upper bound of {fixed_ub}")
        else:
            self.final_value = final_value
        self._lb = random.randint(fixed_lb, self.final_value)
        self._ub = random.randint(self.final_value, fixed_ub)
        self.tightenings: int = 0

    def bounds(self) -> Range:
        return Range(self._lb, self._ub)

    def tighten_bounds(self) -> bool:
        bounds_before = self.bounds()
        lb_diff = self.final_value - self._lb
        ub_diff = self._ub - self.final_value
        if lb_diff == ub_diff == 0:
            return False
        if lb_diff <= 1:
            self._lb = self.final_value
        else:
            self._lb += random.randint(max(int(0.5 * lb_diff), 1), lb_diff)
        if ub_diff <= 1:
            self._ub = self.final_value
        else:
            self._ub -= random.randint(max(int(0.5 * ub_diff), 1), ub_diff)
        if bounds_before.lower_bound < self._lb or bounds_before.upper_bound > self._ub:
            self.tightenings += 1
            return True
        else:
            return False

    def __repr__(self):
        return repr(self.bounds())


class TestBounds(TestCase):
    def test_random_decreasing_range(self):
        for _ in range(1000):
            r = RandomDecreasingRange()
            last_range = r.bounds()
            while r.tighten_bounds():
                next_range = r.bounds()
                self.assertTrue(next_range.lower_bound >= last_range.lower_bound
                                and next_range.upper_bound <= last_range.upper_bound
                                and (
                                    next_range.lower_bound > last_range.lower_bound or
                                    next_range.upper_bound < last_range.upper_bound
                                ))
                last_range = next_range

    def test_sort(self):
        for _ in trange(100):
            ranges = [RandomDecreasingRange() for _ in range(100)]
            sorted_ranges = sorted(ranges, key=lambda r: r.final_value)
            for expected, actual in zip(sorted_ranges, sort(ranges), strict=True):
                self.assertEqual(expected.final_value, actual.final_value)

    def test_make_distinct(self):
        speedups = 0
        tests = 0
        try:
            with trange(0, 100) as t:
                for i in t:
                    ranges = [RandomDecreasingRange() for _ in range(i)]
                    make_distinct(*ranges)
                    last_range = None
                    for r in sort(ranges):
                        rbounds = r.bounds()
                        if last_range is not None:
                            self.assertTrue((last_range.definitive() and rbounds.definitive() and last_range == rbounds) or
                                            last_range.upper_bound < rbounds.lower_bound,
                                            f"{last_range!r} was followed by {rbounds!r}")
                        last_range = rbounds
                    tightenings = sum(r.tightenings for r in ranges)
                    if tightenings > 0:
                        untightened = 0
                        for r in ranges:
                            t_before = r.tightenings
                            while r.tighten_bounds():
                                pass
                            untightened += r.tightenings - t_before
                        t.desc = f"{(untightened + tightenings) / tightenings:.01f}x Speedup"
                        speedups += (untightened + tightenings) / tightenings
                        tests += 1
        finally:
            print(f"Average speedup: {speedups / tests:.01f}x")

    def test_make_distinct_with_identical_initial_bounds(self):
        """Checks the result of ``make_distinct`` when every input starts with the same bounds.

        The interval tree ``make_distinct`` uses keys its intervals on their spans, so inputs that all
        report the same bounds are the degenerate case for it. This asserts the outcome rather than the
        data structure: it catches a rewrite of ``make_distinct`` that leaves two ranges overlapping
        without being definitive, which is the postcondition callers such as the bipartite matcher rely
        on to order edits.
        """
        ranges = [CollidingRange(final_value=i * 5) for i in range(64)]
        make_distinct(*ranges)
        ordered = list(sort(ranges))
        self.assertEqual(len(ranges), len(ordered))
        last_range = None
        for r in ordered:
            rbounds = r.bounds()
            if last_range is not None:
                self.assertTrue(
                    (last_range.definitive() and rbounds.definitive() and last_range == rbounds)
                    or last_range.upper_bound < rbounds.lower_bound,
                    f"{last_range!r} was followed by {rbounds!r}",
                )
            last_range = rbounds

    def test_make_distinct_does_not_probe_quadratically(self):
        """Pins the hash and equality invariant that keeps ``make_distinct`` out of quadratic behavior.

        ``intervaltree.Interval`` hashes on ``(begin, end)`` alone but compares ``data`` as well, and
        ``IntervalTree`` holds its intervals in sets. Intervals that share a span therefore land in one
        hash bucket, and each insertion and lookup turns into a linear scan of equality tests.
        ``IdentityInterval`` mixes ``id(data)`` into the hash to keep those buckets apart.

        This catches a regression to the inherited hash. With it, a 64-element run costs over 700,000
        data comparisons; with ``IdentityInterval`` it costs none at all, because no two intervals share
        a bucket.
        """
        first, second = CollidingRange(1), CollidingRange(2)
        self.assertNotEqual(IdentityInterval(0, 10, first), IdentityInterval(0, 10, second))

        n = 64
        ranges = [CollidingRange(final_value=i * 5) for i in range(n)]
        CollidingRange.comparisons = 0
        try:
            make_distinct(*ranges)
            comparisons = CollidingRange.comparisons
        finally:
            CollidingRange.comparisons = 0
        self.assertLessEqual(
            comparisons,
            4 * n,
            f"make_distinct made {comparisons} equality comparisons over {n} equally bounded inputs, "
            f"which suggests the intervals are colliding into shared hash buckets",
        )
