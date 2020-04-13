import random
from typing import Optional
from unittest import TestCase

from tqdm import trange

from graphtage.bounds import Bounded, make_distinct, Range, sort


class RandomDecreasingRange(Bounded):
    def __init__(self, fixed_lb: int = 0, fixed_ub: int = 2000000, final_value: Optional[int] = None):
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
        lb_diff = self.final_value - self._lb
        ub_diff = self._ub - self.final_value
        if lb_diff == ub_diff == 0:
            return False
        elif lb_diff <= 1 or ub_diff <= 1:
            self.tightenings += 1
            self._lb = self.final_value
            self._ub = self.final_value
            return True
        else:
            self.tightenings += 1
            self._lb += random.randint(1, lb_diff)
            self._ub -= random.randint(1, ub_diff)
            return True

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
        for _ in trange(1000):
            ranges = [RandomDecreasingRange() for _ in range(100)]
            sorted_ranges = sorted(ranges, key=lambda r: r.final_value)
            for expected, actual in zip(sorted_ranges, sort(ranges)):
                self.assertEqual(expected.final_value, actual.final_value)

    def test_make_distinct(self):
        for i in trange(128):
            ranges = [RandomDecreasingRange() for _ in range(i)]
            make_distinct(*ranges)
            last_range = None
            for r in sort(ranges):
                rbounds = r.bounds()
                if last_range is not None:
                    self.assertTrue((last_range.definitive() and rbounds.definitive() and last_range == rbounds) or
                                    last_range.upper_bound < rbounds.lower_bound)
                last_range = rbounds
