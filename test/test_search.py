import random
from unittest import TestCase

from tqdm import trange

from graphtage.bounds import Bounded, Range
from graphtage.search import IterativeTighteningSearch


class RandomDecreasingRange(Bounded):
    def __init__(self):
        self.final_value = random.randint(0, 1000000)
        self._lb = random.randint(0, self.final_value)
        self._ub = random.randint(self.final_value, 2000000)
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


class TestIterativeTighteningSearch(TestCase):
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

    def test_iterative_tightening_search(self):
        speedups = 0
        tests = 0
        try:
            t = trange(1000)
            for test_num in t:
                ranges = [RandomDecreasingRange() for _ in range(100)]
                best_range: RandomDecreasingRange = None
                for r in ranges:
                    if best_range is None or r.final_value < best_range.final_value:
                        best_range = r
                search = IterativeTighteningSearch(iter(ranges))
                # with tqdm(desc=str(search.bounds()), leave=False) as substatus:
                #     while search.tighten_bounds():
                #         substatus.total = search.bounds().upper_bound - search.bounds().lower_bound
                #         substatus.desc = f"Test {test_num}: {search.bounds()!s}"
                #         substatus.update(substatus.total - (search.bounds().upper_bound - search.bounds().lower_bound))
                #     substatus.clear()
                while search.tighten_bounds():
                    pass
                result = search.best_match
                tightenings = sum(r.tightenings for r in ranges)
                untightened = 0
                for r in ranges:
                    t_before = r.tightenings
                    while r.tighten_bounds():
                        pass
                    untightened += r.tightenings - t_before
                t.desc = f"{(untightened + tightenings) / tightenings:.01f}x Speedup"
                speedups += (untightened + tightenings) / tightenings
                tests += 1
                self.assertEqual(best_range.final_value, result.final_value)
        finally:
            print(f"Average speedup: {speedups / tests:.01f}x")
