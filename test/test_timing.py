from unittest import TestCase

from .timing import run_with_time_limit


def infinite_loop():
    while True:
        pass


def limited_infinite_loop():
    with run_with_time_limit(seconds=1):
        infinite_loop()


class TestTiming(TestCase):
    def test_time_limit(self):
        self.assertRaises(TimeoutError, limited_infinite_loop)

    def test_non_infinite_loop(self):
        with run_with_time_limit(seconds=60):
            _ = 10
