import random
from unittest import TestCase

from graphtage.fibonacci import FibonacciHeap


class TestFibonacciHeap(TestCase):
    def test_fibonacci_heap(self):
        heap = FibonacciHeap()
        random_list = [random.randint(0, 1000000) for _ in range(1000)]
        sorted_list = sorted(random_list)
        for rand_int in random_list:
            heap.push(rand_int)
        heap_sorted = [heap.pop() for _ in range(len(random_list))]
        self.assertEqual(sorted_list, heap_sorted)
