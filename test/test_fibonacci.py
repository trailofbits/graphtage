import random
from unittest import TestCase

from graphtage.fibonacci import FibonacciHeap


class TestFibonacciHeap(TestCase):
    def test_duplicate_items(self):
        heap = FibonacciHeap()
        test_list = [2, 1, 2]
        for i in test_list:
            heap.push(i)
        heap_sorted = [heap.pop() for _ in range(len(test_list))]
        self.assertEqual(sorted(test_list), heap_sorted)

    def test_fibonacci_heap(self):
        heap = FibonacciHeap()
        random_list = [random.randint(0, 10000) for _ in range(10000)]
        sorted_list = sorted(random_list)
        for rand_int in random_list:
            heap.push(rand_int)
        heap_sorted = [heap.pop() for _ in range(len(random_list))]
        self.assertEqual(sorted_list, heap_sorted)
