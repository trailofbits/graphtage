import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set
from unittest import TestCase

from tqdm import tqdm, trange

from graphtage.fibonacci import FibonacciHeap, HeapNode, MaxFibonacciHeap


class TestFibonacciHeap(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.random_list: List[int] = [random.randint(0, 10000) for _ in range(10000)]
        cls.sorted_list: List[int] = sorted(cls.random_list)

    def test_duplicate_items(self):
        heap = FibonacciHeap()
        test_list = [2, 1, 2]
        for i in test_list:
            heap.push(i)
        heap_sorted = [heap.pop() for _ in range(len(test_list))]
        self.assertEqual(sorted(test_list), heap_sorted)

    def random_heap(self) -> FibonacciHeap[int, int]:
        heap: FibonacciHeap[int, int] = FibonacciHeap()
        for rand_int in self.random_list:
            heap.push(rand_int)
        return heap

    def random_max_heap(self, key: Optional[Callable[[int], int]] = None) -> MaxFibonacciHeap[int, int]:
        heap: FibonacciHeap[int, int] = MaxFibonacciHeap(key=key)
        for rand_int in self.random_list:
            heap.push(rand_int)
        return heap

    def test_fibonacci_heap(self):
        heap = self.random_heap()
        heap_sorted = [heap.pop() for _ in range(len(self.random_list))]
        self.assertEqual(self.sorted_list, heap_sorted)

    def test_max_fibonacci_heap(self):
        heap = self.random_max_heap()
        heap_sorted = [heap.pop() for _ in range(len(self.random_list))]
        self.assertEqual(list(reversed(self.sorted_list)), heap_sorted)

    def test_max_fibonacci_heap_with_key(self):
        heap = self.random_max_heap(key=lambda i: -i)
        heap_sorted = [heap.pop() for _ in range(len(self.random_list))]
        self.assertEqual(self.sorted_list, heap_sorted)

    def test_node_traversal(self):
        heap = self.random_heap()
        self.assertEqual(sum(1 for _ in heap.nodes()), len(heap))

    def test_manual_node_deletion(self):
        heap = self.random_heap()
        for i in trange(len(self.random_list)//20):
            random_node: HeapNode[int, int] = random.choice(list(heap.nodes()))
            heap.decrease_key(random_node, -1)
            heap.pop()
            random_node.deleted = True
            self.assertEqual(len(heap), len(self.random_list) - i - 1)

    def test_node_deletion(self):
        heap = self.random_heap()
        for i in trange(len(self.random_list)//20):
            random_node: HeapNode[int, int] = random.choice(list(heap.nodes()))
            heap.remove(random_node)
            self.assertEqual(len(heap), len(self.random_list) - i - 1)

    def test_decrease_key(self):
        heap = self.random_heap()
        nodes_by_value: Dict[int, Set[HeapNode[int, int]]] = defaultdict(set)
        for node in heap.nodes():
            nodes_by_value[node.key].add(node)
        changes: Dict[int, int] = {}
        for _ in trange(len(self.random_list)//20):
            while True:
                random_sorted_index = random.randint(0, len(self.random_list) - 1)
                if random_sorted_index not in changes:
                    break
            random_node: HeapNode[int, int] = next(iter(nodes_by_value[self.sorted_list[random_sorted_index]]))
            self.assertEqual(random_node.key, self.sorted_list[random_sorted_index])
            if random_node.key <= 0:
                continue
            new_key = random.randint(0, random_node.key - 1)
            nodes_by_value[random_node.key].remove(random_node)
            nodes_by_value[new_key].add(random_node)
            changes[random_sorted_index] = new_key
            heap.decrease_key(random_node, new_key)
        updated_list = []
        for i, expected in enumerate(self.sorted_list):
            if i in changes:
                updated_list.append(changes[i])
            else:
                updated_list.append(expected)
        expected_list = sorted(updated_list)
        for expected in tqdm(expected_list):
            node = heap.min_node
            heap.pop()
            self.assertEqual(node.key, expected)
