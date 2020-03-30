from typing import Callable, Generic, Iterator, Optional, TypeVar

T = TypeVar('T')
Key = TypeVar('Key')
DefaultKey = object()


class HeapNode(Generic[T, Key]):
    def __init__(self, item: T, key: Key = DefaultKey):
        self.item: T = item
        if id(key) == id(DefaultKey):
            key = item
        self.key: Key = key
        self.parent: Optional[HeapNode[T, Key]] = None
        self.child: Optional[HeapNode[T, Key]] = None
        self.left: HeapNode[T, Key] = self
        self.right: HeapNode[T, Key] = self
        self.degree: int = 0
        self.mark: bool = False
        self.deleted: bool = False

    def add_child(self, node):
        assert node != self
        if self.child is None:
            self.child = node
        else:
            node.right = self.child.right
            node.left = self.child
            self.child.right.left = node
            self.child.right = node
        self.degree += 1

    def remove_child(self, node):
        assert self.child is not None
        if self.child == self.child.right:
            self.child = None
        elif self.child == node:
            self.child = node.right
            node.right.parent = self
        node.left.right = node.right
        node.right.left = node.left
        self.degree -= 1

    @property
    def siblings(self) -> Iterator:
        node = self.right
        while node != self:
            yield node
            node = node.right

    @property
    def children(self) -> Iterator:
        assert (self.degree == 0 and self.child is None) or (self.degree == 1 + sum(1 for _ in self.child.siblings))
        if self.child is not None:
            yield self.child
            yield from self.child.siblings

    def __iter__(self):
        yield self
        if self.child:
            yield from iter(self.child)
        node = self.right
        while node != self:
            yield node
            if node.child is not None:
                yield from iter(node.child)
            node = node.right

    def __lt__(self, other):
        return self.deleted or self.key < other.key

    def __le__(self, other):
        return self < other or self.key == other.key

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return hash(self.item)

    def __repr__(self):
        return f"{self.__class__.__name__}(item={self.item!r}, key={self.key!r})"


class FibonacciHeap(Generic[T, Key]):
    def __init__(self, key: Optional[Callable[[T], Key]] = None):
        if key is None:
            self.key = lambda a: a
        else:
            self.key: Callable[[T], Key] = key
        self._min: Optional[HeapNode[T, Key]] = None
        self._root: Optional[HeapNode[T, Key]] = None
        self._n: int = 0

    def clear(self):
        self._min = None
        self._root = None
        self._n = 0

    def peek(self) -> T:
        while self._min is not None and self._min.deleted:
            self._extract_min()
        return self._min.item

    @property
    def min_node(self) -> HeapNode[T, Key]:
        return self._min

    @property
    def _roots(self) -> Iterator[HeapNode[T, Key]]:
        if self._root is not None:
            yield self._root
            yield from self._root.siblings

    def __len__(self):
        return self._n

    def __bool__(self):
        return self._n > 0

    def __iter__(self) -> Iterator[T]:
        for node in self._root:
            yield node.item

    def nodes(self) -> Iterator[HeapNode[T, Key]]:
        if self._root is None:
            return
        yield from iter(self._root)

    def _extract_min(self) -> HeapNode[T, Key]:
        z = self._min
        if z is not None:
            if z.child is not None:
                for child in list(z.children):
                    self._append_root(child)
                    child.parent = None
            self._remove_root(z)
            if z == z.right:
                self._min = self._root = None
            else:
                self._min = z.right
                self._consolidate()
            self._n -= 1
        return z

    def push(self, item: T) -> HeapNode[T, Key]:
        node = HeapNode(item=item, key=self.key(item))
        node.left = node.right = node
        self._append_root(node)
        if self._min is None or node < self._min:
            self._min = node
        self._n += 1
        return node

    def decrease_key(self, x: HeapNode[T, Key], k: Key):
        if x.key < k:
            raise ValueError(f"The key can only decrease! New key {k!r} > old key {x.key!r}.")
        x.key = k
        y = x.parent
        if y is not None and x < y:
            self._cut(x, y)
            self._cascading_cut(y)
        if x < self._min:
            self._min = x

    def __add__(self, other):
        if not other:
            return self
        elif not self:
            return other
        merged = FibonacciHeap(key=self.key)
        merged._root, merged._min = self._root, self._min
        merged.key = self.key
        last = other._root.left
        other._root.left = merged._root.left
        merged._root.left.right = other._root
        merged._root.left = last
        merged._root.left.right = merged._root
        if other._min < merged._min:
            merged._min = other._min
        merged._n = self._n + other._n
        return merged

    def _cut(self, x: HeapNode[T, Key], y: HeapNode[T, Key]):
        y.remove_child(x)
        self._append_root(x)
        x.parent = None
        x.mark = False

    def _cascading_cut(self, y: HeapNode[T, Key]):
        z = y.parent
        if z is not None:
            if y.mark is False:
                y.mark = True
            else:
                self._cut(y, z)
                self._cascading_cut(z)

    def _consolidate(self):
        a = [None] * self._n
        for x in list(self._roots):
            d = x.degree
            while a[d] is not None:
                y = a[d]
                if y < x:
                    x, y = y, x
                self._link(y, x)
                a[d] = None
                d += 1
            a[d] = x
        for i in range(0, len(a)):
            if a[i] is not None:
                if a[i] <= self._min:
                    self._min = a[i]

    def _link(self, y: HeapNode[T, Key], x: HeapNode[T, Key]):
        self._remove_root(y)
        y.left = y.right = y
        x.add_child(y)
        y.parent = x
        y.mark = False

    def _append_root(self, node: HeapNode[T, Key]):
        if self._root is None:
            self._root = node
        else:
            node.right = self._root.right
            node.left = self._root
            self._root.right.left = node
            self._root.right = node

    def _remove_root(self, node: HeapNode[T, Key]):
        if node == self._root:
            self._root = node.right
        node.left.right = node.right
        node.right.left = node.left

    def pop(self) -> T:
        while self._min is not None and self._min.deleted:
            self._extract_min()
        return self._extract_min().item


if __name__ == '__main__':
    import random

    heap = FibonacciHeap()
    random_list = [random.randint(0, 1000000) for _ in range(1000)]
    sorted_list = sorted(random_list)
    for rand_int in random_list:
        heap.push(rand_int)
    print(sorted_list)
    print([element for element in heap])
    heap_sorted = [heap.pop() for _ in range(len(random_list))]
    print(heap_sorted)
    assert sorted_list == heap_sorted
