"""A pure Python implementation of a `Fibonacci Heap`_.

Many of the algorithms in Graphtage only require partially sorting collections, so we can get a speedup from using a
Fibonacci Heap that has amortized constant time insertion.

.. _Fibonacci Heap:
    https://en.wikipedia.org/wiki/Fibonacci_heap

"""

from typing import Callable, Generic, Iterator, Optional, TypeVar

T = TypeVar('T')
Key = TypeVar('Key')
DefaultKey = object()


class HeapNode(Generic[T, Key]):
    """A node in a :class:`FibonacciHeap`."""
    def __init__(self, item: T, key: Key = DefaultKey):
        """Initializes a Fibonacci heap node.

        Args:
            item: The heap item associated with the node.
            key: An optional key to use for the item in sorting. If omitted, the item itself will be used.

        """
        self.item: T = item
        """The item associated with this heap node."""
        if id(key) == id(DefaultKey):
            key = item
        self.key: Key = key
        """The key to be used when sorting this heap node."""
        self.parent: Optional[HeapNode[T, Key]] = None
        """The node's parent."""
        self.child: Optional[HeapNode[T, Key]] = None
        """The node's child."""
        self.left: HeapNode[T, Key] = self
        """The left sibling of this node, or :obj:`self` if it has no left sibling."""
        self.right: HeapNode[T, Key] = self
        """The right sibling of this node, or :obj:`self` if it has no left sibling."""
        self.degree: int = 0
        """The degree of this node (*i.e.*, the number of its children)."""
        self.mark: bool = False
        """The node's marked state."""
        self.deleted: bool = False
        """Whether the node has been deleted.
        
        This is to prevent nodes from being manipulated after they have been removed from a heap.
        
        Warning:
            Do not set :attr:`HeapNode.deleted` to :const:`True` unless the node has already been removed from the heap.

        """

    def add_child(self, node):
        """Adds a child to this heap node, incrementing its degree."""
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
        """Removes a child from this heap node, decrementing its degree."""
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
    def siblings(self) -> Iterator['HeapNode[T, Key]']:
        """Iterates over this node's siblings.

        Equivalent to::

            node = self.right
            while node != self:
                yield node
                node = node.right

        """
        node = self.right
        while node != self:
            yield node
            node = node.right

    @property
    def children(self) -> Iterator['HeapNode[T, Key]']:
        """Iterates over this node's children.

        Equivalent to::

            if self.child is not None:
                yield self.child
                yield from self.child.siblings

        """
        assert (self.degree == 0 and self.child is None) or (self.degree == 1 + sum(1 for _ in self.child.siblings))
        if self.child is not None:
            yield self.child
            yield from self.child.siblings

    def __iter__(self) -> Iterator['HeapNode[T, Key]']:
        """Iterates over all of this node's descendants, including itself."""
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
        return (self.deleted and not other.deleted) or self.key < other.key

    def __le__(self, other):
        return self < other or self.key == other.key

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return hash(self.item)

    def __repr__(self):
        return f"{self.__class__.__name__}(item={self.item!r}, key={self.key!r})"


class FibonacciHeap(Generic[T, Key]):
    """A Fibonacci Heap."""
    def __init__(self, key: Optional[Callable[[T], Key]] = None):
        """Initializes a Fibonacci heap.

        Args:
            key: An optional function that accepts an item and returns the key to be used for comparing that item.
                If omitted, it is equivalent to::

                    lambda item: item

        """
        if key is None:
            self.key = lambda a: a
            """The function to extract comparison keys from items."""
        else:
            self.key: Callable[[T], Key] = key
        self._min: Optional[HeapNode[T, Key]] = None
        self._root: Optional[HeapNode[T, Key]] = None
        self._n: int = 0

    def clear(self):
        """Removes all items from this heap."""
        self._min = None
        self._root = None
        self._n = 0

    def peek(self) -> T:
        """Returns the smallest element of the heap without removing it.

        Returns:
            T: The smallest element of the heap.

        """
        while self._min is not None and self._min.deleted:
            self._extract_min()
        return self._min.item

    def remove(self, node: HeapNode[T, Key]):
        """Removes the given node from this heap.

        Args:
            node: The node to be removed.

        Warning:
            This function assumes that the provided node is actually a member of this heap. It also assumes (but does
            not check) that :attr:`node.deleted <HeapNode.deleted>` is :const:`False`. If either of these assumptions
            is incorrect, it will lead to undefined behavior and corruption of the heap.

        """
        node.deleted = True
        y = node.parent
        if y is not None and node < y:
            self._cut(node, y)
            self._cascading_cut(y)
        self._min = node
        self._extract_min()

    @property
    def min_node(self) -> HeapNode[T, Key]:
        """Returns the heap node associated with the smallest item in the heap, without removing it."""
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
        """Iterates over all of the heap nodes in this heap."""
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
        """Adds a new item to this heap.

        Returns:
            HeapNode[T, Key]: The heap node created to store the new item.

        """
        node = HeapNode(item=item, key=self.key(item))
        node.left = node.right = node
        self._append_root(node)
        if self._min is None or node < self._min:
            self._min = node
        self._n += 1
        return node

    def decrease_key(self, x: HeapNode[T, Key], k: Key):
        """Decreases the key value associated with the given node.

        Args:
            x: The node to modify.
            k: The new key value.

        Raises:
            ValueError: If :attr:`x.key <HeapNode.key>` is less than :obj:`k`.

        """
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
        """Returns and removes the smallest item from this heap."""
        while self._min is not None and self._min.deleted:
            self._extract_min()
        return self._extract_min().item


class ReversedComparator(Generic[Key]):
    """A wrapper that reverses the semantics of its comparison operators."""
    def __init__(self, key: Key):
        self.key = key

    def __lt__(self, other):
        return self.key > other.key

    def __le__(self, other):
        return self.key >= other.key

    def __eq__(self, other):
        return self.key == other.key

    def __hash__(self):
        return hash(self.key)


class MaxFibonacciHeap(Generic[T, Key], FibonacciHeap[T, ReversedComparator[Key]]):
    """A Fibonacci Heap that yields items in decreasing order, using a :class:`ReversedComparator`."""
    def __init__(self, key: Optional[Callable[[T], Key]] = None):
        if key is None:
            def key(n: T):
                return n
        super().__init__(key=lambda n: ReversedComparator(key(n)))
