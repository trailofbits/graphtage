"""Generic utility functions and classes."""

import os
import sys
import tempfile as tf
from collections import Counter, OrderedDict
from collections.abc import Iterable
from collections.abc import Sized as AbstractSized
from typing import Any, Callable, Dict, Generic, Optional, IO, Iterator, Mapping, MutableMapping, Tuple, TypeVar, Union
from typing import Iterable as IterableType
from typing_extensions import Protocol
import typing

from .fibonacci import FibonacciHeap, MaxFibonacciHeap

T = TypeVar('T')


class Sized(Protocol):
    """A protocol for objects that have a ``getsizeof`` function."""
    getsizeof: Callable[[], int]
    """Returns the size of this object."""


def getsizeof(obj) -> int:
    """A function to calculate the memory footprint of an object.

    If the object implements the :class:`Sized` protocol (*i.e.*, if it implements a ``getsizeof`` method), then this is
    used::

        if hasattr(obj, 'getsizeof'):
            return obj.getsizeof()

    If the object is a list or tuple, then::

        return sys.getsizeof(obj) + sum(getsizeof(i) for i in obj)

    If the object is a dict, then::

        return sys.getsizeof(obj) + sum(getsizeof(key) + getsizeof(value) for key, value in obj.items())

    Otherwise::

        return sys.getsizeof(obj)

    Args:
        obj: The object to measure.

    Returns:
        int: An approximation of the memory footprint of the object, in bytes.

    """
    if hasattr(obj, 'getsizeof'):
        return obj.getsizeof()
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return sys.getsizeof(obj) + sum(getsizeof(i) for i in obj)
    elif isinstance(obj, dict):
        return sys.getsizeof(obj) + sum(getsizeof(key) + getsizeof(value) for key, value in obj.items())
    else:
        return sys.getsizeof(obj)


class HashableCounter(Generic[T], typing.Counter[T], Counter):
    """A :class:`Counter` that supports being hashed even though it is mutable."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __hash__(self):
        h = 0
        for key, value in self.items():
            h ^= hash((key, value))
        return h

    def elements(self) -> Iterator:
        """Iterator over elements repeating each as many times as its count.

        Examples:

        .. code-block:: python

            >>> c = Counter('ABCABC')
            >>> sorted(c.elements())
            ['A', 'A', 'B', 'B', 'C', 'C']

            # Knuth's example for prime factors of 1836:  2**2 * 3**3 * 17**1
            >>> prime_factors = Counter({2: 2, 3: 3, 17: 1})
            >>> product = 1
            >>> for factor in prime_factors.elements():     # loop over factors
            ...     product *= factor                       # and multiply them
            >>> product
            1836

        Note:
            If an element's count has been set to zero or is a negative number, elements() will ignore it.

        """
        # Extending this solely to fix the broken docstring in Counter.elements!
        return super().elements()


class OrderedCounter(Counter, OrderedDict):
    """A counter that remembers the order elements are first encountered"""

    def __hash__(self):
        h = 0
        for key, value in self.items():
            h ^= hash((key, value))
        return h

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)

    def elements(self) -> Iterator:
        """Iterator over elements repeating each as many times as its count.

        Examples:

        .. code-block:: python

            >>> c = Counter('ABCABC')
            >>> sorted(c.elements())
            ['A', 'A', 'B', 'B', 'C', 'C']

            # Knuth's example for prime factors of 1836:  2**2 * 3**3 * 17**1
            >>> prime_factors = Counter({2: 2, 3: 3, 17: 1})
            >>> product = 1
            >>> for factor in prime_factors.elements():     # loop over factors
            ...     product *= factor                       # and multiply them
            >>> product
            1836

        Note:
            If an element's count has been set to zero or is a negative number, elements() will ignore it.

        """
        # Extending this solely to fix the broken docstring in Counter.elements!
        return super().elements()


class SparseMatrix(Sized, Generic[T], Mapping[int, MutableMapping[int, Optional[T]]]):
    """A sparse matrix that can store arbitrary Python objects.

    For sparse matrices storing homogeneous items and/or native types, it is more efficient to use an implementation
    like a `scipy sparse matrix <https://docs.scipy.org/doc/scipy/reference/sparse.html>`__.

    """
    class SparseMatrixRow(Sized, MutableMapping[int, Optional[T]]):
        """A row of a sparse matrix."""
        def __init__(
                self,
                row_num: int,
                num_cols: Optional[int] = None,
                default_value: Optional[T] = None
        ):
            """Initializes a sparse matrix row.

            Args:
                row_num: The index of this row.
                num_cols: An optional number of columns in this row. If omitted, this row will be unbounded and allow
                    insertion and access at any column index.
                default_value: An optional default value to use if a column is accessed before it is assigned.
            """
            self.row_num: int = row_num
            """The index of this row."""
            self.row: Dict[int, Optional[T]] = {}
            """Data structure holding the contents of this row."""
            self.num_cols: Optional[int] = num_cols
            """The number of columns in this row."""
            self.default_value: Optional[T] = default_value
            """The default value for this row."""

        def shape(self) -> int:
            """The number of columns in this row.

            This is equivalent to::

                if self.num_cols is None:
                    if self.row:
                        return max(self.row.keys()) + 1
                    else:
                        return 0
                else:
                    return self.num_cols

            """
            if self.num_cols is None:
                if self.row:
                    return max(self.row.keys()) + 1
                else:
                    return 0
            else:
                return self.num_cols

        def clear(self):
            """Clears the contents of this row."""
            self.row = {}

        def getsizeof(self) -> int:
            """Returns an approximation of the number of bytes used by this row in memory.

            This is equivalent to::

                return sys.getsizeof(self) + getsizeof(self.row)

            """
            return sys.getsizeof(self) + getsizeof(self.row)

        def __len__(self) -> int:
            """Returns the number of cells defined this row."""
            return len(self.row)

        def __iter__(self) -> Iterator[int]:
            """Iterates over the indexes of columns that have been set in this row."""
            return iter(self.row)

        def __getitem__(self, col: int) -> Optional[T]:
            """Returns the value of the given column of this row.

            Args:
                col: The index of the column to get.

            Returns:
                Optional[T]: The value of the column, or the default value if it has not yet been set.

            Raises:
                IndexError: If :attr:`self.num_cols <SparseMatrixRow.num_cols>` is not :const:`None` and :obj:`col` is
                    greater than or equal to it.

            """
            if col not in self.row:
                if self.num_cols is not None and col >= self.num_cols:
                    raise IndexError((self.row_num, col))
                return self.default_value
            return self.row[col]

        def __setitem__(self, col: int, value: T):
            if self.num_cols is not None and col >= self.num_cols:
                raise IndexError((self.row_num, col))
            self.row[col] = value

        def __delitem__(self, col: int):
            if col in self.row:
                del self.row[col]

    def __init__(
            self,
            num_rows: Optional[int] = None,
            num_cols: Optional[int] = None,
            default_value: Optional[T] = None
    ):
        """Initializes a sparse matrix.

        Args:
            num_rows: An optional bound on the number of rows.
            num_cols: An optional bound on the number of columns.
            default_value: An optional default value to return if cells are accessed before they are set.
        """
        self.rows: Dict[int, SparseMatrix[T].SparseMatrixRow] = {}
        """The rows of this matrix."""
        self.num_rows: Optional[int] = num_rows
        """The number of rows in this matrix, or :const:`None` if there is no bound."""
        self.num_cols: Optional[int] = num_cols
        """The number of columns in this matrox, or :const:`None` if there is no bound."""
        self.default_value: Optional[T] = default_value
        """The default value to return if cells are accessed before they are set."""
        self._max_row: Optional[int] = None

    def clear(self):
        """Clears the contents of this matrix."""
        self.rows = {}

    def getsizeof(self) -> int:
        """Calculates the approximate memory footprint of this matrix in bytes."""
        return sys.getsizeof(self) + getsizeof(self.rows)

    def __getitem__(self, row: int) -> MutableMapping[int, Optional[T]]:
        """Returns the value of the given row of this matrix.

        Args:
            row: The index of the row to get.

        Returns:
            MutableMapping[int, Optional[T]]: The contents of the row.

        Raises:
            IndexError: If :attr:`self.num_rows <SparseMatrix.num_rows>` is not :const:`None` and :obj:`row` is
                greater than or equal to it.

        """
        if row in self.rows:
            return self.rows[row]
        elif self.num_rows is not None and row >= self.num_rows:
            raise IndexError(row)
        new_row = SparseMatrix.SparseMatrixRow(row, self.num_cols, default_value=self.default_value)
        self.rows[row] = new_row
        if self._max_row is None:
            self._max_row = row
        else:
            self._max_row = max(self._max_row, row)
        return new_row

    def __len__(self) -> int:
        """Returns the number of rows in this matrix.

        If :attr:`self.num_rows <SparseMatrix.num_rows>` is :const:`None`, this will return the maximum row index of a
        cell that has been set, plus one.

        """
        if self.num_rows is not None:
            return self.num_rows
        elif self._max_row is not None:
            return self._max_row + 1
        else:
            return 0

    def __iter__(self) -> Iterator[MutableMapping[int, Optional[T]]]:
        for i in range(len(self)):
            yield self[i]

    def num_filled_elements(self) -> int:
        """Counts the number of elements in this matrix that have been explicitly set."""
        return sum(len(row.row) for row in self.rows.values())

    def shape(self) -> Tuple[int, int]:
        """Returns the (number of rows, number of columns) of this matrix."""
        if self.num_rows is None:
            if self.rows:
                rows = max(self.rows.keys()) + 1
            else:
                rows = 0
        else:
            rows = self.num_rows
        if self.num_cols is None:
            if self.rows:
                cols = max(row.shape() for row in self.rows.values())
            else:
                cols = 0
        else:
            cols = self.num_cols
        return rows, cols


class Tempfile:
    """Creates a temporary file containing a given byte sequence.

    The file will automatically be cleaned up after its use. This is useful for interacting with functions and
    libraries that only accept a path and not a stream object.

    Examples:

        >>> from graphtage.utils import Tempfile
        >>> with Tempfile(b"foo") as tmp:
        ...     print(tmp)
        /var/folders/bs/hrvzrctx6tg2_j17gb6wckph0000gn/T/tmpkza5fvr_
        >>> with Tempfile(b"foo") as tmp:
        ...     with open(tmp, 'r') as f:
        ...         print(f.read())
        foo

    """
    def __init__(self, contents: bytes, prefix: Optional[str] = None, suffix: Optional[str] = None):
        """Initializes a Tempfile

        Args:
            contents: The contents to be populated in the file.
            prefix: An optional prefix for the filename.
            suffix: An optional suffix for the filename.
        """
        self._temp: Optional[IO] = None
        self._data: bytes = contents
        self._prefix: Optional[str] = prefix
        self._suffix: Optional[str] = suffix

    def __enter__(self) -> str:
        """Constructs a tempfile, returning the path to the file."""
        self._temp = tf.NamedTemporaryFile(prefix=self._prefix, suffix=self._suffix, delete=False)
        self._temp.write(self._data)
        self._temp.flush()
        self._temp.close()
        return self._temp.name

    def __exit__(self, type, value, traceback):
        """Automatically cleans up the tempfile."""
        if self._temp is not None:
            os.unlink(self._temp.name)
            self._temp = None


def smallest(*sequence: Union[T, IterableType[T]], n: int = 1, key: Optional[Callable[[T], Any]] = None) -> Iterator[T]:
    if len(sequence) == 1 and isinstance(sequence, Iterable):
        sequence = sequence[0]

    if isinstance(sequence, AbstractSized) and len(sequence) <= n:
        yield from sequence
        return

    heap: FibonacciHeap[T, T] = FibonacciHeap(key=key)

    for s in sequence:
        heap.push(s)

    for _ in range(n):
        if not heap:
            break
        yield heap.pop()


def largest(*sequence: Union[T, IterableType[T]], n: int = 1, key: Optional[Callable[[T], Any]] = None) -> Iterator[T]:
    if len(sequence) == 1 and isinstance(sequence[0], Iterable):
        sequence = sequence[0]

    if isinstance(sequence, AbstractSized) and len(sequence) <= n:
        yield from sequence
        return

    heap: MaxFibonacciHeap[T, Any] = MaxFibonacciHeap(key=key)

    for s in sequence:
        heap.push(s)

    for _ in range(n):
        if not heap:
            break
        yield heap.pop()
