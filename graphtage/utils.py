import os
import sys
import tempfile as tf
from collections import Counter, OrderedDict
from typing import Callable, Dict, Generic, Optional, IO, Iterator, Mapping, MutableMapping, Tuple, TypeVar, Union
from typing_extensions import Protocol
import typing

T = TypeVar('T')


class Sized(Protocol):
    getsizeof: Callable[[], int]


def getsizeof(obj) -> int:
    if hasattr(obj, 'getsizeof'):
        return obj.getsizeof()
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return sys.getsizeof(obj) + sum(getsizeof(i) for i in obj)
    elif isinstance(obj, dict):
        return sys.getsizeof(obj) + sum(getsizeof(key) + getsizeof(value) for key, value in obj.items())
    else:
        return sys.getsizeof(obj)


class HashableCounter(Generic[T], typing.Counter[T], Counter):
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
    class SparseMatrixRow(MutableMapping[int, Optional[T]]):
        def __init__(
                self,
                row_num: int,
                num_cols: Optional[int] = None,
                default_value: Optional[T] = None
        ):
            self.row_num: int = row_num
            self.row: Dict[int, Optional[T]] = {}
            self.num_cols: Optional[int] = num_cols
            self.default_value: Optional[T] = default_value

        def shape(self) -> int:
            if self.num_cols is None:
                if self.row:
                    return max(self.row.keys()) + 1
                else:
                    return 0
            else:
                return self.num_cols

        def clear(self):
            self.row = {}

        def getsizeof(self) -> int:
            return sys.getsizeof(self) + getsizeof(self.row)

        def __len__(self) -> int:
            return len(self.row)

        def __iter__(self) -> Iterator[int]:
            return iter(self.row)

        def __getitem__(self, col: int) -> Optional[T]:
            if col not in self.row:
                if self.num_cols is not None and col >= self.num_cols:
                    raise ValueError((self.row_num, col))
                return self.default_value
            return self.row[col]

        def __setitem__(self, col: int, value: T):
            if col not in self.row:
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
        self.rows: Dict[int, SparseMatrix[T].SparseMatrixRow] = {}
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.default_value = default_value
        self._max_row: Optional[int] = None

    def clear(self):
        self.rows = {}

    def getsizeof(self) -> int:
        return sys.getsizeof(self) + getsizeof(self.rows)

    def __getitem__(self, row: int) -> MutableMapping[int, Optional[T]]:
        if row in self.rows:
            return self.rows[row]
        elif self.num_rows is not None and row >= self.num_rows:
            raise ValueError(row)
        new_row = SparseMatrix.SparseMatrixRow(row, self.num_cols, default_value=self.default_value)
        self.rows[row] = new_row
        return new_row

    def __len__(self) -> int:
        if self.num_rows is not None:
            return self.num_rows
        elif self._max_row is not None:
            return self._max_row
        else:
            return 0

    def __iter__(self) -> Iterator[MutableMapping[int, Optional[T]]]:
        for i in range(len(self)):
            yield self[i]

    def num_filled_elements(self) -> int:
        return sum(len(row.row) for row in self.rows.values())

    def shape(self) -> Tuple[int, int]:
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
    def __init__(self, contents: Union[str, bytes], prefix: Optional[str] = None, suffix: Optional[str] = None):
        self._temp: Optional[IO] = None
        self._data: Union[str, bytes] = contents
        self._prefix: Optional[str] = prefix
        self._suffix: Optional[str] = suffix

    def __enter__(self) -> str:
        self._temp = tf.NamedTemporaryFile(prefix=self._prefix, suffix=self._suffix, delete=False)
        self._temp.write(self._data)
        self._temp.flush()
        self._temp.close()
        return self._temp.name

    def __exit__(self, type, value, traceback):
        if self._temp is not None:
            os.unlink(self._temp.name)
            self._temp = None
