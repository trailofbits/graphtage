from typing import Dict, Generic, Optional, Iterator, Mapping, MutableMapping, TypeVar

T = TypeVar('T')


class SparseMatrix(Generic[T], Mapping[int, MutableMapping[int, Optional[T]]]):
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
