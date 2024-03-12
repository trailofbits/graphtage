from enum import Enum
from typing import Dict, List, Sequence, Tuple, TypeVar

from .printer import DEFAULT_PRINTER

T = TypeVar("T")


class Edit(Enum):
    INSERT = "INSERT"
    REMOVE = "REMOVE"
    KEEP = "KEEP"

    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name}"


def myers(from_seq: Sequence[T], to_seq: Sequence[T]) -> List[Tuple[Edit, T]]:
    fringe: Dict[int, Tuple[int, List[Tuple[Edit, T]]]] = {1: (0, [])}

    for d in DEFAULT_PRINTER.trange(0, len(from_seq) + len(to_seq) + 1, desc="Diffing Sequences", leave=False,
                                    delay=2.0):
        for k in range(-d, d + 1, 2):
            down = k == -d or (k != d and fringe[k - 1][0] < fringe[k + 1][0])

            if down:
                old_from_index, history = fringe[k + 1]
                from_index = old_from_index
            else:
                old_from_index, history = fringe[k - 1]
                from_index = old_from_index + 1
            to_index = from_index - k

            history = list(history)

            if 1 <= to_index <= len(to_seq) and down:
                history.append((Edit.INSERT, to_seq[to_index - 1]))
            elif 1 <= from_index <= len(from_seq):
                history.append((Edit.REMOVE, from_seq[from_index - 1]))

            while from_index < len(from_seq) and to_index < len(to_seq) and from_seq[from_index] == to_seq[to_index]:
                from_index += 1
                to_index += 1
                history.append((Edit.KEEP, from_seq[from_index - 1]))

            if from_index >= len(from_seq) and to_index >= len(to_seq):
                return history

            fringe[k] = from_index, history

    raise NotImplementedError("This should not be reachable")
