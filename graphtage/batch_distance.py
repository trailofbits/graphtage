"""Exact Levenshtein distances for every pair drawn from two collections of strings.

:func:`graphtage.levenshtein.levenshtein_distance` prices one pair of strings with a pure Python dynamic program,
so a caller that needs the whole cross product of two collections pays the interpreter cost of every matrix cell.
This module answers the same question for a whole batch at once, and dispatches to whichever registered backend is
fastest for the size of the problem.

The answers are exact. Every backend returns what :func:`graphtage.levenshtein.levenshtein_distance` returns for the
same pair, and ``test/test_batch_distance.py`` pins that agreement over a randomized corpus.

Two backends ship with Graphtage:

``python``
    Calls :func:`graphtage.levenshtein.levenshtein_distance` once per pair. It is the oracle that the other backends
    are checked against, and the fallback for batches too small for vectorization to pay off.

``numpy``
    A batched min-plus scan over :mod:`numpy` arrays. It advances every pair in the batch through one row of its
    Levenshtein matrix per array pass, so a batch costs as many passes as the longest string is long, rather than as
    many interpreter steps as the batch has matrix cells.

Set the ``GRAPHTAGE_BATCH_BACKEND`` environment variable to a backend name to pin the choice, which is useful for
benchmarking one backend against another and for testing that they agree.

Graphtage reaches the batch through :func:`cost`, which prices a single pair the way
:func:`graphtage.levenshtein.levenshtein_distance` does but answers from a pre-computed block when one holds the
pair. :class:`graphtage.multiset.MultiSetEdit` and :class:`graphtage.levenshtein.EditDistance` each face a cross
product of node pairs whose edits are constructed one at a time, so they call :func:`preprice` or
:func:`preprice_product` before constructing any of them and let every later :func:`cost` read the answer out of the
block.

"""

import os
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from .levenshtein import levenshtein_distance

BACKEND_ENV_VAR = 'GRAPHTAGE_BATCH_BACKEND'
"""The environment variable that pins backend selection, overriding the automatic choice."""

FALLBACK_BACKEND = 'python'
"""The backend used when no faster one is both available and willing to take the batch."""

NUMPY_MIN_PAIRS = 32
"""The smallest batch the ``numpy`` backend accepts.

Below it, laying out the arrays can cost more than the scalar dynamic program it replaces. Where the two meet
depends on how long the strings are: measured on an Apple M-series laptop, the ``numpy`` backend overtakes the
``python`` one at about 9 pairs of 8 to 25 character strings but not until about 50 pairs of 2 to 5 character ones.
This threshold covers the shorter case, and either backend answers a batch that small in well under a millisecond.

"""

NUMPY_MAX_CHUNK_BYTES = 256 * 1024
"""The working set budget for one chunk of the ``numpy`` backend's dynamic program, in bytes.

A batch is split into chunks small enough that ``pairs * (longest + 1) * 4`` stays under this, which both bounds
peak memory and keeps the rows the scan sweeps back and forth over small enough to stay in cache. Measured on an
Apple M-series laptop, raising it to 8 MiB costs about a third of the throughput, and lowering it to 64 KiB costs a
few percent to the extra per-chunk bookkeeping.

"""

_KERNEL_ITEM_BYTES = 4
"""The width of one dynamic programming cell, which the scan holds as :class:`numpy.int32`."""

BLOCK_STACK_LIMIT = 8
"""The number of pre-priced blocks :func:`cost` searches before the oldest is dropped.

A block is a pure cache of a deterministic function, so dropping one costs a recomputation and nothing else. The
limit is what keeps the structure bounded: an edit installs a block in its constructor and reads from it across
however many later calls it takes to tighten its bounds, and nothing tells the cache when that edit is finished.

"""

PREPRICE_MIN_PAIRS = 32
"""The number of pairs below which pre-pricing is skipped and every pair goes through the scalar dynamic program.

Building a block costs a pass over the pairs, a batch, and the two dictionaries that index it, none of which the
scalar path pays. Measured on an Apple M-series laptop over lists of 6 to 24 character strings, diffing gets
faster from about 36 pairs and slower below about 25, which is the same crossing as :const:`NUMPY_MIN_PAIRS`:
below it a batch is answered by the ``python`` backend, so a block buys the overhead of vectorizing without the
vectorization.

"""

PREPRICE_MAX_CELLS = 4 * 1024 * 1024
"""The largest block, in cells, that pre-pricing will build.

The pairs a caller offers need not fill the rectangle their distinct sides span. A collection holding both leaves
and key/value pairs, for instance, spans a rectangle several times larger than the number of pairs in it. Cells
that no pair fills cost four bytes each and hold no answer, so a batch whose rectangle is this large is left to the
scalar path rather than given a block of 16 MB.

"""

VECTORIZED_MIN_CELLS = 4096
"""The size of Levenshtein matrix above which one pair alone is worth handing to a vectorized backend.

The batched scan costs one array pass per row rather than one interpreter step per cell, so it beats the scalar
dynamic program on a single pair of long strings even though there is nothing to amortize the array setup over.
Measured on an Apple M-series laptop, the two meet at about 52 by 52 characters and the scan is 2.4 times faster
by 128 by 128. This threshold, 64 by 64, sits just above the crossing.

"""

_UNPRICED = -1
"""The value of a block cell that no pair filled. Distances are never negative, so it cannot be mistaken for one."""

_STR_PAD = 0x110000
"""The padding symbol for :class:`str` batches: the first value that is not a Unicode code point."""

_BYTES_PAD = 0x100
"""The padding symbol for :class:`bytes` batches: the first value that is not a byte."""

Pair = tuple[str, str] | tuple[bytes, bytes]
"""Two strings to compare.

Both sides must be the same type. A pair that mixes :class:`str` with :class:`bytes` is rejected rather than
compared, because ``ord('a')`` and ``b'a'[0]`` are both ``97`` while ``'a' == b'a'`` is :const:`False`, so the two
readings of such a pair disagree.

"""


class BatchBackend(Protocol):
    """A strategy for computing the Levenshtein distance of every pair in a batch."""

    name: str
    """The name this backend is registered under, and that callers and the environment variable select it by."""

    min_pairs: int
    """The smallest batch this backend accepts. The dispatcher skips it for anything smaller."""

    speed_rank: int
    """Relative throughput, higher being faster.

    The dispatcher picks the highest-ranked available backend whose :attr:`BatchBackend.min_pairs` the batch meets.

    """

    def is_available(self) -> bool:
        """Returns whether this backend can run in the current environment."""
        raise NotImplementedError(f"Class {self.__class__.__name__} must implement is_available")

    def distances(self, pairs: Sequence[Pair]) -> np.ndarray:
        """Computes the Levenshtein distance of each pair.

        Args:
            pairs: The pairs to price. Every pair has two sides of the same type, and neither side is empty.

        Returns:
            numpy.ndarray: A one-dimensional array of :class:`numpy.int64` holding one distance per pair, in the
            order the pairs were given.

        """
        raise NotImplementedError(f"Class {self.__class__.__name__} must implement distances")


class PythonBackend:
    """The reference backend, which applies :func:`graphtage.levenshtein.levenshtein_distance` to each pair."""

    name = FALLBACK_BACKEND
    min_pairs = 0
    speed_rank = 0

    def is_available(self) -> bool:
        """Returns :const:`True`, because the pure Python backend has no requirements."""
        return True

    def distances(self, pairs: Sequence[Pair]) -> np.ndarray:
        """Computes the Levenshtein distance of each pair, one at a time.

        Args:
            pairs: The pairs to price.

        Returns:
            numpy.ndarray: One distance per pair, in the order the pairs were given.

        """
        return np.fromiter(
            (levenshtein_distance(left, right) for left, right in pairs), dtype=np.int64, count=len(pairs)
        )


def _encode(value: str | bytes) -> np.ndarray:
    """Turns a string into the array of symbols the scan compares.

    A :class:`str` becomes one :class:`numpy.uint32` per code point, which is what Python itself iterates over, so
    characters outside the Basic Multilingual Plane need no special handling. A :class:`bytes` becomes one
    :class:`numpy.uint16` per byte. The two widths differ so that a batch of one kind cannot be concatenated with a
    batch of the other without an explicit cast.

    Args:
        value: The string to encode.

    Returns:
        numpy.ndarray: The symbols of ``value``.

    """
    if isinstance(value, str):
        return np.frombuffer(value.encode('utf-32-le', 'surrogatepass'), dtype=np.uint32)
    return np.frombuffer(value, dtype=np.uint8).astype(np.uint16)


def _pad(rows: list[np.ndarray], width: int, pad_value: int) -> np.ndarray:
    """Stacks encoded strings of differing lengths into one rectangular array.

    Args:
        rows: The encoded strings, all of the same dtype.
        width: The number of columns, which is at least the length of the longest row.
        pad_value: The symbol to fill the unused columns with.

    Returns:
        numpy.ndarray: A ``(len(rows), width)`` array.

    """
    matrix = np.full((len(rows), width), pad_value, dtype=rows[0].dtype)
    for index, row in enumerate(rows):
        matrix[index, :row.shape[0]] = row
    return matrix


def _scan(
        short: np.ndarray,
        short_lengths: np.ndarray,
        long: np.ndarray,
        long_lengths: np.ndarray,
) -> np.ndarray:
    """Advances every pair in a chunk through its Levenshtein matrix, one row per array pass.

    The horizontal dependency ``current[j] = min(base[j], current[j - 1] + 1)`` is what normally stops a row from
    being vectorized. Subtracting ``j`` from both sides turns it into a running minimum of ``base[j] - j``, which
    :func:`numpy.minimum.accumulate` computes in a single pass, so a row costs a constant number of array
    operations instead of one interpreter step per cell.

    Pairs of differing lengths share the chunk. A pair is read out of the row in which ``i`` reaches the length of
    its shorter side, at the column given by the length of its longer side, so the padding beyond those never
    reaches the answer.

    Args:
        short: The padded symbols of each pair's shorter side, which drives the row loop.
        short_lengths: The unpadded length of each row of ``short``.
        long: The padded symbols of each pair's longer side.
        long_lengths: The unpadded length of each row of ``long``.

    Returns:
        numpy.ndarray: One distance per row, in the order the rows were given.

    """
    pairs, rows = short.shape
    columns = long.shape[1] + 1
    column_index = np.arange(columns, dtype=np.int32)
    previous = np.tile(column_index, (pairs, 1))
    current = np.empty_like(previous)
    result = np.zeros(pairs, dtype=np.int64)
    for i in range(1, rows + 1):
        substitution = (short[:, i - 1][:, None] != long).astype(np.int32)
        current[:, 0] = i
        np.minimum(previous[:, :-1] + substitution, previous[:, 1:] + 1, out=current[:, 1:])
        current -= column_index
        np.minimum.accumulate(current, axis=1, out=current)
        current += column_index
        finished = short_lengths == i
        if finished.any():
            result[finished] = current[finished, long_lengths[finished]]
        previous, current = current, previous
    return result


def _chunk_bounds(long_lengths: np.ndarray, max_bytes: int) -> list[tuple[int, int]]:
    """Splits a length-ordered batch into chunks that each fit the working set budget.

    Padding a chunk costs its widest member's length in every one of its rows, so a single long string among short
    ones would otherwise make the scan do far more work than the scalar dynamic program it replaces. Splitting a
    batch that is already ordered by length keeps the long string with other long strings.

    Args:
        long_lengths: The length of each pair's longer side, in the order the batch will be processed, which
            :func:`_kind_distances` has sorted so that similar lengths sit together.
        max_bytes: The budget for one chunk, as described by :const:`NUMPY_MAX_CHUNK_BYTES`.

    Returns:
        List[Tuple[int, int]]: Half-open ``(start, stop)`` index ranges covering the whole batch.

    """
    bounds: list[tuple[int, int]] = []
    start = 0
    widest = 0
    for position, length in enumerate(long_lengths):
        candidate = max(widest, int(length))
        fits = (position - start + 1) * (candidate + 1) * _KERNEL_ITEM_BYTES <= max_bytes
        if position > start and not fits:
            bounds.append((start, position))
            start = position
            widest = int(length)
        else:
            widest = candidate
    bounds.append((start, len(long_lengths)))
    return bounds


def _kind_distances(pairs: Sequence[Pair], pad_value: int) -> np.ndarray:
    """Computes the distances of a batch whose pairs are all of the same type.

    Args:
        pairs: The pairs to price, every one of them either two :class:`str` or two :class:`bytes`, and with each
            side already oriented so that the shorter string comes first.
        pad_value: The padding symbol for this type of string.

    Returns:
        numpy.ndarray: One distance per pair, in the order the pairs were given.

    """
    short_lengths = np.array([len(short) for short, _ in pairs], dtype=np.intp)
    long_lengths = np.array([len(long) for _, long in pairs], dtype=np.intp)
    order = np.lexsort((long_lengths, short_lengths))
    encodings: dict[str | bytes, np.ndarray] = {}
    result = np.empty(len(pairs), dtype=np.int64)
    for start, stop in _chunk_bounds(long_lengths[order], NUMPY_MAX_CHUNK_BYTES):
        selected = order[start:stop]
        chunk = [pairs[index] for index in selected]
        for pair in chunk:
            for value in pair:
                if value not in encodings:
                    encodings[value] = _encode(value)
        result[selected] = _scan(
            _pad([encodings[short] for short, _ in chunk], int(short_lengths[selected].max()), pad_value),
            short_lengths[selected],
            _pad([encodings[long] for _, long in chunk], int(long_lengths[selected].max()), pad_value),
            long_lengths[selected],
        )
    return result


class NumpyBackend:
    """A batched min-plus scan over :mod:`numpy` arrays.

    The batch is prepared in three steps before :func:`_scan` sees it:

    1. Each pair is oriented so that its shorter string drives the row loop, which is free because the Levenshtein
       distance is symmetric. That gives the fewest, widest array passes.
    2. Pairs of :class:`str` and pairs of :class:`bytes` are separated, because the two encodings assign different
       meanings to the same number.
    3. Pairs are sorted by length and split into chunks, so that padding one long string does not widen the rows of
       every short pair that shares its batch.

    """

    name = 'numpy'
    min_pairs = NUMPY_MIN_PAIRS
    speed_rank = 10

    def is_available(self) -> bool:
        """Returns :const:`True`, because :mod:`numpy` is a hard dependency of Graphtage."""
        return True

    def distances(self, pairs: Sequence[Pair]) -> np.ndarray:
        """Computes the Levenshtein distance of each pair.

        Args:
            pairs: The pairs to price.

        Returns:
            numpy.ndarray: One distance per pair, in the order the pairs were given.

        """
        oriented = [(left, right) if len(left) <= len(right) else (right, left) for left, right in pairs]
        result = np.zeros(len(pairs), dtype=np.int64)
        for kind, pad_value in ((str, _STR_PAD), (bytes, _BYTES_PAD)):
            selected = [index for index, (short, _) in enumerate(oriented) if isinstance(short, kind)]
            if selected:
                result[selected] = _kind_distances([oriented[index] for index in selected], pad_value)
        return result


_BACKENDS: dict[str, BatchBackend] = {}


def register_backend(backend: BatchBackend) -> None:
    """Adds a backend to the registry.

    Args:
        backend: The backend to register. Its :attr:`BatchBackend.name` becomes the value that the ``backend``
            argument and the ``GRAPHTAGE_BATCH_BACKEND`` environment variable accept.

    Raises:
        ValueError: If a backend of that name is already registered.

    """
    if backend.name in _BACKENDS:
        raise ValueError(f"A batch distance backend named {backend.name!r} is already registered")
    _BACKENDS[backend.name] = backend


def available_backends() -> tuple[str, ...]:
    """Returns the names of the backends usable in this environment, fastest first.

    Returns:
        Tuple[str, ...]: The names, ordered by descending :attr:`BatchBackend.speed_rank` and then by name.

    """
    usable = sorted(
        (backend for backend in _BACKENDS.values() if backend.is_available()),
        key=lambda backend: (-backend.speed_rank, backend.name),
    )
    return tuple(backend.name for backend in usable)


register_backend(PythonBackend())
register_backend(NumpyBackend())


def _select_backend(requested: str | None, num_pairs: int) -> BatchBackend:
    """Chooses the backend to run a batch of a given size.

    An explicit request wins over the ``GRAPHTAGE_BATCH_BACKEND`` environment variable, which in turn wins over the
    automatic choice. Both named paths ignore :attr:`BatchBackend.min_pairs`, so that a benchmark or a test can
    force a backend onto a batch it would not otherwise be given.

    Args:
        requested: The name the caller asked for, or :const:`None` to consult the environment and then choose.
        num_pairs: The number of pairs in the batch.

    Returns:
        BatchBackend: The backend to run.

    Raises:
        ValueError: If a name was given that is not registered or not available here.

    """
    name = requested if requested is not None else os.environ.get(BACKEND_ENV_VAR)
    if name is not None:
        if name not in _BACKENDS:
            raise ValueError(
                f"Unknown batch distance backend {name!r}; the available backends are {available_backends()!r}"
            )
        backend = _BACKENDS[name]
        if not backend.is_available():
            raise ValueError(f"The {name!r} batch distance backend is not available in this environment")
        return backend
    best = _BACKENDS[FALLBACK_BACKEND]
    for backend in _BACKENDS.values():
        if backend.is_available() and num_pairs >= backend.min_pairs and backend.speed_rank > best.speed_rank:
            best = backend
    return best


def _kind_of(value: str | bytes) -> type:
    """Returns :class:`str` or :class:`bytes` for a value, rejecting anything else.

    Args:
        value: The value to classify.

    Returns:
        type: :class:`str` or :class:`bytes`.

    Raises:
        TypeError: If ``value`` is neither.

    """
    if isinstance(value, str):
        return str
    elif isinstance(value, bytes):
        return bytes
    raise TypeError(f"Batch distances need str or bytes, but got {type(value).__name__}: {value!r}")


def _pair_distances(pairs: Sequence[Pair], backend: str | None) -> np.ndarray:
    """Prices a list of pairs, answering the trivial ones directly and handing the rest to a backend.

    A pair whose sides are equal, or either of whose sides is empty, has an answer that needs no dynamic program.
    Settling those here keeps them out of the batch, which both saves the work and lets :func:`_scan` assume that
    every row it sees has at least one symbol.

    Args:
        pairs: The pairs to price.
        backend: The name of the backend to use, or :const:`None` to choose one.

    Returns:
        numpy.ndarray: One distance per pair, in the order the pairs were given.

    Raises:
        TypeError: If a side is neither :class:`str` nor :class:`bytes`, or if a pair mixes the two.

    """
    result = np.zeros(len(pairs), dtype=np.int64)
    remaining: list[int] = []
    for index, (left, right) in enumerate(pairs):
        if _kind_of(left) is not _kind_of(right):
            raise TypeError(f"Cannot compare a {type(left).__name__} to a {type(right).__name__}: {left!r}, {right!r}")
        if left == right:
            continue
        elif not left:
            result[index] = len(right)
        elif not right:
            result[index] = len(left)
        else:
            remaining.append(index)
    if remaining:
        selected = _select_backend(backend, len(remaining))
        result[remaining] = selected.distances([pairs[index] for index in remaining])
    return result


def _dedupe(values: Sequence[object]) -> tuple[list, np.ndarray]:
    """Collapses repeated values, returning the distinct ones and where each input found its own.

    Args:
        values: The values to deduplicate.

    Returns:
        Tuple[list, numpy.ndarray]: The distinct values in first-seen order, and an index array mapping each
        position of ``values`` onto that list.

    """
    positions: dict[object, int] = {}
    index = np.empty(len(values), dtype=np.intp)
    for position, value in enumerate(values):
        index[position] = positions.setdefault(value, len(positions))
    return list(positions), index


def all_pairs(
        from_strings: Sequence[str | bytes],
        to_strings: Sequence[str | bytes],
        *,
        backend: str | None = None,
) -> np.ndarray:
    """Computes the Levenshtein distance for every pair drawn from two collections.

    Repeated strings are computed once and shared, so a collection with few distinct values costs little more than
    the values themselves.

    Args:
        from_strings: The strings to measure from, which index the rows of the result.
        to_strings: The strings to measure to, which index the columns of the result.
        backend: The name of the backend to use. The default, :const:`None`, reads ``GRAPHTAGE_BATCH_BACKEND`` from
            the environment and otherwise picks the fastest backend that will take the batch.

    Returns:
        numpy.ndarray: A ``(len(from_strings), len(to_strings))`` array of :class:`numpy.int64` in which entry
        ``(i, j)`` equals ``levenshtein_distance(from_strings[i], to_strings[j])``.

    Raises:
        TypeError: If a string is neither :class:`str` nor :class:`bytes`, or if a pair mixes the two.
        ValueError: If ``backend`` names a backend that is not registered or not available here.

    """
    unique_from, from_index = _dedupe(from_strings)
    unique_to, to_index = _dedupe(to_strings)
    pairs = [(left, right) for left in unique_from for right in unique_to]
    grid = _pair_distances(pairs, backend).reshape(len(unique_from), len(unique_to))
    return grid[np.ix_(from_index, to_index)]


def all_flat(
        a: Sequence[str | bytes],
        b: Sequence[str | bytes],
        *,
        backend: str | None = None,
) -> np.ndarray:
    """Computes the Levenshtein distance of two collections position by position.

    Args:
        a: The strings to measure from.
        b: The strings to measure to, the same number of them as ``a`` has.
        backend: The name of the backend to use. The default, :const:`None`, reads ``GRAPHTAGE_BATCH_BACKEND`` from
            the environment and otherwise picks the fastest backend that will take the batch.

    Returns:
        numpy.ndarray: A one-dimensional array of :class:`numpy.int64` of length ``len(a)`` in which entry ``i``
        equals ``levenshtein_distance(a[i], b[i])``.

    Raises:
        TypeError: If a string is neither :class:`str` nor :class:`bytes`, or if a pair mixes the two.
        ValueError: If the two collections differ in length, or if ``backend`` names a backend that is not
            registered or not available here.

    """
    if len(a) != len(b):
        raise ValueError(f"all_flat needs two collections of the same length, but got {len(a)} and {len(b)}")
    unique_pairs, index = _dedupe(list(zip(a, b, strict=True)))
    return _pair_distances(unique_pairs, backend)[index]


class _Block:
    """A rectangle of pre-computed distances, indexed by the strings themselves.

    A :class:`dict` keyed on pairs of strings would hold the same answers, but it would hold each key too: for the
    cross product of two collections of four hundred strings that is tens of megabytes of tuples and hashes against
    the ``rows * columns * 4`` bytes of the matrix here.

    """

    __slots__ = ('_columns', '_matrix', '_rows')

    def __init__(self, rows: dict[str | bytes, int], columns: dict[str | bytes, int], matrix: np.ndarray):
        """Initializes the block.

        Args:
            rows: Each string that indexes a row, mapped to its row.
            columns: Each string that indexes a column, mapped to its column.
            matrix: A ``(len(rows), len(columns))`` array of :class:`numpy.int32`, whose unfilled cells hold
                :const:`_UNPRICED`.

        """
        self._rows = rows
        self._columns = columns
        self._matrix = matrix

    def lookup(self, a: str | bytes, b: str | bytes) -> int | None:
        """Returns the distance between two strings, or :const:`None` if this block does not hold it.

        Args:
            a: The string to measure from.
            b: The string to measure to.

        Returns:
            Optional[int]: The distance, or :const:`None`.

        """
        row = self._rows.get(a)
        if row is None:
            return None
        column = self._columns.get(b)
        if column is None:
            return None
        value = int(self._matrix[row, column])
        if value == _UNPRICED:
            return None
        return value


_blocks: tuple[_Block, ...] = ()
"""The pre-priced blocks, newest first. Rebound rather than mutated, so that a reader never sees a partial update."""


def _install(block: _Block) -> None:
    """Adds a block to the front of the stack, dropping the oldest once the stack is full.

    The stack is replaced rather than mutated. A reader holding the old tuple still sees a consistent stack, and a
    writer that loses a race only costs the block it was installing, which is a recomputation rather than a wrong
    answer.

    Args:
        block: The block to install.

    """
    global _blocks
    _blocks = (block, *_blocks[:BLOCK_STACK_LIMIT - 1])


def clear() -> None:
    """Discards every pre-priced block.

    :func:`cost` answers the same afterwards, because the blocks only hold what it would otherwise compute.

    """
    global _blocks
    _blocks = ()


def cost(a: str | bytes, b: str | bytes) -> int:
    """Returns the Levenshtein distance between two strings, reading a pre-priced block when one holds it.

    This is what :func:`graphtage.levenshtein.exact_string_distance` calls, and it returns exactly what
    :func:`graphtage.levenshtein.levenshtein_distance` returns for the same pair.

    A pair that no block holds is computed and *not* recorded. Recording it would make the cache grow with the
    number of distinct pairs a diff asks about, which is the whole cross product, and nothing would ever evict it.
    Blocks are installed deliberately, by a caller that knows it is about to ask for a whole rectangle of pairs.

    Args:
        a: The string to measure from.
        b: The string to measure to.

    Returns:
        int: The Levenshtein edit distance between the two strings.

    """
    if a == b:
        return 0
    elif not a:
        return len(b)
    elif not b:
        return len(a)
    for block in _blocks:
        value = block.lookup(a, b)
        if value is not None:
            return value
    return _uncached_cost(a, b)


def _strip_shared_affixes(a: str | bytes, b: str | bytes) -> tuple[str | bytes, str | bytes]:
    """Removes the shared prefix and suffix of two strings, which quadratically shrinks the matrix between them.

    Dropping them cannot change the distance: a character that aligns with itself is free, and no optimal alignment
    crosses such a position.

    Args:
        a: The string to measure from.
        b: The string to measure to.

    Returns:
        Tuple[str | bytes, str | bytes]: What is left of each string.

    """
    overlap = min(len(a), len(b))
    prefix = 0
    while prefix < overlap and a[prefix] == b[prefix]:
        prefix += 1
    suffix = 0
    while suffix < overlap - prefix and a[len(a) - suffix - 1] == b[len(b) - suffix - 1]:
        suffix += 1
    return a[prefix:len(a) - suffix], b[prefix:len(b) - suffix]


def _same_kind(a: str | bytes, b: str | bytes) -> bool:
    """Returns whether two strings are both :class:`str` or both :class:`bytes`.

    A pair that mixes the two still has a well defined distance, because a :class:`str` character never equals a
    :class:`bytes` element, but the backends reject it rather than pick one of the two readings of its symbols.

    Args:
        a: The string to measure from.
        b: The string to measure to.

    Returns:
        bool: Whether a backend will accept the pair.

    """
    if isinstance(a, str):
        return isinstance(b, str)
    return isinstance(a, bytes) and isinstance(b, bytes)


def _preferred_backend() -> str:
    """Returns the backend to run a batch whose pairs are big but whose pair count is small.

    :func:`_select_backend` skips a backend whose :attr:`BatchBackend.min_pairs` a batch does not meet, which is the
    right rule for a batch of short strings and the wrong one for a batch of one long pair. Naming the backend
    outright bypasses that rule while still honoring ``GRAPHTAGE_BATCH_BACKEND``.

    Returns:
        str: The name of the fastest available backend, or of the one the environment pins.

    """
    return os.environ.get(BACKEND_ENV_VAR) or available_backends()[0]


def _uncached_cost(a: str | bytes, b: str | bytes) -> int:
    """Prices one pair that no block holds.

    Args:
        a: The string to measure from, which differs from ``b`` and is not empty.
        b: The string to measure to, which is not empty.

    Returns:
        int: The Levenshtein edit distance between the two strings.

    """
    left, right = _strip_shared_affixes(a, b)
    if not left:
        return len(right)
    elif not right:
        return len(left)
    elif len(left) * len(right) < VECTORIZED_MIN_CELLS or not _same_kind(left, right):
        return levenshtein_distance(left, right)
    return int(_pair_distances([(left, right)], _preferred_backend())[0])


def _index(values: Sequence[str | bytes]) -> tuple[list[str | bytes], dict[str | bytes, int]]:
    """Numbers the distinct values of a collection in first-seen order.

    Args:
        values: The strings to number.

    Returns:
        Tuple[List[str | bytes], Dict[str | bytes, int]]: The distinct strings, and each one mapped to its position.

    """
    positions: dict[str | bytes, int] = {}
    for value in values:
        if value not in positions:
            positions[value] = len(positions)
    return list(positions), positions


def _product_matrix(rows: list[str | bytes], columns: list[str | bytes]) -> np.ndarray:
    """Prices a whole rectangle, one batch per kind of string.

    :class:`str` and :class:`bytes` rows are batched separately, and the cells where one meets the other are left
    unfilled so that :meth:`_Block.lookup` reports them as absent.

    Args:
        rows: The distinct strings indexing the rows.
        columns: The distinct strings indexing the columns.

    Returns:
        numpy.ndarray: A ``(len(rows), len(columns))`` array of :class:`numpy.int32`.

    """
    matrix = np.full((len(rows), len(columns)), _UNPRICED, dtype=np.int32)
    for kind in (str, bytes):
        row_positions = [position for position, value in enumerate(rows) if isinstance(value, kind)]
        column_positions = [position for position, value in enumerate(columns) if isinstance(value, kind)]
        if row_positions and column_positions:
            matrix[np.ix_(row_positions, column_positions)] = all_pairs(
                [rows[position] for position in row_positions],
                [columns[position] for position in column_positions],
            )
    return matrix


def preprice_product(
        from_strings: Sequence[str | bytes],
        to_strings: Sequence[str | bytes],
) -> None:
    """Prices every pair drawn from two collections and keeps the answers for :func:`cost`.

    Call this before constructing the edits that will ask for those pairs. The block outlives this call, because
    the edits that read it are constructed and tightened long afterwards.

    Nothing is installed for a rectangle larger than :const:`PREPRICE_MAX_CELLS`, and the pairs then go through the
    scalar path one at a time.

    Args:
        from_strings: The strings to measure from, which may repeat.
        to_strings: The strings to measure to, which may repeat.

    """
    rows, row_index = _index(from_strings)
    columns, column_index = _index(to_strings)
    if not rows or not columns or len(rows) * len(columns) > PREPRICE_MAX_CELLS:
        return
    _install(_Block(row_index, column_index, _product_matrix(rows, columns)))


def preprice(pairs: Sequence[Pair]) -> None:
    """Prices a list of pairs and keeps the answers for :func:`cost`.

    This is the form for a caller whose pairs are not a whole rectangle, such as
    :class:`graphtage.multiset.MultiSetEdit`, which draws one pair from two leaves but two from two key/value
    pairs. Pairs that mix :class:`str` with :class:`bytes` are left out and go through the scalar path.

    Nothing is installed for a rectangle larger than :const:`PREPRICE_MAX_CELLS`.

    Args:
        pairs: The pairs to price, which may repeat.

    """
    priceable = [(left, right) for left, right in pairs if _same_kind(left, right)]
    if not priceable:
        return
    rows, row_index = _index([left for left, _ in priceable])
    columns, column_index = _index([right for _, right in priceable])
    if len(rows) * len(columns) > PREPRICE_MAX_CELLS:
        return
    matrix = np.full((len(rows), len(columns)), _UNPRICED, dtype=np.int32)
    matrix[
        [row_index[left] for left, _ in priceable],
        [column_index[right] for _, right in priceable],
    ] = all_flat([left for left, _ in priceable], [right for _, right in priceable])
    _install(_Block(row_index, column_index, matrix))
