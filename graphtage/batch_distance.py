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

Nothing in Graphtage calls this module yet.

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
