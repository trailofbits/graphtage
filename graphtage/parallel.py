"""Parallel execution backends for Graphtage.

This module provides a backend abstraction layer for parallel execution of
computationally expensive operations in Graphtage. It supports multiple backends:

- **SEQUENTIAL**: Standard single-threaded execution (always available)
- **THREADED**: Multi-threaded execution using Python 3.14+ free-threading (PEP 703)
- **WEBGPU**: GPU-accelerated execution using WebGPU via wgpu-py (optional)

The module automatically detects available backends and selects the optimal one
based on problem size and system capabilities.

Example:
    >>> from graphtage.parallel import get_config, configure, Backend
    >>> configure(preferred_backend=Backend.THREADED, max_workers=4)
    >>> config = get_config()
    >>> print(config.preferred_backend)
    Backend.THREADED

"""

import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

T = TypeVar("T")
B = TypeVar("B")


class Backend(Enum):
    """Available parallel execution backends."""

    SEQUENTIAL = auto()
    THREADED = auto()
    WEBGPU = auto()


def _detect_free_threading() -> bool:
    """Detect if running on Python 3.14+ with GIL disabled.

    Returns:
        True if free-threading is available and GIL is disabled.

    """
    if sys.version_info < (3, 13):
        return False
    # Python 3.13+ has sys._is_gil_enabled()
    # In 3.14+, free-threading is no longer experimental
    try:
        return not sys._is_gil_enabled()  # type: ignore[attr-defined]
    except AttributeError:
        return False


def _detect_webgpu() -> bool:
    """Detect if wgpu is available and functional.

    Returns:
        True if WebGPU backend is available.

    """
    try:
        import wgpu

        # Verify we can get an adapter
        adapter = wgpu.gpu.request_adapter_sync()
        _ = wgpu  # Ensure wgpu is used
        return adapter is not None
    except (ImportError, Exception):
        return False


# Cache for detected backends
_AVAILABLE_BACKENDS: Optional[List[Backend]] = None
_FREE_THREADING_AVAILABLE: Optional[bool] = None
_WEBGPU_AVAILABLE: Optional[bool] = None


def _reset_backend_cache() -> None:
    """Reset the backend detection cache. Used for testing."""
    global _AVAILABLE_BACKENDS, _FREE_THREADING_AVAILABLE, _WEBGPU_AVAILABLE
    _AVAILABLE_BACKENDS = None
    _FREE_THREADING_AVAILABLE = None
    _WEBGPU_AVAILABLE = None


def is_free_threading_available() -> bool:
    """Check if free-threading is available.

    Returns:
        True if Python 3.14+ free-threading is available.

    """
    global _FREE_THREADING_AVAILABLE
    if _FREE_THREADING_AVAILABLE is None:
        _FREE_THREADING_AVAILABLE = _detect_free_threading()
    return _FREE_THREADING_AVAILABLE


def is_webgpu_available() -> bool:
    """Check if WebGPU backend is available.

    Returns:
        True if wgpu is installed and functional.

    """
    global _WEBGPU_AVAILABLE
    if _WEBGPU_AVAILABLE is None:
        _WEBGPU_AVAILABLE = _detect_webgpu()
    return _WEBGPU_AVAILABLE


def available_backends() -> List[Backend]:
    """Get list of available parallel backends.

    Returns:
        List of Backend enum values that are available on this system.

    """
    global _AVAILABLE_BACKENDS
    if _AVAILABLE_BACKENDS is None:
        _AVAILABLE_BACKENDS = [Backend.SEQUENTIAL]
        if is_free_threading_available():
            _AVAILABLE_BACKENDS.append(Backend.THREADED)
        if is_webgpu_available():
            _AVAILABLE_BACKENDS.append(Backend.WEBGPU)
    return _AVAILABLE_BACKENDS


@dataclass
class ParallelConfig:
    """Configuration for parallel execution.

    Attributes:
        preferred_backend: The preferred backend to use. If None, auto-selects.
        max_workers: Maximum number of worker threads. None means auto-detect.
        webgpu_min_size: Minimum problem size to use WebGPU (default 10000).
        threaded_min_size: Minimum problem size to use threading (default 500).
        enabled: Whether parallel execution is enabled at all.

    """

    preferred_backend: Optional[Backend] = None
    max_workers: Optional[int] = None
    webgpu_min_size: int = 10000
    threaded_min_size: int = 500
    enabled: bool = True


# Global configuration
_config: ParallelConfig = ParallelConfig()


def configure(
    preferred_backend: Optional[Backend] = None,
    max_workers: Optional[int] = None,
    webgpu_min_size: int = 10000,
    threaded_min_size: int = 500,
    enabled: bool = True,
) -> None:
    """Configure parallel execution globally.

    Args:
        preferred_backend: The preferred backend to use.
        max_workers: Maximum number of worker threads.
        webgpu_min_size: Minimum problem size for WebGPU.
        threaded_min_size: Minimum problem size for threading.
        enabled: Whether to enable parallel execution.

    """
    global _config
    _config = ParallelConfig(
        preferred_backend=preferred_backend,
        max_workers=max_workers,
        webgpu_min_size=webgpu_min_size,
        threaded_min_size=threaded_min_size,
        enabled=enabled,
    )


def get_config() -> ParallelConfig:
    """Get the current parallel configuration.

    Returns:
        The current ParallelConfig instance.

    """
    return _config


def get_backend(
    preferred: Optional[Backend] = None, problem_size: int = 0
) -> Backend:
    """Select the optimal backend based on availability and problem size.

    Args:
        preferred: Preferred backend to use if available.
        problem_size: Size of the problem (e.g., matrix dimensions).

    Returns:
        The selected Backend.

    """
    config = get_config()

    if not config.enabled:
        return Backend.SEQUENTIAL

    available = available_backends()

    # Use explicit preference if specified and available
    if preferred is not None and preferred in available:
        # Still respect minimum size thresholds
        if preferred == Backend.WEBGPU and problem_size < config.webgpu_min_size:
            pass  # Fall through to auto-selection
        elif preferred == Backend.THREADED and problem_size < config.threaded_min_size:
            return Backend.SEQUENTIAL
        else:
            return preferred

    # Use config preference if specified and available
    if config.preferred_backend is not None and config.preferred_backend in available:
        if (
            config.preferred_backend == Backend.WEBGPU
            and problem_size >= config.webgpu_min_size
        ):
            return Backend.WEBGPU
        elif (
            config.preferred_backend == Backend.THREADED
            and problem_size >= config.threaded_min_size
        ):
            return Backend.THREADED

    # Auto-select based on problem size thresholds
    if Backend.WEBGPU in available and problem_size >= config.webgpu_min_size:
        return Backend.WEBGPU
    if Backend.THREADED in available and problem_size >= config.threaded_min_size:
        return Backend.THREADED

    return Backend.SEQUENTIAL


def compute_edge_matrix_sequential(
    from_nodes: Sequence[T],
    to_nodes: Sequence[T],
    get_edge: Callable[[T, T], Optional[B]],
) -> List[List[Optional[B]]]:
    """Compute edge matrix sequentially (baseline implementation).

    Args:
        from_nodes: Source nodes.
        to_nodes: Target nodes.
        get_edge: Function to compute edge between two nodes.

    Returns:
        2D list of edges (or None for missing edges).

    """
    return [
        [get_edge(from_node, to_node) for to_node in to_nodes]
        for from_node in from_nodes
    ]


def compute_edge_matrix_threaded(
    from_nodes: Sequence[T],
    to_nodes: Sequence[T],
    get_edge: Callable[[T, T], Optional[B]],
    max_workers: Optional[int] = None,
) -> List[List[Optional[B]]]:
    """Compute edge matrix using thread pool.

    Args:
        from_nodes: Source nodes.
        to_nodes: Target nodes.
        get_edge: Function to compute edge between two nodes.
        max_workers: Maximum number of worker threads.

    Returns:
        2D list of edges (or None for missing edges).

    """
    m, n = len(from_nodes), len(to_nodes)

    if m == 0 or n == 0:
        return [[] for _ in range(m)]

    result: List[List[Optional[B]]] = [[None] * n for _ in range(m)]

    def compute_cell(i: int, j: int) -> Tuple[int, int, Optional[B]]:
        return (i, j, get_edge(from_nodes[i], to_nodes[j]))

    # Generate all (i, j) pairs
    indices = [(i, j) for i in range(m) for j in range(n)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compute_cell, i, j) for i, j in indices]
        for future in futures:
            i, j, edge = future.result()
            result[i][j] = edge

    return result


def compute_edge_matrix(
    from_nodes: Sequence[T],
    to_nodes: Sequence[T],
    get_edge: Callable[[T, T], Optional[B]],
    preferred_backend: Optional[Backend] = None,
) -> List[List[Optional[B]]]:
    """Compute edge matrix using the optimal backend.

    Automatically selects the best available backend based on problem size
    and system capabilities.

    Args:
        from_nodes: Source nodes.
        to_nodes: Target nodes.
        get_edge: Function to compute edge between two nodes.
        preferred_backend: Override the backend selection.

    Returns:
        2D list of edges (or None for missing edges).

    """
    config = get_config()
    problem_size = len(from_nodes) * len(to_nodes)
    backend = get_backend(preferred_backend, problem_size)

    if backend == Backend.WEBGPU:
        # WebGPU implementation would go here
        # For now, fall back to threaded
        backend = Backend.THREADED if Backend.THREADED in available_backends() else Backend.SEQUENTIAL

    if backend == Backend.THREADED:
        return compute_edge_matrix_threaded(
            from_nodes, to_nodes, get_edge, max_workers=config.max_workers
        )

    return compute_edge_matrix_sequential(from_nodes, to_nodes, get_edge)


def tighten_diagonal_sequential(
    cells: Sequence[Tuple[int, int]],
    get_edit: Callable[[int, int], "Bounded"],  # noqa: F821
) -> bool:
    """Tighten all cells in a diagonal sequentially.

    Args:
        cells: List of (row, col) tuples for cells in the diagonal.
        get_edit: Function to get the edit at a given position.

    Returns:
        True if any cell was tightened.

    """
    any_tightened = False
    for row, col in cells:
        edit = get_edit(row, col)
        while edit.tighten_bounds():
            any_tightened = True
    return any_tightened


def tighten_diagonal_threaded(
    cells: Sequence[Tuple[int, int]],
    get_edit: Callable[[int, int], "Bounded"],  # noqa: F821
    max_workers: Optional[int] = None,
) -> bool:
    """Tighten all cells in a diagonal using thread pool.

    Args:
        cells: List of (row, col) tuples for cells in the diagonal.
        get_edit: Function to get the edit at a given position.
        max_workers: Maximum number of worker threads.

    Returns:
        True if any cell was tightened.

    """
    if not cells:
        return False

    def tighten_cell(row: int, col: int) -> bool:
        edit = get_edit(row, col)
        tightened = False
        while edit.tighten_bounds():
            tightened = True
        return tightened

    any_tightened = False
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(tighten_cell, row, col) for row, col in cells]
        for future in futures:
            if future.result():
                any_tightened = True

    return any_tightened


def tighten_diagonal(
    cells: Sequence[Tuple[int, int]],
    get_edit: Callable[[int, int], "Bounded"],  # noqa: F821
    preferred_backend: Optional[Backend] = None,
) -> bool:
    """Tighten all cells in a diagonal using the optimal backend.

    Args:
        cells: List of (row, col) tuples for cells in the diagonal.
        get_edit: Function to get the edit at a given position.
        preferred_backend: Override the backend selection.

    Returns:
        True if any cell was tightened.

    """
    config = get_config()
    problem_size = len(cells)
    backend = get_backend(preferred_backend, problem_size)

    if backend == Backend.THREADED:
        return tighten_diagonal_threaded(cells, get_edit, max_workers=config.max_workers)

    return tighten_diagonal_sequential(cells, get_edit)


def make_distinct_sequential(bounded_items: List["Bounded"]) -> None:  # noqa: F821
    """Make all bounded items distinct sequentially (baseline implementation).

    This is the original algorithm from bounds.py.

    Args:
        bounded_items: List of Bounded objects to make distinct.

    """
    from intervaltree import Interval, IntervalTree

    tree: IntervalTree = IntervalTree()
    for b in bounded_items:
        if not b.bounds().finite:
            b.tighten_bounds()
            if not b.bounds().finite:
                raise ValueError(f"Could not tighten {b!r} to a finite bound")
        tree.add(Interval(b.bounds().lower_bound, b.bounds().upper_bound + 1, b))

    while len(tree) > 1:
        # Find the biggest interval in the tree
        biggest: Optional[Interval] = None
        for m in tree:
            m_size = m.end - m.begin
            if biggest is None or m_size > biggest.end - biggest.begin:
                biggest = m
        assert biggest is not None
        if biggest.data.bounds().definitive():
            break
        tree.remove(biggest)
        matching = tree[biggest.begin:biggest.end]
        if len(matching) < 1:
            continue
        # Find the biggest other interval that intersects with biggest
        second_biggest: Optional[Interval] = None
        for m in matching:
            m_size = m.end - m.begin
            if second_biggest is None or m_size > second_biggest.end - second_biggest.begin:
                second_biggest = m
        assert second_biggest is not None
        tree.remove(second_biggest)
        # Shrink the two biggest intervals until they are distinct
        while True:
            biggest_bound = biggest.data.bounds()
            second_biggest_bound = second_biggest.data.bounds()
            if (
                (biggest_bound.definitive() and second_biggest_bound.definitive())
                or biggest_bound.upper_bound < second_biggest_bound.lower_bound
                or second_biggest_bound.upper_bound < biggest_bound.lower_bound
            ):
                break
            biggest.data.tighten_bounds()
            second_biggest.data.tighten_bounds()
        new_interval = Interval(
            begin=biggest.data.bounds().lower_bound,
            end=biggest.data.bounds().upper_bound + 1,
            data=biggest.data,
        )
        if tree.overlaps(new_interval.begin, new_interval.end):
            tree.add(new_interval)
        new_interval = Interval(
            begin=second_biggest.data.bounds().lower_bound,
            end=second_biggest.data.bounds().upper_bound + 1,
            data=second_biggest.data,
        )
        if tree.overlaps(new_interval.begin, new_interval.end):
            tree.add(new_interval)


def _find_independent_pairs(
    tree: "IntervalTree",  # noqa: F821
) -> List[Tuple["Bounded", "Bounded"]]:  # noqa: F821
    """Find pairs of overlapping intervals that can be tightened independently.

    Two pairs are independent if they share no common bounded items.
    This allows them to be processed in parallel.

    Args:
        tree: IntervalTree containing bounded items.

    Returns:
        List of (bounded1, bounded2) pairs that can be processed in parallel.

    """
    pairs = []
    used = set()

    # Sort intervals by size (largest first) for greedy matching
    intervals = sorted(tree, key=lambda m: m.end - m.begin, reverse=True)

    for interval in intervals:
        if id(interval.data) in used:
            continue
        if interval.data.bounds().definitive():
            continue

        # Find overlapping intervals
        matching = tree[interval.begin:interval.end]
        for other in matching:
            if other is interval:
                continue
            if id(other.data) in used:
                continue

            # Found an independent pair
            pairs.append((interval.data, other.data))
            used.add(id(interval.data))
            used.add(id(other.data))
            break

    return pairs


def make_distinct_threaded(
    bounded_items: List["Bounded"],  # noqa: F821
    max_workers: Optional[int] = None,
) -> None:
    """Make all bounded items distinct using parallel processing.

    This parallelizes the make_distinct algorithm by processing multiple
    independent pairs of overlapping intervals concurrently.

    Args:
        bounded_items: List of Bounded objects to make distinct.
        max_workers: Maximum number of worker threads.

    """
    from intervaltree import Interval, IntervalTree

    def tighten_pair(b1: "Bounded", b2: "Bounded") -> None:  # noqa: F821
        """Tighten a pair of bounded items until they are distinct."""
        while True:
            b1_bound = b1.bounds()
            b2_bound = b2.bounds()
            if (
                (b1_bound.definitive() and b2_bound.definitive())
                or b1_bound.upper_bound < b2_bound.lower_bound
                or b2_bound.upper_bound < b1_bound.lower_bound
            ):
                break
            b1.tighten_bounds()
            b2.tighten_bounds()

    def rebuild_tree(items: List["Bounded"]) -> IntervalTree:  # noqa: F821
        """Rebuild the interval tree with current bounds."""
        tree = IntervalTree()
        for b in items:
            bounds = b.bounds()
            if not bounds.definitive():
                tree.add(Interval(bounds.lower_bound, bounds.upper_bound + 1, b))
        return tree

    # Initial tightening to ensure finite bounds
    for b in bounded_items:
        if not b.bounds().finite:
            b.tighten_bounds()
            if not b.bounds().finite:
                raise ValueError(f"Could not tighten {b!r} to a finite bound")

    # Build initial tree
    tree = rebuild_tree(bounded_items)

    while len(tree) > 1:
        # Find independent pairs that can be processed in parallel
        pairs = _find_independent_pairs(tree)

        if not pairs:
            # No pairs found - all remaining intervals are either distinct
            # or definitive
            break

        # Process pairs in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(tighten_pair, b1, b2) for b1, b2 in pairs]
            for future in futures:
                future.result()

        # Rebuild tree with updated bounds
        tree = rebuild_tree(bounded_items)


def make_distinct(
    bounded_items: List["Bounded"],  # noqa: F821
    preferred_backend: Optional[Backend] = None,
) -> None:
    """Make all bounded items distinct using the optimal backend.

    Ensures that all bounded items are tightened until they are finite and
    either definitive or non-overlapping with any of the other items.

    Note: Profiling shows that the threaded implementation is slower than
    sequential due to thread overhead exceeding the benefit of parallelism.
    The work per pair (tightening bounds) is too fast to benefit from
    threading. Therefore, this always uses the sequential implementation.

    Args:
        bounded_items: List of Bounded objects to make distinct.
        preferred_backend: Ignored (sequential is always faster).

    """
    # Always use sequential - profiling shows threaded is slower due to
    # thread overhead exceeding the benefit of parallelism for this operation
    make_distinct_sequential(bounded_items)
