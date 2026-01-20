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
        threaded_min_size: Minimum problem size to use threading (default 100).
        enabled: Whether parallel execution is enabled at all.

    """

    preferred_backend: Optional[Backend] = None
    max_workers: Optional[int] = None
    webgpu_min_size: int = 10000
    threaded_min_size: int = 100
    enabled: bool = True


# Global configuration
_config: ParallelConfig = ParallelConfig()


def configure(
    preferred_backend: Optional[Backend] = None,
    max_workers: Optional[int] = None,
    webgpu_min_size: int = 10000,
    threaded_min_size: int = 100,
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
