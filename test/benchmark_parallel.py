#!/usr/bin/env python3
"""Benchmark script for parallel execution performance.

Run with:
    # GIL enabled (standard Python)
    uv run python test/benchmark_parallel.py

    # GIL disabled (free-threaded Python)
    uv run --python cpython-3.14.0+freethreaded python test/benchmark_parallel.py
"""

import sys
import time
from statistics import mean, stdev

# Add project root to path
sys.path.insert(0, ".")

from graphtage.parallel import (
    Backend,
    compute_edge_matrix_sequential,
    compute_edge_matrix_threaded,
    configure,
    is_free_threading_available,
    make_distinct_sequential,
    make_distinct_threaded,
    tighten_diagonal_sequential,
    tighten_diagonal_threaded,
)
from test.test_bounds import RandomDecreasingRange


def get_gil_status() -> str:
    """Get GIL status string."""
    if hasattr(sys, "_is_gil_enabled"):
        return "disabled" if not sys._is_gil_enabled() else "enabled"
    return "enabled (pre-3.13)"


def benchmark(func, *args, iterations: int = 5, warmup: int = 1, **kwargs) -> dict:
    """Run a benchmark and return timing statistics."""
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean": mean(times),
        "stdev": stdev(times) if len(times) > 1 else 0,
        "min": min(times),
        "max": max(times),
        "iterations": iterations,
    }


def create_random_edges(m: int, n: int) -> list:
    """Create a matrix of RandomDecreasingRange objects."""
    return [[RandomDecreasingRange() for _ in range(n)] for _ in range(m)]


def expensive_get_edge(from_node, to_node, work_factor: int = 1000):
    """Simulate an expensive edge computation (like computing tree edits)."""
    # Simulate work by doing some computation
    result = 0
    for i in range(work_factor):
        result += (from_node * to_node + i) % 1000
    return RandomDecreasingRange(fixed_lb=0, fixed_ub=result + 1000, final_value=result % 500)


def benchmark_edge_matrix(sizes: list[int]) -> dict:
    """Benchmark edge matrix computation at various sizes (trivial work)."""
    results = {}

    for size in sizes:
        print(f"  Edge matrix {size}x{size} (trivial)...", end=" ", flush=True)

        from_nodes = list(range(size))
        to_nodes = list(range(size))
        edges = create_random_edges(size, size)

        def get_edge(f, t):
            return edges[f][t]

        # Sequential benchmark
        seq_result = benchmark(
            compute_edge_matrix_sequential,
            from_nodes,
            to_nodes,
            get_edge,
            iterations=3,
        )

        # Threaded benchmark
        thr_result = benchmark(
            compute_edge_matrix_threaded,
            from_nodes,
            to_nodes,
            get_edge,
            max_workers=4,
            iterations=3,
        )

        speedup = seq_result["mean"] / thr_result["mean"] if thr_result["mean"] > 0 else 0

        results[size] = {
            "sequential": seq_result,
            "threaded": thr_result,
            "speedup": speedup,
        }

        print(f"seq={seq_result['mean']*1000:.1f}ms, thr={thr_result['mean']*1000:.1f}ms, speedup={speedup:.2f}x")

    return results


def benchmark_edge_matrix_expensive(sizes: list[int], work_factor: int = 500) -> dict:
    """Benchmark edge matrix computation with expensive edge computation."""
    results = {}

    for size in sizes:
        print(f"  Edge matrix {size}x{size} (expensive, work={work_factor})...", end=" ", flush=True)

        from_nodes = list(range(size))
        to_nodes = list(range(size))

        def get_edge(f, t):
            return expensive_get_edge(f, t, work_factor)

        # Sequential benchmark
        seq_result = benchmark(
            compute_edge_matrix_sequential,
            from_nodes,
            to_nodes,
            get_edge,
            iterations=3,
        )

        # Threaded benchmark
        thr_result = benchmark(
            compute_edge_matrix_threaded,
            from_nodes,
            to_nodes,
            get_edge,
            max_workers=4,
            iterations=3,
        )

        speedup = seq_result["mean"] / thr_result["mean"] if thr_result["mean"] > 0 else 0

        results[size] = {
            "sequential": seq_result,
            "threaded": thr_result,
            "speedup": speedup,
        }

        print(f"seq={seq_result['mean']*1000:.1f}ms, thr={thr_result['mean']*1000:.1f}ms, speedup={speedup:.2f}x")

    return results


def benchmark_diagonal_tightening(sizes: list[int]) -> dict:
    """Benchmark diagonal tightening at various sizes."""
    results = {}

    for size in sizes:
        print(f"  Diagonal tightening (diagonal size {size})...", end=" ", flush=True)

        # Create a diagonal of edits
        edits = [[RandomDecreasingRange() for _ in range(size)] for _ in range(size)]
        diagonal_cells = [(i, size - 1 - i) for i in range(size)]

        def run_sequential():
            # Reset edits for fair comparison
            for row, col in diagonal_cells:
                edits[row][col] = RandomDecreasingRange()
            tighten_diagonal_sequential(diagonal_cells, lambda r, c: edits[r][c])

        def run_threaded():
            # Reset edits for fair comparison
            for row, col in diagonal_cells:
                edits[row][col] = RandomDecreasingRange()
            tighten_diagonal_threaded(diagonal_cells, lambda r, c: edits[r][c], max_workers=4)

        # Sequential benchmark
        seq_result = benchmark(run_sequential, iterations=3)

        # Threaded benchmark
        thr_result = benchmark(run_threaded, iterations=3)

        speedup = seq_result["mean"] / thr_result["mean"] if thr_result["mean"] > 0 else 0

        results[size] = {
            "sequential": seq_result,
            "threaded": thr_result,
            "speedup": speedup,
        }

        print(f"seq={seq_result['mean']*1000:.1f}ms, thr={thr_result['mean']*1000:.1f}ms, speedup={speedup:.2f}x")

    return results


def benchmark_make_distinct(sizes: list[int]) -> dict:
    """Benchmark make_distinct at various sizes."""
    import random

    results = {}

    for size in sizes:
        print(f"  make_distinct (n={size} items)...", end=" ", flush=True)

        def run_sequential():
            random.seed(42)
            items = [RandomDecreasingRange() for _ in range(size)]
            make_distinct_sequential(items)

        def run_threaded():
            random.seed(42)
            items = [RandomDecreasingRange() for _ in range(size)]
            make_distinct_threaded(items, max_workers=4)

        # Sequential benchmark
        seq_result = benchmark(run_sequential, iterations=3)

        # Threaded benchmark
        thr_result = benchmark(run_threaded, iterations=3)

        speedup = seq_result["mean"] / thr_result["mean"] if thr_result["mean"] > 0 else 0

        results[size] = {
            "sequential": seq_result,
            "threaded": thr_result,
            "speedup": speedup,
        }

        print(f"seq={seq_result['mean']*1000:.1f}ms, thr={thr_result['mean']*1000:.1f}ms, speedup={speedup:.2f}x")

    return results


def benchmark_weighted_bipartite_matcher(sizes: list[int]) -> dict:
    """Benchmark WeightedBipartiteMatcher end-to-end."""
    from graphtage.matching import WeightedBipartiteMatcher

    results = {}

    for size in sizes:
        print(f"  WeightedBipartiteMatcher {size}x{size}...", end=" ", flush=True)

        def run_matcher(use_parallel: bool):
            from_nodes = list(range(size))
            to_nodes = list(range(size))
            edges = create_random_edges(size, size)

            # Force diagonal to have zero cost for deterministic matching
            for i in range(min(size, size)):
                edges[i][i] = RandomDecreasingRange(fixed_lb=0, fixed_ub=100000, final_value=0)

            configure(enabled=use_parallel, threaded_min_size=1)

            matcher = WeightedBipartiteMatcher(
                from_nodes=from_nodes,
                to_nodes=to_nodes,
                get_edge=lambda n1, n2: edges[n1][n2],
            )

            # Trigger edge computation and matching
            _ = matcher.edges
            while matcher.tighten_bounds():
                pass

            return matcher.bounds()

        # Sequential benchmark
        configure(enabled=False)
        seq_result = benchmark(lambda: run_matcher(False), iterations=3)

        # Parallel benchmark
        configure(enabled=True, threaded_min_size=1)
        par_result = benchmark(lambda: run_matcher(True), iterations=3)

        speedup = seq_result["mean"] / par_result["mean"] if par_result["mean"] > 0 else 0

        results[size] = {
            "sequential": seq_result,
            "parallel": par_result,
            "speedup": speedup,
        }

        print(f"seq={seq_result['mean']*1000:.1f}ms, par={par_result['mean']*1000:.1f}ms, speedup={speedup:.2f}x")

    return results


def print_summary_table(
    edge_results: dict,
    edge_expensive_results: dict,
    diagonal_results: dict,
    make_distinct_results: dict,
    matcher_results: dict,
):
    """Print a summary table of all results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Python version: {sys.version.split()[0]}")
    print(f"GIL status: {get_gil_status()}")
    print(f"Free-threading available: {is_free_threading_available()}")
    print()

    print("Edge Matrix Computation (trivial work per edge):")
    print("-" * 60)
    print(f"{'Size':>10} {'Sequential':>12} {'Threaded':>12} {'Speedup':>10}")
    print("-" * 60)
    for size, data in edge_results.items():
        seq_ms = data["sequential"]["mean"] * 1000
        thr_ms = data["threaded"]["mean"] * 1000
        speedup = data["speedup"]
        print(f"{size:>10} {seq_ms:>10.1f}ms {thr_ms:>10.1f}ms {speedup:>9.2f}x")
    print()

    print("Edge Matrix Computation (expensive work per edge):")
    print("-" * 60)
    print(f"{'Size':>10} {'Sequential':>12} {'Threaded':>12} {'Speedup':>10}")
    print("-" * 60)
    for size, data in edge_expensive_results.items():
        seq_ms = data["sequential"]["mean"] * 1000
        thr_ms = data["threaded"]["mean"] * 1000
        speedup = data["speedup"]
        print(f"{size:>10} {seq_ms:>10.1f}ms {thr_ms:>10.1f}ms {speedup:>9.2f}x")
    print()

    print("Diagonal Tightening:")
    print("-" * 60)
    print(f"{'Size':>10} {'Sequential':>12} {'Threaded':>12} {'Speedup':>10}")
    print("-" * 60)
    for size, data in diagonal_results.items():
        seq_ms = data["sequential"]["mean"] * 1000
        thr_ms = data["threaded"]["mean"] * 1000
        speedup = data["speedup"]
        print(f"{size:>10} {seq_ms:>10.1f}ms {thr_ms:>10.1f}ms {speedup:>9.2f}x")
    print()

    print("make_distinct (overlapping interval tightening):")
    print("-" * 60)
    print(f"{'Items':>10} {'Sequential':>12} {'Threaded':>12} {'Speedup':>10}")
    print("-" * 60)
    for size, data in make_distinct_results.items():
        seq_ms = data["sequential"]["mean"] * 1000
        thr_ms = data["threaded"]["mean"] * 1000
        speedup = data["speedup"]
        print(f"{size:>10} {seq_ms:>10.1f}ms {thr_ms:>10.1f}ms {speedup:>9.2f}x")
    print()

    print("WeightedBipartiteMatcher (end-to-end):")
    print("-" * 60)
    print(f"{'Size':>10} {'Sequential':>12} {'Parallel':>12} {'Speedup':>10}")
    print("-" * 60)
    for size, data in matcher_results.items():
        seq_ms = data["sequential"]["mean"] * 1000
        par_ms = data["parallel"]["mean"] * 1000
        speedup = data["speedup"]
        print(f"{size:>10} {seq_ms:>10.1f}ms {par_ms:>10.1f}ms {speedup:>9.2f}x")
    print()


def main():
    print("=" * 80)
    print("GRAPHTAGE PARALLEL EXECUTION BENCHMARKS")
    print("=" * 80)
    print(f"Python version: {sys.version}")
    print(f"GIL status: {get_gil_status()}")
    print(f"Free-threading available: {is_free_threading_available()}")
    print()

    # Sizes to benchmark
    edge_sizes = [10, 25, 50]
    edge_expensive_sizes = [10, 20, 30, 40]
    diagonal_sizes = [10, 25, 50, 75]
    make_distinct_sizes = [20, 50, 100, 200]
    matcher_sizes = [10, 15, 20]

    print("Running edge matrix benchmarks (trivial work)...")
    edge_results = benchmark_edge_matrix(edge_sizes)

    print("\nRunning edge matrix benchmarks (expensive work)...")
    edge_expensive_results = benchmark_edge_matrix_expensive(edge_expensive_sizes, work_factor=500)

    print("\nRunning diagonal tightening benchmarks...")
    diagonal_results = benchmark_diagonal_tightening(diagonal_sizes)

    print("\nRunning make_distinct benchmarks...")
    make_distinct_results = benchmark_make_distinct(make_distinct_sizes)

    print("\nRunning WeightedBipartiteMatcher benchmarks...")
    matcher_results = benchmark_weighted_bipartite_matcher(matcher_sizes)

    # Print summary
    print_summary_table(
        edge_results,
        edge_expensive_results,
        diagonal_results,
        make_distinct_results,
        matcher_results,
    )

    # Reset config
    configure()


if __name__ == "__main__":
    main()
