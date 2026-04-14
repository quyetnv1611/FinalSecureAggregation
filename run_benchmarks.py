"""
run_benchmarks.py — Unified benchmark entry point
==================================================
Runs one or more benchmark suites for the SecureAggregation project.

Usage
-----
    # Run all benchmarks
    python run_benchmarks.py

    # Run specific benchmarks
    python run_benchmarks.py --only 4           # crypto params only (fastest)
    python run_benchmarks.py --only 2           # scalability
    python run_benchmarks.py --only 3           # original vs PQ comparison
    python run_benchmarks.py --only 1           # FL accuracy (slowest)
    python run_benchmarks.py --only 1 2 4       # multiple

Benchmark map
-------------
  1  bench_accuracy.py   — FL accuracy across 4 datasets (requires torch)
  2  bench_scalability.py — SecAgg timing vs n_clients × threshold
    3  bench_orig_vs_pq.py  — Original SecAgg vs PQ comparison
  4  bench_crypto_params.py — KEM + SIG primitive comparison

Results are written to the ``results/`` directory as CSV files.
"""

import argparse
import sys
import time
from pathlib import Path


def _import_and_run(module_path: str, fn: str = "run") -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location("_bench", module_path)
    mod  = importlib.util.load_from_spec(spec)
    spec.loader.exec_module(mod)
    getattr(mod, fn)()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SecureAggregation benchmark suites"
    )
    parser.add_argument(
        "--only",
        nargs="+",
        type=int,
        metavar="N",
        choices=[1, 2, 3, 4],
        help="Run only the specified benchmark(s): 1=accuracy, 2=scalability, 3=orig-vs-pq, 4=crypto",
    )
    args = parser.parse_args()
    requested = set(args.only) if args.only else {1, 2, 3, 4}

    root = Path(__file__).parent
    bench_dir = root / "experiments" / "benchmarks"

    schedule = [
        (4, "Crypto param comparison",  bench_dir / "bench_crypto_params.py"),
        (3, "Original vs PQ comparison", bench_dir / "bench_orig_vs_pq.py"),
        (2, "Scalability benchmark",    bench_dir / "bench_scalability.py"),
        (1, "FL accuracy benchmark",    bench_dir / "bench_accuracy.py"),
    ]

    for bid, label, path in schedule:
        if bid not in requested:
            continue
        print(f"\n{'='*60}")
        print(f"  Benchmark {bid}: {label}")
        print(f"{'='*60}")
        t0 = time.perf_counter()
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("_bench", str(path))
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.run()
        except Exception as exc:
            print(f"  [ERROR] Benchmark {bid} failed: {exc}", file=sys.stderr)
            import traceback; traceback.print_exc()
        elapsed = time.perf_counter() - t0
        print(f"\n  Benchmark {bid} completed in {elapsed:.1f}s")

    print(f"\n{'='*60}")
    print("  All requested benchmarks complete.")
    print(f"  Results written to: {root / 'results'}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
