

import math
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from secagg.config import (
    DH_GENERATOR,
    DH_PRIME,
    K_THRESHOLD,
    MODEL_SHAPE,
    N_CLIENTS,
)


CLIENT_SEEDS = list(range(1, N_CLIENTS + 1))
TIMEOUT = max(60, N_CLIENTS * 6)  # scales: n=10→60s, n=50→300s, n=100→600s

ROOT = Path(__file__).parent
PYTHON = sys.executable




def _expected_weights(seeds: list[int]) -> np.ndarray:
    """Return the sum of gradients produced by *seeds*."""
    total = np.zeros(MODEL_SHAPE, dtype=np.float64)
    for s in seeds:
        rng = np.random.default_rng(s)
        total += rng.uniform(0.1, 1.0, MODEL_SHAPE).astype(np.float64)
    return total




def _stream_output(proc: subprocess.Popen, prefix: str) -> None:
    """Print each line from *proc.stdout* prefixed with *prefix*."""
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.decode(errors="replace").rstrip()
        print(f"[{prefix}] {line}", flush=True)


def _launch(script: str, args: list[str], prefix: str) -> subprocess.Popen:
    env = {"PYTHONUNBUFFERED": "1"}
    import os
    merged = {**os.environ, **env}
    proc = subprocess.Popen(
        [PYTHON, "-u", script, *args],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=merged,
    )
    t = threading.Thread(target=_stream_output, args=(proc, prefix), daemon=True)
    t.start()
    return proc




def main() -> None:
    print("=" * 60)
    print(f"Secure Aggregation experiment")
    print(f"  N_CLIENTS  = {N_CLIENTS}")
    print(f"  K_THRESHOLD= {K_THRESHOLD}  (ceil(2n/3)={math.ceil(2*N_CLIENTS/3)})")
    print(f"  MODEL_SHAPE= {MODEL_SHAPE}")
    print("=" * 60)

    # ---- expected aggregates -----------------------------------------------
    expected_k = _expected_weights(CLIENT_SEEDS[:K_THRESHOLD])
    expected_all = _expected_weights(CLIENT_SEEDS)
    print(f"\nExpected aggregate (first {K_THRESHOLD} clients):")
    print(f"  sum: {expected_k.sum():.4f}  |  max: {expected_k.max():.4f}")
    print(f"Expected aggregate (all {N_CLIENTS} clients):")
    print(f"  sum: {expected_all.sum():.4f}  |  max: {expected_all.max():.4f}\n")

    # ---- launch server -------------------------------------------------------
    server_proc = _launch("server_main.py", [], "SERVER")
    time.sleep(2)  # wait for server to bind

    # ---- launch clients ------------------------------------------------------
    client_procs = []
    for seed in CLIENT_SEEDS:
        p = _launch("client_main.py", [str(seed)], f"C{seed:02d}")
        client_procs.append(p)
        time.sleep(0.5)  # 0.5s stagger: gives server time to process each handshake

    # ---- wait for all clients ------------------------------------------------
    start = time.perf_counter()
    for i, p in enumerate(client_procs):
        seed = CLIENT_SEEDS[i]
        try:
            p.wait(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            print(f"[WARNING] Client (seed={seed}) timed out — killing.", flush=True)
            p.kill()

    elapsed = time.perf_counter() - start
    print(f"\n[RUN] All clients finished in {elapsed:.2f}s", flush=True)

    # ---- shut down server ----------------------------------------------------
    time.sleep(1)
    server_proc.terminate()
    try:
        server_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server_proc.kill()

    print("[RUN] Experiment complete.", flush=True)


if __name__ == "__main__":
    main()
