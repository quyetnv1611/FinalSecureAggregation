"""
experiments/benchmarks/bench_orig_vs_pq.py
==========================================
Benchmark 5 - Original SecAgg vs PQ SecAgg comparison.

This benchmark keeps the existing benchmark files untouched and adds a new
comparison-focused suite for the two protocol variants that correspond to the
figures shared by the user:

* Original protocol: DH + classic ECDSA
* PQ protocol:      ML-KEM-768 + ML-DSA-65

The benchmark sweeps three views of the protocol:

* participant count at several dropout rates
* dropout rate at a fixed participant count
* vector size at a fixed participant count and no dropout

It produces both CSV summaries and PDF figures under ``results/`` and
``figures/``.

Outputs
-------
``results/bench_orig_vs_pq_timing.csv``
    Per-phase timing rows for both protocol variants.

``results/bench_orig_vs_pq_summary.csv``
    Scenario-level timing and estimated communication summary.

``figures/bench_orig_vs_pq_runtime_clients.pdf``
``figures/bench_orig_vs_pq_comm_clients.pdf``
``figures/bench_orig_vs_pq_runtime_dropout.pdf``
``figures/bench_orig_vs_pq_comm_dropout.pdf``
``figures/bench_orig_vs_pq_runtime_vector.pdf``
``figures/bench_orig_vs_pq_comm_vector.pdf``
"""

from __future__ import annotations

import csv
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from experiments.fl_simulator import run_secagg_timing
from secagg.config import DH_PRIME
from secagg.crypto_mlkem import SecAggregatorMLKEM
from secagg.sig_pq import make_signer


RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

OUT_TIMING = RESULTS_DIR / "bench_orig_vs_pq_timing.csv"
OUT_SUMMARY = RESULTS_DIR / "bench_orig_vs_pq_summary.csv"

ORIGINAL = {
    "algorithm": "original",
    "label": "Original (DH + ECDSA)",
    "kem_backend": "DH",
    "sig_backend": "classic",
}

PQ = {
    "algorithm": "pq",
    "label": "PQ (ML-KEM-768 + ML-DSA-65)",
    "kem_backend": "ML-KEM-768",
    "sig_backend": "ML-DSA-65",
}

ALGORITHMS = [ORIGINAL, PQ]

CLIENT_COUNTS = [100, 200, 300, 400, 500]
DROPOUT_RATES = [0.0, 0.1, 0.2, 0.3]
# Extended vector sizes to match Figure 6 (10k-50k range with more data points)
VECTOR_SIZES = [10_000, 15_000, 20_000, 25_000, 30_000, 35_000, 40_000, 45_000, 50_000]

BITS_PER_ENTRY = 62
N_REPEAT = 1
REFERENCE_CLIENTS = 100
REFERENCE_DROPOUT = 0.0

# Checkpoint settings for resumable runs
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
OUT_CHECKPOINT = CHECKPOINT_DIR / "bench_orig_vs_pq_progress.json"


def _vector_shape(vector_size: int) -> tuple[int, ...]:
    return (vector_size,)


@lru_cache(maxsize=None)
def _classic_signature_sizes() -> tuple[int, int, int]:
    signer = make_signer("classic")
    pk, sk = signer.keygen()
    sig = signer.sign(sk, b"bench")
    return len(pk), len(sk), len(sig)


@lru_cache(maxsize=None)
def _pq_signature_sizes(sig_backend: str) -> tuple[int, int, int]:
    signer = make_signer(sig_backend)
    pk, sk = signer.keygen()
    sig = signer.sign(sk, b"bench")
    return len(pk), len(sk), len(sig)


@lru_cache(maxsize=None)
def _pq_kem_sizes(kem_backend: str) -> tuple[int, int]:
    client = SecAggregatorMLKEM(shape=_vector_shape(1), security_level=kem_backend)
    peer = SecAggregatorMLKEM(shape=_vector_shape(1), security_level=kem_backend)
    ciphertexts = client.generate_ciphertexts({"A": client.public_key, "B": peer.public_key}, "A")
    ct_bytes = len(ciphertexts["B"])
    return client.encapsulation_key_size, ct_bytes


def _dh_public_key_size_bytes() -> int:
    return (DH_PRIME.bit_length() + 7) // 8


def _run_timing(
    *,
    algorithm: dict[str, str],
    n_clients: int,
    dropout_rate: float,
    vector_size: int,
) -> dict[str, float | int | str]:
    timer = run_secagg_timing(
        n_clients=n_clients,
        grad_shape=_vector_shape(vector_size),
        kem_backend=algorithm["kem_backend"],
        sig_backend=algorithm["sig_backend"],
        n_repeat=N_REPEAT,
        dropout_rate=dropout_rate,
    )
    return {
        "advertise_keys": timer.advertise_keys,
        "share_keys": timer.share_keys,
        "verify_sigs": timer.verify_sigs,
        "masked_input": timer.masked_input,
        "unmasking": timer.unmasking,
        "total": timer.total,
    }


def _estimate_comm_bytes(
    *,
    algorithm: dict[str, str],
    n_clients: int,
    dropout_rate: float,
    vector_size: int,
) -> dict[str, float]:
    survivors = max(1, min(n_clients, int(round(n_clients * (1.0 - dropout_rate)))))
    survivor_ratio = survivors / n_clients
    masked_input_bytes = vector_size * (BITS_PER_ENTRY / 8.0)

    if algorithm["algorithm"] == "original":
        pk_bytes = _dh_public_key_size_bytes()
        sig_pk_bytes, _, sig_bytes = _classic_signature_sizes()
        advertise_bytes = pk_bytes + sig_pk_bytes + sig_bytes
        share_bytes = 0.0
    else:
        kem_pk_bytes, ct_bytes = _pq_kem_sizes(algorithm["kem_backend"])
        sig_pk_bytes, _, sig_bytes = _pq_signature_sizes(algorithm["sig_backend"])
        advertise_bytes = kem_pk_bytes + sig_pk_bytes + sig_bytes
        share_bytes = ((survivors - 1) / 2.0) * ct_bytes

    advertise_avg = survivor_ratio * advertise_bytes
    share_avg = survivor_ratio * share_bytes
    masked_avg = survivor_ratio * masked_input_bytes
    total_avg = advertise_avg + share_avg + masked_avg

    return {
        "advertise_mb": advertise_avg / 1_000_000.0,
        "share_mb": share_avg / 1_000_000.0,
        "masked_mb": masked_avg / 1_000_000.0,
        "total_mb": total_avg / 1_000_000.0,
    }


def _build_scenarios() -> list[dict[str, object]]:
    scenarios: list[dict[str, object]] = []

    for dropout_rate in DROPOUT_RATES:
        for n_clients in CLIENT_COUNTS:
            scenarios.append({
                "scenario": "clients",
                "n_clients": n_clients,
                "dropout_rate": dropout_rate,
                "vector_size": VECTOR_SIZES[-1],
            })

    for dropout_rate in DROPOUT_RATES:
        scenarios.append({
            "scenario": "dropout",
            "n_clients": REFERENCE_CLIENTS,
            "dropout_rate": dropout_rate,
            "vector_size": VECTOR_SIZES[-1],
        })

    for vector_size in VECTOR_SIZES:
        scenarios.append({
            "scenario": "vector",
            "n_clients": REFERENCE_CLIENTS,
            "dropout_rate": REFERENCE_DROPOUT,
            "vector_size": vector_size,
        })

    return scenarios


def _load_checkpoint() -> tuple[set[tuple], list[dict], list[dict]]:
    """Load checkpoint from JSON. Returns (completed_set, timing_rows, summary_rows)."""
    if not OUT_CHECKPOINT.exists():
        return set(), [], []
    
    try:
        with OUT_CHECKPOINT.open("r", encoding="utf-8") as f:
            data = json.load(f)
        completed = set(tuple(item) for item in data.get("completed", []))
        timing_rows = data.get("timing_rows", [])
        summary_rows = data.get("summary_rows", [])
        print(f"[orig_vs_pq] Loaded checkpoint: {len(completed)} scenarios completed")
        return completed, timing_rows, summary_rows
    except Exception as e:
        print(f"[orig_vs_pq] Warning: Failed to load checkpoint: {e}")
        return set(), [], []


def _save_checkpoint(completed: set[tuple], timing_rows: list[dict], summary_rows: list[dict]) -> None:
    """Save checkpoint to JSON."""
    try:
        data = {
            "completed": [list(item) for item in completed],
            "timing_rows": timing_rows,
            "summary_rows": summary_rows,
            "timestamp": time.time(),
        }
        with OUT_CHECKPOINT.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[orig_vs_pq] Warning: Failed to save checkpoint: {e}")


def _make_scenario_key(scenario_name: str, n_clients: int, dropout_rate: float, vector_size: int, algorithm: str) -> tuple:
    """Create a hashable key for scenario tracking."""
    return (scenario_name, n_clients, dropout_rate, vector_size, algorithm)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_clients(summary: pd.DataFrame, metric: str, title: str, ylabel: str, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    for axis, dropout_rate in zip(axes, DROPOUT_RATES):
        subset = summary[(summary["scenario"] == "clients") & (summary["dropout_rate"] == dropout_rate)]
        for algorithm in ALGORITHMS:
            alg_subset = subset[subset["algorithm"] == algorithm["algorithm"]].sort_values("n_clients")
            axis.plot(
                alg_subset["n_clients"],
                alg_subset[metric],
                marker="o",
                linewidth=2,
                label=algorithm["label"],
            )
        axis.set_title(f"Dropout Rate: {int(dropout_rate * 100)}%")
        axis.grid(True, alpha=0.3)
        axis.set_xlabel("#Participants")
        axis.set_ylabel(ylabel)
    axes[0].legend(frameon=True)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_dropout(summary: pd.DataFrame, metric: str, title: str, ylabel: str, out_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    subset = summary[summary["scenario"] == "dropout"].sort_values("dropout_rate")
    for algorithm in ALGORITHMS:
        alg_subset = subset[subset["algorithm"] == algorithm["algorithm"]]
        axis.plot(
            alg_subset["dropout_rate"],
            alg_subset[metric],
            marker="o",
            linewidth=2,
            label=algorithm["label"],
        )
    axis.set_xlabel("Dropout Rate")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_vector(summary: pd.DataFrame, metric: str, title: str, ylabel: str, out_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    subset = summary[summary["scenario"] == "vector"].sort_values("vector_size")
    for algorithm in ALGORITHMS:
        alg_subset = subset[subset["algorithm"] == algorithm["algorithm"]]
        axis.plot(
            alg_subset["vector_size"],
            alg_subset[metric],
            marker="o",
            linewidth=2,
            label=algorithm["label"],
        )
    axis.set_xlabel("Data vector size")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    # Load checkpoint to support resumable runs
    completed, timing_rows, summary_rows = _load_checkpoint()
    
    scenarios = _build_scenarios()
    total_scenarios = len(scenarios) * len(ALGORITHMS)
    completed_count = len(completed)
    
    print(f"[orig_vs_pq] Found {completed_count} / {total_scenarios} scenarios already completed")
    
    for scenario_idx, scenario in enumerate(scenarios):
        scenario_name = str(scenario["scenario"])
        n_clients = int(scenario["n_clients"])
        dropout_rate = float(scenario["dropout_rate"])
        vector_size = int(scenario["vector_size"])

        for algorithm in ALGORITHMS:
            scenario_key = _make_scenario_key(scenario_name, n_clients, dropout_rate, vector_size, algorithm["algorithm"])
            
            # Skip if already completed
            if scenario_key in completed:
                print(f"[orig_vs_pq] Skipping (already done): scenario={scenario_name} n={n_clients} "
                      f"dropout={dropout_rate:.1f} vector={vector_size} algo={algorithm['algorithm']}")
                continue
            
            # Run timing benchmark
            print(f"[orig_vs_pq] scenario={scenario_name} n={n_clients} "
                  f"dropout={dropout_rate:.1f} vector={vector_size} algo={algorithm['algorithm']}")
            
            try:
                timings = _run_timing(
                    algorithm=algorithm,
                    n_clients=n_clients,
                    dropout_rate=dropout_rate,
                    vector_size=vector_size,
                )
                comm = _estimate_comm_bytes(
                    algorithm=algorithm,
                    n_clients=n_clients,
                    dropout_rate=dropout_rate,
                    vector_size=vector_size,
                )

                total_time = float(timings["total"])
                
                # Add timing rows for all phases
                timing_rows.append({
                    "scenario": scenario_name,
                    "algorithm": algorithm["algorithm"],
                    "backend_label": algorithm["label"],
                    "kem_backend": algorithm["kem_backend"],
                    "sig_backend": algorithm["sig_backend"],
                    "n_clients": n_clients,
                    "dropout_rate": dropout_rate,
                    "vector_size": vector_size,
                    "phase": "advertise_keys",
                    "time_sec": round(float(timings["advertise_keys"]), 6),
                })
                timing_rows.append({
                    "scenario": scenario_name,
                    "algorithm": algorithm["algorithm"],
                    "backend_label": algorithm["label"],
                    "kem_backend": algorithm["kem_backend"],
                    "sig_backend": algorithm["sig_backend"],
                    "n_clients": n_clients,
                    "dropout_rate": dropout_rate,
                    "vector_size": vector_size,
                    "phase": "share_keys",
                    "time_sec": round(float(timings["share_keys"]), 6),
                })
                timing_rows.append({
                    "scenario": scenario_name,
                    "algorithm": algorithm["algorithm"],
                    "backend_label": algorithm["label"],
                    "kem_backend": algorithm["kem_backend"],
                    "sig_backend": algorithm["sig_backend"],
                    "n_clients": n_clients,
                    "dropout_rate": dropout_rate,
                    "vector_size": vector_size,
                    "phase": "verify_sigs",
                    "time_sec": round(float(timings["verify_sigs"]), 6),
                })
                timing_rows.append({
                    "scenario": scenario_name,
                    "algorithm": algorithm["algorithm"],
                    "backend_label": algorithm["label"],
                    "kem_backend": algorithm["kem_backend"],
                    "sig_backend": algorithm["sig_backend"],
                    "n_clients": n_clients,
                    "dropout_rate": dropout_rate,
                    "vector_size": vector_size,
                    "phase": "masked_input",
                    "time_sec": round(float(timings["masked_input"]), 6),
                })
                timing_rows.append({
                    "scenario": scenario_name,
                    "algorithm": algorithm["algorithm"],
                    "backend_label": algorithm["label"],
                    "kem_backend": algorithm["kem_backend"],
                    "sig_backend": algorithm["sig_backend"],
                    "n_clients": n_clients,
                    "dropout_rate": dropout_rate,
                    "vector_size": vector_size,
                    "phase": "unmasking",
                    "time_sec": round(float(timings["unmasking"]), 6),
                })
                timing_rows.append({
                    "scenario": scenario_name,
                    "algorithm": algorithm["algorithm"],
                    "backend_label": algorithm["label"],
                    "kem_backend": algorithm["kem_backend"],
                    "sig_backend": algorithm["sig_backend"],
                    "n_clients": n_clients,
                    "dropout_rate": dropout_rate,
                    "vector_size": vector_size,
                    "phase": "total",
                    "time_sec": round(total_time, 6),
                })

                summary_rows.append({
                    "scenario": scenario_name,
                    "algorithm": algorithm["algorithm"],
                    "backend_label": algorithm["label"],
                    "kem_backend": algorithm["kem_backend"],
                    "sig_backend": algorithm["sig_backend"],
                    "n_clients": n_clients,
                    "dropout_rate": dropout_rate,
                    "vector_size": vector_size,
                    "advertise_keys_sec": round(float(timings["advertise_keys"]), 6),
                    "share_keys_sec": round(float(timings["share_keys"]), 6),
                    "verify_sigs_sec": round(float(timings["verify_sigs"]), 6),
                    "masked_input_sec": round(float(timings["masked_input"]), 6),
                    "unmasking_sec": round(float(timings["unmasking"]), 6),
                    "total_time_sec": round(total_time, 6),
                    "advertise_mb": round(comm["advertise_mb"], 6),
                    "share_mb": round(comm["share_mb"], 6),
                    "masked_mb": round(comm["masked_mb"], 6),
                    "total_comm_mb": round(comm["total_mb"], 6),
                })
                
                # Mark as completed and save checkpoint
                completed.add(scenario_key)
                _save_checkpoint(completed, timing_rows, summary_rows)
                
            except Exception as e:
                print(f"[orig_vs_pq] ERROR in scenario: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                # Save progress before exiting
                _save_checkpoint(completed, timing_rows, summary_rows)
                print(f"[orig_vs_pq] Checkpoint saved before exit. Restart to resume.", file=sys.stderr)
                raise

    # All scenarios completed: write final CSV and generate figures
    timing_fieldnames = [
        "scenario",
        "algorithm",
        "backend_label",
        "kem_backend",
        "sig_backend",
        "n_clients",
        "dropout_rate",
        "vector_size",
        "phase",
        "time_sec",
    ]
    summary_fieldnames = [
        "scenario",
        "algorithm",
        "backend_label",
        "kem_backend",
        "sig_backend",
        "n_clients",
        "dropout_rate",
        "vector_size",
        "advertise_keys_sec",
        "share_keys_sec",
        "verify_sigs_sec",
        "masked_input_sec",
        "unmasking_sec",
        "total_time_sec",
        "advertise_mb",
        "share_mb",
        "masked_mb",
        "total_comm_mb",
    ]

    _write_csv(OUT_TIMING, timing_fieldnames, timing_rows)
    _write_csv(OUT_SUMMARY, summary_fieldnames, summary_rows)

    summary = pd.DataFrame(summary_rows)
    _plot_clients(
        summary,
        metric="total_time_sec",
        title="Original vs PQ: Total Runtime vs Participants",
        ylabel="Total runtime (sec)",
        out_path=FIGURES_DIR / "bench_orig_vs_pq_runtime_clients.pdf",
    )
    _plot_clients(
        summary,
        metric="total_comm_mb",
        title="Original vs PQ: Communication Cost vs Participants",
        ylabel="Communication per client (MB)",
        out_path=FIGURES_DIR / "bench_orig_vs_pq_comm_clients.pdf",
    )
    _plot_dropout(
        summary,
        metric="total_time_sec",
        title="Original vs PQ: Total Runtime vs Dropout Rate",
        ylabel="Total runtime (sec)",
        out_path=FIGURES_DIR / "bench_orig_vs_pq_runtime_dropout.pdf",
    )
    _plot_dropout(
        summary,
        metric="total_comm_mb",
        title="Original vs PQ: Communication Cost vs Dropout Rate",
        ylabel="Communication per client (MB)",
        out_path=FIGURES_DIR / "bench_orig_vs_pq_comm_dropout.pdf",
    )
    _plot_vector(
        summary,
        metric="total_time_sec",
        title="Original vs PQ: Total Runtime vs Vector Size",
        ylabel="Total runtime (sec)",
        out_path=FIGURES_DIR / "bench_orig_vs_pq_runtime_vector.pdf",
    )
    _plot_vector(
        summary,
        metric="total_comm_mb",
        title="Original vs PQ: Communication Cost vs Vector Size",
        ylabel="Communication per client (MB)",
        out_path=FIGURES_DIR / "bench_orig_vs_pq_comm_vector.pdf",
    )

    print(f"[orig_vs_pq] Timing results  -> {OUT_TIMING}")
    print(f"[orig_vs_pq] Summary results -> {OUT_SUMMARY}")
    print(f"[orig_vs_pq] Figures saved in -> {FIGURES_DIR}")
    
    # Clean up checkpoint after successful completion
    if OUT_CHECKPOINT.exists():
        OUT_CHECKPOINT.unlink()
        print(f"[orig_vs_pq] Checkpoint removed (all scenarios completed)")


if __name__ == "__main__":
    run()