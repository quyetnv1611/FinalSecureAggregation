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
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

import gc


ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from experiments.fl_simulator import run_secagg_timing
from secagg.config import DH_PRIME
from secagg.crypto_backend import configure_backend_environment
from secagg.crypto_backend import cuda_kem_available, cuda_sig_available, resolve_mode
from secagg.crypto_mlkem import SecAggregatorMLKEM
from secagg.sig_pq import make_signer


RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

_OUT_SUFFIX = os.getenv("BENCH_ORIG_VS_PQ_SUFFIX", "")
if _OUT_SUFFIX:
    OUT_TIMING = RESULTS_DIR / f"bench_orig_vs_pq_timing_{_OUT_SUFFIX}.csv"
    OUT_SUMMARY = RESULTS_DIR / f"bench_orig_vs_pq_summary_{_OUT_SUFFIX}.csv"
else:
    OUT_TIMING = 
    
    
    
    dRESULTS_DIR / "bench_orig_vs_pq_timing.csv"
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
ONLY_ALGORITHM = None  # If set, only run this algorithm (e.g., "original" or "pq")

CLIENT_COUNTS = [100, 200, 300, 400, 500]
DROPOUT_RATES = [0.0, 0.1, 0.2, 0.3]
# Extended vector sizes to match Figure 6 (10k-50k range with more data points)
VECTOR_SIZES = [10_000, 15_000, 20_000, 25_000, 30_000, 35_000, 40_000, 45_000, 50_000]

BITS_PER_ENTRY = 62
N_REPEAT = 3
REFERENCE_CLIENTS = 100
REFERENCE_DROPOUT = 0.0

# Checkpoint settings for resumable runs
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
OUT_CHECKPOINT = CHECKPOINT_DIR / "bench_orig_vs_pq_progress.json"


def _parse_int_list(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return [int(item) for item in values]


def _parse_float_list(raw: str) -> list[float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return [float(item) for item in values]


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


def _run_timing_once(
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
        n_repeat=1,
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


def _load_checkpoint() -> tuple[dict[str, dict[str, object]], list[dict], list[dict]]:
    """Load checkpoint from JSON. Returns (progress_map, timing_rows, summary_rows)."""
    if not OUT_CHECKPOINT.exists():
        return {}, [], []
    
    try:
        with OUT_CHECKPOINT.open("r", encoding="utf-8") as f:
            data = json.load(f)
        progress = data.get("progress", {})
        timing_rows = data.get("timing_rows", [])
        summary_rows = data.get("summary_rows", [])
        print(f"[orig_vs_pq] Loaded checkpoint: {len(progress)} scenarios tracked")
        return progress, timing_rows, summary_rows
    except Exception as e:
        print(f"[orig_vs_pq] Warning: Failed to load checkpoint: {e}")
        return {}, [], []

def _load_existing_csv_results() -> tuple[dict[str, dict[str, object]], list[dict], list[dict]]:
    """Load any already-written CSV rows so reruns can skip finished scenarios.

    This lets the benchmark resume from a partially completed or previously
    interrupted run even when the JSON checkpoint is missing or stale.
    """
    progress: dict[str, dict[str, object]] = {}
    timing_rows: list[dict] = []
    summary_rows: list[dict] = []

    if OUT_TIMING.exists():
        try:
            timing_df = pd.read_csv(OUT_TIMING)
            timing_rows = timing_df.to_dict(orient="records")
        except Exception as e:
            print(f"[orig_vs_pq] Warning: Failed to read timing CSV: {e}")

    if OUT_SUMMARY.exists():
        try:
            summary_df = pd.read_csv(OUT_SUMMARY)
            summary_rows = summary_df.to_dict(orient="records")
            for row in summary_rows:
                key = _make_scenario_key(
                    str(row["scenario"]),
                    int(row["n_clients"]),
                    float(row["dropout_rate"]),
                    int(row["vector_size"]),
                    str(row["algorithm"]),
                )
                progress[str(key)] = {
                    "completed_repeats": N_REPEAT,
                    "scenario": str(row["scenario"]),
                    "n_clients": int(row["n_clients"]),
                    "dropout_rate": float(row["dropout_rate"]),
                    "vector_size": int(row["vector_size"]),
                    "algorithm": str(row["algorithm"]),
                }
            print(f"[orig_vs_pq] Loaded existing CSV results: {len(progress)} scenarios completed")
        except Exception as e:
            print(f"[orig_vs_pq] Warning: Failed to read summary CSV: {e}")

    if timing_rows:
        timing_df = pd.DataFrame(timing_rows)
        if "repeat" in timing_df.columns:
            for _, group in timing_df.groupby(["scenario", "algorithm", "n_clients", "dropout_rate", "vector_size"]):
                first = group.iloc[0]
                key = _make_scenario_key(
                    str(first["scenario"]),
                    int(first["n_clients"]),
                    float(first["dropout_rate"]),
                    int(first["vector_size"]),
                    str(first["algorithm"]),
                )
                progress.setdefault(str(key), {
                    "completed_repeats": 0,
                    "scenario": str(first["scenario"]),
                    "n_clients": int(first["n_clients"]),
                    "dropout_rate": float(first["dropout_rate"]),
                    "vector_size": int(first["vector_size"]),
                    "algorithm": str(first["algorithm"]),
                })
                progress[str(key)]["completed_repeats"] = max(
                    int(progress[str(key)]["completed_repeats"]),
                    int(group["repeat"].nunique()),
                )

    return progress, timing_rows, summary_rows


def _save_checkpoint(progress: dict[str, dict[str, object]], timing_rows: list[dict], summary_rows: list[dict]) -> None:
    """Save checkpoint to JSON."""
    try:
        data = {
            "progress": progress,
            "timing_rows": timing_rows,
            "summary_rows": summary_rows,
            "timestamp": time.time(),
        }
        with OUT_CHECKPOINT.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[orig_vs_pq] Warning: Failed to save checkpoint: {e}")

def _save_outputs(timing_rows: list[dict], summary_rows: list[dict]) -> None:
    """Persist CSV outputs incrementally so completed rows survive interruptions."""
    timing_fieldnames = [
        "scenario",
        "algorithm",
        "repeat",
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


def _make_scenario_key(scenario_name: str, n_clients: int, dropout_rate: float, vector_size: int, algorithm: str) -> tuple:
    """Create a hashable key for scenario tracking."""
    return (scenario_name, n_clients, dropout_rate, vector_size, algorithm)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _log_backend_state() -> tuple[bool, bool, str, str]:
    runtime_gpu = torch.cuda.is_available()
    requested_mode = os.getenv("SECAGG_CRYPTO_ACCEL", "auto")
    effective_mode = resolve_mode(requested_mode)
    kem_cuda = cuda_kem_available()
    sig_cuda = cuda_sig_available()
    print(f"[orig_vs_pq] Torch CUDA available: {runtime_gpu}")
    if runtime_gpu:
        print(f"[orig_vs_pq] Torch GPU device: {torch.cuda.get_device_name(0)}")
    print(
        f"[orig_vs_pq] Crypto accel request={requested_mode!r} effective={effective_mode} "
        f"kem_cuda={kem_cuda} sig_cuda={sig_cuda}"
    )
    return kem_cuda, sig_cuda, requested_mode, effective_mode


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


def _render_plots(summary: pd.DataFrame, plot_formats: list[str] | None = None) -> None:
    formats = plot_formats or ["pdf"]
    for ext in formats:
        suffix = ext.lower().lstrip(".")
        _plot_clients(
            summary,
            metric="total_time_sec",
            title="Original vs PQ: Total Runtime vs Participants",
            ylabel="Total runtime (sec)",
            out_path=FIGURES_DIR / f"bench_orig_vs_pq_runtime_clients.{suffix}",
        )
        _plot_clients(
            summary,
            metric="total_comm_mb",
            title="Original vs PQ: Communication Cost vs Participants",
            ylabel="Communication per client (MB)",
            out_path=FIGURES_DIR / f"bench_orig_vs_pq_comm_clients.{suffix}",
        )
        _plot_dropout(
            summary,
            metric="total_time_sec",
            title="Original vs PQ: Total Runtime vs Dropout Rate",
            ylabel="Total runtime (sec)",
            out_path=FIGURES_DIR / f"bench_orig_vs_pq_runtime_dropout.{suffix}",
        )
        _plot_dropout(
            summary,
            metric="total_comm_mb",
            title="Original vs PQ: Communication Cost vs Dropout Rate",
            ylabel="Communication per client (MB)",
            out_path=FIGURES_DIR / f"bench_orig_vs_pq_comm_dropout.{suffix}",
        )
        _plot_vector(
            summary,
            metric="total_time_sec",
            title="Original vs PQ: Total Runtime vs Vector Size",
            ylabel="Total runtime (sec)",
            out_path=FIGURES_DIR / f"bench_orig_vs_pq_runtime_vector.{suffix}",
        )
        _plot_vector(
            summary,
            metric="total_comm_mb",
            title="Original vs PQ: Communication Cost vs Vector Size",
            ylabel="Communication per client (MB)",
            out_path=FIGURES_DIR / f"bench_orig_vs_pq_comm_vector.{suffix}",
        )


def plot_from_summary_csv(summary_csv: Path, plot_formats: list[str] | None = None) -> None:
    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")
    summary = pd.read_csv(summary_csv)
    required_cols = {"scenario", "algorithm", "total_time_sec", "total_comm_mb", "n_clients", "dropout_rate", "vector_size"}
    missing = required_cols.difference(set(summary.columns))
    if missing:
        raise ValueError(f"Summary CSV missing required columns: {sorted(missing)}")
    _render_plots(summary, plot_formats=plot_formats)
    print(f"[orig_vs_pq] Plots generated from summary CSV -> {summary_csv}")
    print(f"[orig_vs_pq] Figures saved in -> {FIGURES_DIR}")


def run(
    device: str = None,
    require_cuda_backend: bool = False,
    require_full_cuda_backend: bool = False,
    only_scenario: str = "",
) -> None:
    # Automatically select GPU if available, otherwise CPU
    if device is None:
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cuda"

    import secagg.config
    secagg.config.CRYPTO_ACCEL = "cuda"   
    kem_cuda, sig_cuda, requested_mode, _effective_mode = _log_backend_state()
    
    effective_mode = "cuda"
    if require_cuda_backend and requested_mode == "cuda" and not (kem_cuda or sig_cuda):
        raise RuntimeError(
            "CUDA crypto backend required, but no CUDA adapter is available "
            "(kem_cuda=False, sig_cuda=False). "
            "Install and configure a CUDA adapter module first."
        )
    if require_full_cuda_backend and requested_mode == "cuda" and not (kem_cuda and sig_cuda):
        raise RuntimeError(
            "Full CUDA crypto backend required, but KEM/SIG CUDA adapters are incomplete "
            f"(kem_cuda={kem_cuda}, sig_cuda={sig_cuda}). "
            "Install and configure both CUDA KEM and CUDA SIG adapters first."
        )

    print(f"[orig_vs_pq] Using device: {device}")

    progress, timing_rows, summary_rows = _load_checkpoint()
    csv_progress, csv_timing_rows, csv_summary_rows = _load_existing_csv_results()
    if csv_progress:
        progress.update(csv_progress)
        if not timing_rows:
            timing_rows = csv_timing_rows
        if not summary_rows:
            summary_rows = csv_summary_rows

    scenarios = _build_scenarios()
    
    # Filter scenarios if only_scenario is specified
    if only_scenario:
        scenarios = [s for s in scenarios if s["scenario"] == only_scenario]
        print(f"[orig_vs_pq] Filtered to ONLY '{only_scenario}' scenarios: {len(scenarios)} total")
    
    total_scenarios = len(scenarios) * len(ALGORITHMS)
    print(f"[orig_vs_pq] Found {len(progress)} / {total_scenarios} scenarios already completed")
    print("[orig_vs_pq] CSV output files are being initialized now; first scenario may take several minutes.", flush=True)
    _save_outputs(timing_rows, summary_rows)

    for scenario in scenarios:
        scenario_name = str(scenario["scenario"])
        n_clients = int(scenario["n_clients"])
        dropout_rate = float(scenario["dropout_rate"])
        vector_size = int(scenario["vector_size"])

        for algorithm in ALGORITHMS:
            scenario_key = _make_scenario_key(
                scenario_name,
                n_clients,
                dropout_rate,
                vector_size,
                algorithm["algorithm"],
            )

            scenario_state = progress.setdefault(str(scenario_key), {
                "completed_repeats": 0,
                "scenario": scenario_name,
                "n_clients": n_clients,
                "dropout_rate": dropout_rate,
                "vector_size": vector_size,
                "algorithm": algorithm["algorithm"],
            })
            completed_repeats = int(scenario_state.get("completed_repeats", 0))

            if completed_repeats >= N_REPEAT:
                print(
                    f"[orig_vs_pq] Skipping (already done): scenario={scenario_name} "
                    f"n={n_clients} dropout={dropout_rate:.1f} vector={vector_size} "
                    f"algo={algorithm['algorithm']}"
                )
                continue

            print(
                f"[orig_vs_pq] scenario={scenario_name} n={n_clients} "
                f"dropout={dropout_rate:.1f} vector={vector_size} algo={algorithm['algorithm']}"
            , flush=True)

            try:
                accumulated = {
                    "advertise_keys": 0.0,
                    "share_keys": 0.0,
                    "verify_sigs": 0.0,
                    "masked_input": 0.0,
                    "unmasking": 0.0,
                    "total": 0.0,
                }

                # 1. Chạy mồi (Warm-up) 1 lần duy nhất để nạp thư viện C/C++
                if completed_repeats == 0:
                    _run_timing_once(
                        algorithm=algorithm,
                        n_clients=min(n_clients, 5), # Chạy nháp với số client nhỏ
                        dropout_rate=dropout_rate,
                        vector_size=vector_size,
                    )

                # 2. Vòng lặp đo lường chính thức
                for repeat_index in range(completed_repeats, N_REPEAT):
                    print(
                        f"[orig_vs_pq] repeat {repeat_index + 1}/{N_REPEAT} for scenario={scenario_name} "
                        f"n={n_clients} dropout={dropout_rate:.1f} vector={vector_size} algo={algorithm['algorithm']}",
                        flush=True,
                    )

                    gc.disable()  # Tắt Garbage Collector trước khi đo
                    
                    timings = _run_timing_once(
                        algorithm=algorithm,
                        n_clients=n_clients,
                        dropout_rate=dropout_rate,
                        vector_size=vector_size,
                    )
                    
                    gc.enable()   # Bật lại Garbage Collector
                    
                    # Dọn dẹp bộ nhớ RAM/VRAM để tránh quá tải
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                    for phase_name in accumulated:
                        accumulated[phase_name] += float(timings[phase_name])

                    timing_rows.extend([
                        {
                            "scenario": scenario_name,
                            "algorithm": algorithm["algorithm"],
                            "repeat": repeat_index + 1,
                            "backend_label": algorithm["label"],
                            "kem_backend": algorithm["kem_backend"],
                            "sig_backend": algorithm["sig_backend"],
                            "n_clients": n_clients,
                            "dropout_rate": dropout_rate,
                            "vector_size": vector_size,
                            "phase": "advertise_keys",
                            "time_sec": round(float(timings["advertise_keys"]), 6),
                        },
                        {
                            "scenario": scenario_name,
                            "algorithm": algorithm["algorithm"],
                            "repeat": repeat_index + 1,
                            "backend_label": algorithm["label"],
                            "kem_backend": algorithm["kem_backend"],
                            "sig_backend": algorithm["sig_backend"],
                            "n_clients": n_clients,
                            "dropout_rate": dropout_rate,
                            "vector_size": vector_size,
                            "phase": "share_keys",
                            "time_sec": round(float(timings["share_keys"]), 6),
                        },
                        {
                            "scenario": scenario_name,
                            "algorithm": algorithm["algorithm"],
                            "repeat": repeat_index + 1,
                            "backend_label": algorithm["label"],
                            "kem_backend": algorithm["kem_backend"],
                            "sig_backend": algorithm["sig_backend"],
                            "n_clients": n_clients,
                            "dropout_rate": dropout_rate,
                            "vector_size": vector_size,
                            "phase": "verify_sigs",
                            "time_sec": round(float(timings["verify_sigs"]), 6),
                        },
                        {
                            "scenario": scenario_name,
                            "algorithm": algorithm["algorithm"],
                            "repeat": repeat_index + 1,
                            "backend_label": algorithm["label"],
                            "kem_backend": algorithm["kem_backend"],
                            "sig_backend": algorithm["sig_backend"],
                            "n_clients": n_clients,
                            "dropout_rate": dropout_rate,
                            "vector_size": vector_size,
                            "phase": "masked_input",
                            "time_sec": round(float(timings["masked_input"]), 6),
                        },
                        {
                            "scenario": scenario_name,
                            "algorithm": algorithm["algorithm"],
                            "repeat": repeat_index + 1,
                            "backend_label": algorithm["label"],
                            "kem_backend": algorithm["kem_backend"],
                            "sig_backend": algorithm["sig_backend"],
                            "n_clients": n_clients,
                            "dropout_rate": dropout_rate,
                            "vector_size": vector_size,
                            "phase": "unmasking",
                            "time_sec": round(float(timings["unmasking"]), 6),
                        },
                        {
                            "scenario": scenario_name,
                            "algorithm": algorithm["algorithm"],
                            "repeat": repeat_index + 1,
                            "backend_label": algorithm["label"],
                            "kem_backend": algorithm["kem_backend"],
                            "sig_backend": algorithm["sig_backend"],
                            "n_clients": n_clients,
                            "dropout_rate": dropout_rate,
                            "vector_size": vector_size,
                            "phase": "total",
                            "time_sec": round(float(timings["total"]), 6),
                        },
                    ])

                    scenario_state["completed_repeats"] = repeat_index + 1
                    _save_checkpoint(progress, timing_rows, summary_rows)
                    _save_outputs(timing_rows, summary_rows)

                comm = _estimate_comm_bytes(
                    algorithm=algorithm,
                    n_clients=n_clients,
                    dropout_rate=dropout_rate,
                    vector_size=vector_size,
                )

                total_time = accumulated["total"] / N_REPEAT
                summary_rows.append({
                    "scenario": scenario_name,
                    "algorithm": algorithm["algorithm"],
                    "backend_label": algorithm["label"],
                    "kem_backend": algorithm["kem_backend"],
                    "sig_backend": algorithm["sig_backend"],
                    "n_clients": n_clients,
                    "dropout_rate": dropout_rate,
                    "vector_size": vector_size,
                    "advertise_keys_sec": round(accumulated["advertise_keys"] / N_REPEAT, 6),
                    "share_keys_sec": round(accumulated["share_keys"] / N_REPEAT, 6),
                    "verify_sigs_sec": round(accumulated["verify_sigs"] / N_REPEAT, 6),
                    "masked_input_sec": round(accumulated["masked_input"] / N_REPEAT, 6),
                    "unmasking_sec": round(accumulated["unmasking"] / N_REPEAT, 6),
                    "total_time_sec": round(total_time, 6),
                    "advertise_mb": round(comm["advertise_mb"], 6),
                    "share_mb": round(comm["share_mb"], 6),
                    "masked_mb": round(comm["masked_mb"], 6),
                    "total_comm_mb": round(comm["total_mb"], 6),
                })

                _save_checkpoint(progress, timing_rows, summary_rows)
                _save_outputs(timing_rows, summary_rows)

            except Exception as e:
                print(f"[orig_vs_pq] ERROR in scenario: {e}", file=sys.stderr)
                import traceback

                traceback.print_exc()
                _save_checkpoint(progress, timing_rows, summary_rows)
                _save_outputs(timing_rows, summary_rows)
                print(
                    f"[orig_vs_pq] Checkpoint saved before exit. Restart to resume.",
                    file=sys.stderr,
                )
                raise

    _save_outputs(timing_rows, summary_rows)

    summary = pd.DataFrame(summary_rows)
    _render_plots(summary)

    print(f"[orig_vs_pq] Timing results  -> {OUT_TIMING}")
    print(f"[orig_vs_pq] Summary results -> {OUT_SUMMARY}")
    print(f"[orig_vs_pq] Figures saved in -> {FIGURES_DIR}")

    if OUT_CHECKPOINT.exists():
        OUT_CHECKPOINT.unlink()
        print(f"[orig_vs_pq] Checkpoint removed (all scenarios completed)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Original vs PQ benchmark")
    parser.add_argument("--crypto-accel", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Crypto accel mode [default: auto]")
    parser.add_argument("--cuda-kem-module", type=str, default="", help="Optional CUDA KEM module name")
    parser.add_argument("--cuda-sig-module", type=str, default="", help="Optional CUDA SIG module name")
    parser.add_argument("--cuda-kem-library", type=str, default="", help="Optional path to a real CUDA ML-KEM shared library (e.g. liboqs.so built with cuPQC)")
    parser.add_argument("--cuda-sig-library", type=str, default="", help="Optional path to a real CUDA signature shared library (e.g. cuDilithium .so/.dll)")
    parser.add_argument("--cpu-kem-module", type=str, default="", help="Optional CPU KEM module name (e.g. oqs)")
    parser.add_argument("--cpu-sig-module", type=str, default="", help="Optional CPU SIG module name (e.g. oqs)")
    parser.add_argument("--prefer-liboqs", action="store_true", help="Prefer liboqs CPU adapter when available")
    parser.add_argument("--clients", type=str, default="", help="Comma-separated client counts, e.g. 100,200,300")
    parser.add_argument("--dropouts", type=str, default="", help="Comma-separated dropout rates, e.g. 0.0,0.1,0.3")
    parser.add_argument("--vector-sizes", type=str, default="", help="Comma-separated vector sizes, e.g. 50000,100000,200000")
    parser.add_argument("--n-repeat", type=int, default=N_REPEAT, help="Number of timing repeats per scenario")
    parser.add_argument("--reference-clients", type=int, default=REFERENCE_CLIENTS, help="Reference clients for dropout/vector sweeps")
    parser.add_argument("--reference-dropout", type=float, default=REFERENCE_DROPOUT, help="Reference dropout for vector sweep")
    parser.add_argument("--reset-checkpoint", action="store_true", help="Delete existing checkpoint before running")
    parser.add_argument("--only-scenario", type=str, default="", choices=["", "clients", "dropout", "vector"], help="Run only this scenario type; empty = all")
    parser.add_argument("--only-algorithm", type=str, default="", choices=["", "original", "pq"], help="Run only this algorithm (original or pq); empty = both")
    parser.add_argument("--plots-only", action="store_true", help="Only render figures from an existing summary CSV")
    parser.add_argument("--summary-csv", type=str, default=str(OUT_SUMMARY), help="Path to summary CSV used by --plots-only")
    parser.add_argument("--plot-formats", type=str, default="pdf", help="Comma-separated output figure formats, e.g. pdf,png")
    parser.add_argument("--require-cuda-backend", action="store_true", help="Fail fast if --crypto-accel cuda but no CUDA KEM/SIG adapter is available")
    parser.add_argument("--require-full-cuda-backend", action="store_true", help="Fail fast if --crypto-accel cuda but both CUDA KEM and CUDA SIG adapters are not available")
    args = parser.parse_args()

    plot_formats = [item.strip().lower().lstrip(".") for item in args.plot_formats.split(",") if item.strip()]
    if not plot_formats:
        raise ValueError("--plot-formats must contain at least one format")

    if args.plots_only:
        plot_from_summary_csv(Path(args.summary_csv), plot_formats=plot_formats)
        raise SystemExit(0)

    if args.clients:
        CLIENT_COUNTS = _parse_int_list(args.clients)
    if args.dropouts:
        DROPOUT_RATES = _parse_float_list(args.dropouts)
    if args.vector_sizes:
        VECTOR_SIZES = _parse_int_list(args.vector_sizes)
    if args.n_repeat < 1:
        raise ValueError("--n-repeat must be >= 1")
    N_REPEAT = args.n_repeat
    REFERENCE_CLIENTS = args.reference_clients
    REFERENCE_DROPOUT = args.reference_dropout

    if args.reset_checkpoint and OUT_CHECKPOINT.exists():
        OUT_CHECKPOINT.unlink()
        print(f"[orig_vs_pq] Removed checkpoint: {OUT_CHECKPOINT}")

    print(
        f"[orig_vs_pq] Sweep config: clients={CLIENT_COUNTS} dropouts={DROPOUT_RATES} "
        f"vector_sizes={VECTOR_SIZES} n_repeat={N_REPEAT} "
        f"ref_clients={REFERENCE_CLIENTS} ref_dropout={REFERENCE_DROPOUT}"
    )

    # Filter algorithms if user specified --only-algorithm
    if args.only_algorithm:
        if args.only_algorithm == "original":
            ALGORITHMS[:] = [ORIGINAL]
            print(f"[orig_vs_pq] Running ONLY Original backend (DH + ECDSA)")
        elif args.only_algorithm == "pq":
            ALGORITHMS[:] = [PQ]
            print(f"[orig_vs_pq] Running ONLY PQ backend (ML-KEM-768 + ML-DSA-65)")

    configure_backend_environment(
        crypto_accel=args.crypto_accel,
        cuda_kem_module=args.cuda_kem_module or None,
        cuda_sig_module=args.cuda_sig_module or None,
        cuda_kem_library=args.cuda_kem_library or None,
        cuda_sig_library=args.cuda_sig_library or None,
        cpu_kem_module=args.cpu_kem_module or None,
        cpu_sig_module=args.cpu_sig_module or None,
        prefer_liboqs=args.prefer_liboqs,
    )

    run(
        require_cuda_backend=args.require_cuda_backend,
        require_full_cuda_backend=args.require_full_cuda_backend,
        only_scenario=args.only_scenario,
    )