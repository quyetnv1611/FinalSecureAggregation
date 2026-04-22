"""
experiments/benchmarks/bench_accuracy.py
==========================================
Benchmark 1 — FL accuracy + SecAgg crypto overhead across four datasets
and five KEM×SIG backend combinations.

Strategy
--------
* FedAvg accuracy: run once per dataset (all 50 clients, 20 rounds).
* SecAgg timing:   run once per (dataset, backend) with SECAGG_N= 10
  representative clients to avoid O(n²) slowdown.
* Output rows carry both accuracy metrics AND per-phase SecAgg timings.

Datasets
--------
* MNIST        — 50 clients, 10 classes, CNN
* CIFAR-10     — 50 clients, 10 classes, CNN
* SMS Spam     — 50 clients, 2  classes, MLP
* Web Attack   — 50 clients, 2  classes, MLP (NSL-KDD)

Backends (5 combos)
-------------------
1. DH        + ECDSA   — Classic (Bonawitz baseline)
2. ML-KEM-768 + ECDSA  — PQ-KEM only
3. DH        + ML-DSA-65 — PQ-SIG only
4. ML-KEM-768 + ML-DSA-65 — Full PQ (Kyber + Dilithium)
5. ML-KEM-768 + SLH-DSA-shake_128f — Full PQ (Kyber + SPHINCS+)

Output
------
``results/bench_accuracy.csv``

Columns: dataset, kem_backend, sig_backend, backend_label, round,
         loss, accuracy, fl_time_sec,
         time_adv_keys, time_share_keys, time_verify_sigs,
         time_masked, time_unmask, time_secagg_total
"""


from __future__ import annotations

import random
import os
import numpy as np

# Hàm set_seed cho random, numpy, torch
def set_seed(seed):
    if seed == 'random':
        seed = int(time.time())
        print(f"[bench_accuracy] Using random seed: {seed}")
    else:
        seed = int(seed)
        print(f"[bench_accuracy] Using fixed seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))


from experiments.fl_simulator import run_secagg_timing
from experiments.fl_simulator import FLSimulator
from secagg.crypto_backend import configure_backend_environment

# Model CNN mới cho MNIST: 2 conv + 2 pool + 1 linear
class CustomMnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(7*7*64, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x
from experiments.models.cifar_model import CifarCNN
from experiments.models.mlp_model import MLP
from sklearn.metrics import f1_score

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUT_CSV = RESULTS_DIR / "bench_accuracy.csv"

# Checkpoint settings for resumable runs
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
OUT_CHECKPOINT = CHECKPOINT_DIR / "bench_accuracy_progress.json"

# ---------------------------------------------------------------------------
# Checkpoint functions
# ---------------------------------------------------------------------------

def _load_checkpoint() -> tuple[set[tuple], list[dict]]:
    """Load checkpoint from JSON. Returns (completed_set, rows)."""
    if not OUT_CHECKPOINT.exists():
        return set(), []
    
    try:
        with OUT_CHECKPOINT.open("r", encoding="utf-8") as f:
            data = json.load(f)
        completed = set(tuple(item) for item in data.get("completed", []))
        rows = data.get("rows", [])
        print(f"[bench_accuracy] Loaded checkpoint: {len(completed)} (ds,kem,sig) combos completed, {len(rows)} rows")
        return completed, rows
    except Exception as e:
        print(f"[bench_accuracy] Warning: Failed to load checkpoint: {e}")
        return set(), []


def _save_checkpoint(completed: set[tuple], rows: list[dict]) -> None:
    """Save checkpoint to JSON."""
    try:
        data = {
            "completed": [list(item) for item in completed],
            "rows": rows,
            "timestamp": time.time(),
        }
        with OUT_CHECKPOINT.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[bench_accuracy] Warning: Failed to save checkpoint: {e}")


# ---------------------------------------------------------------------------
# Config
LR           = 0.01 
BATCH_SIZE   = 128
DROPOUT      = 0.1  # ty le client dropout moi vong 
TIMING_REPS  = 2      # so lan chay roi lay trung binh

# Default config - có thể thay đổi qua command line
N_CLIENTS    = 500    # Tong so client tham gia (test: 5, 10, 20, 50, 100)
SECAGG_N     = 10    # So client su dung de do thời gian thuat toan ma hoa (n² optimization)
N_ROUNDS     = 20    # So vong giao tiep giua client va server (từ paper)
LOCAL_EPOCHS = 3     # so vong client tu huan luyen

# Uncomment nếu muốn test với config khác
# N_CLIENTS    = 10
# N_ROUNDS     = 5
# LOCAL_EPOCHS = 3

# (kem_backend, sig_backend, human label)
BACKENDS = [
    ("DH",          "classic",             "Classic (DH+ECDSA)"),
    # ("ML-KEM-768",  "classic",             "PQ-KEM only"),
    # ("DH",          "ML-DSA-65",           "PQ-SIG only"),
    ("ML-KEM-768",  "ML-DSA-65",           "Full PQ (ML-KEM+ML-DSA)"),
    ("ML-KEM-768",  "SLH-DSA-shake_128f",  "Full PQ (ML-KEM+SPHINCS+)"),
]

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

def _get_mnist():
    from experiments.datasets.mnist_loader import load_mnist
    tl, vl = load_mnist(n_clients=N_CLIENTS, batch_size=BATCH_SIZE)
    return tl, vl, lambda: CustomMnistCNN(), nn.CrossEntropyLoss()


def _get_cifar10():
    from experiments.datasets.cifar_loader import load_cifar10
    tl, vl = load_cifar10(n_clients=N_CLIENTS, batch_size=BATCH_SIZE)
    return tl, vl, lambda: CifarCNN(), nn.CrossEntropyLoss()


def _get_spam():
    from experiments.datasets.spam_loader import load_spam
    tl, vl, idim = load_spam(n_clients=N_CLIENTS, batch_size=BATCH_SIZE)
    return tl, vl, lambda: MLP(idim), nn.CrossEntropyLoss()


def _get_webattack():
    from experiments.datasets.webattack_loader import load_webattack
    tl, vl, idim = load_webattack(n_clients=N_CLIENTS, batch_size=BATCH_SIZE)
    return tl, vl, lambda: MLP(idim), nn.CrossEntropyLoss()


DATASETS = {
    "mnist":      _get_mnist,
    "cifar10":    _get_cifar10,
    "spam":       _get_spam,
    "webattack":  _get_webattack,
}

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _get_device(preferred: str = "auto") -> str:
    """Select the tensor device for FL training and evaluation."""
    preferred = (preferred or "auto").strip().lower()
    if preferred not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unknown device mode: {preferred}")

    if preferred == "cpu":
        print("[bench_accuracy] Using device: cpu")
        return "cpu"

    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build or switch back to --device auto."
            )
        print(f"[bench_accuracy] GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"[bench_accuracy] CUDA capability: {torch.cuda.get_device_capability(0)}")
        print("[bench_accuracy] Using device: cuda")
        return "cuda"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"[bench_accuracy] GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"[bench_accuracy] CUDA capability: {torch.cuda.get_device_capability(0)}")
    print(f"[bench_accuracy] Using device: {device}")
    return device

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(dataset_names: list[str] | None = None, device_mode: str = "auto") -> None:
    names = dataset_names or list(DATASETS.keys())
    
    device = _get_device(device_mode)
    
    # Load checkpoint to support resumable runs
    completed, rows = _load_checkpoint()
    print(f"[bench_accuracy] Starting: {len(names)} datasets, {len(BACKENDS)} backends")

    FIELDNAMES = [
        "dataset", "kem_backend", "sig_backend", "backend_label",
        "round", "loss", "accuracy", "f1_score", "fl_time_sec", "train_time_sec",
        "learning_rate", "batch_size",
        "time_adv_keys", "time_share_keys", "time_verify_sigs",
        "time_masked", "time_unmask", "time_secagg_total",
    ]

    for ds_name in names:
        print(f"\n{'='*60}")
        print(f"[bench_accuracy] Dataset: {ds_name}")
        loader_fn = DATASETS[ds_name]
        train_loaders, test_loader, model_fn, loss_fn = loader_fn()

        # Infer gradient shape from the base model
        _tmp = model_fn()
        import torch
        _flat = torch.cat([p.data.flatten() for p in _tmp.parameters()])
        grad_shape = (_flat.numel(),)
        del _tmp, _flat

        for kem, sig, label in BACKENDS:
            combo_key = (ds_name, kem, sig)
            
            # Skip if already completed
            if combo_key in completed:
                print(f"\n  Backend: {label}  (KEM={kem}, SIG={sig})  [SKIPPED - already done]")
                continue
            
            # Try-except to save progress if error occurs
            try:
                set_seed(RUN_SEED)
                print(f"\n  Backend: {label}  (KEM={kem}, SIG={sig})")

                # ── SecAgg timing (once per backend per dataset) ──────────────
                print(f"    Measuring SecAgg timing (n={SECAGG_N}, reps={TIMING_REPS})…", end=" ", flush=True)
                timer = run_secagg_timing(
                    n_clients   = SECAGG_N,
                    grad_shape  = grad_shape,
                    kem_backend = kem,
                    sig_backend = sig,
                    n_repeat    = TIMING_REPS,
                )
                print(f"done  (total={timer.total:.3f}s)")

                set_seed(RUN_SEED)
               

                # ── FedAvg training (fresh model per backend) ─────────────────
                sim = FLSimulator(
                    model_fn       = model_fn,
                    loss_fn        = loss_fn,
                    kem_backend    = kem,
                    sig_backend    = sig,
                    secagg_n       = SECAGG_N,
                    device         = device,
                    n_local_epochs = LOCAL_EPOCHS,
                    lr             = LR,
                )
                if device == "cuda":
                    model_device = next(sim.global_model.parameters()).device
                    print(f"    Model device: {model_device}")

                if RUN_SEED == 'random':
                    base_seed = int(time.time())
                else:
                    base_seed = int(RUN_SEED)

                t_start = time.perf_counter()
                for rnd in range(1, N_ROUNDS + 1):
                    set_seed(base_seed + rnd)
                    t_train = time.perf_counter()   
                    sim.train_round(train_loaders, dropout_rate=DROPOUT)
                    train_time = time.perf_counter() - t_train
                    loss, acc = sim.evaluate(test_loader)
                    # Tính F1-score nếu là bài toán phân loại
                    y_true, y_pred = None, None
                    try:
                        y_true, y_pred = sim.get_true_and_pred(test_loader)
                        f1 = f1_score(y_true, y_pred, average="macro")
                    except Exception:
                        f1 = None
                    fl_elapsed = time.perf_counter() - t_start
                    if device == "cuda":
                        torch.cuda.synchronize()
                        mem_now = torch.cuda.memory_allocated() / (1024 ** 2)
                        mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
                        gpu_stat = f" gpu_mem={mem_now:.1f}MB peak={mem_peak:.1f}MB"
                    else:
                        gpu_stat = ""
                    print(f"    Round {rnd:3d}  loss={loss:.4f}  acc={acc:.4f}  f1={f1 if f1 is not None else 'NA'}  fl_t={fl_elapsed:.1f}s train_t={train_time:.2f}s{gpu_stat}")
                    rows.append({
                        "dataset":           ds_name,
                        "kem_backend":       kem,
                        "sig_backend":       sig,
                        "backend_label":     label,
                        "round":             rnd,
                        "loss":              round(loss, 6),
                        "accuracy":          round(acc, 6),
                        "f1_score":          round(f1, 6) if f1 is not None else None,
                        "fl_time_sec":       round(fl_elapsed, 3),
                        "train_time_sec":    round(train_time, 3),
                        "learning_rate":     LR,
                        "batch_size":        BATCH_SIZE,
                        "time_adv_keys":     round(timer.advertise_keys, 6),
                        "time_share_keys":   round(timer.share_keys, 6),
                        "time_verify_sigs":  round(timer.verify_sigs, 6),
                        "time_masked":       round(timer.masked_input, 6),
                        "time_unmask":       round(timer.unmasking, 6),
                        "time_secagg_total": round(timer.total, 6),
                    })
                
                # Mark as completed and save checkpoint
                completed.add(combo_key)
                _save_checkpoint(completed, rows)
                
            except Exception as e:
                print(f"[bench_accuracy] ERROR in {ds_name} + {label}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                # Save progress before re-raising
                _save_checkpoint(completed, rows)
                print(f"[bench_accuracy] Checkpoint saved before exit. Restart to resume.", file=sys.stderr)
                raise

    # All datasets and backends completed: write final CSV and clean up checkpoint
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    if OUT_CHECKPOINT.exists():
        OUT_CHECKPOINT.unlink()
        print(f"[bench_accuracy] Checkpoint removed (all datasets completed)")

    print(f"\n[bench_accuracy] Results saved → {OUT_CSV}")


def _parse_int_csv(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_float_csv(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_local_grid(raw: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        b_str, e_str = token.split(":", 1)
        pairs.append((int(b_str.strip()), int(e_str.strip())))
    return pairs


def _run_mnist_curve(
    *,
    iid: bool,
    n_clients: int,
    batch_size: int,
    local_epochs: int,
    dropout: float,
    lr: float,
    milestones: list[int],
    seed: str,
    device_mode: str,
) -> list[dict[str, object]]:
    from experiments.datasets.mnist_loader import load_mnist

    train_loaders, test_loader = load_mnist(n_clients=n_clients, batch_size=batch_size, iid=iid)
    sim = FLSimulator(
        model_fn=CustomMnistCNN,
        loss_fn=nn.CrossEntropyLoss(),
        kem_backend="DH",
        sig_backend="classic",
        secagg_n=min(10, max(3, n_clients // 5)),
        device=_get_device(device_mode),
        n_local_epochs=local_epochs,
        lr=lr,
    )

    max_round = max(milestones)
    rows: list[dict[str, object]] = []

    if seed == "random":
        base_seed = int(time.time())
    else:
        base_seed = int(seed)

    for rnd in range(1, max_round + 1):
        set_seed(base_seed + rnd)
        sim.train_round(train_loaders, dropout_rate=dropout)
        loss, acc = sim.evaluate(test_loader)
        if rnd in milestones:
            rows.append(
                {
                    "round": rnd,
                    "loss": float(loss),
                    "accuracy": float(acc),
                }
            )
    return rows


def _plot_study(rows: list[dict[str, object]], study: str, out_path: Path) -> None:
    if not rows:
        return

    partitions = ["IID", "Non-IID"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for idx, part in enumerate(partitions):
        ax = axes[idx]
        part_rows = [r for r in rows if r["partition"] == part]
        labels = sorted(set(r["setting_label"] for r in part_rows))
        for label in labels:
            curve = sorted((r for r in part_rows if r["setting_label"] == label), key=lambda x: int(x["round"]))
            xs = [int(r["round"]) for r in curve]
            ys = [float(r["accuracy"]) for r in curve]
            ax.plot(xs, ys, marker="o", linewidth=1.8, label=label)
        ax.set_title(f"MNIST CNN {part}")
        ax.set_xlabel("Rounds")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.set_ylabel("Accuracy")
        ax.legend(fontsize=8)

    fig.suptitle(f"MNIST study: {study}")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_mnist_studies(
    *,
    studies: list[str],
    iid_mode: str,
    milestones: list[int],
    local_grid: list[tuple[int, int]],
    clients_grid: list[int],
    dropout_grid: list[float],
    fixed_clients: int,
    fixed_batch: int,
    fixed_epochs: int,
    fixed_dropout: float,
    lr: float,
    seed: str,
    device_mode: str,
) -> None:
    results_path = RESULTS_DIR / "bench_accuracy_mnist_studies.csv"
    figures_dir = ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)

    if iid_mode == "both":
        partitions = [(True, "IID"), (False, "Non-IID")]
    elif iid_mode == "iid":
        partitions = [(True, "IID")]
    else:
        partitions = [(False, "Non-IID")]

    rows: list[dict[str, object]] = []
    for iid, part_name in partitions:
        for study in studies:
            if study == "local_params":
                for bsz, loc_ep in local_grid:
                    curve = _run_mnist_curve(
                        iid=iid,
                        n_clients=fixed_clients,
                        batch_size=bsz,
                        local_epochs=loc_ep,
                        dropout=fixed_dropout,
                        lr=lr,
                        milestones=milestones,
                        seed=seed,
                        device_mode=device_mode,
                    )
                    for point in curve:
                        rows.append(
                            {
                                "study": study,
                                "partition": part_name,
                                "setting_label": f"B={bsz} E={loc_ep}",
                                "n_clients": fixed_clients,
                                "batch_size": bsz,
                                "local_epochs": loc_ep,
                                "dropout": fixed_dropout,
                                **point,
                            }
                        )
            elif study == "clients":
                for n_clients in clients_grid:
                    curve = _run_mnist_curve(
                        iid=iid,
                        n_clients=n_clients,
                        batch_size=fixed_batch,
                        local_epochs=fixed_epochs,
                        dropout=fixed_dropout,
                        lr=lr,
                        milestones=milestones,
                        seed=seed,
                        device_mode=device_mode,
                    )
                    for point in curve:
                        rows.append(
                            {
                                "study": study,
                                "partition": part_name,
                                "setting_label": f"C={n_clients}",
                                "n_clients": n_clients,
                                "batch_size": fixed_batch,
                                "local_epochs": fixed_epochs,
                                "dropout": fixed_dropout,
                                **point,
                            }
                        )
            elif study == "dropout":
                for drop in dropout_grid:
                    curve = _run_mnist_curve(
                        iid=iid,
                        n_clients=fixed_clients,
                        batch_size=fixed_batch,
                        local_epochs=fixed_epochs,
                        dropout=drop,
                        lr=lr,
                        milestones=milestones,
                        seed=seed,
                        device_mode=device_mode,
                    )
                    for point in curve:
                        rows.append(
                            {
                                "study": study,
                                "partition": part_name,
                                "setting_label": f"D={int(drop * 100)}%",
                                "n_clients": fixed_clients,
                                "batch_size": fixed_batch,
                                "local_epochs": fixed_epochs,
                                "dropout": drop,
                                **point,
                            }
                        )
            else:
                raise ValueError(f"Unknown study: {study}")

    fieldnames = [
        "study",
        "partition",
        "setting_label",
        "n_clients",
        "batch_size",
        "local_epochs",
        "dropout",
        "round",
        "loss",
        "accuracy",
    ]
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for study in studies:
        out_fig = figures_dir / f"bench_accuracy_mnist_{study}.png"
        _plot_study([r for r in rows if r["study"] == study], study, out_fig)

    print(f"[bench_accuracy] MNIST study CSV saved -> {results_path}")
    print(f"[bench_accuracy] MNIST study figures saved in -> {figures_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark FL accuracy + SecAgg overhead",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bench_accuracy.py                           # All datasets, 50 clients, 20 rounds
  python bench_accuracy.py --dataset mnist           # MNIST only
  python bench_accuracy.py --n_clients 10 --n_rounds 5  # Quick test: 10 clients, 5 rounds
  python bench_accuracy.py --n_clients 100 --dropout 0.2  # Test 100 clients, 20% dropout
        """
    )
    parser.add_argument("--dataset", type=str, help="Dataset name (mnist, cifar10, spam, webattack)")
    parser.add_argument("--seed", type=str, default="42", help="Seed for reproducibility (int or 'random') [default: 42]")
    parser.add_argument("--n_clients", type=int, default=50, help="Number of clients [default: 50]")
    parser.add_argument("--n_rounds", type=int, default=20, help="Number of FL rounds [default: 20]")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate [default: 0.1]")
    parser.add_argument("--local_epochs", type=int, default=1, help="Local training epochs [default: 1]")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device for FL training [default: auto]")
    parser.add_argument("--crypto-accel", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Crypto accel mode [default: auto]")
    parser.add_argument("--cuda-kem-module", type=str, default="", help="Optional CUDA KEM module name")
    parser.add_argument("--cuda-sig-module", type=str, default="", help="Optional CUDA SIG module name")
    parser.add_argument("--cpu-kem-module", type=str, default="", help="Optional CPU KEM module name (e.g. oqs)")
    parser.add_argument("--cpu-sig-module", type=str, default="", help="Optional CPU SIG module name (e.g. oqs)")
    parser.add_argument("--prefer-liboqs", action="store_true", help="Prefer liboqs CPU adapter when available")
    parser.add_argument("--mnist-studies", action="store_true", help="Run MNIST study suite (local params / clients / dropout) instead of legacy benchmark")
    parser.add_argument("--studies", type=str, default="local_params,clients,dropout", help="Comma-separated MNIST studies")
    parser.add_argument("--mnist-iid-mode", type=str, default="both", choices=["both", "iid", "noniid"], help="MNIST partition mode for study suite")
    parser.add_argument("--milestones", type=str, default="5,10,20,50,100", help="Round milestones for MNIST study")
    parser.add_argument("--local-grid", type=str, default="10:1,10:5,10:20,50:1,50:5,50:20", help="Local parameter grid as batch:epochs pairs")
    parser.add_argument("--clients-grid", type=str, default="5,10,20,50,100", help="Client-count grid for MNIST study")
    parser.add_argument("--dropout-grid", type=str, default="0.0,0.05,0.1,0.3,0.5", help="Dropout grid for MNIST study")
    parser.add_argument("--study-fixed-clients", type=int, default=50, help="Fixed clients for local/dropout studies")
    parser.add_argument("--study-fixed-batch", type=int, default=10, help="Fixed batch size for clients/dropout studies")
    parser.add_argument("--study-fixed-epochs", type=int, default=5, help="Fixed local epochs for clients/dropout studies")
    parser.add_argument("--study-fixed-dropout", type=float, default=0.0, help="Fixed dropout for local/clients studies")
    parser.add_argument("--study-lr", type=float, default=0.01, help="Learning rate for MNIST study suite")
    
    args = parser.parse_args()

    RUN_SEED = args.seed
    N_CLIENTS = args.n_clients
    N_ROUNDS = args.n_rounds
    DROPOUT = args.dropout
    LOCAL_EPOCHS = args.local_epochs
    SECAGG_N = min(10, max(3, N_CLIENTS // 5))  # Adaptive: 10 for large n, but at least 3

    configure_backend_environment(
        crypto_accel=args.crypto_accel,
        cuda_kem_module=args.cuda_kem_module or None,
        cuda_sig_module=args.cuda_sig_module or None,
        cpu_kem_module=args.cpu_kem_module or None,
        cpu_sig_module=args.cpu_sig_module or None,
        prefer_liboqs=args.prefer_liboqs,
    )
    
    print(f"[bench_accuracy] Config: clients={N_CLIENTS}, rounds={N_ROUNDS}, dropout={DROPOUT}, local_epochs={LOCAL_EPOCHS}")
    
    if args.mnist_studies:
        studies = [item.strip() for item in args.studies.split(",") if item.strip()]
        run_mnist_studies(
            studies=studies,
            iid_mode=args.mnist_iid_mode,
            milestones=_parse_int_csv(args.milestones),
            local_grid=_parse_local_grid(args.local_grid),
            clients_grid=_parse_int_csv(args.clients_grid),
            dropout_grid=_parse_float_csv(args.dropout_grid),
            fixed_clients=args.study_fixed_clients,
            fixed_batch=args.study_fixed_batch,
            fixed_epochs=args.study_fixed_epochs,
            fixed_dropout=args.study_fixed_dropout,
            lr=args.study_lr,
            seed=args.seed,
            device_mode=args.device,
        )
    else:
        if args.dataset:
            run([args.dataset], device_mode=args.device)
        else:
            run(device_mode=args.device)
