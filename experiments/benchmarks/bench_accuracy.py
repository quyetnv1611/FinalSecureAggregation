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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))


from experiments.fl_simulator import run_secagg_timing
from experiments.fl_simulator import FLSimulator

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
BATCH_SIZE   = 32
DROPOUT      = 0.1  # ty le client dropout moi vong 
TIMING_REPS  = 2      # so lan chay roi lay trung binh



# N_CLIENTS    = 4    # Tong so client tham gia.
# SECAGG_N     = 3   # So client su dung de do thời gian thuat toan ma hoa 
# N_ROUNDS     = 2    # So vong giao tiep giua client va server
# LOCAL_EPOCHS = 1   # so vong client tu huan luyen
# LR           = 0.01 
# BATCH_SIZE   = 32
# DROPOUT      = 0.1  # ty le client dropout moi vong 
# TIMING_REPS  = 2      # so lan chay roi lay trung binh

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

def _get_device() -> str:
    """Automatically select GPU if available, otherwise CPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"[bench_accuracy] GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"[bench_accuracy] Using device: {device}")
    return device

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(dataset_names: list[str] | None = None) -> None:
    names = dataset_names or list(DATASETS.keys())
    
    # Automatically select best device (GPU if available)
    device = _get_device()
    
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
                    print(f"    Round {rnd:3d}  loss={loss:.4f}  acc={acc:.4f}  f1={f1 if f1 is not None else 'NA'}  fl_t={fl_elapsed:.1f}s train_t={train_time:.2f}s")
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset name (mnist, cifar10, spam, webattack)")
    parser.add_argument("--seed", type=str, default="42", help="Seed for reproducibility (int or 'random') [default: 42]")
    args = parser.parse_args()
    global RUN_SEED
    RUN_SEED = args.seed
    if args.dataset:
        run([args.dataset])
    else:
        run()
