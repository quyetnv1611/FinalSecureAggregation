"""
experiments/benchmarks/bench_II_2_scaling_clients.py
=====================================================
Benchmark II.2 — FL accuracy scaling with varying client counts.

Measures how accuracy and convergence behavior change as the number of clients
(participants) increases, with optional variation in local/global epochs.

Configurations
--------------
* Client counts:    5, 10, 20, 50, 100
* Local epochs:     1, 3 (default 1)
* Global epochs:    Configurable (default 20)
* Dropout rates:    0%, 5%, 10%, 30%, 50%
* Backends:         DH+ECDSA vs ML-KEM-768+ML-DSA-65

Datasets
--------
* MNIST   — Image classification, 10 classes
* CIFAR-10 — Image classification, 10 classes
* Spam    — Text classification, 2 classes

Models
------
* CNN     — For MNIST, CIFAR
* MLP     — For Spam, WebAttack

Output
------
``results/bench_II_2_scaling_clients_summary.csv``
    Aggregated accuracy/loss per configuration.

``results/bench_II_2_scaling_clients_per_round.csv``
    Per-round loss/accuracy for each configuration.
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from experiments.fl_simulator import FLSimulator, run_secagg_timing
from experiments.models.mnist_model import MnistCNN
from experiments.models.cifar_model import CifarCNN
from experiments.models.mlp_model import MLP
from secagg.crypto_backend import configure_backend_environment


def set_seed(seed: int = 42) -> None:
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

OUT_SUMMARY = RESULTS_DIR / "bench_II_2_scaling_clients_summary.csv"
OUT_PER_ROUND = RESULTS_DIR / "bench_II_2_scaling_clients_per_round.csv"

CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
OUT_CHECKPOINT = CHECKPOINT_DIR / "bench_II_2_progress.json"


# ============================================================================
# Configuration
# ============================================================================

# Sweep parameters
CLIENT_COUNTS = [5, 10, 20, 50, 100]
LOCAL_EPOCHS = [1, 3]
DROPOUT_RATES = [0.0, 0.05, 0.1, 0.3, 0.5]

# Default parameters
N_ROUNDS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.01

# Backends to test
BACKENDS = [
    {"algorithm": "original", "label": "Original (DH+ECDSA)", "kem": "DH", "sig": "classic"},
    {"algorithm": "pq", "label": "PQ (ML-KEM-768+ML-DSA-65)", "kem": "ML-KEM-768", "sig": "ML-DSA-65"},
]

# Datasets
DATASETS = ["mnist", "cifar10"]  # "spam" if needed


# ============================================================================
# Models
# ============================================================================

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
        self.fc = nn.Linear(7 * 7 * 64, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x


# ============================================================================
# Dataset loading
# ============================================================================

def load_dataset(dataset_name: str, n_clients: int):
    """Load dataset and return (train_loaders, val_loader, model_fn, loss_fn)."""
    if dataset_name == "mnist":
        from experiments.datasets.mnist_loader import load_mnist
        train_loaders, val_loader = load_mnist(n_clients=n_clients, batch_size=BATCH_SIZE)
        model_fn = CustomMnistCNN
        loss_fn = nn.CrossEntropyLoss()
    elif dataset_name == "cifar10":
        from experiments.datasets.cifar_loader import load_cifar10
        train_loaders, val_loader = load_cifar10(n_clients=n_clients, batch_size=BATCH_SIZE)
        model_fn = CifarCNN
        loss_fn = nn.CrossEntropyLoss()
    elif dataset_name == "spam":
        from experiments.datasets.spam_loader import load_spam
        train_loaders, val_loader, input_dim = load_spam(n_clients=n_clients, batch_size=BATCH_SIZE)
        model_fn = lambda: MLP(input_dim)
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_loaders, val_loader, model_fn, loss_fn


# ============================================================================
# Checkpoint management
# ============================================================================

def load_checkpoint() -> tuple[dict, list, list]:
    """Load checkpoint. Returns (completed_configs, summary_rows, per_round_rows)."""
    if not OUT_CHECKPOINT.exists():
        return {}, [], []
    
    try:
        with OUT_CHECKPOINT.open("r") as f:
            data = json.load(f)
        completed = data.get("completed", {})
        summary_rows = data.get("summary_rows", [])
        per_round_rows = data.get("per_round_rows", [])
        print(f"[bench_II_2] Loaded checkpoint: {len(completed)} configs completed")
        return completed, summary_rows, per_round_rows
    except Exception as e:
        print(f"[bench_II_2] Warning: Failed to load checkpoint: {e}")
        return {}, [], []


def save_checkpoint(completed: dict, summary_rows: list, per_round_rows: list) -> None:
    """Save checkpoint."""
    try:
        data = {
            "completed": completed,
            "summary_rows": summary_rows,
            "per_round_rows": per_round_rows,
            "timestamp": time.time(),
        }
        with OUT_CHECKPOINT.open("w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[bench_II_2] Warning: Failed to save checkpoint: {e}")


def save_outputs(summary_rows: list, per_round_rows: list) -> None:
    """Persist CSV outputs."""
    summary_fields = [
        "dataset", "n_clients", "local_epochs", "n_rounds", "dropout_rate",
        "backend", "backend_label",
        "final_accuracy", "final_loss", "best_accuracy", "best_loss", "best_round",
        "convergence_time_sec", "total_time_sec", "secagg_overhead_sec",
    ]
    per_round_fields = [
        "dataset", "n_clients", "local_epochs", "dropout_rate", "backend",
        "round", "loss", "accuracy",
    ]
    
    with OUT_SUMMARY.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    
    with OUT_PER_ROUND.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_round_fields)
        writer.writeheader()
        writer.writerows(per_round_rows)


# ============================================================================
# Main benchmark
# ============================================================================

def run_config(
    dataset: str,
    n_clients: int,
    local_epochs: int,
    dropout_rate: float,
    backend: dict,
) -> tuple[float, float, float, float, int, float, float]:
    """
    Run one FL training configuration.
    
    Returns: (final_accuracy, final_loss, best_accuracy, best_loss, best_round, 
              convergence_time_sec, total_time_sec)
    """
    set_seed(42)
    
    print(
        f"[bench_II_2] Running: dataset={dataset}, n_clients={n_clients}, "
        f"local_epochs={local_epochs}, dropout={dropout_rate:.0%}, "
        f"backend={backend['label']}",
        flush=True,
    )
    
    # Load dataset
    train_loaders, val_loader, model_fn, loss_fn = load_dataset(dataset, n_clients)
    
    # Create FL simulator
    simulator = FLSimulator(
        model_fn=model_fn,
        loss_fn=loss_fn,
        lr=LEARNING_RATE,
        n_local_epochs=local_epochs,
        kem_backend=backend["kem"],
        sig_backend=backend["sig"],
    )
    
    # Run training
    start_time = time.time()
    per_round_metrics = []
    best_accuracy = 0.0
    best_loss = float('inf')
    best_round = 0
    convergence_time = 0.0
    
    for round_idx in range(N_ROUNDS):
        # Train round with SecAgg
        simulator.train_round(
            train_loaders=train_loaders,
            dropout_rate=dropout_rate,
        )
        
        # Evaluate on validation set
        loss, accuracy = simulator.evaluate(val_loader)
        
        per_round_metrics.append({
            "round": round_idx + 1,
            "loss": loss,
            "accuracy": accuracy,
        })
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_loss = loss
            best_round = round_idx + 1
            convergence_time = time.time() - start_time
        
        print(f"  Round {round_idx + 1}/{N_ROUNDS}: loss={loss:.4f}, acc={accuracy:.4f}", flush=True)
    
    total_time = time.time() - start_time
    
    # Estimate SecAgg overhead (measure 10 clients, n_repeats=1)
    timer = run_secagg_timing(
        n_clients=min(n_clients, 10),
        grad_shape=(100000,),
        kem_backend=backend["kem"],
        sig_backend=backend["sig"],
        n_repeat=1,
        dropout_rate=0.0,
    )
    secagg_overhead = timer.total
    
    final_accuracy = per_round_metrics[-1]["accuracy"]
    final_loss = per_round_metrics[-1]["loss"]
    
    return (
        final_accuracy,
        final_loss,
        best_accuracy,
        best_loss,
        best_round,
        convergence_time,
        total_time,
        secagg_overhead,
        per_round_metrics,
    )


def main(reset_checkpoint: bool = False) -> None:
    """Run full benchmark."""
    if reset_checkpoint and OUT_CHECKPOINT.exists():
        OUT_CHECKPOINT.unlink()
        print(f"[bench_II_2] Removed checkpoint: {OUT_CHECKPOINT}")
    
    completed, summary_rows, per_round_rows = load_checkpoint()
    
    # Generate all configurations
    configs = []
    for dataset in DATASETS:
        for n_clients in CLIENT_COUNTS:
            for local_epochs in LOCAL_EPOCHS:
                for dropout_rate in DROPOUT_RATES:
                    for backend in BACKENDS:
                        config_key = f"{dataset}_{n_clients}_{local_epochs}_{dropout_rate:.2f}_{backend['algorithm']}"
                        configs.append({
                            "key": config_key,
                            "dataset": dataset,
                            "n_clients": n_clients,
                            "local_epochs": local_epochs,
                            "dropout_rate": dropout_rate,
                            "backend": backend,
                        })
    
    total_configs = len(configs)
    print(f"[bench_II_2] Total configurations: {total_configs}")
    print(f"[bench_II_2] Already completed: {len(completed)}")
    
    for idx, config in enumerate(configs):
        key = config["key"]
        
        if key in completed:
            print(f"[bench_II_2] ({idx+1}/{total_configs}) Skipping (completed): {key}")
            continue
        
        print(f"[bench_II_2] ({idx+1}/{total_configs}) Starting: {key}")
        
        try:
            result = run_config(
                dataset=config["dataset"],
                n_clients=config["n_clients"],
                local_epochs=config["local_epochs"],
                dropout_rate=config["dropout_rate"],
                backend=config["backend"],
            )
            
            (final_accuracy, final_loss, best_accuracy, best_loss, best_round,
             convergence_time, total_time, secagg_overhead, per_round_metrics) = result
            
            # Add summary row
            summary_rows.append({
                "dataset": config["dataset"],
                "n_clients": config["n_clients"],
                "local_epochs": config["local_epochs"],
                "n_rounds": N_ROUNDS,
                "dropout_rate": config["dropout_rate"],
                "backend": config["backend"]["algorithm"],
                "backend_label": config["backend"]["label"],
                "final_accuracy": round(final_accuracy, 6),
                "final_loss": round(final_loss, 6),
                "best_accuracy": round(best_accuracy, 6),
                "best_loss": round(best_loss, 6),
                "best_round": best_round,
                "convergence_time_sec": round(convergence_time, 3),
                "total_time_sec": round(total_time, 3),
                "secagg_overhead_sec": round(secagg_overhead, 3),
            })
            
            # Add per-round rows
            for metrics in per_round_metrics:
                per_round_rows.append({
                    "dataset": config["dataset"],
                    "n_clients": config["n_clients"],
                    "local_epochs": config["local_epochs"],
                    "dropout_rate": config["dropout_rate"],
                    "backend": config["backend"]["algorithm"],
                    "round": metrics["round"],
                    "loss": round(metrics["loss"], 6),
                    "accuracy": round(metrics["accuracy"], 6),
                })
            
            completed[key] = True
            save_checkpoint(completed, summary_rows, per_round_rows)
            save_outputs(summary_rows, per_round_rows)
            
        except Exception as e:
            print(f"[bench_II_2] ERROR in config {key}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            save_checkpoint(completed, summary_rows, per_round_rows)
            save_outputs(summary_rows, per_round_rows)
            raise
    
    print(f"[bench_II_2] All configurations completed!")
    print(f"[bench_II_2] Summary  -> {OUT_SUMMARY}")
    print(f"[bench_II_2] Per-round -> {OUT_PER_ROUND}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark II.2 - Scaling clients")
    parser.add_argument("--datasets", type=str, default="mnist,cifar10", help="Comma-separated datasets")
    parser.add_argument("--clients", type=str, default="5,10,20,50,100", help="Comma-separated client counts")
    parser.add_argument("--local-epochs", type=str, default="1,3", help="Comma-separated local epochs")
    parser.add_argument("--dropouts", type=str, default="0.0,0.05,0.1,0.3,0.5", help="Comma-separated dropout rates")
    parser.add_argument("--n-rounds", type=int, default=20, help="Number of FL rounds")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--reset-checkpoint", action="store_true", help="Reset checkpoint and restart")
    parser.add_argument("--crypto-accel", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    
    args = parser.parse_args()
    
    # Parse arguments
    DATASETS = [s.strip() for s in args.datasets.split(",") if s.strip()]
    CLIENT_COUNTS = [int(s.strip()) for s in args.clients.split(",") if s.strip()]
    LOCAL_EPOCHS = [int(s.strip()) for s in args.local_epochs.split(",") if s.strip()]
    DROPOUT_RATES = [float(s.strip()) for s in args.dropouts.split(",") if s.strip()]
    N_ROUNDS = args.n_rounds
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    
    configure_backend_environment(crypto_accel=args.crypto_accel)
    
    print(
        f"[bench_II_2] Config: datasets={DATASETS}, clients={CLIENT_COUNTS}, "
        f"local_epochs={LOCAL_EPOCHS}, dropouts={DROPOUT_RATES}, "
        f"n_rounds={N_ROUNDS}, lr={LEARNING_RATE}, batch_size={BATCH_SIZE}"
    )
    
    main(reset_checkpoint=args.reset_checkpoint)
