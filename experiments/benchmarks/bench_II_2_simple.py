"""
experiments/benchmarks/bench_II_2_simple.py
============================================
Simplified Benchmark II.2 — FL accuracy scaling WITHOUT SecAgg overhead.

Only measures accuracy/convergence with varying client counts,
using standard FedAvg (no secure aggregation).

This is a quick test to verify the benchmark framework works.
Real benchmark with SecAgg can be added later.
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

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

OUT_SUMMARY = RESULTS_DIR / "bench_II_2_simple_summary.csv"
OUT_PER_ROUND = RESULTS_DIR / "bench_II_2_simple_per_round.csv"

CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
OUT_CHECKPOINT = CHECKPOINT_DIR / "bench_II_2_simple_progress.json"


# ============================================================================
# Configuration
# ============================================================================

CLIENT_COUNTS = [5, 10, 20, 50, 100]
DROPOUT_RATES = [0.0, 0.05, 0.1, 0.3, 0.5]
LOCAL_EPOCHS = [1, 3]

N_ROUNDS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.01
DATASETS = ["mnist"]  # Start with MNIST only


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
        print(f"[bench_II_2_simple] Loaded checkpoint: {len(completed)} configs completed")
        return completed, summary_rows, per_round_rows
    except Exception as e:
        print(f"[bench_II_2_simple] Warning: Failed to load checkpoint: {e}")
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
        print(f"[bench_II_2_simple] Warning: Failed to save checkpoint: {e}")


def save_outputs(summary_rows: list, per_round_rows: list) -> None:
    """Persist CSV outputs."""
    summary_fields = [
        "dataset", "n_clients", "local_epochs", "n_rounds", "dropout_rate",
        "final_accuracy", "final_loss", "best_accuracy", "best_loss", "best_round",
        "convergence_time_sec", "total_time_sec",
    ]
    per_round_fields = [
        "dataset", "n_clients", "local_epochs", "dropout_rate", "round", "loss", "accuracy",
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
# Main FL training (simple FedAvg without SecAgg)
# ============================================================================

def fedavg_training(
    dataset: str,
    n_clients: int,
    local_epochs: int,
    dropout_rate: float,
) -> tuple:
    """
    Run FedAvg training.
    
    Returns: (final_accuracy, final_loss, best_accuracy, best_loss, best_round,
              convergence_time_sec, total_time_sec, per_round_metrics)
    """
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(
        f"[bench_II_2_simple] Running: dataset={dataset}, n_clients={n_clients}, "
        f"local_epochs={local_epochs}, dropout={dropout_rate:.0%}",
        flush=True,
    )
    
    # Load dataset
    train_loaders, val_loader, model_fn, loss_fn = load_dataset(dataset, n_clients)
    
    # Initialize global model
    global_model = model_fn().to(device)
    
    # Training loop
    start_time = time.time()
    per_round_metrics = []
    best_accuracy = 0.0
    best_loss = float('inf')
    best_round = 0
    convergence_time = 0.0
    
    for round_idx in range(N_ROUNDS):
        # Simulate dropout
        alive_indices = [i for i in range(n_clients) if np.random.rand() > dropout_rate]
        
        if not alive_indices:
            print(f"  Round {round_idx + 1}: All clients dropped out, skipping")
            continue
        
        # Local training
        client_updates = []
        for idx in alive_indices:
            local_model = deepcopy(global_model)
            opt = torch.optim.SGD(local_model.parameters(), lr=LEARNING_RATE)
            local_model.train()
            
            for _ in range(local_epochs):
                for X, y in train_loaders[idx]:
                    X, y = X.to(device), y.to(device)
                    opt.zero_grad()
                    loss = loss_fn(local_model(X), y)
                    loss.backward()
                    opt.step()
            
            # Get model weights
            client_weights = torch.cat([p.data.flatten() for p in local_model.parameters()])
            client_updates.append(client_weights)
        
        # FedAvg: average client updates
        global_weights = torch.cat([p.data.flatten() for p in global_model.parameters()])
        avg_update = torch.stack(client_updates).mean(dim=0)
        
        # Update global model
        offset = 0
        for p in global_model.parameters():
            numel = p.numel()
            p.data.copy_(avg_update[offset: offset + numel].view(p.shape))
            offset += numel
        
        # Evaluate
        global_model.eval()
        total_loss, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = global_model(X)
                total_loss += loss_fn(logits, y).item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                n += len(y)
        
        loss = total_loss / n
        accuracy = correct / n
        
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
        
        if (round_idx + 1) % 5 == 0:
            print(f"  Round {round_idx + 1}/{N_ROUNDS}: loss={loss:.4f}, acc={accuracy:.4f}")
    
    total_time = time.time() - start_time
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
        per_round_metrics,
    )


def main(reset_checkpoint: bool = False) -> None:
    """Run full benchmark."""
    if reset_checkpoint and OUT_CHECKPOINT.exists():
        OUT_CHECKPOINT.unlink()
        print(f"[bench_II_2_simple] Removed checkpoint: {OUT_CHECKPOINT}")
    
    completed, summary_rows, per_round_rows = load_checkpoint()
    
    # Generate all configurations
    configs = []
    for dataset in DATASETS:
        for n_clients in CLIENT_COUNTS:
            for local_epochs in LOCAL_EPOCHS:
                for dropout_rate in DROPOUT_RATES:
                    config_key = f"{dataset}_{n_clients}_{local_epochs}_{dropout_rate:.2f}"
                    configs.append({
                        "key": config_key,
                        "dataset": dataset,
                        "n_clients": n_clients,
                        "local_epochs": local_epochs,
                        "dropout_rate": dropout_rate,
                    })
    
    total_configs = len(configs)
    print(f"[bench_II_2_simple] Total configurations: {total_configs}")
    print(f"[bench_II_2_simple] Already completed: {len(completed)}")
    
    for idx, config in enumerate(configs):
        key = config["key"]
        
        if key in completed:
            print(f"[bench_II_2_simple] ({idx+1}/{total_configs}) Skipping (completed): {key}")
            continue
        
        print(f"[bench_II_2_simple] ({idx+1}/{total_configs}) Starting: {key}")
        
        try:
            result = fedavg_training(
                dataset=config["dataset"],
                n_clients=config["n_clients"],
                local_epochs=config["local_epochs"],
                dropout_rate=config["dropout_rate"],
            )
            
            (final_accuracy, final_loss, best_accuracy, best_loss, best_round,
             convergence_time, total_time, per_round_metrics) = result
            
            # Add summary row
            summary_rows.append({
                "dataset": config["dataset"],
                "n_clients": config["n_clients"],
                "local_epochs": config["local_epochs"],
                "n_rounds": N_ROUNDS,
                "dropout_rate": config["dropout_rate"],
                "final_accuracy": round(final_accuracy, 6),
                "final_loss": round(final_loss, 6),
                "best_accuracy": round(best_accuracy, 6),
                "best_loss": round(best_loss, 6),
                "best_round": best_round,
                "convergence_time_sec": round(convergence_time, 3),
                "total_time_sec": round(total_time, 3),
            })
            
            # Add per-round rows
            for metrics in per_round_metrics:
                per_round_rows.append({
                    "dataset": config["dataset"],
                    "n_clients": config["n_clients"],
                    "local_epochs": config["local_epochs"],
                    "dropout_rate": config["dropout_rate"],
                    "round": metrics["round"],
                    "loss": round(metrics["loss"], 6),
                    "accuracy": round(metrics["accuracy"], 6),
                })
            
            completed[key] = True
            save_checkpoint(completed, summary_rows, per_round_rows)
            save_outputs(summary_rows, per_round_rows)
            
        except Exception as e:
            print(f"[bench_II_2_simple] ERROR in config {key}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            save_checkpoint(completed, summary_rows, per_round_rows)
            save_outputs(summary_rows, per_round_rows)
            raise
    
    print(f"[bench_II_2_simple] All configurations completed!")
    print(f"[bench_II_2_simple] Summary    -> {OUT_SUMMARY}")
    print(f"[bench_II_2_simple] Per-round  -> {OUT_PER_ROUND}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simplified Benchmark II.2")
    parser.add_argument("--datasets", type=str, default="mnist", help="Datasets")
    parser.add_argument("--clients", type=str, default="5,10,20,50,100", help="Client counts")
    parser.add_argument("--local-epochs", type=str, default="1,3", help="Local epochs")
    parser.add_argument("--dropouts", type=str, default="0.0,0.05,0.1,0.3,0.5", help="Dropout rates")
    parser.add_argument("--n-rounds", type=int, default=20, help="Number of FL rounds")
    parser.add_argument("--reset-checkpoint", action="store_true", help="Reset checkpoint")
    
    args = parser.parse_args()
    
    # Parse arguments
    DATASETS = [s.strip() for s in args.datasets.split(",") if s.strip()]
    CLIENT_COUNTS = [int(s.strip()) for s in args.clients.split(",") if s.strip()]
    LOCAL_EPOCHS = [int(s.strip()) for s in args.local_epochs.split(",") if s.strip()]
    DROPOUT_RATES = [float(s.strip()) for s in args.dropouts.split(",") if s.strip()]
    N_ROUNDS = args.n_rounds
    
    print(
        f"[bench_II_2_simple] Config: datasets={DATASETS}, clients={CLIENT_COUNTS}, "
        f"local_epochs={LOCAL_EPOCHS}, dropouts={DROPOUT_RATES}, n_rounds={N_ROUNDS}"
    )
    
    configure_backend_environment(crypto_accel="cpu")
    main(reset_checkpoint=args.reset_checkpoint)
