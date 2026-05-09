"""
Test script: Compare different model architectures on SMS Spam dataset.
Finds the best model for improving accuracy from 0.867145 to 0.9x%.
"""
import sys
import time
from pathlib import Path
import csv

import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from experiments.datasets.spam_loader import load_spam
from experiments.models.mlp_model import MLP
from experiments.models.spam_model import ImprovedMLP, DeepMLP, TextCNN


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(model, train_loaders, test_loader, n_rounds=20, learning_rate=0.01, device="cpu"):
    """Train a model on spam dataset and return final accuracy."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"    Training for {n_rounds} rounds (lr={learning_rate})...")
    
    best_accuracy = 0.0
    best_round = 0
    accuracies = []
    
    for round_idx in range(1, n_rounds + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        total_samples = 0
        
        for client_loader in train_loaders:
            for batch_x, batch_y in client_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(batch_y)
                total_samples += len(batch_y)
        
        avg_train_loss = train_loss / max(total_samples, 1)
        
        # Evaluation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.to(device)
                
                logits = model(batch_x)
                _, predicted = torch.max(logits, 1)
                correct += (predicted == batch_y).sum().item()
                total += len(batch_y)
        
        accuracy = correct / total if total > 0 else 0.0
        accuracies.append(accuracy)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_round = round_idx
        
        if round_idx % 5 == 0 or round_idx == 1:
            print(f"      Round {round_idx:2d}: loss={avg_train_loss:.6f}, accuracy={accuracy:.6f}")
    
    print(f"    Final: accuracy={accuracies[-1]:.6f}, best={best_accuracy:.6f} (round {best_round})")
    return {
        "final_accuracy": accuracies[-1],
        "best_accuracy": best_accuracy,
        "best_round": best_round,
        "accuracies": accuracies,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Load dataset
    print("[spam_model_test] Loading SMS Spam dataset (50 clients)...")
    set_seed(42)
    train_loaders, test_loader, input_dim = load_spam(
        n_clients=50,
        batch_size=64,
        n_features=5000,
        test_ratio=0.2,
        seed=42,
    )
    print(f"  Input dim: {input_dim}, Test set size: {len(test_loader.dataset)}\n")
    
    # Models to test
    models = {
        "MLP (Original)": MLP(input_dim, hidden_dim=256, n_classes=2),
        "ImprovedMLP": ImprovedMLP(input_dim),
        "DeepMLP": DeepMLP(input_dim),
        "TextCNN": TextCNN(input_dim),
    }
    
    results = {}
    output_csv = ROOT / "results" / "spam_model_comparison.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("MODEL ACCURACY COMPARISON ON SMS SPAM DATASET")
    print("=" * 70 + "\n")
    
    for model_name, model in models.items():
        print(f"[{model_name}]")
        set_seed(42)
        result = train_model(
            model,
            train_loaders,
            test_loader,
            n_rounds=20,
            learning_rate=0.01,
            device=device,
        )
        results[model_name] = result
        print()
    
    # Save results
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<25} {'Final Acc':<12} {'Best Acc':<12} {'Best Round':<12}")
    print("-" * 65)
    
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "final_accuracy", "best_accuracy", "best_round"])
        
        for model_name in sorted(results.keys(), key=lambda x: results[x]["final_accuracy"], reverse=True):
            result = results[model_name]
            final_acc = result["final_accuracy"]
            best_acc = result["best_accuracy"]
            best_round = result["best_round"]
            
            print(f"{model_name:<25} {final_acc:.6f}      {best_acc:.6f}      {best_round:<12}")
            writer.writerow([model_name, final_acc, best_acc, best_round])
    
    print(f"\nResults saved to: {output_csv}")
    
    # Find best model
    best_model = max(results.keys(), key=lambda x: results[x]["final_accuracy"])
    best_acc = results[best_model]["final_accuracy"]
    print(f"\n✓ BEST MODEL: {best_model} with accuracy {best_acc:.6f}")
    
    if best_acc > 0.90:
        print(f"✓ TARGET MET: accuracy {best_acc:.6f} > 0.90 ✓")
    else:
        print(f"⚠ Target not met: {best_acc:.6f} < 0.90 (gap: {0.90 - best_acc:.6f})")


if __name__ == "__main__":
    main()
