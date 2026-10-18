from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.dataset import LabDatasetManager, LabGraphData
from src.models import LabGCN, LabGAT, MLPBaseline


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unpack_scenario(item):
    """Accept either a LabGraphData object or a dict with the same fields."""
    get = item.get if isinstance(item, dict) else lambda key, default=None: getattr(item, key, default)
    x = get("x")
    adj = get("adj")
    y_binary = get("y_binary")
    y_severity = get("y_severity")
    if any(v is None for v in (x, adj, y_binary, y_severity)):
        raise ValueError("Each scenario must provide x, adj, y_binary, and y_severity.")
    return x.float(), adj.float(), y_binary.float(), y_severity.long()


def model_forward(model, x, adj):
    output = model(x, adj)
    if not isinstance(output, (tuple, list)) or len(output) < 2:
        raise ValueError("Model forward must return (binary_logits, severity_logits, ...).")
    return output[0], output[1]


def compute_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Computes accuracy, F1, and ROC-AUC for binary cascade prediction."""
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred_prob = np.asarray(y_pred_prob).astype(float).reshape(-1)
    y_pred_binary = (y_pred_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    if len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_pred_prob))
        except Exception:
            auc = 0.5
    else:
        auc = 0.5

    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "roc_auc": float(auc)
    }


@torch.no_grad()
def evaluate(model, scenarios, device, pos_weight):
    model.eval()
    all_y, all_p = [], []
    total_loss, count = 0.0, 0
    pos_weight_tensor = torch.as_tensor(pos_weight, dtype=torch.float32, device=device).reshape(())
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    ce = nn.CrossEntropyLoss()

    for item in scenarios:
        x, adj, y, sev = unpack_scenario(item)
        x, adj, y, sev = x.to(device), adj.to(device), y.to(device), sev.to(device)
        logits, sev_logits = model_forward(model, x, adj)
        loss = bce(logits, y) + 0.5 * ce(sev_logits, sev)
        total_loss += float(loss.item())
        count += 1
        all_y.extend(y.cpu().numpy().flatten().tolist())
        all_p.extend(torch.sigmoid(logits).cpu().numpy().flatten().tolist())

    if not all_y:
        return {"loss": 0.0, "accuracy": 0.0, "f1": 0.0, "roc_auc": 0.5}

    metrics = compute_metrics(np.array(all_y), np.array(all_p))
    metrics["loss"] = total_loss / max(count, 1)
    return metrics


def train_model(
    model: nn.Module,
    model_name: str,
    train_data: List[Any],
    val_data: List[Any],
    pos_weight: float,
    epochs: int = 40,
    lr: float = 0.005,
    patience: int = 10,
    device: torch.device | None = None,
    save_dir: str = "models"
) -> Dict[str, Any]:
    """Trains a model with early stopping and returns training history and checkpoint info."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"{model_name.lower()}_best.pth")

    pos_weight_tensor = torch.as_tensor(pos_weight, dtype=torch.float32, device=device).reshape(())
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-5)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    ce = nn.CrossEntropyLoss()

    best_score = -float("inf")
    best_epoch = 0
    bad_epochs = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_roc_auc": []
    }

    print(f"\n=======================================================")
    print(f" Training {model_name} (Epochs: {epochs}, LR: {lr})")
    print(f"=======================================================")

    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(len(train_data))
        running_loss = 0.0
        n_batches = 0

        for idx in perm:
            x, adj, y, sev = unpack_scenario(train_data[int(idx)])
            x, adj, y, sev = x.to(device), adj.to(device), y.to(device), sev.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, sev_logits = model_forward(model, x, adj)
            loss = bce(logits, y) + 0.5 * ce(sev_logits, sev)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        avg_train_loss = running_loss / max(n_batches, 1)

        val_metrics = evaluate(model, val_data, device, pos_weight)
        score = val_metrics["f1"] + val_metrics["roc_auc"]

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_roc_auc"].append(val_metrics["roc_auc"])

        if (epoch % 5 == 0) or (epoch == 1) or (epoch == epochs):
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.3f} | F1: {val_metrics['f1']:.3f} | ROC-AUC: {val_metrics['roc_auc']:.3f}")

        if score > best_score:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[*] Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch}.")
                break

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print(f"[+] Loaded best {model_name} checkpoint from epoch {best_epoch}.")

    return {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "checkpoint_path": checkpoint_path,
        "history": history
    }


def run_training_pipeline(
    dataset_path: str = "data/lab_cascade_dataset.pt",
    epochs: int = 35
) -> Dict[str, Any]:
    """End-to-end training pipeline for LabGCN, LabGAT, and MLP Baseline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_mgr = LabDatasetManager(data_dir="data")
    dataset = dataset_mgr.create_or_load_dataset(cache_path=dataset_path)

    train_data = dataset["train"]
    val_data = dataset["val"]
    pos_weight = dataset["pos_weight"]
    num_features = dataset.get("num_features", 10)

    # 1. Initialize models
    gcn = LabGCN(in_features=num_features, hidden_dim=64, num_classes=3, dropout=0.25)
    gat = LabGAT(in_features=num_features, hidden_dim=64, num_classes=3, dropout=0.25)
    mlp = MLPBaseline(in_features=num_features, hidden_dim=64, num_classes=3, dropout=0.25)

    # 2. Train models
    gcn_res = train_model(gcn, "LabGCN", train_data, val_data, pos_weight, epochs=epochs, lr=0.006, device=device)
    gat_res = train_model(gat, "LabGAT", train_data, val_data, pos_weight, epochs=epochs, lr=0.005, device=device)
    mlp_res = train_model(mlp, "MLP_Baseline", train_data, val_data, pos_weight, epochs=epochs, lr=0.006, device=device)

    # 3. Save training histories
    training_summary = {
        "LabGCN": gcn_res["history"],
        "LabGAT": gat_res["history"],
        "MLP_Baseline": mlp_res["history"]
    }
    os.makedirs("results", exist_ok=True)
    with open("results/training_history.json", "w") as f:
        json.dump(training_summary, f, indent=2)

    print(f"\n[Training Complete] Models trained and checkpoints saved in models/.")
    return {
        "gcn": gcn,
        "gat": gat,
        "mlp": mlp,
        "results": {
            "LabGCN": gcn_res,
            "LabGAT": gat_res,
            "MLP_Baseline": mlp_res
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--cache", default="data/lab_cascade_dataset.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    run_training_pipeline(dataset_path=args.cache, epochs=args.epochs)


if __name__ == "__main__":
    main()
