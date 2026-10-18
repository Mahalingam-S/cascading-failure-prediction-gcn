import os
import json
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from src.models import LabGCN, LabGAT, MLPBaseline
from src.dataset import LabGraphData


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def get_model_device(model: torch.nn.Module) -> torch.device:
    """Return the device on which the model parameters reside."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    """Calculate ROC-AUC when both target classes are present."""
    if len(np.unique(y_true)) < 2:
        return None

    return float(roc_auc_score(y_true, y_prob))


def calculate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Calculate classification metrics from binary labels and scores."""

    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob).astype(float).reshape(-1)

    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty test dataset.")

    if len(y_true) != len(y_prob):
        raise ValueError("Targets and prediction scores have different lengths.")

    if not np.isfinite(y_prob).all():
        raise ValueError("Prediction scores contain NaN or infinity.")

    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = safe_auc(y_true, y_prob)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "roc_auc": roc_auc,
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
    }


def calculate_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    """Calculate ROC and precision-recall curve coordinates."""

    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob).astype(float).reshape(-1)

    # ROC curve requires both positive and negative samples.
    if len(np.unique(y_true)) < 2:
        fpr, tpr = [], []
    else:
        fpr_values, tpr_values, _ = roc_curve(y_true, y_prob)
        fpr = fpr_values.tolist()
        tpr = tpr_values.tolist()

    precision_values, recall_values, _ = precision_recall_curve(
        y_true, y_prob
    )

    return {
        "fpr": fpr,
        "tpr": tpr,
        "precision_curve": precision_values.tolist(),
        "recall_curve": recall_values.tolist(),
        "y_true": y_true.tolist(),
        "y_prob": y_prob.tolist(),
    }


# ---------------------------------------------------------
# Rule-based baseline
# ---------------------------------------------------------

def degree_heuristic_predict(
    dataset: List[LabGraphData],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rule-based baseline combining:
    - In-degree and out-degree
    - Temperature feature (index 0)
    - Error feature (index 2)

    The output is a heuristic score, not a calibrated probability.
    Features are assumed to be in the same normalized format used
    by the dataset pipeline.
    """

    all_probs = []
    all_targets = []

    for item in dataset:
        adj = item.adj.detach().cpu().numpy()
        x = item.x.detach().cpu().numpy()
        targets = item.y_binary.detach().cpu().numpy().flatten()

        # Sum incoming and outgoing weighted connections.
        degrees = np.sum(adj, axis=0) + np.sum(adj, axis=1)

        max_degree = float(np.max(degrees)) if degrees.size else 0.0

        if max_degree > 0:
            deg_norm = degrees / (max_degree + 1e-5)
        else:
            deg_norm = np.zeros_like(degrees)

        # Numerically stable sigmoid.
        temp_values = np.clip(x[:, 0], -50, 50)
        error_values = np.clip(x[:, 2], -50, 50)

        temp_sig = 1.0 / (1.0 + np.exp(-temp_values))
        err_sig = 1.0 / (1.0 + np.exp(-error_values))

        # Weighted heuristic score.
        heuristic_score = (
            0.40 * deg_norm
            + 0.35 * temp_sig
            + 0.25 * err_sig
        )

        heuristic_score = np.clip(heuristic_score, 0.0, 1.0)

        all_probs.extend(heuristic_score.tolist())
        all_targets.extend(targets.tolist())

    return (
        np.asarray(all_targets, dtype=int),
        np.asarray(all_probs, dtype=float),
    )


# ---------------------------------------------------------
# Neural model prediction collection
# ---------------------------------------------------------

def collect_model_predictions(
    model: torch.nn.Module,
    dataset: List[LabGraphData],
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a neural model on test graphs.

    Expected model output:
        binary_logits, severity_logits, extra_output

    Returns:
        true binary labels,
        predicted binary probabilities,
        predicted severity classes.
    """

    if not dataset:
        raise ValueError("The test dataset is empty.")

    if device is None:
        device = get_model_device(model)

    model.eval()

    all_targets = []
    all_probs = []
    all_severities = []

    with torch.no_grad():
        for item in dataset:
            x = item.x.to(device)
            adj = item.adj.to(device)

            outputs = model(x, adj)

            if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
                raise ValueError(
                    "Expected model(x, adj) to return at least "
                    "(binary_logits, severity_logits)."
                )

            binary_logits = outputs[0]
            severity_logits = outputs[1]

            probs = (
                torch.sigmoid(binary_logits)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            severity_predictions = (
                torch.argmax(severity_logits, dim=-1)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            targets = (
                item.y_binary.detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            if len(targets) != len(probs):
                raise ValueError(
                    "Model prediction count does not match label count."
                )

            all_probs.extend(probs.tolist())
            all_severities.extend(severity_predictions.tolist())
            all_targets.extend(targets.tolist())

    return (
        np.asarray(all_targets, dtype=int),
        np.asarray(all_probs, dtype=float),
        np.asarray(all_severities, dtype=int),
    )


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

def evaluate_all_models(
    models_dict: Dict[str, torch.nn.Module],
    test_data: List[LabGraphData],
    output_dir: str = "results",
) -> Dict[str, Any]:
    """
    Benchmark neural models and the degree heuristic on test scenarios.

    Saves:
        model_comparison.json
        model_comparison.csv
        curves_data.json
    """

    if not test_data:
        raise ValueError("No test scenarios were provided.")

    os.makedirs(output_dir, exist_ok=True)

    results = {}
    curves_data = {}

    # 1. Evaluate neural models.
    for name, model in models_dict.items():
        print(f"[Evaluation] Evaluating {name}...")

        y_true, y_prob, _ = collect_model_predictions(
            model=model,
            dataset=test_data,
        )

        results[name] = calculate_metrics(y_true, y_prob)
        curves_data[name] = calculate_curves(y_true, y_prob)

    # 2. Evaluate rule-based heuristic.
    print("[Evaluation] Evaluating Degree_Heuristic...")

    heuristic_true, heuristic_scores = degree_heuristic_predict(
        test_data
    )

    results["Degree_Heuristic"] = calculate_metrics(
        heuristic_true,
        heuristic_scores,
    )

    curves_data["Degree_Heuristic"] = calculate_curves(
        heuristic_true,
        heuristic_scores,
    )

    # 3. Save JSON outputs.
    metrics_json_path = os.path.join(
        output_dir, "model_comparison.json"
    )

    curves_json_path = os.path.join(
        output_dir, "curves_data.json"
    )

    with open(metrics_json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, allow_nan=False)

    with open(curves_json_path, "w", encoding="utf-8") as file:
        json.dump(curves_data, file, indent=2, allow_nan=False)

    # 4. Build a readable summary table.
    records = []

    metric_columns = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1-Score"),
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
    ]

    for model_name, metrics in results.items():
        row = {"Model": model_name}

        for key, display_name in metric_columns:
            value = metrics[key]
            row[display_name] = (
                f"{value:.4f}" if value is not None else "N/A"
            )

        records.append(row)

    df_results = pd.DataFrame(records)

    csv_path = os.path.join(
        output_dir, "model_comparison.csv"
    )

    df_results.to_csv(csv_path, index=False)

    print("\n" + "=" * 78)
    print("MODEL BENCHMARK RESULTS ON UNSEEN TEST SCENARIOS")
    print("=" * 78)
    print(df_results.to_string(index=False))
    print("=" * 78)

    print(f"Metrics saved to: {metrics_json_path}")
    print(f"Curves saved to:  {curves_json_path}")
    print(f"CSV saved to:     {csv_path}")

    return {
        "summary": results,
        "dataframe": df_results,
        "curves": curves_data,
    }


# ---------------------------------------------------------
# Qualitative cascading-failure case study
# ---------------------------------------------------------

def find_cascading_case_study(
    gcn_model: torch.nn.Module,
    mlp_model: torch.nn.Module,
    test_scenario: LabGraphData,
) -> Dict[str, Any]:
    """
    Find nodes where:
    - The ground-truth binary label indicates risk.
    - GCN predicts high risk.
    - MLP predicts low risk.

    This identifies examples consistent with the hypothesis that
    graph structure can help identify cascading risk.

    It does not, by itself, prove that message passing caused the
    performance difference.
    """

    gcn_device = get_model_device(gcn_model)
    mlp_device = get_model_device(mlp_model)

    gcn_model.eval()
    mlp_model.eval()

    with torch.no_grad():
        gcn_outputs = gcn_model(
            test_scenario.x.to(gcn_device),
            test_scenario.adj.to(gcn_device),
        )

        mlp_outputs = mlp_model(
            test_scenario.x.to(mlp_device),
            test_scenario.adj.to(mlp_device),
        )

        gcn_logits = gcn_outputs[0]
        mlp_logits = mlp_outputs[0]

        gcn_probs = (
            torch.sigmoid(gcn_logits)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        mlp_probs = (
            torch.sigmoid(mlp_logits)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

    targets = (
        test_scenario.y_binary.detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    nodes = test_scenario.node_names

    if not (
        len(nodes) == len(targets)
        == len(gcn_probs) == len(mlp_probs)
    ):
        raise ValueError(
            "Node names, labels, and model predictions must have "
            "the same number of entries."
        )

    insights = []

    for index, node_name in enumerate(nodes):
        target = int(targets[index])
        gcn_probability = float(gcn_probs[index])
        mlp_probability = float(mlp_probs[index])

        # Positive target, high GCN score, low MLP score.
        if (
            target == 1
            and gcn_probability >= 0.55
            and mlp_probability < 0.45
        ):
            graph = test_scenario.graph

            if graph is not None and node_name in graph.nodes:
                node_data = graph.nodes[node_name]
            else:
                node_data = {}

            insights.append({
                "node_id": str(node_name),
                "node_name": str(
                    node_data.get("name", node_name)
                ),
                "type": str(
                    node_data.get("role", "device")
                ),
                "ground_truth": "Failed in Cascade",
                "gcn_predicted_risk": (
                    f"{gcn_probability * 100:.1f}%"
                ),
                "mlp_predicted_risk": (
                    f"{mlp_probability * 100:.1f}%"
                ),
                "explanation": (
                    f"Node {node_name} has a positive ground-truth "
                    f"risk label. The GCN assigned a risk score of "
                    f"{gcn_probability:.3f}, while the MLP assigned "
                    f"{mlp_probability:.3f}. This is a candidate "
                    f"example for investigating whether graph "
                    f"connectivity helps identify cascading risk. "
                    f"The trigger recorded for this scenario is "
                    f"{test_scenario.trigger_node}."
                ),
            })

    return {
        "trigger": test_scenario.trigger_node,
        "demonstrations": insights[:3],
        "num_matching_nodes": len(insights),
    }


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    print(
        "evaluate.py contains evaluation functions. "
        "Load trained models and call evaluate_all_models() "
        "from your training or evaluation script."
    )

