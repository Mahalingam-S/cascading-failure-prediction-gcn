from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.topology import (
    NODE_TYPE_POWER, NODE_TYPE_SWITCH, NODE_TYPE_COMPUTER, NODE_TYPE_SERVICE,
    EDGE_POWER, EDGE_NETWORK, EDGE_SERVICE
)

TYPE_COLORS = {
    NODE_TYPE_POWER: "#e6ab02",
    NODE_TYPE_SWITCH: "#1b9e77",
    NODE_TYPE_COMPUTER: "#7570b3",
    NODE_TYPE_SERVICE: "#d95f02",
}


def ensure_parent(path):
    parent = os.path.dirname(os.fspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def plot_lab_topology(graph: nx.DiGraph, output_path="results/lab_topology.png"):
    ensure_parent(output_path)
    pos = nx.spring_layout(graph, seed=42, k=0.8 / np.sqrt(max(graph.number_of_nodes(), 1)))
    node_colors = [TYPE_COLORS.get(graph.nodes[n].get("type"), "#999999") for n in graph.nodes]
    fig, ax = plt.subplots(figsize=(15, 10))
    edge_styles = {
        EDGE_POWER: ("#d73027", "solid"),
        EDGE_NETWORK: ("#4575b4", "solid"),
        EDGE_SERVICE: ("#666666", "dashed"),
    }
    for edge_type, (color, style) in edge_styles.items():
        edges = [(u, v) for u, v, d in graph.edges(data=True)
                 if d.get("edge_type") == edge_type]
        nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color=color,
                               style=style, alpha=0.45, arrows=False, ax=ax)
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=110, ax=ax)
    labels = {n: str(n).replace("Power_", "P_").replace("Switch_", "SW_")
              for n in graph.nodes}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=6, ax=ax)
    ax.set_title("Computer Laboratory Dependency Topology")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_cascade_progression(sim_result: dict, output_path="results/cascade_progression.png"):
    ensure_parent(output_path)
    trace = sim_result.get("step_trace", [])
    if not trace:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No step trace recorded for this simulation",
                ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return output_path
    steps = [entry.get("step", i) for i, entry in enumerate(trace)]
    newly = [len(entry.get("newly_failed", [])) for entry in trace]
    cumulative = [len(entry.get("failed_nodes", [])) for entry in trace]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(steps, newly, label="New failures")
    ax.plot(steps, cumulative, marker="o", label="Cumulative failures")
    ax.set(xlabel="Simulation step", ylabel="Number of nodes",
           title=f"Cascade progression (trigger: {sim_result.get('trigger_node', 'unknown')})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_risk_heatmap(graph: nx.Graph, risk_probs, node_names=None,
                      output_path="results/risk_heatmap.png"):
    ensure_parent(output_path)
    risk_probs = np.asarray(risk_probs, dtype=float).reshape(-1)
    nodes = list(graph.nodes())
    if len(nodes) != len(risk_probs):
        raise ValueError("risk_probs length must match graph node count/order.")
    pos = nx.spring_layout(graph, seed=42)
    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx_edges(graph, pos, alpha=0.2, ax=ax)
    nx.draw_networkx_nodes(graph, pos, node_color=risk_probs, cmap=plt.cm.Reds,
                           vmin=0, vmax=1, node_size=230, ax=ax)
    labels = {}
    for i, n in enumerate(nodes):
        label = node_names[i] if node_names is not None else str(n)
        if risk_probs[i] >= 0.5:
            labels[n] = f"{label}\n{risk_probs[i]:.2f}"
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7, ax=ax)
    ax.set_title("Predicted Cascading-Failure Risk")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_roc_and_pr_curves(curves_data_path="results/curves_data.json",
                           output_path="results/model_curves.png"):
    ensure_parent(output_path)
    with open(curves_data_path, encoding="utf-8") as f:
        data = json.load(f)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
    plotted = False
    for name, item in data.items():
        y = np.asarray(item["y_true"], dtype=int)
        p = np.asarray(item["y_prob"], dtype=float)
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
        ax_pr.plot(recall, precision, label=f"{name} (AP={average_precision_score(y, p):.3f})")
        plotted = True
    ax_roc.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall curves")
    if plotted:
        ax_roc.legend()
        ax_pr.legend()
    else:
        ax_roc.text(0.5, 0.5, "No curves available (test set may have one class)", ha="center")
        ax_pr.text(0.5, 0.5, "No curves available (test set may have one class)", ha="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_feature_importance(model, feature_names=None, output_path="results/feature_importance.png"):
    """Plot first-layer weight magnitudes as a feature importance diagnostic."""
    if isinstance(feature_names, str):
        output_path = feature_names
        feature_names = None

    ensure_parent(output_path)
    layer = getattr(model, "gcn1", None)
    if layer is None:
        raise ValueError("Expected a LabGCN model with gcn1 layer.")

    if hasattr(layer, "weight"):
        weights = layer.weight.detach().cpu().numpy()
        importance = np.abs(weights).mean(axis=1)
    elif hasattr(layer, "linear"):
        weights = layer.linear.weight.detach().cpu().numpy()
        importance = np.abs(weights).mean(axis=0)
    else:
        raise ValueError("Expected gcn1 to have weight or linear attribute.")

    try:
        from src.topology import FEATURE_NAMES
        names = feature_names or FEATURE_NAMES
    except Exception:
        names = feature_names or [f"feature_{i}" for i in range(len(importance))]

    if len(names) != len(importance):
        names = [f"feature_{i}" for i in range(len(importance))]

    order = np.argsort(importance)
    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))
    ax.barh(np.asarray(names)[order], importance[order])
    ax.set(xlabel="Mean absolute first-layer weight", title="GCN Input Feature Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", default="results/curves_data.json")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.exists(args.curves):
        plot_roc_and_pr_curves(args.curves, os.path.join(args.output_dir, "model_curves.png"))
        print("Saved model curves.")
    else:
        print(f"Curve file not found: {args.curves}; run evaluate.py first.")


if __name__ == "__main__":
    main()
