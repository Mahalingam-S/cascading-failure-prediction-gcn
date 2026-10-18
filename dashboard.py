"""
dashboard.py: Interactive Flask Web Dashboard for University IT Teams.
Features:
- Live cyber-physical laboratory dependency graph
- Interactive failure injection and step-by-step cascade animation
- Real-time GCN risk prediction and telemetry inspector
- Preventive maintenance recommendations
"""

import os
import json
import torch
import numpy as np
import networkx as nx
from flask import Flask, render_template, jsonify, request, send_from_directory

from src.topology import (
    create_lab_topology,
    graph_to_feature_matrix,
    NODE_TYPE_NAMES,
    FEATURE_NAMES
)
from src.simulator import CascadeSimulator
from src.models import LabGCN
from src.dataset import LabDatasetManager

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Global state for interactive session
MASTER_GRAPH = None
TRAINED_GCN = None
DATASET_MGR = None


def load_resources():
    global MASTER_GRAPH, TRAINED_GCN, DATASET_MGR
    print("[Dashboard] Initializing master lab topology and models...")
    MASTER_GRAPH = create_lab_topology(num_labs=2, pcs_per_lab=15, seed=42, stress_seed_node=True)
    DATASET_MGR = LabDatasetManager(data_dir="data")
    
    # Try to load dataset normalizer
    if os.path.exists("data/lab_cascade_dataset.pt"):
        cached = torch.load("data/lab_cascade_dataset.pt", weights_only=False)
        DATASET_MGR.mean = cached["mean"]
        DATASET_MGR.std = cached["std"]
    else:
        print("[Dashboard] Dataset cache not found. Generating baseline normalizer...")
        DATASET_MGR.create_or_load_dataset(num_scenarios=15, cache_path="data/lab_cascade_dataset.pt")
        
    # Load trained GCN if exists
    TRAINED_GCN = LabGCN(in_features=10, hidden_dim=64, num_classes=3)
    weights_path = "models/labgcn_best.pth"
    if os.path.exists(weights_path):
        TRAINED_GCN.load_state_dict(torch.load(weights_path, weights_only=True))
        print(f"[Dashboard] Loaded trained GCN weights from {weights_path}")
    TRAINED_GCN.eval()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/topology")
def get_topology():
    """
    Returns graph nodes and edges with current telemetry for D3/SVG frontend rendering.
    """
    nodes = []
    edges = []
    
    for n, d in MASTER_GRAPH.nodes(data=True):
        nodes.append({
            "id": n,
            "name": d.get("name", n),
            "type": d["type"],
            "typeName": NODE_TYPE_NAMES.get(d["type"], "Unknown"),
            "role": d.get("role", ""),
            "room": d.get("room", ""),
            "temperature": round(float(d.get("baseline_temp", 40.0)), 1),
            "uptime": round(float(d.get("baseline_uptime", 100.0)), 1),
            "errors": int(d.get("baseline_errors", 0)),
            "load": round(float(d.get("baseline_load", 30.0)), 1),
            "power": round(float(d.get("baseline_power", 150.0)), 1),
            "criticality": float(d.get("criticality", 0.5)),
            "state": d.get("state", "OPERATIONAL")
        })
        
    for u, v, d in MASTER_GRAPH.edges(data=True):
        edges.append({
            "source": u,
            "target": v,
            "edgeType": d.get("edge_type", "network"),
            "weight": d.get("weight", 1.0)
        })
        
    return jsonify({
        "nodes": nodes,
        "edges": edges,
        "totalNodes": len(nodes),
        "totalEdges": len(edges)
    })


@app.route("/api/simulate_cascade", methods=["POST"])
def simulate_cascade():
    """
    Simulates cascading failure starting from a user-specified trigger node.
    """
    data = request.json or {}
    trigger_node = data.get("trigger_node", "Power_UPS_Lab_1")
    
    sim = CascadeSimulator()
    result = sim.simulate(MASTER_GRAPH, trigger_node=trigger_node, max_steps=5, record_trace=True)
    
    # Format response
    trace_steps = []
    for t in result["step_trace"]:
        trace_steps.append({
            "step": t["step"],
            "failed_nodes": t["failed_nodes"],
            "newly_failed": t["newly_failed"],
            "active_count": t["active_count"]
        })
        
    return jsonify({
        "trigger_node": result["trigger_node"],
        "total_steps": result["total_steps"],
        "total_failed": result["total_failed"],
        "step_trace": trace_steps,
        "final_status": result["status"]
    })


@app.route("/api/gcn_predict", methods=["GET", "POST"])
def gcn_predict():
    """
    Runs GCN inference on the current graph topology to predict failure risk scores.
    """
    nodes = list(MASTER_GRAPH.nodes())
    X_raw, _ = graph_to_feature_matrix(MASTER_GRAPH)
    
    # Transform
    X_norm = DATASET_MGR.transform(X_raw)
    x_tensor = torch.tensor(X_norm, dtype=torch.float32)
    
    # Adjacency
    adj_np = nx.to_numpy_array(MASTER_GRAPH, nodelist=nodes, dtype=np.float32)
    adj_tensor = torch.tensor(adj_np, dtype=torch.float32)
    
    TRAINED_GCN.eval()
    with torch.no_grad():
        b_logits, s_logits, _ = TRAINED_GCN(x_tensor, adj_tensor)
        probs = torch.sigmoid(b_logits).numpy().flatten()
        severities = torch.argmax(s_logits, dim=-1).numpy().flatten()
        
    predictions = {}
    high_risk_list = []
    
    for i, n in enumerate(nodes):
        prob = float(probs[i])
        sev = int(severities[i])
        predictions[n] = {
            "risk_score": round(prob, 4),
            "risk_percent": round(prob * 100, 1),
            "severity": "Critical" if sev == 2 or prob > 0.7 else ("Warning" if sev == 1 or prob > 0.4 else "Normal"),
            "status": "AT_RISK" if prob > 0.5 else "SAFE"
        }
        if prob > 0.4:
            high_risk_list.append({
                "id": n,
                "name": MASTER_GRAPH.nodes[n].get("name", n),
                "role": MASTER_GRAPH.nodes[n].get("role", ""),
                "risk_percent": round(prob * 100, 1),
                "temp": round(float(MASTER_GRAPH.nodes[n].get("baseline_temp", 40.0)), 1),
                "errors": int(MASTER_GRAPH.nodes[n].get("baseline_errors", 0))
            })
            
    high_risk_list.sort(key=lambda x: x["risk_percent"], reverse=True)
    
    # Generate preventive maintenance advice
    recommendations = []
    for hr in high_risk_list[:4]:
        rec_text = ""
        if "Power" in hr["id"] or "UPS" in hr["id"]:
            rec_text = f"Priority Action: Inspect battery cooling for {hr['name']}. Temp is {hr['temp']}°C. Prepare backup bypass breaker."
        elif "Switch" in hr["id"]:
            rec_text = f"Priority Action: Port buffer overflow on {hr['name']}. Reroute VLAN traffic to redundant trunk to prevent row drop."
        else:
            rec_text = f"Notice: Workstation {hr['name']} sits downstream of unstable uplink. Isolate session state to prevent corruption."
        recommendations.append(rec_text)

    return jsonify({
        "predictions": predictions,
        "high_risk_equipment": high_risk_list,
        "recommendations": recommendations,
        "total_at_risk": len(high_risk_list)
    })


@app.route("/api/metrics")
def get_metrics():
    """
    Returns benchmark comparison metrics between GCN, GAT, and Baselines.
    """
    metrics_path = "results/model_comparison.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            data = json.load(f)
        return jsonify(data)
    else:
        return jsonify({"message": "Metrics not yet generated. Run main.py first."})


@app.route("/results/<path:filename>")
def serve_results(filename):
    """
    Serves generated chart images.
    """
    return send_from_directory("results", filename)


if __name__ == "__main__":
    load_resources()
    app.run(host="127.0.0.1", port=5000, debug=False)
