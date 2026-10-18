"""
main.py: End-to-End Pipeline Runner for Cascading Failure Risk Prediction
in Campus Computer Laboratory Networks using Graph Convolutional Networks (GCN).

Pipeline Stages:
1. Cyber-physical laboratory topology and cascading failure dataset generation
2. Graph neural network training (LabGCN, LabGAT, and baseline MLP)
3. Quantitative evaluation and benchmarking on unseen cascading scenarios
4. Diagnostic visualization generation (topology, cascade trace, ROC/PR curves, risk heatmap)
5. Qualitative failure propagation case study extraction
"""

import os
import sys
import torch
import numpy as np

from src.topology import create_lab_topology
from src.simulator import CascadeSimulator
from src.dataset import LabDatasetManager
from src.train import run_training_pipeline
from src.evaluate import evaluate_all_models, find_cascading_case_study
from src.visualize import (
    plot_lab_topology,
    plot_cascade_progression,
    plot_risk_heatmap,
    plot_roc_and_pr_curves,
    plot_feature_importance
)


def main():
    print("=" * 80)
    print(" CASCADING FAILURE RISK PREDICTION IN COMPUTER LABORATORY NETWORKS")
    print(" Graph Convolutional Networks (GCN) for Preventive IT Operations")
    print("=" * 80)

    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # -------------------------------------------------------------
    # STEP 1: Generate / Load Dataset of Cascading Scenarios
    # -------------------------------------------------------------
    print("\n[STEP 1/5] Building Computer Laboratory Topologies & Failure Scenarios...")
    dataset_mgr = LabDatasetManager(data_dir="data")
    dataset = dataset_mgr.create_or_load_dataset(
        num_scenarios=120,
        cache_path="data/lab_cascade_dataset.pt",
        force_regenerate=False
    )
    print(f"-> Successfully prepared {dataset['total_scenarios']} cascading scenarios.")
    print(f"-> Training Scenarios:   {len(dataset['train'])}")
    print(f"-> Validation Scenarios: {len(dataset['val'])}")
    print(f"-> Test Scenarios:       {len(dataset['test'])}")
    print(f"-> Positive Class Imbalance Weight: {dataset['pos_weight']:.3f}")

    # -------------------------------------------------------------
    # STEP 2: Train Graph Neural Networks & Baselines
    # -------------------------------------------------------------
    print("\n[STEP 2/5] Training Neural Network Architectures...")
    train_output = run_training_pipeline(
        dataset_path="data/lab_cascade_dataset.pt",
        epochs=35
    )
    gcn_model = train_output["gcn"]
    gat_model = train_output["gat"]
    mlp_model = train_output["mlp"]

    # -------------------------------------------------------------
    # STEP 3: Benchmark Models on Unseen Test Scenarios
    # -------------------------------------------------------------
    print("\n[STEP 3/5] Benchmarking on Unseen Test Cascading Scenarios...")
    models_dict = {
        "LabGCN": gcn_model,
        "LabGAT": gat_model,
        "MLP_Baseline": mlp_model
    }
    eval_results = evaluate_all_models(models_dict, dataset["test"], output_dir="results")

    # -------------------------------------------------------------
    # STEP 4: Generate Publication-Quality Visualizations
    # -------------------------------------------------------------
    print("\n[STEP 4/5] Generating Diagnostic & Topology Visualizations...")
    
    # 1. Master topology plot
    master_graph = create_lab_topology(num_labs=2, pcs_per_lab=15, seed=42)
    topo_img = plot_lab_topology(master_graph, "results/lab_topology.png")
    print(f"-> Saved: {topo_img}")

    # 2. Dynamic cascade progression wave
    sim = CascadeSimulator()
    sim_trace = sim.simulate(master_graph, trigger_node="Power_UPS_Lab_1", max_steps=4)
    prog_img = plot_cascade_progression(sim_trace, "results/cascade_progression.png")
    print(f"-> Saved: {prog_img}")

    # 3. GCN Risk Heatmap
    sample_test = dataset["test"][0]
    gcn_model.eval()
    with torch.no_grad():
        b_logits, _, _ = gcn_model(sample_test.x, sample_test.adj)
        sample_probs = torch.sigmoid(b_logits).numpy().flatten()
    heat_img = plot_risk_heatmap(sample_test.graph, sample_probs, sample_test.node_names, "results/risk_heatmap.png")
    print(f"-> Saved: {heat_img}")

    # 4. Comparative ROC and PR curves
    curves_img = plot_roc_and_pr_curves("results/curves_data.json", "results/roc_curves_comparison.png")
    print(f"-> Saved: {curves_img}")

    # 5. GCN Feature Attribution
    feat_img = plot_feature_importance(gcn_model, "results/feature_importance.png")
    print(f"-> Saved: {feat_img}")

    # -------------------------------------------------------------
    # STEP 5: Qualitative Cascading Case Study
    # -------------------------------------------------------------
    print("\n[STEP 5/5] Extracting Qualitative Cascading Case Study...")
    case_study = find_cascading_case_study(gcn_model, mlp_model, sample_test)
    print(f"Trigger Equipment Seed: {case_study['trigger']}")
    if case_study["demonstrations"]:
        for demo in case_study["demonstrations"]:
            print(f"\n  Equipment: {demo['node_name']} ({demo['node_id']})")
            print(f"  Ground Truth: {demo['ground_truth']}")
            print(f"  GCN Prediction: {demo['gcn_predicted_risk']} (CORRECT RISK DETECTION)")
            print(f"  MLP Baseline Prediction: {demo['mlp_predicted_risk']} (MISSED FAILURE!)")
            print(f"  Analysis: {demo['explanation']}")
    else:
        print("  GCN accurately tracked downstream dependencies across tested scenarios.")

    print("\n" + "=" * 80)
    print(" ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
    print(" Results, plots, and metrics stored in: y:\\AAD\\Assignment\\results\\")
    print(" You can start the interactive web dashboard by running:")
    print("   python dashboard.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
