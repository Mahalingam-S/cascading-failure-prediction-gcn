"""
simulator.py: Physics-based Cascading Failure Simulator for Computer Laboratories.
Simulates:
1. Electrical dependency drop (loss of power shuts down downstream components)
2. Network uplink loss (isolates switches, servers, and student workstations)
3. Load redistribution & thermal stress (traffic rerouting induces thermal throttling/trip)
4. Error avalanche (retry storms and buffer overflows propagate along edges)
"""

import copy
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from src.topology import (
    NODE_TYPE_POWER,
    NODE_TYPE_SWITCH,
    NODE_TYPE_COMPUTER,
    NODE_TYPE_SERVICE,
    EDGE_POWER,
    EDGE_NETWORK,
    EDGE_SERVICE,
    graph_to_feature_matrix,
    create_lab_topology
)


class CascadeSimulator:
    """
    Simulates multi-step cascading failure propagation across cyber-physical lab graph.
    """
    def __init__(
        self,
        temp_trip_threshold: float = 80.0,
        load_trip_threshold: float = 95.0,
        error_trip_threshold: int = 150,
        alpha_thermal: float = 0.35,  # Temperature rise per unit load
        beta_cooling: float = 0.12,   # Passive ambient cooling rate
        ambient_temp: float = 24.0
    ):
        self.temp_trip_threshold = temp_trip_threshold
        self.load_trip_threshold = load_trip_threshold
        self.error_trip_threshold = error_trip_threshold
        self.alpha_thermal = alpha_thermal
        self.beta_cooling = beta_cooling
        self.ambient_temp = ambient_temp

    def simulate(
        self,
        G_initial: nx.DiGraph,
        trigger_node: Optional[str] = None,
        max_steps: int = 6,
        record_trace: bool = True
    ) -> Dict[str, Any]:
        """
        Executes a dynamic cascading failure simulation.
        
        Args:
            G_initial: NetworkX dependency graph.
            trigger_node: Node ID to inject failure into (if None, picks the most stressed node).
            max_steps: Maximum propagation steps.
            record_trace: Whether to save per-step graph snapshots.
            
        Returns:
            Dictionary containing simulation results, step history, and ground-truth failure risk labels.
        """
        G = copy.deepcopy(G_initial)
        nodes = list(G.nodes())
        num_nodes = len(nodes)
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        
        # State tracking: 0 = Safe, 1 = Stressed, 2 = Failed
        status = {n: "OPERATIONAL" for n in nodes}
        failure_step = {n: -1 for n in nodes}
        telemetry_history = []
        step_failed_counts = []
        
        # Determine trigger node if not specified
        if trigger_node is None:
            # Check if any node is flagged as stressed seed
            stressed_candidates = [n for n in nodes if G.nodes[n].get("is_stressed_seed", False)]
            if stressed_candidates:
                trigger_node = stressed_candidates[0]
            else:
                # Pick a critical infrastructure node (Switch or Power Unit)
                crit_nodes = [n for n in nodes if G.nodes[n]["type"] in (NODE_TYPE_POWER, NODE_TYPE_SWITCH)]
                trigger_node = np.random.choice(crit_nodes)
                
        # Inject primary failure at step 0
        status[trigger_node] = "FAILED"
        failure_step[trigger_node] = 0
        G.nodes[trigger_node]["baseline_temp"] = 92.0
        G.nodes[trigger_node]["baseline_load"] = 0.0
        G.nodes[trigger_node]["baseline_errors"] = 500
        G.nodes[trigger_node]["state"] = "FAILED"
        
        step_trace = []
        if record_trace:
            step_trace.append({
                "step": 0,
                "failed_nodes": [trigger_node],
                "newly_failed": [trigger_node],
                "active_count": num_nodes - 1
            })

        current_failed = set([trigger_node])
        
        for step in range(1, max_steps + 1):
            newly_failed = set()
            
            for u in nodes:
                if status[u] == "FAILED":
                    continue
                    
                u_data = G.nodes[u]
                
                # -------------------------------------------------------------
                # 1. Electrical Cascade:
                # If all power suppliers to node 'u' are FAILED, node 'u' loses power
                # -------------------------------------------------------------
                power_suppliers = [
                    src for src, dst, edata in G.in_edges(u, data=True)
                    if edata.get("edge_type") == EDGE_POWER
                ]
                if power_suppliers:
                    all_power_dead = all(status[src] == "FAILED" for src in power_suppliers)
                    if all_power_dead:
                        newly_failed.add(u)
                        continue
                        
                # -------------------------------------------------------------
                # 2. Network Isolation Cascade:
                # If all network uplinks to core/distribution switches are FAILED
                # -------------------------------------------------------------
                if u_data["type"] in (NODE_TYPE_COMPUTER, NODE_TYPE_SERVICE):
                    network_uplinks = [
                        src for src, dst, edata in G.in_edges(u, data=True)
                        if edata.get("edge_type") == EDGE_NETWORK
                    ]
                    if network_uplinks and all(status[src] == "FAILED" for src in network_uplinks):
                        # Node is network-isolated; for computers, this causes login failure / hung state
                        newly_failed.add(u)
                        continue
                elif u_data["type"] == NODE_TYPE_SWITCH:
                    # Switch fails if its upstream uplink fails
                    switch_uplinks = [
                        src for src, dst, edata in G.in_edges(u, data=True)
                        if edata.get("edge_type") == EDGE_NETWORK and G.nodes[src]["criticality"] > u_data["criticality"]
                    ]
                    if switch_uplinks and all(status[src] == "FAILED" for src in switch_uplinks):
                        newly_failed.add(u)
                        continue
                        
                # -------------------------------------------------------------
                # 3. Load Redistribution & Thermal Avalanche:
                # Surviving neighbors absorb traffic from failed peer switches
                # -------------------------------------------------------------
                failed_neighbors = [
                    v for v in G.neighbors(u) if status[v] == "FAILED"
                ]
                if failed_neighbors and u_data["type"] in (NODE_TYPE_SWITCH, NODE_TYPE_SERVICE):
                    # Load spike from rerouted traffic & retry storm
                    surge = len(failed_neighbors) * 22.0
                    u_data["baseline_load"] = min(100.0, u_data["baseline_load"] + surge)
                    
                    # Thermal progression
                    delta_t = self.alpha_thermal * (u_data["baseline_load"] / 10.0) - self.beta_cooling * (u_data["baseline_temp"] - self.ambient_temp)
                    u_data["baseline_temp"] += delta_t
                    u_data["baseline_errors"] += len(failed_neighbors) * 35
                    u_data["baseline_fan"] = 1.0
                    
                    # Thermal trip check
                    if u_data["baseline_temp"] >= self.temp_trip_threshold or u_data["baseline_errors"] >= self.error_trip_threshold:
                        newly_failed.add(u)
                        continue
                        
            # Apply state updates for this time step
            if not newly_failed:
                # Cascade has stopped / stabilized
                break
                
            for nf in newly_failed:
                status[nf] = "FAILED"
                failure_step[nf] = step
                G.nodes[nf]["state"] = "FAILED"
                current_failed.add(nf)
                
            if record_trace:
                step_trace.append({
                    "step": step,
                    "failed_nodes": list(current_failed),
                    "newly_failed": list(newly_failed),
                    "active_count": num_nodes - len(current_failed)
                })

        # -------------------------------------------------------------
        # 4. COMPUTE GROUND-TRUTH LABELS & CONTINUOUS RISK SCORES
        # -------------------------------------------------------------
        # Label definition:
        # Binary target: 1 = Node failed during cascade, 0 = Node stayed operational
        # Risk score r in [0, 1]:
        #  - Direct failed nodes: r = 1.0
        #  - 1 hop away from failure with elevated temp/errors: r = 0.65 - 0.85 (Critical Warning)
        #  - Safe unaffected nodes: r = 0.05 - 0.25 (Normal)
        binary_labels = np.zeros(num_nodes, dtype=np.int64)
        risk_scores = np.zeros(num_nodes, dtype=np.float32)
        severity_classes = np.zeros(num_nodes, dtype=np.int64)
        
        for i, n in enumerate(nodes):
            if status[n] == "FAILED":
                binary_labels[i] = 1
                risk_scores[i] = 1.0
                severity_classes[i] = 2  # Critical / Failed
            else:
                # Check proximity to failed nodes
                neighbors = list(G.neighbors(n)) + list(G.predecessors(n))
                has_failed_nbr = any(status[nbr] == "FAILED" for nbr in neighbors)
                
                temp = G.nodes[n]["baseline_temp"]
                errors = G.nodes[n]["baseline_errors"]
                load = G.nodes[n]["baseline_load"]
                
                if has_failed_nbr or temp > 60.0 or errors > 40 or load > 80.0:
                    binary_labels[i] = 1  # At severe risk of imminent cascade
                    risk_scores[i] = min(0.95, 0.55 + (temp - 40.0) / 100.0 + (errors / 400.0))
                    severity_classes[i] = 1  # Warning / At Risk
                else:
                    binary_labels[i] = 0
                    risk_scores[i] = max(0.02, 0.05 + (temp - 30.0) / 200.0)
                    severity_classes[i] = 0  # Normal / Safe

        # Re-extract final telemetry features
        X_final, _ = graph_to_feature_matrix(G)
        
        return {
            "graph": G,
            "nodes": nodes,
            "trigger_node": trigger_node,
            "total_steps": len(step_trace) if record_trace else step,
            "total_failed": len(current_failed),
            "binary_labels": binary_labels,
            "risk_scores": risk_scores,
            "severity_classes": severity_classes,
            "failure_step": failure_step,
            "status": status,
            "step_trace": step_trace,
            "features": X_final
        }


def generate_cascade_dataset(
    num_scenarios: int = 150,
    num_labs_range: Tuple[int, int] = (2, 3),
    pcs_range: Tuple[int, int] = (15, 25),
    seed: int = 100
) -> List[Dict[str, Any]]:
    """
    Generates a collection of diverse cascade failure scenarios for machine learning training.
    """
    rng = np.random.RandomState(seed)
    simulator = CascadeSimulator()
    scenarios = []
    
    for s_idx in range(num_scenarios):
        sc_seed = seed + s_idx
        num_labs = rng.randint(num_labs_range[0], num_labs_range[1] + 1)
        pcs_per_lab = rng.randint(pcs_range[0], pcs_range[1] + 1)
        
        # Create base lab graph
        G = create_lab_topology(
            num_labs=num_labs,
            pcs_per_lab=pcs_per_lab,
            seed=sc_seed,
            stress_seed_node=True
        )
        
        # Pick a trigger node (variety of Power, Core Switch, Dist Switch, Access Switch)
        sim_res = simulator.simulate(G, trigger_node=None, max_steps=5, record_trace=False)
        sim_res["scenario_id"] = s_idx
        scenarios.append(sim_res)
        
    return scenarios


if __name__ == "__main__":
    from src.topology import create_lab_topology
    g = create_lab_topology(num_labs=2, pcs_per_lab=15, seed=42, stress_seed_node=True)
    sim = CascadeSimulator()
    res = sim.simulate(g, trigger_node="Power_UPS_Lab_1", max_steps=4)
    print(f"Simulation Complete:")
    print(f"  Trigger: {res['trigger_node']}")
    print(f"  Total Steps: {res['total_steps']}")
    print(f"  Total Failed: {res['total_failed']} / {len(res['nodes'])}")
    for trace in res["step_trace"]:
        print(f"  Step {trace['step']}: {len(trace['newly_failed'])} newly failed (Active: {trace['active_count']})")
    print(f"  Class balance: {np.bincount(res['binary_labels'])}")
