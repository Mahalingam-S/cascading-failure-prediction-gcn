"""
topology.py: Generates realistic University Computer Laboratory Topologies.
Models cyber-physical dependencies:
- Power distribution hierarchy (Main PDU -> UPS -> ePDU -> Devices)
- Network switching hierarchy (Core Switch -> Distribution Switch -> Access Switch -> Endpoints)
- Service dependencies (Workstations -> DNS/DHCP, NFS Storage, License Server)
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any

# Node types enum/constants
NODE_TYPE_POWER = 0
NODE_TYPE_SWITCH = 1
NODE_TYPE_COMPUTER = 2
NODE_TYPE_SERVICE = 3

NODE_TYPE_NAMES = {
    NODE_TYPE_POWER: "Power Unit",
    NODE_TYPE_SWITCH: "Switch",
    NODE_TYPE_COMPUTER: "Computer",
    NODE_TYPE_SERVICE: "Service Server"
}

# Edge dependency types
EDGE_POWER = "power_dependency"
EDGE_NETWORK = "network_dependency"
EDGE_SERVICE = "service_dependency"

# Feature indices
FEATURE_NAMES = [
    "temperature_c",       # Telemetry: Temperature in Celsius (30°C - 95°C)
    "uptime_hours",        # Telemetry: Hours since reboot (0 - 10,000 hrs)
    "error_counts",        # Telemetry: CRC / ECC / packet drop events in last hour
    "load_percent",        # Telemetry: CPU / Port bandwidth utilization (0 - 100%)
    "power_draw_watts",    # Telemetry: Power draw in Watts (50W - 3500W)
    "fan_speed_ratio",     # Telemetry: Fan PWM duty cycle (0.2 - 1.0)
    "type_is_power",       # One-hot: Is Power Unit
    "type_is_switch",      # One-hot: Is Switch
    "type_is_computer",    # One-hot: Is Computer
    "type_is_service"      # One-hot: Is Service Server
]
NUM_FEATURES = len(FEATURE_NAMES)


def create_lab_topology(
    num_labs: int = 2,
    pcs_per_lab: int = 20,
    seed: int = 42,
    stress_seed_node: bool = False
) -> nx.DiGraph:
    """
    Constructs a realistic multi-room university computer laboratory infrastructure graph.
    
    Args:
        num_labs: Number of computer lab rooms (e.g. Lab A, Lab B).
        pcs_per_lab: Number of student workstations per lab room.
        seed: Random seed for telemetry generation.
        stress_seed_node: If True, introduces thermal/error anomalies into a random critical node.
        
    Returns:
        nx.DiGraph: NetworkX directed dependency graph with node attributes and features.
    """
    rng = np.random.RandomState(seed)
    G = nx.DiGraph()
    
    # -------------------------------------------------------------
    # 1. POWER INFRASTRUCTURE HIERARCHY
    # -------------------------------------------------------------
    # Main Building PDU (Root of power grid)
    main_pdu_id = "Power_Main_PDU"
    G.add_node(
        main_pdu_id,
        name="Campus Building Substation PDU",
        type=NODE_TYPE_POWER,
        role="main_pdu",
        room="Utility_Basement",
        criticality=1.0,
        baseline_temp=38.0 + rng.uniform(-2, 3),
        baseline_uptime=4500.0 + rng.uniform(0, 1000),
        baseline_errors=0,
        baseline_load=45.0 + rng.uniform(-5, 5),
        baseline_power=15000.0,
        baseline_fan=0.45
    )
    
    # -------------------------------------------------------------
    # 2. NETWORK CORE INFRASTRUCTURE
    # -------------------------------------------------------------
    core_switch_id = "Switch_Core_01"
    G.add_node(
        core_switch_id,
        name="Backbone Campus Core Switch",
        type=NODE_TYPE_SWITCH,
        role="core_switch",
        room="Server_Room",
        criticality=1.0,
        baseline_temp=46.0 + rng.uniform(-2, 4),
        baseline_uptime=3200.0 + rng.uniform(0, 800),
        baseline_errors=rng.randint(0, 5),
        baseline_load=55.0 + rng.uniform(-5, 10),
        baseline_power=650.0,
        baseline_fan=0.60
    )
    
    # Core Switch powered by Main PDU
    G.add_edge(main_pdu_id, core_switch_id, edge_type=EDGE_POWER, weight=1.0)
    
    # -------------------------------------------------------------
    # 3. CRITICAL LAB SERVICES / SERVERS
    # -------------------------------------------------------------
    services = [
        ("Srv_DNS_DHCP", "Lab Core DNS & DHCP Server", 0.95, 320.0),
        ("Srv_NFS_Storage", "Central Roaming Profile NFS Storage", 0.90, 480.0),
        ("Srv_License_Mgr", "Floating Academic License Manager", 0.85, 260.0),
        ("Srv_Gateway_Proxy", "Lab Firewall & Proxy Gateway", 0.90, 310.0)
    ]
    
    for srv_id, srv_name, crit, p_watts in services:
        G.add_node(
            srv_id,
            name=srv_name,
            type=NODE_TYPE_SERVICE,
            role="service_server",
            room="Server_Room",
            criticality=crit,
            baseline_temp=42.0 + rng.uniform(-2, 5),
            baseline_uptime=2800.0 + rng.uniform(0, 500),
            baseline_errors=rng.randint(0, 8),
            baseline_load=40.0 + rng.uniform(-10, 15),
            baseline_power=p_watts + rng.uniform(-20, 30),
            baseline_fan=0.50
        )
        # Servers powered by Main PDU & linked directly to Core Switch
        G.add_edge(main_pdu_id, srv_id, edge_type=EDGE_POWER, weight=1.0)
        G.add_edge(core_switch_id, srv_id, edge_type=EDGE_NETWORK, weight=1.0)
        G.add_edge(srv_id, core_switch_id, edge_type=EDGE_NETWORK, weight=1.0)
        
    # -------------------------------------------------------------
    # 4. PER-LAB INFRASTRUCTURE (Racks, UPS, Switches, PCs)
    # -------------------------------------------------------------
    for lab_idx in range(1, num_labs + 1):
        lab_code = f"Lab_{lab_idx}"
        
        # Rack UPS for this lab
        ups_id = f"Power_UPS_{lab_code}"
        G.add_node(
            ups_id,
            name=f"Online Rackmount UPS ({lab_code})",
            type=NODE_TYPE_POWER,
            role="rack_ups",
            room=lab_code,
            criticality=0.85,
            baseline_temp=39.0 + rng.uniform(-2, 3),
            baseline_uptime=1800.0 + rng.uniform(0, 600),
            baseline_errors=rng.randint(0, 3),
            baseline_load=50.0 + rng.uniform(-5, 10),
            baseline_power=3200.0,
            baseline_fan=0.55
        )
        # UPS fed by Main PDU
        G.add_edge(main_pdu_id, ups_id, edge_type=EDGE_POWER, weight=1.0)
        
        # Distribution Switch for this lab floor/room
        dist_sw_id = f"Switch_Dist_{lab_code}"
        G.add_node(
            dist_sw_id,
            name=f"Distribution Switch ({lab_code})",
            type=NODE_TYPE_SWITCH,
            role="dist_switch",
            room=lab_code,
            criticality=0.80,
            baseline_temp=44.0 + rng.uniform(-3, 4),
            baseline_uptime=2100.0 + rng.uniform(0, 400),
            baseline_errors=rng.randint(0, 6),
            baseline_load=48.0 + rng.uniform(-8, 12),
            baseline_power=280.0,
            baseline_fan=0.50
        )
        # Dist switch powered by UPS & connected to Core Switch
        G.add_edge(ups_id, dist_sw_id, edge_type=EDGE_POWER, weight=1.0)
        G.add_edge(core_switch_id, dist_sw_id, edge_type=EDGE_NETWORK, weight=1.0)
        G.add_edge(dist_sw_id, core_switch_id, edge_type=EDGE_NETWORK, weight=1.0)
        
        # Split workstations into rows/racks (e.g. 10-15 PCs per Access Switch)
        pcs_per_access = 10
        num_access = int(np.ceil(pcs_per_lab / pcs_per_access))
        
        for acc_idx in range(1, num_access + 1):
            access_sw_id = f"Switch_Access_{lab_code}_Row{acc_idx}"
            epdu_id = f"Power_ePDU_{lab_code}_Row{acc_idx}"
            
            # Smart PDU for this row
            G.add_node(
                epdu_id,
                name=f"Smart Row ePDU ({lab_code} Row {acc_idx})",
                type=NODE_TYPE_POWER,
                role="row_pdu",
                room=lab_code,
                criticality=0.65,
                baseline_temp=35.0 + rng.uniform(-1, 3),
                baseline_uptime=1500.0 + rng.uniform(0, 500),
                baseline_errors=0,
                baseline_load=55.0 + rng.uniform(-10, 10),
                baseline_power=1400.0,
                baseline_fan=0.30
            )
            G.add_edge(ups_id, epdu_id, edge_type=EDGE_POWER, weight=1.0)
            
            # Access Switch for this row
            G.add_node(
                access_sw_id,
                name=f"Access Switch ({lab_code} Row {acc_idx})",
                type=NODE_TYPE_SWITCH,
                role="access_switch",
                room=lab_code,
                criticality=0.70,
                baseline_temp=43.0 + rng.uniform(-2, 4),
                baseline_uptime=1900.0 + rng.uniform(0, 300),
                baseline_errors=rng.randint(0, 5),
                baseline_load=45.0 + rng.uniform(-10, 15),
                baseline_power=180.0,
                baseline_fan=0.45
            )
            # Access switch powered by Row ePDU and uplinked to Dist Switch
            G.add_edge(epdu_id, access_sw_id, edge_type=EDGE_POWER, weight=1.0)
            G.add_edge(dist_sw_id, access_sw_id, edge_type=EDGE_NETWORK, weight=1.0)
            G.add_edge(access_sw_id, dist_sw_id, edge_type=EDGE_NETWORK, weight=1.0)
            
            # Student Workstations connected to this Access Switch and ePDU
            pc_start = (acc_idx - 1) * pcs_per_access + 1
            pc_end = min(acc_idx * pcs_per_access, pcs_per_lab)
            
            for pc_num in range(pc_start, pc_end + 1):
                pc_id = f"PC_{lab_code}_{pc_num:02d}"
                # Every 5th PC is a GPU workstation with higher power/temp
                is_gpu = (pc_num % 5 == 0)
                pc_temp = (48.0 if is_gpu else 38.0) + rng.uniform(-3, 5)
                pc_load = (65.0 if is_gpu else 35.0) + rng.uniform(-15, 20)
                pc_power = (280.0 if is_gpu else 110.0) + rng.uniform(-15, 20)
                
                G.add_node(
                    pc_id,
                    name=f"Student {'GPU ' if is_gpu else ''}Workstation #{pc_num:02d} ({lab_code})",
                    type=NODE_TYPE_COMPUTER,
                    role="workstation",
                    room=lab_code,
                    criticality=0.40,
                    baseline_temp=pc_temp,
                    baseline_uptime=rng.uniform(10.0, 720.0), # Workstations reboot more often
                    baseline_errors=rng.randint(0, 4),
                    baseline_load=pc_load,
                    baseline_power=pc_power,
                    baseline_fan=0.35 + (0.3 if is_gpu else 0.0)
                )
                
                # Workstation powered by Row ePDU
                G.add_edge(epdu_id, pc_id, edge_type=EDGE_POWER, weight=1.0)
                # Workstation network link to Row Access Switch (Bidirectional dependency)
                G.add_edge(access_sw_id, pc_id, edge_type=EDGE_NETWORK, weight=1.0)
                G.add_edge(pc_id, access_sw_id, edge_type=EDGE_NETWORK, weight=1.0)
                
                # Logical service dependencies: Workstation depends on License Manager, NFS, DNS
                G.add_edge(srv_id, pc_id, edge_type=EDGE_SERVICE, weight=0.5)

    # -------------------------------------------------------------
    # 5. ATTACH FEATURE VECTORS TO ALL NODES
    # -------------------------------------------------------------
    # If stress_seed_node is requested, select a random switch or power unit to show thermal/error anomaly
    stressed_node = None
    if stress_seed_node:
        candidates = [n for n, d in G.nodes(data=True) if d["type"] in (NODE_TYPE_POWER, NODE_TYPE_SWITCH)]
        stressed_node = rng.choice(candidates)
        G.nodes[stressed_node]["baseline_temp"] += rng.uniform(25.0, 40.0)  # Overheating
        G.nodes[stressed_node]["baseline_errors"] += rng.randint(40, 150)   # High packet drops / CRC
        G.nodes[stressed_node]["baseline_load"] += rng.uniform(30.0, 45.0)   # Heavy load
        G.nodes[stressed_node]["baseline_fan"] = 1.0                         # Max fan RPM
        G.nodes[stressed_node]["is_stressed_seed"] = True
    
    for n, data in G.nodes(data=True):
        t_type = data["type"]
        feat = np.array([
            float(data["baseline_temp"]),
            float(data["baseline_uptime"]),
            float(data["baseline_errors"]),
            float(np.clip(data["baseline_load"], 0.0, 100.0)),
            float(data["baseline_power"]),
            float(np.clip(data["baseline_fan"], 0.0, 1.0)),
            1.0 if t_type == NODE_TYPE_POWER else 0.0,
            1.0 if t_type == NODE_TYPE_SWITCH else 0.0,
            1.0 if t_type == NODE_TYPE_COMPUTER else 0.0,
            1.0 if t_type == NODE_TYPE_SERVICE else 0.0
        ], dtype=np.float32)
        
        G.nodes[n]["features"] = feat
        G.nodes[n]["state"] = "OPERATIONAL"  # OPERATIONAL, AT_RISK, FAILED
        if "is_stressed_seed" not in G.nodes[n]:
            G.nodes[n]["is_stressed_seed"] = False
            
    return G


def graph_to_feature_matrix(G: nx.DiGraph) -> Tuple[np.ndarray, List[str]]:
    """
    Extracts the node feature matrix X and node ID ordering.
    Returns:
        X: numpy array of shape (N, NUM_FEATURES)
        node_order: list of node IDs corresponding to row indices
    """
    node_order = list(G.nodes())
    X = np.stack([G.nodes[n]["features"] for n in node_order], axis=0)
    return X, node_order


if __name__ == "__main__":
    # Quick sanity test
    g = create_lab_topology(num_labs=2, pcs_per_lab=15, seed=42, stress_seed_node=True)
    print(f"Constructed Lab Infrastructure Graph:")
    print(f"  Total Nodes: {g.number_of_nodes()}")
    print(f"  Total Edges: {g.number_of_edges()}")
    type_counts = {}
    for n, d in g.nodes(data=True):
        tname = NODE_TYPE_NAMES[d['type']]
        type_counts[tname] = type_counts.get(tname, 0) + 1
    for k, v in type_counts.items():
        print(f"  - {k}: {v} devices")
    X, order = graph_to_feature_matrix(g)
    print(f"  Feature Matrix shape: {X.shape}")
