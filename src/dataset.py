import os
from typing import Optional

import networkx as nx
import numpy as np
import torch

from src.simulator import generate_cascade_dataset
from src.topology import NUM_FEATURES


class LabGraphData:
    """Stores one graph scenario and its labels."""

    def __init__(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        y_binary: torch.Tensor,
        y_severity: torch.Tensor,
        risk_scores: torch.Tensor,
        node_names: list,
        graph: Optional[nx.DiGraph] = None,
        trigger_node=None,
    ):
        self.x = x
        self.adj = adj
        self.y_binary = y_binary
        self.y_severity = y_severity
        self.risk_scores = risk_scores
        self.node_names = node_names
        self.graph = graph
        self.trigger_node = trigger_node
        self.num_nodes = x.size(0)


class LabDatasetManager:
    """Creates, splits, normalizes, and caches graph scenarios."""

    def __init__(self, data_dir="data", split_seed=123):
        self.data_dir = data_dir
        self.split_seed = split_seed
        os.makedirs(data_dir, exist_ok=True)

        self.mean = None
        self.std = None

    def fit_normalizer(self, feature_matrices):
        """Fit feature statistics using training data only."""
        if not feature_matrices:
            raise ValueError("Cannot fit normalizer on an empty dataset.")

        all_x = np.concatenate(feature_matrices, axis=0).astype(
            np.float32
        )

        if all_x.ndim != 2 or all_x.shape[1] != NUM_FEATURES:
            raise ValueError(
                f"Expected feature matrices with {NUM_FEATURES} features."
            )

        # The original implementation standardizes the first six features.
        continuous_x = all_x[:, :6]

        self.mean = continuous_x.mean(axis=0)
        self.std = continuous_x.std(axis=0)

        # Avoid division by zero for constant features.
        self.std[self.std < 1e-6] = 1.0

    def transform(self, x):
        """Normalize continuous features using training statistics."""
        x_norm = np.asarray(x, dtype=np.float32).copy()

        if x_norm.ndim != 2 or x_norm.shape[1] != NUM_FEATURES:
            raise ValueError(
                f"Expected a 2D feature matrix with {NUM_FEATURES} columns."
            )

        if self.mean is not None and self.std is not None:
            x_norm[:, :6] = (
                x_norm[:, :6] - self.mean
            ) / self.std

        return x_norm

    @staticmethod
    def _split_scenarios(scenarios, seed):
        """Split complete scenarios to prevent graph-level leakage."""
        count = len(scenarios)

        if count < 3:
            raise ValueError(
                "At least three scenarios are required for train/val/test."
            )

        rng = np.random.default_rng(seed)
        indices = rng.permutation(count)

        n_train = max(1, int(0.70 * count))
        n_val = max(1, int(0.15 * count))

        # Ensure that at least one scenario remains for testing.
        if n_train + n_val >= count:
            n_train = count - 2
            n_val = 1

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        train = [scenarios[i] for i in train_indices]
        val = [scenarios[i] for i in val_indices]
        test = [scenarios[i] for i in test_indices]

        return train, val, test

    @staticmethod
    def _to_graph_data(scenario, normalizer):
        """Convert one generated scenario into tensors."""
        graph = scenario["graph"]
        nodes = list(scenario["nodes"])
        num_nodes = len(nodes)

        if num_nodes == 0:
            raise ValueError("A scenario contains no graph nodes.")

        features = np.asarray(
            scenario["features"], dtype=np.float32
        )

        if features.shape != (num_nodes, NUM_FEATURES):
            raise ValueError(
                f"Feature shape {features.shape} does not match "
                f"({num_nodes}, {NUM_FEATURES})."
            )

        adjacency = nx.to_numpy_array(
            graph,
            nodelist=nodes,
            weight="weight",
            dtype=np.float32,
        )

        # Preserve the original behavior: symmetrize non-power edges.
        node_index = {node: i for i, node in enumerate(nodes)}

        for u, v, edge_data in graph.edges(data=True):
            if edge_data.get("edge_type") != "power_dependency":
                i = node_index[u]
                j = node_index[v]
                weight = max(adjacency[i, j], adjacency[j, i])
                adjacency[i, j] = weight
                adjacency[j, i] = weight

        normalized_features = normalizer.transform(features)

        binary_labels = np.asarray(
            scenario["binary_labels"], dtype=np.float32
        ).reshape(-1, 1)

        severity_labels = np.asarray(
            scenario["severity_classes"], dtype=np.int64
        ).reshape(-1)

        risk_scores = np.asarray(
            scenario["risk_scores"], dtype=np.float32
        ).reshape(-1, 1)

        if len(binary_labels) != num_nodes:
            raise ValueError("Binary-label count does not match node count.")

        if len(severity_labels) != num_nodes:
            raise ValueError("Severity-label count does not match node count.")

        if len(risk_scores) != num_nodes:
            raise ValueError("Risk-score count does not match node count.")

        return LabGraphData(
            x=torch.tensor(normalized_features, dtype=torch.float32),
            adj=torch.tensor(adjacency, dtype=torch.float32),
            y_binary=torch.tensor(binary_labels, dtype=torch.float32),
            y_severity=torch.tensor(severity_labels, dtype=torch.long),
            risk_scores=torch.tensor(risk_scores, dtype=torch.float32),
            node_names=nodes,
            graph=graph,
            trigger_node=scenario.get("trigger_node"),
        )

    def create_or_load_dataset(
        self,
        num_scenarios=120,
        cache_path="data/lab_cascade_dataset.pt",
        force_regenerate=False,
    ):
        """Generate or load the graph dataset."""
        config = {
            "num_scenarios": int(num_scenarios),
            "split_seed": int(self.split_seed),
            "feature_count": int(NUM_FEATURES),
        }

        if os.path.exists(cache_path) and not force_regenerate:
            try:
                # This cache contains Python objects; only load trusted files.
                cached = torch.load(
                    cache_path,
                    map_location="cpu",
                    weights_only=False,
                )

                if cached.get("config") == config:
                    self.mean = cached["mean"]
                    self.std = cached["std"]
                    print(f"[Dataset] Loaded cache: {cache_path}")
                    return cached

                print("[Dataset] Cache configuration differs; regenerating.")

            except (OSError, RuntimeError, ValueError, KeyError) as error:
                print(f"[Dataset] Could not use cache: {error}")

        print(
            f"[Dataset] Generating {num_scenarios} cascade scenarios..."
        )

        raw_scenarios = generate_cascade_dataset(
            num_scenarios=num_scenarios,
            num_labs_range=(2, 3),
            pcs_range=(15, 25),
            seed=42,
        )

        if len(raw_scenarios) < 3:
            raise ValueError(
                "The simulator must generate at least three scenarios."
            )

        train_raw, val_raw, test_raw = self._split_scenarios(
            raw_scenarios, self.split_seed
        )

        # Important: fit normalization only on training scenarios.
        self.fit_normalizer(
            [scenario["features"] for scenario in train_raw]
        )

        train_data = [
            self._to_graph_data(scenario, self)
            for scenario in train_raw
        ]
        val_data = [
            self._to_graph_data(scenario, self)
            for scenario in val_raw
        ]
        test_data = [
            self._to_graph_data(scenario, self)
            for scenario in test_raw
        ]

        train_labels = torch.cat(
            [data.y_binary for data in train_data], dim=0
        )

        num_positive = float((train_labels == 1.0).sum())
        num_negative = float((train_labels == 0.0).sum())

        pos_weight = num_negative / max(1.0, num_positive)

        payload = {
            "train": train_data,
            "val": val_data,
            "test": test_data,
            "pos_weight": float(pos_weight),
            "mean": self.mean,
            "std": self.std,
            "num_features": NUM_FEATURES,
            "total_scenarios": len(raw_scenarios),
            "config": config,
        }

        cache_directory = os.path.dirname(cache_path)
        if cache_directory:
            os.makedirs(cache_directory, exist_ok=True)

        torch.save(payload, cache_path)

        print(
            f"[Dataset] Saved cache: {cache_path}\n"
            f"  Train scenarios: {len(train_data)}\n"
            f"  Validation scenarios: {len(val_data)}\n"
            f"  Test scenarios: {len(test_data)}\n"
            f"  Positive-class weight: {pos_weight:.3f}"
        )

        return payload


if __name__ == "__main__":
    manager = LabDatasetManager(data_dir="data")

    dataset = manager.create_or_load_dataset(
        num_scenarios=20,
        force_regenerate=True,
    )

    print("\nDataset summary")
    print("----------------")
    print("Total scenarios:", dataset["total_scenarios"])
    print("Training graphs:", len(dataset["train"]))
    print("Validation graphs:", len(dataset["val"]))
    print("Test graphs:", len(dataset["test"]))
    print("Number of features:", dataset["num_features"])

