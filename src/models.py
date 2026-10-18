from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Graph normalization
# ---------------------------------------------------------

def compute_normalized_laplacian(adj: torch.Tensor) -> torch.Tensor:
    """
    Compute symmetric normalized adjacency:

        A_hat = D^(-1/2) (A + I) D^(-1/2)

    Note:
        This is normalized adjacency, not the normalized
        graph Laplacian.

    Supports:
        (N, N)
        (B, N, N)
    """

    if adj.dim() not in (2, 3):
        raise ValueError(
            f"Expected a 2D or 3D adjacency matrix; got {adj.shape}."
        )

    if adj.size(-1) != adj.size(-2):
        raise ValueError("Adjacency matrices must be square.")

    n = adj.size(-1)

    identity = torch.eye(
        n,
        device=adj.device,
        dtype=adj.dtype,
    )

    if adj.dim() == 2:
        adj_with_self_loops = adj + identity
        degree = adj_with_self_loops.sum(dim=1)

        degree_inv_sqrt = torch.rsqrt(degree.clamp(min=1e-6))

        normalized_adj = (
            degree_inv_sqrt.unsqueeze(1)
            * adj_with_self_loops
            * degree_inv_sqrt.unsqueeze(0)
        )

        return normalized_adj

    # Batched adjacency: (B, N, N)
    identity = identity.unsqueeze(0)
    adj_with_self_loops = adj + identity

    degree = adj_with_self_loops.sum(dim=2)
    degree_inv_sqrt = torch.rsqrt(degree.clamp(min=1e-6))

    normalized_adj = (
        degree_inv_sqrt.unsqueeze(2)
        * adj_with_self_loops
        * degree_inv_sqrt.unsqueeze(1)
    )

    return normalized_adj


# ---------------------------------------------------------
# Batch normalization helper
# ---------------------------------------------------------

def safe_batch_norm(
    x: torch.Tensor,
    batch_norm: nn.BatchNorm1d,
) -> torch.Tensor:
    """
    Apply BatchNorm1d safely when training with only one sample.

    When the effective batch contains one sample, use the
    layer's stored running statistics rather than estimating
    variance from a single sample.
    """

    if batch_norm.training and x.size(0) == 1:
        return F.batch_norm(
            x,
            batch_norm.running_mean,
            batch_norm.running_var,
            batch_norm.weight,
            batch_norm.bias,
            training=False,
            momentum=0.0,
            eps=batch_norm.eps,
        )

    return batch_norm(x)


# ---------------------------------------------------------
# Graph convolution layer
# ---------------------------------------------------------

class GraphConvLayer(nn.Module):
    """
    Graph convolution:

        H' = A_norm @ H @ W + b

    Supports:
        x:        (N, in_features) or (B, N, in_features)
        adj_norm: (N, N) or (B, N, N)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(in_features, out_features)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        adj_norm: torch.Tensor,
    ) -> torch.Tensor:

        support = torch.matmul(x, self.weight)

        if x.dim() == 2:
            if adj_norm.is_sparse:
                output = torch.sparse.mm(adj_norm, support)
            else:
                output = torch.mm(adj_norm, support)

        elif x.dim() == 3:
            output = torch.bmm(adj_norm, support)

        else:
            raise ValueError(
                f"Unsupported feature dimensions: {x.shape}"
            )

        if self.bias is not None:
            output = output + self.bias

        return output


# ---------------------------------------------------------
# LabGCN
# ---------------------------------------------------------

class LabGCN(nn.Module):
    """
    Multi-layer Graph Convolutional Network for lab equipment
    cascading-failure prediction.

    Architecture:
        Input
          -> GCN1 -> BatchNorm -> ReLU -> Dropout
          -> GCN2 + residual projection
             -> BatchNorm -> ReLU -> Dropout
          -> GCN3 -> BatchNorm -> ReLU
          -> Binary risk head
          -> Severity classification head

    Severity classes:
        0 = Safe
        1 = Warning
        2 = Critical
    """

    def __init__(
        self,
        in_features: int = 10,
        hidden_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.25,
    ):
        super().__init__()

        if in_features < 1:
            raise ValueError("in_features must be positive.")

        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2.")

        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        self.gcn1 = GraphConvLayer(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.gcn2 = GraphConvLayer(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.gcn3 = GraphConvLayer(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)

        # Project raw input features to the residual dimension.
        self.skip_proj = nn.Linear(in_features, hidden_dim)

        # Binary cascade-risk head: one raw logit per node.
        self.binary_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # Multiclass severity head.
        self.severity_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if x.dim() != 2:
            raise ValueError(
                f"LabGCN expects x with shape (N, F); got {x.shape}."
            )

        if adj.dim() != 2:
            raise ValueError(
                f"LabGCN expects adj with shape (N, N); got {adj.shape}."
            )

        if adj.size(0) != x.size(0) or adj.size(1) != x.size(0):
            raise ValueError(
                "Adjacency dimensions must match the number of nodes."
            )

        adj_norm = compute_normalized_laplacian(adj)

        # Layer 1
        h1 = self.gcn1(x, adj_norm)
        h1 = safe_batch_norm(h1, self.bn1)
        h1 = F.relu(h1)
        h1 = F.dropout(
            h1,
            p=self.dropout_rate,
            training=self.training,
        )

        # Layer 2 with projected input residual.
        skip = self.skip_proj(x)

        h2 = self.gcn2(h1, adj_norm) + skip
        h2 = safe_batch_norm(h2, self.bn2)
        h2 = F.relu(h2)
        h2 = F.dropout(
            h2,
            p=self.dropout_rate,
            training=self.training,
        )

        # Layer 3
        h3 = self.gcn3(h2, adj_norm)
        h3 = safe_batch_norm(h3, self.bn3)
        h3 = F.relu(h3)

        binary_logits = self.binary_head(h3)
        severity_logits = self.severity_head(h3)

        return binary_logits, severity_logits, h3


# ---------------------------------------------------------
# Graph attention layer
# ---------------------------------------------------------

class GraphAttentionLayer(nn.Module):
    """
    Single-head graph attention layer.

    Nodes attend to their neighbors according to the nonzero
    entries in the adjacency matrix. Edge weights are currently
    treated as connectivity indicators, not attention multipliers.

    Returns:
        Updated node representations
        Attention-weight matrix
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.2,
        alpha: float = 0.2,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(
            torch.empty(in_features, out_features)
        )

        self.a = nn.Parameter(
            torch.empty(2 * out_features, 1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if h.dim() != 2 or adj.dim() != 2:
            raise ValueError(
                "GraphAttentionLayer expects 2D features and adjacency."
            )

        n = h.size(0)

        if adj.shape != (n, n):
            raise ValueError(
                "Adjacency dimensions must match the number of nodes."
            )

        # Linear transformation.
        Wh = torch.mm(h, self.W)

        # Attention terms.
        attention_left = torch.matmul(
            Wh, self.a[:self.out_features]
        )

        attention_right = torch.matmul(
            Wh, self.a[self.out_features:]
        )

        # Pairwise node attention scores: (N, N).
        scores = F.leaky_relu(
            attention_left + attention_right.T,
            negative_slope=self.alpha,
        )

        # Add self-loops to the connectivity mask.
        identity = torch.eye(
            n,
            device=adj.device,
            dtype=adj.dtype,
        )

        adjacency_with_self_loops = adj + identity
        mask = adjacency_with_self_loops == 0

        scores = scores.masked_fill(mask, float("-inf"))

        # Normalize attention across each destination row.
        attention = F.softmax(scores, dim=1)

        # Drop attention coefficients during training.
        attention_used = F.dropout(
            attention,
            p=self.dropout,
            training=self.training,
        )

        output = torch.matmul(attention_used, Wh)

        return output, attention


# ---------------------------------------------------------
# LabGAT
# ---------------------------------------------------------

class LabGAT(nn.Module):
    """
    Graph Attention Network for lab equipment.

    The architecture uses two sequential attention layers and
    separate binary-risk and severity heads.

    The current implementation uses single-head attention
    in each layer.
    """

    def __init__(
        self,
        in_features: int = 10,
        hidden_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.25,
    ):
        super().__init__()

        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2.")

        self.gat1 = GraphAttentionLayer(
            in_features,
            hidden_dim,
            dropout=dropout,
        )

        self.gat2 = GraphAttentionLayer(
            hidden_dim,
            hidden_dim // 2,
            dropout=dropout,
        )

        self.binary_head = nn.Linear(hidden_dim // 2, 1)
        self.severity_head = nn.Linear(
            hidden_dim // 2, num_classes
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        h1, attention1 = self.gat1(x, adj)
        h1 = F.elu(h1)

        h2, attention2 = self.gat2(h1, adj)
        h2 = F.elu(h2)

        binary_logits = self.binary_head(h2)
        severity_logits = self.severity_head(h2)

        # Attention matrices are available for debugging or analysis.
        # The standard evaluator expects three outputs.
        return binary_logits, severity_logits, h2


# ---------------------------------------------------------
# MLP baseline
# ---------------------------------------------------------

class MLPBaseline(nn.Module):
    """
    Topology-agnostic multilayer perceptron.

    Uses node features only. The adjacency argument is accepted
    for compatibility with the shared evaluator but is ignored.
    """

    def __init__(
        self,
        in_features: int = 10,
        hidden_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.25,
    ):
        super().__init__()

        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2.")

        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)

        self.dropout = nn.Dropout(dropout)

        self.binary_head = nn.Linear(hidden_dim // 2, 1)
        self.severity_head = nn.Linear(
            hidden_dim // 2, num_classes
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        h = self.fc1(x)
        h = safe_batch_norm(h, self.bn1)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.fc2(h)
        h = safe_batch_norm(h, self.bn2)
        h = F.relu(h)
        h = self.dropout(h)

        h = F.relu(self.fc3(h))

        binary_logits = self.binary_head(h)
        severity_logits = self.severity_head(h)

        return binary_logits, severity_logits, h


# ---------------------------------------------------------
# Basic forward-pass test
# ---------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(19)

    num_nodes = 45
    num_features = 10

    x = torch.randn(num_nodes, num_features)

    # Create an undirected test adjacency matrix.
    random_adj = (
        torch.rand(num_nodes, num_nodes) > 0.90
    ).float()

    adj = torch.maximum(random_adj, random_adj.T)
    adj.fill_diagonal_(0)

    models = {
        "LabGCN": LabGCN(in_features=num_features),
        "LabGAT": LabGAT(in_features=num_features),
        "MLPBaseline": MLPBaseline(in_features=num_features),
    }

    for model_name, model in models.items():
        model.eval()

        with torch.no_grad():
            binary_logits, severity_logits, embeddings = model(
                x, adj
            )

        print(f"\n{model_name} forward pass successful")
        print("  Binary logits:", tuple(binary_logits.shape))
        print("  Severity logits:", tuple(severity_logits.shape))
        print("  Embeddings:", tuple(embeddings.shape))
