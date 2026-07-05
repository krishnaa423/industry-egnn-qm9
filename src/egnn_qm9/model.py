from __future__ import annotations


def _require_modules() -> tuple[object, object, object, object, object]:
    try:
        import torch
        import torch.nn as nn
        from e3nn import o3
        from e3nn.o3 import FullyConnectedTensorProduct, Linear
        from torch_geometric.nn import MessagePassing
    except Exception as exc:
        raise RuntimeError("Torch, torch-geometric, and e3nn must import cleanly.") from exc
    return torch, nn, o3, FullyConnectedTensorProduct, Linear, MessagePassing


torch, nn, o3, FullyConnectedTensorProduct, Linear, MessagePassing = _require_modules()


def scalar_slice(irreps: o3.Irreps) -> slice:
    width = 0
    for mul, ir in irreps:
        if ir.l != 0:
            break
        width += mul * ir.dim
    return slice(0, width)


class TensorProductMessagePassing(MessagePassing):
    """Message passing layer with learnable tensor-product edge mixing."""

    def __init__(self, node_irreps: str, edge_attr_irreps: str, edge_feature_dim: int) -> None:
        super().__init__(aggr="add", node_dim=0)
        self.node_irreps = o3.Irreps(node_irreps)
        self.edge_attr_irreps = o3.Irreps(edge_attr_irreps)
        self.scalar_only_irreps = o3.Irreps(f"{scalar_slice(self.node_irreps).stop}x0e")
        self.self_linear = Linear(self.node_irreps, self.node_irreps)
        self.tensor_product = FullyConnectedTensorProduct(
            self.node_irreps,
            self.edge_attr_irreps,
            self.node_irreps,
            shared_weights=False,
        )
        self.edge_weight = nn.Sequential(
            nn.Linear(edge_feature_dim, edge_feature_dim),
            nn.SiLU(),
            nn.Linear(edge_feature_dim, self.tensor_product.weight_numel),
        )
        scalar_width = scalar_slice(self.node_irreps).stop
        self.edge_update_mlp = nn.Sequential(
            nn.Linear(2 * scalar_width + edge_feature_dim, edge_feature_dim),
            nn.SiLU(),
            nn.Linear(edge_feature_dim, edge_feature_dim),
        )

    def forward(
        self,
        node_attr: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        messages = self.propagate(
            edge_index,
            node_attr=node_attr,
            edge_attr=edge_attr,
            edge_features=edge_features,
        )
        updated_nodes = self.self_linear(node_attr) + messages

        src, dst = edge_index
        scalar_part = scalar_slice(self.node_irreps)
        edge_inputs = torch.cat(
            [updated_nodes[src][:, scalar_part], updated_nodes[dst][:, scalar_part], edge_features],
            dim=-1,
        )
        updated_edge_features = edge_features + self.edge_update_mlp(edge_inputs)
        return updated_nodes, updated_edge_features

    def message(
        self,
        node_attr_j: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        weights = self.edge_weight(edge_features)
        return self.tensor_product(node_attr_j, edge_attr, weights)


class QM9EnergyModel(nn.Module):
    def __init__(
        self,
        max_atomic_number: int = 10,
        node_irreps: str = "32x0e + 8x1o",
        edge_attr_irreps: str = "1x0e + 1x1o",
        edge_feature_dim: int = 32,
        radial_basis_dim: int = 8,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.node_irreps = o3.Irreps(node_irreps)
        self.edge_attr_irreps = o3.Irreps(edge_attr_irreps)
        self.scalar_width = scalar_slice(self.node_irreps).stop
        self.atomic_embedding = nn.Embedding(max_atomic_number + 1, self.scalar_width)
        self.radial_basis_dim = radial_basis_dim
        self.edge_feature_proj = nn.Sequential(
            nn.Linear(radial_basis_dim, edge_feature_dim),
            nn.SiLU(),
            nn.Linear(edge_feature_dim, edge_feature_dim),
        )
        self.layers = nn.ModuleList(
            [
                TensorProductMessagePassing(
                    node_irreps=node_irreps,
                    edge_attr_irreps=edge_attr_irreps,
                    edge_feature_dim=edge_feature_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.energy_head = nn.Sequential(
            nn.Linear(self.scalar_width, self.scalar_width),
            nn.SiLU(),
            nn.Linear(self.scalar_width, 1),
        )

    def radial_basis(self, edge_length: torch.Tensor) -> torch.Tensor:
        centers = torch.linspace(0.0, 5.0, self.radial_basis_dim, device=edge_length.device, dtype=edge_length.dtype)
        widths = 0.75
        return torch.exp(-((edge_length - centers) ** 2) / widths**2)

    def build_node_attr(self, z: torch.Tensor, num_nodes: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        node_attr = torch.zeros((num_nodes, self.node_irreps.dim), device=device, dtype=dtype)
        node_attr[:, : self.scalar_width] = self.atomic_embedding(z.long()).to(dtype)
        return node_attr

    def forward(self, data: object) -> torch.Tensor:
        z = data.z
        pos = data.pos
        edge_index = data.edge_index
        src, dst = edge_index
        edge_vec = pos[dst] - pos[src]
        edge_len = edge_vec.norm(dim=-1, keepdim=True)
        edge_attr = o3.spherical_harmonics(
            self.edge_attr_irreps,
            edge_vec,
            normalize=True,
            normalization="component",
        )
        radial = self.radial_basis(edge_len)
        edge_features = self.edge_feature_proj(radial)
        node_attr = self.build_node_attr(z, data.num_nodes, pos.device, pos.dtype)

        for layer in self.layers:
            node_attr, edge_features = layer(node_attr, edge_index, edge_attr, edge_features)

        node_energy = self.energy_head(node_attr[:, : self.scalar_width]).squeeze(-1)
        batch = getattr(data, "batch", torch.zeros(data.num_nodes, dtype=torch.long, device=pos.device))
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        graph_energy = torch.zeros(num_graphs, device=pos.device, dtype=node_energy.dtype)
        graph_energy.index_add_(0, batch, node_energy)
        return graph_energy
