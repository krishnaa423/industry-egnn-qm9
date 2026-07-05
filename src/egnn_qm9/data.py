from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from .config import DatasetConfig

QM9_TARGETS = {
    0: "mu",
    1: "alpha",
    2: "homo",
    3: "lumo",
    4: "gap",
    5: "r2",
    6: "zpve",
    7: "U0",
    8: "U",
    9: "H",
    10: "G",
    11: "Cv",
    12: "U0_atom",
    13: "U_atom",
    14: "H_atom",
    15: "G_atom",
    16: "A",
    17: "B",
    18: "C",
}


def load_qm9_dataset(config: DatasetConfig) -> QM9:
    return QM9(root=str(config.root / "raw"))


def prepare_splits(config: DatasetConfig) -> dict[str, Any]:
    dataset = load_qm9_dataset(config)
    total_requested = config.train_size + config.val_size + config.test_size
    if total_requested > len(dataset):
        raise ValueError(f"Requested {total_requested} samples, but QM9 only has {len(dataset)}.")

    generator = torch.Generator().manual_seed(config.seed)
    indices = torch.randperm(len(dataset), generator=generator)

    train_idx = indices[: config.train_size]
    val_idx = indices[config.train_size : config.train_size + config.val_size]
    test_idx = indices[
        config.train_size + config.val_size : config.train_size + config.val_size + config.test_size
    ]

    subsets = {
        "train": dataset.index_select(train_idx),
        "val": dataset.index_select(val_idx),
        "test": dataset.index_select(test_idx),
    }

    loaders = {
        split: DataLoader(subset, batch_size=config.batch_size, shuffle=(split == "train"))
        for split, subset in subsets.items()
    }

    target_name = QM9_TARGETS.get(config.target_index, f"target_{config.target_index}")
    metadata = {
        "dataset_name": "QM9",
        "dataset_size": len(dataset),
        "target_index": config.target_index,
        "target_name": target_name,
        "train_size": len(subsets["train"]),
        "val_size": len(subsets["val"]),
        "test_size": len(subsets["test"]),
        "batch_size": config.batch_size,
    }
    return {
        "dataset": dataset,
        "subsets": subsets,
        "loaders": loaders,
        "metadata": metadata,
    }


def write_split_metadata(config: DatasetConfig, metadata: dict[str, Any]) -> None:
    config.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": asdict(config),
        **metadata,
    }
    config.metadata_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def summarize_batch_targets(loader: DataLoader, target_index: int) -> dict[str, float]:
    values = []
    for batch in loader:
        values.append(batch.y[:, target_index])
    targets = torch.cat(values, dim=0)
    return {
        "mean": float(targets.mean()),
        "std": float(targets.std().clamp_min(1e-8)),
    }


def rotation_matrix_z_90(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
