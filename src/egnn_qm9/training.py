from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .config import ExperimentConfig
from .data import rotation_matrix_z_90


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_r2_score(references: np.ndarray, predictions: np.ndarray) -> float:
    references = np.asarray(references, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    ss_res = float(np.sum((references - predictions) ** 2))
    ss_tot = float(np.sum((references - references.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def prepare_targets(batch: object, target_index: int) -> torch.Tensor:
    return batch.y[:, target_index]


def train_model(config: ExperimentConfig, bundle: dict[str, Any]) -> dict[str, Any]:
    from .model import QM9EnergyModel

    set_seed(config.training.seed)
    train_loader = bundle["loaders"]["train"]
    val_loader = bundle["loaders"]["val"]
    test_loader = bundle["loaders"]["test"]
    writer = SummaryWriter(log_dir=str(config.output.runs_dir / "qm9_energy_model"))

    model = QM9EnergyModel(
        max_atomic_number=config.model.max_atomic_number,
        node_irreps=config.model.node_irreps,
        edge_attr_irreps=config.model.edge_attr_irreps,
        edge_feature_dim=config.model.edge_feature_dim,
        radial_basis_dim=config.model.radial_basis_dim,
        num_layers=config.model.num_layers,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)

    target_stats = bundle["target_stats"]
    history = {"train_loss": [], "val_loss": []}
    best = {"val_loss": float("inf"), "state_dict": None}

    for epoch in range(config.training.epochs):
        model.train()
        train_losses = []
        for step, batch in enumerate(train_loader, start=1):
            target = prepare_targets(batch, config.dataset.target_index)
            pred = model(batch)
            normalized_error = (pred - target) / target_stats["std"]
            loss = normalized_error.square().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
            writer.add_scalar("batch/train_loss", float(loss.detach()), epoch * 1000 + step)

        val_metrics = evaluate_loader(model, val_loader, config.dataset.target_index, target_stats["std"])
        train_loss = float(np.mean(train_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        writer.add_scalar("epoch/train_loss", train_loss, epoch)
        writer.add_scalar("epoch/val_loss", val_metrics["loss"], epoch)

        if val_metrics["loss"] < best["val_loss"]:
            best["val_loss"] = val_metrics["loss"]
            best["state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])

    test_metrics = evaluate_loader(
        model,
        test_loader,
        config.dataset.target_index,
        target_stats["std"],
        with_rotation_check=True,
    )
    metrics = {
        "energy_mse": test_metrics["mse"],
        "energy_mae": test_metrics["mae"],
        "energy_r2": test_metrics["r2"],
        "rotation_invariance_mae": test_metrics["rotation_invariance_mae"],
        "rotation_errors": test_metrics["rotation_errors"],
        "predictions": test_metrics["predictions"],
        "references": test_metrics["references"],
        "history": history,
        "target_stats": target_stats,
        "config": asdict(config),
        "dataset_metadata": bundle["metadata"],
    }
    writer.close()
    return metrics


def evaluate_loader(
    model: torch.nn.Module,
    loader: object,
    target_index: int,
    target_std: float,
    with_rotation_check: bool = False,
) -> dict[str, Any]:
    model.eval()
    losses = []
    preds = []
    refs = []
    rotation_errors: list[float] = []

    for batch in loader:
        target = prepare_targets(batch, target_index)
        pred = model(batch)
        normalized_error = (pred - target) / target_std
        losses.append(float(normalized_error.square().mean().detach()))
        preds.append(pred.detach().cpu())
        refs.append(target.detach().cpu())

        if with_rotation_check:
            rotated = batch.clone()
            rot = rotation_matrix_z_90(device=batch.pos.device, dtype=batch.pos.dtype)
            rotated.pos = batch.pos @ rot.T
            rotated_pred = model(rotated)
            rotation_errors.extend((rotated_pred - pred).abs().detach().cpu().tolist())

    pred_tensor = torch.cat(preds)
    ref_tensor = torch.cat(refs)
    pred_np = pred_tensor.numpy()
    ref_np = ref_tensor.numpy()
    return {
        "loss": float(np.mean(losses)),
        "mse": float(np.mean((pred_np - ref_np) ** 2)),
        "mae": float(np.mean(np.abs(pred_np - ref_np))),
        "r2": float(compute_r2_score(ref_np, pred_np)),
        "rotation_invariance_mae": float(np.mean(rotation_errors)) if rotation_errors else 0.0,
        "rotation_errors": rotation_errors,
        "predictions": pred_np.tolist(),
        "references": ref_np.tolist(),
    }


def write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
