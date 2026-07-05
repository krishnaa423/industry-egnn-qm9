from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set_theme(style="whitegrid")
plt.style.use("seaborn-v0_8-whitegrid")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_training_curve(history: dict[str, list[float]], output_path: Path) -> None:
    ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["train_loss"], label="Train", linewidth=2.2, color="#166534")
    ax.plot(history["val_loss"], label="Validation", linewidth=2.2, color="#2563eb")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized MSE")
    ax.set_title("QM9 energy training curve")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_energy_parity(references: list[float], predictions: list[float], output_path: Path) -> None:
    references = np.asarray(references)
    predictions = np.asarray(predictions)
    ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(references, predictions, s=22, alpha=0.65, color="#0f766e", edgecolors="none")
    bounds = [float(min(references.min(), predictions.min())), float(max(references.max(), predictions.max()))]
    ax.plot(bounds, bounds, linestyle="--", color="#7c2d12", linewidth=1.5)
    ax.set_xlabel("Reference energy")
    ax.set_ylabel("Predicted energy")
    ax.set_title("Energy parity plot")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_rotation_invariance(errors: list[float], output_path: Path) -> None:
    errors = np.asarray(errors)
    ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(np.arange(len(errors)), errors, color="#7c3aed")
    ax.set_xlabel("Test graph")
    ax.set_ylabel("|E(Rx) - E(x)|")
    ax.set_title("Rotation invariance check")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
