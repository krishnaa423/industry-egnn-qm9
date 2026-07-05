from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class DatasetConfig:
    root: Path = Path("data")
    metadata_name: str = "qm9_energy_split.json"
    target_index: int = 7
    train_size: int = 500
    val_size: int = 100
    test_size: int = 100
    batch_size: int = 32
    seed: int = 7

    @property
    def metadata_path(self) -> Path:
        return self.root / "processed" / self.metadata_name


@dataclass(slots=True)
class ModelConfig:
    node_irreps: str = "32x0e + 8x1o"
    edge_attr_irreps: str = "1x0e + 1x1o"
    edge_feature_dim: int = 32
    radial_basis_dim: int = 8
    cutoff: float = 6.0
    max_atomic_number: int = 10
    num_layers: int = 4


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = 40
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    log_every: int = 5
    seed: int = 7


@dataclass(slots=True)
class OutputConfig:
    docs_dir: Path = Path("docs")
    figures_dirname: str = "figures"
    runs_dir: Path = Path("runs")
    metrics_filename: str = "metrics.json"
    parity_filename: str = "energy_parity.png"
    training_curve_filename: str = "training_curve.png"
    equivariance_filename: str = "rotation_invariance.png"

    @property
    def figures_dir(self) -> Path:
        return self.docs_dir / self.figures_dirname

    @property
    def metrics_path(self) -> Path:
        return self.docs_dir / self.metrics_filename


@dataclass(slots=True)
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
