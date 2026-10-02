"""Typed experiment configuration, serialized with every run."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .models import ModelConfig


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 1111
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    patience: int = 6
    gradient_clip: float = 1.0
    device: str = "auto"
    num_workers: int = 0
    threads: int = 4

    def __post_init__(self):
        if (
            min(self.epochs, self.batch_size, self.patience, self.threads) < 1
            or self.num_workers < 0
        ):
            raise ValueError(
                "epochs, batch_size, patience and threads must be positive; workers >= 0"
            )
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("Invalid optimizer configuration")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "experiment"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def config_from_dict(value: dict) -> ExperimentConfig:
    if set(value) - {"name", "model", "training"}:
        raise ValueError("Unknown top-level config keys")
    return ExperimentConfig(
        name=value.get("name", "experiment"),
        model=ModelConfig(**value.get("model", {})),
        training=TrainingConfig(**value.get("training", {})),
    )


def load_config(path: Path) -> ExperimentConfig:
    with Path(path).open("rb") as stream:
        return config_from_dict(tomllib.load(stream))
