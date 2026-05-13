"""Configuration dataclasses and default loader."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClassesConfig:
    polymer: int = 0
    background: int = 1
    cell_body: int = 2
    cell_border: int = 3


@dataclass
class InstancesConfig:
    closing_radius: int = 2
    peak_min_distance: int = 12
    peak_threshold_rel: float = 0.1
    min_area: int = 550
    max_area: int = 6000
    drop_edge_touching: bool = True


@dataclass
class Config:
    classes: ClassesConfig = None  # type: ignore[assignment]
    instances: InstancesConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.classes is None:
            self.classes = ClassesConfig()
        if self.instances is None:
            self.instances = InstancesConfig()


def load_config(*_overrides) -> Config:
    """Return a Config with default values."""
    return Config()
