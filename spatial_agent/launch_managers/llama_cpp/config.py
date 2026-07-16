"""Load and validate models.json configuration for llama.cpp."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ModelConfig:
    name: str
    model: str
    served_name: str
    port: int = 8081
    is_mtp: bool = True


@dataclass
class ManagerConfig:
    models: List[ModelConfig] = field(default_factory=list)


def load_config(config_path: Path = None) -> ManagerConfig:
    """Load models.json from package directory (or custom path)."""
    if config_path is None:
        config_path = Path(__file__).parent / "models.json"

    if not config_path.exists():
        return ManagerConfig()

    with open(config_path, "r") as f:
        raw = json.load(f)

    models = []
    for m in raw.get("models", []):
        models.append(ModelConfig(
            name=m["name"],
            model=m["model"],
            served_name=m["served_name"],
            port=m.get("port", 8081),
            is_mtp=m.get("is_mtp", True),
        ))

    return ManagerConfig(models=models)
