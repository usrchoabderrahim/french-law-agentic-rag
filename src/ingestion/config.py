from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load the project YAML configuration."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration introuvable : {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}
