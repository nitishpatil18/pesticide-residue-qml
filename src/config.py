"""Project configuration helpers."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "default.json"


def load_config(path: Path | str | None = None) -> dict:
    with open(path or CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


CFG = load_config()
SEED = CFG["seed"]


def path(key: str) -> Path:
    """Absolute path for a key in cfg['paths'] (directory is created if needed)."""
    p = ROOT / CFG["paths"][key]
    if key.endswith("_dir"):
        p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_json(obj, file: Path | str) -> None:
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(file, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=default)


def load_json(file: Path | str):
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)
