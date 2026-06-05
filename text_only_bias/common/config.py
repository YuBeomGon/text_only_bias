"""Minimal YAML config loader."""
from __future__ import annotations

import os

import yaml

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "mvp.yaml")


def load_config(path: str | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, "r") as f:
        return yaml.safe_load(f)
