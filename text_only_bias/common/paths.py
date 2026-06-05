"""Output path resolution relative to the package root.

Outputs are organized per experiment:
  outputs/common/eval_reports/   (shared, e.g. baseline)
  outputs/exp01/{checkpoints,eval_reports}
  outputs/exp02/...
  outputs/exp03/...
"""
from __future__ import annotations

import os

# text_only_bias/  (parent of common/)
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def out_dir(exp: str, kind: str) -> str:
    """Return (and create) outputs/<exp>/<kind>. kind in {checkpoints, eval_reports}."""
    d = os.path.join(PKG_ROOT, "outputs", exp, kind)
    os.makedirs(d, exist_ok=True)
    return d
