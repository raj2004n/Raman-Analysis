"""Golden-snapshot plumbing.

A snapshot records the numbers the current code produces, so a later change can be
shown to have preserved them. Each value is stored as a small fingerprint rather
than the raw array: shape plus sum/min/max rounded to six decimals. That is stable
across machines and BLAS builds, and when a test fails the recorded and current
fingerprints print side by side, so the failure says *what* moved rather than just
that something did.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
GOLDEN_FILE = GOLDEN_DIR / "snapshots.json"

ROUND_TO = 6


def load_snapshots() -> dict:
    if not GOLDEN_FILE.exists():
        return {}
    return json.loads(GOLDEN_FILE.read_text())


def save_snapshots(snapshots: dict) -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    GOLDEN_FILE.write_text(json.dumps(snapshots, indent=2, sort_keys=True) + "\n")


def fingerprint(value) -> dict:
    """Describe a value so that a change to it is both detectable and readable."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"value": round(value, ROUND_TO) if isinstance(value, float) else value}

    if isinstance(value, dict):
        return {"items": {str(key): fingerprint(item) for key, item in sorted(value.items())}}

    array = np.asarray(value)

    if array.ndim == 0:
        return {"value": _plain(array.item())}

    if array.dtype.kind in "OSU" or array.dtype == object:
        # Strings and mixed content have no meaningful sum, so record them literally.
        return {"shape": list(array.shape), "values": [_plain(item) for item in array.ravel().tolist()]}

    if array.dtype.kind in "iub":
        return {
            "shape": list(array.shape),
            "sum": _plain(array.sum()),
            "min": _plain(array.min()),
            "max": _plain(array.max()),
        }

    rounded = np.round(array.astype(float), ROUND_TO)
    return {
        "shape": list(array.shape),
        "sum": round(float(rounded.sum()), ROUND_TO),
        "min": round(float(rounded.min()), ROUND_TO),
        "max": round(float(rounded.max()), ROUND_TO),
    }


def _plain(value):
    """numpy scalars are neither JSON-serialisable nor ``int``/``float`` instances."""
    return value.item() if isinstance(value, np.generic) else value
