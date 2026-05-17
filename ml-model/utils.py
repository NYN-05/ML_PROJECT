"""General utilities for the ML pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json


def ensure_dir(path: Path) -> None:
    """Ensure a directory exists.

    Args:
        path: Directory path to create.
    """
    path.mkdir(parents=True, exist_ok=True)


def save_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    """Save metadata to disk as JSON.

    Args:
        path: Output file path.
        metadata: Metadata dictionary.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_metadata(path: Path) -> Dict[str, Any]:
    """Load metadata from disk.

    Args:
        path: Metadata file path.

    Returns:
        Metadata dictionary.
    """
    return json.loads(path.read_text(encoding="utf-8"))
