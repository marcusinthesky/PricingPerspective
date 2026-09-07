"""Shared access to the complete ``params.yaml`` document."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from pathlib import Path


def load_params(path: Path) -> dict[str, Any]:
    """Load the full ``params.yaml`` document as a dict."""
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)
