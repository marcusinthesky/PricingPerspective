"""Stable programmatic API for checking, previewing, and applying semflow state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from semflow.config import Config, load_config
from semflow.dvc import apply_patch, apply_refresh, build_patch, build_refresh

if TYPE_CHECKING:
    from pathlib import Path

    from semflow.model import PatchPlan


def check(config: Config | Path | None = None) -> PatchPlan:
    """Return the changes required to make semantic fingerprints current."""
    return build_patch(_resolve_config(config))


def patch(
    config: Config | Path | None = None,
    *,
    apply: bool = False,
) -> PatchPlan:
    """Build a patch and optionally apply it to the configured repository."""
    plan = check(config)
    if apply and not plan.clean:
        apply_patch(plan)
    return plan


def refresh(
    config: Config | Path | None = None,
    *,
    apply: bool = False,
) -> PatchPlan:
    """Build or apply a JSON-only fingerprint refresh plan."""
    plan = build_refresh(_resolve_config(config))
    if apply and not plan.clean:
        apply_refresh(plan)
    return plan


def _resolve_config(config: Config | Path | None) -> Config:
    if isinstance(config, Config):
        return config
    return load_config(config)
