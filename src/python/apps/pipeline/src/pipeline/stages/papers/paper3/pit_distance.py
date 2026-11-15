"""Point-in-time squared W2 matrices for Paper 3.

Each annual vintage is a typed, rooted ``W_2`` artifact computed from article
embedding laws observed by the declared cutoff.  Paper 3's covariance envelope
uses the quadratic transport cost, so this module performs the single governed
conversion ``W_2 -> W_2^2`` while assembling the vintage archive.

The article laws are observable characteristic laws ``C_i``.  They are not
silently relabelled as latent exposure laws ``P_i``; the manuscript's
transmission restriction is what connects the two geometries.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
from typing import TYPE_CHECKING, Any, NoReturn, cast

import numpy as np
import yaml

from pipeline.io.contracts import write_manifest
from pipeline.io.semantic_identity import entrypoint_code_identity
from pipeline.io.typed_analysis import (
    StatisticalDistanceSpec,
    load_typed_analysis,
    typed_analysis_parameter_identity,
)
from pipeline.jax_cache import configure_persistent_cache
from pipeline.stages.shared.typed_artifacts import read_typed_distance_artifact

if TYPE_CHECKING:
    from pathlib import Path

configure_persistent_cache()

logger = logging.getLogger(__name__)

PIT_PROVIDER_ID = "qwen3-embedding-8b"
PIT_REPRESENTATION_ID = "qwen3-embedding-8b-unit"


class PitDistanceError(ValueError):
    """Raised when the Paper 3 W2 vintage contract is violated."""


def _pit_error(message: str) -> NoReturn:
    raise PitDistanceError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vintage_roster(params_all: dict[str, Any]) -> list[tuple[str, str]]:
    """Read ``(vintage, distance_id)`` pairs from the Paper 3 registry."""
    grid = params_all["paper3"]["pit_vintage_grid"]
    if not isinstance(grid, list) or not grid:
        _pit_error("paper3.pit_vintage_grid must be a nonempty list")
    roster: list[tuple[str, str]] = []
    for entry in grid:
        if not isinstance(entry, dict) or set(entry) != {"vintage", "distance_id"}:
            _pit_error(
                "each pit_vintage_grid entry must carry exactly vintage and "
                f"distance_id, got {entry!r}"
            )
        roster.append((str(entry["vintage"]), str(entry["distance_id"])))
    if len({vintage for vintage, _ in roster}) != len(roster):
        _pit_error("pit_vintage_grid repeats a vintage date")
    if len({distance_id for _, distance_id in roster}) != len(roster):
        _pit_error("pit_vintage_grid repeats a distance ID")
    return roster


def _parameter_identity(params_file: Path, roster: list[tuple[str, str]]) -> str:
    typed_identity = typed_analysis_parameter_identity(
        params_file,
        provider_ids=(PIT_PROVIDER_ID,),
        representation_ids=(PIT_REPRESENTATION_ID,),
        distance_ids=tuple(distance_id for _vintage, distance_id in roster),
    )
    payload = json.dumps(
        {
            "paper3.pit_vintage_grid": [
                {"vintage": vintage, "distance_id": distance_id}
                for vintage, distance_id in roster
            ],
            "typed_analysis_parameter_identity": typed_identity,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_w2(distance: StatisticalDistanceSpec) -> None:
    if (
        distance.family != "wasserstein"
        or distance.estimator != "balanced_wasserstein_2"
        or distance.normalization != "rooted"
        or distance.value_semantics != "statistical_distance"
    ):
        _pit_error(
            f"{distance.distance_id!r} must be rooted balanced W2, got "
            f"{distance.family}/{distance.estimator}/{distance.normalization}"
        )


def _square_rooted_w2(values: np.ndarray) -> np.ndarray:
    """Convert rooted W2 values to the quadratic cost used by polarization."""
    rooted = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(rooted)) or np.any(rooted < 0.0):
        _pit_error("rooted W2 matrix contains a nonfinite or negative value")
    squared = np.square(rooted, dtype=np.float64)
    np.fill_diagonal(squared, 0.0)
    return squared


def _load_vintage_matrix(
    distance_root: Path,
    vintage: str,
    distance: StatisticalDistanceSpec,
    expected_end: str,
    expected_items: tuple[str, ...] | None,
) -> tuple[np.ndarray, tuple[str, ...], str]:
    """Read one typed rooted-W2 artifact and return its squared matrix."""
    _require_w2(distance)
    if vintage != expected_end:
        _pit_error(
            f"vintage {vintage!r} disagrees with the window end {expected_end!r} "
            f"declared by distance {distance.distance_id!r}"
        )
    directory = (
        distance_root / PIT_PROVIDER_ID / PIT_REPRESENTATION_ID / distance.distance_id
    )
    frame, summary = read_typed_distance_artifact(
        directory,
        expected_identity={
            "provider_id": PIT_PROVIDER_ID,
            "representation_id": PIT_REPRESENTATION_ID,
            "distance_id": distance.distance_id,
            "value_semantics": "statistical_distance",
        },
    )
    item_ids = summary.get("item_ids")
    metadata = summary.get("metadata")
    if not isinstance(item_ids, list) or not isinstance(metadata, dict):
        _pit_error(f"{distance.distance_id!r} has malformed typed metadata")
    items = tuple(str(item) for item in item_ids)
    if expected_items is not None and items != expected_items:
        _pit_error(
            f"{distance.distance_id!r} covers a different item roster than the "
            "first vintage"
        )
    sample_window = metadata.get("sample_window")
    expected_window = distance.sample_window
    if sample_window != expected_window:
        _pit_error(
            f"{distance.distance_id!r} artifact window {sample_window!r} disagrees "
            f"with registry window {expected_window!r}"
        )
    rooted = frame["value"].to_numpy(dtype=np.float64).reshape(len(items), len(items))
    return _square_rooted_w2(rooted), items, str(sample_window)


def compute_paper3_pit_distance(
    params_file: Path,
    out_path: Path,
    distance_root: Path,
) -> None:
    """Assemble Paper 3's annual squared-W2 archive from typed artifacts."""
    with params_file.open(encoding="utf-8") as params_handle:
        params_all = yaml.safe_load(params_handle)
    config = load_typed_analysis(params_file)
    roster = _vintage_roster(params_all)

    matrices: dict[str, np.ndarray] = {}
    windows: dict[str, str] = {}
    vintage_identities: dict[str, dict[str, object]] = {}
    items: tuple[str, ...] | None = None
    for vintage, distance_id in roster:
        distance = config.distances.get(distance_id)
        if not isinstance(distance, StatisticalDistanceSpec):
            _pit_error(f"{distance_id!r} is not a registered statistical distance")
        window = config.window_for(distance)
        if window is None or window.end is None:
            _pit_error(f"{distance_id!r} declares no closed point-in-time window")
        values, items_seen, window_id = _load_vintage_matrix(
            distance_root, vintage, distance, window.end.isoformat(), items
        )
        items = items_seen
        matrices[vintage] = values
        windows[vintage] = window_id
        vintage_identities[vintage] = {
            "provider_id": PIT_PROVIDER_ID,
            "representation_id": PIT_REPRESENTATION_ID,
            "distance_id": distance.distance_id,
            "family": distance.family,
            "ground_distance": distance.ground_distance,
            "estimator": distance.estimator,
            "normalization": distance.normalization,
            "sample_design": distance.sample_design,
            "sample_size": distance.sample_size,
            "sample_window_id": distance.sample_window,
            "window_start": (
                window.start.isoformat() if window.start is not None else None
            ),
            "window_end": window.end.isoformat(),
            "source_scale": "rooted_wasserstein_2",
            "archive_scale": "squared_wasserstein_2_transport_cost",
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **cast("dict[str, Any]", dict(matrices)))
    write_manifest(
        out_path.with_name("provenance.manifest.json"),
        {
            "schema_version": "1.0",
            "stage_key": "p3_pit_w2",
            "lane": "p3",
            "phase": "prepare",
            "protocol_id": "paper3.pit-w2.v2",
            "code_identity": {
                "kind": "sha256",
                "value": entrypoint_code_identity("paper3-pit-w2"),
            },
            "parameter_identity": {
                "kind": "scoped-params-sha256",
                "value": _parameter_identity(params_file, roster),
            },
            "seed_policy": "deterministic-from-declared-inputs",
            "upstream_artifacts": [
                {
                    "kind": "path",
                    "value": str(
                        distance_root
                        / PIT_PROVIDER_ID
                        / PIT_REPRESENTATION_ID
                        / distance_id
                    ),
                    "vintage": vintage,
                    "distance_id": distance_id,
                    "provider_id": PIT_PROVIDER_ID,
                    "representation_id": PIT_REPRESENTATION_ID,
                }
                for vintage, distance_id in roster
            ],
            "environment": {"python": platform.python_version()},
            "metadata": {
                "item_ids": list(items or ()),
                "vintage_windows": windows,
                "vintage_identity_schema": "pit_vintage_identity.v1",
                "vintage_identities": vintage_identities,
                "source_scale": "rooted_wasserstein_2",
                "archive_scale": "squared_wasserstein_2_transport_cost",
                "observed_object": "characteristic_law_C",
            },
            "outputs": [
                {
                    "path": str(out_path),
                    "kind": "npz",
                    "sha256": _sha256(out_path),
                    "bytes": out_path.stat().st_size,
                }
            ],
            "risk_class": "high",
        },
    )
    logger.info("wrote PIT squared-W2 matrices to %s", out_path)
