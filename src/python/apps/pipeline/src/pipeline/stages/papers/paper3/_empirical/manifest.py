"""Provenance-manifest emission for the Paper 3 empirical stage."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pipeline.io.contracts import write_manifest

if TYPE_CHECKING:
    from pipeline.stages.papers.paper3._empirical.contracts import (
        Paper3EmpiricalConfig,
        Paper3EmpiricalPaths,
    )


def _sha256_path(path: Path) -> str:
    """Hash a file or directory with a deterministic path-aware digest."""
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if path.is_dir():
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(child.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with child.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\0")
        return digest.hexdigest()
    raise FileNotFoundError(path)


def _stage_source_digest() -> str:
    """Digest the whole Paper 3 empirical stage source, facade plus step package.

    Before the t42 split this was ``_sha256_path(Path(__file__))`` over the single
    1439-line ``empirical.py``. ``__file__`` is now one step module, so hashing it
    alone would certify a ``code_identity`` that stops moving when the panel,
    identity, backtest, or driver code changes -- a silent provenance regression
    on a ``risk_class: high`` manifest. ``_sha256_path`` over the package
    directory is not an option either: it ``rglob``s, so ``__pycache__`` would
    make the digest non-reproducible.

    Returns:
        Hex digest over every stage source file, keyed by its stable label.

    """
    package = Path(__file__).resolve().parent
    sources = [("empirical.py", package.parent / "empirical.py")]
    sources += [(f"{package.name}/{p.name}", p) for p in sorted(package.glob("*.py"))]
    digest = hashlib.sha256()
    for label, source in sources:
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_path(source).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _output_record(path: Path, root: Path, kind: str) -> dict[str, object]:
    """Return the contract output record for a materialized file."""
    return {
        "path": path.relative_to(root).as_posix(),
        "kind": kind,
        "sha256": _sha256_path(path),
        "bytes": path.stat().st_size,
    }


def _write_paper3_manifest(
    paths: Paper3EmpiricalPaths,
    config: Paper3EmpiricalConfig,
) -> None:
    """Write the deterministic provenance sidecar for Paper 3 empirics."""
    parameters = {
        "n_boot": config.n_boot,
        "n_clusters": config.n_clusters,
        "n_mc": config.n_mc,
        "seed": config.seed,
        "windows": list(config.windows),
        "provider_id": config.provider_id,
        "representation_id": config.representation_id,
        "distance_id": config.distance_id,
    }
    parameter_bytes = json.dumps(
        parameters, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest = {
        "schema_version": "1.0",
        "stage_key": "p3_empirical",
        "lane": "paper3",
        "phase": "empirical",
        "protocol_id": "paper3.random-functional-w2.v2",
        "code_identity": {
            "kind": "sha256",
            "value": _stage_source_digest(),
            "label": "paper3 empirical stage source",
        },
        "parameter_identity": {
            "kind": "sha256",
            "value": hashlib.sha256(parameter_bytes).hexdigest(),
            "label": "paper3 empirical invocation parameters",
        },
        "seed_policy": f"fixed:{config.seed}",
        "upstream_artifacts": [
            {
                "kind": "sha256",
                "value": _sha256_path(paths.returns_dir),
                "label": "shared returns directory",
            },
            {
                "kind": "sha256",
                "value": _sha256_path(paths.distance_artifact_dir),
                "label": "typed distance artifact directory",
            },
            {
                "kind": "sha256",
                "value": _sha256_path(
                    paths.output_dir.parent / "pit_w2" / "matrices.npz"
                ),
                "label": "point-in-time squared-W2 matrices",
            },
        ],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": sys.platform,
        },
        "outputs": [
            _output_record(
                paths.output_dir / "results.parquet", paths.output_dir, "parquet"
            ),
            _output_record(paths.output_dir / "summary.yaml", paths.output_dir, "yaml"),
            _output_record(
                paths.output_dir / "robustness_gel_gms.yaml",
                paths.output_dir,
                "yaml",
            ),
        ],
        "risk_class": "high",
        "metadata": {
            "parameters": parameters,
            "observed_object": "characteristic_law_C",
            "latent_object": "exposure_law_P=Law(T(X,U))",
        },
    }
    write_manifest(paths.output_dir / "provenance.manifest.json", manifest)
