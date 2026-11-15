"""Persistence boundary for Paper 3 certificate-feasibility artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from pipeline.io.contracts import write_manifest

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import pandas as pd


def sha256_path(path: Path) -> str:
    """Hash a file or directory deterministically by relative path and bytes."""
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    for child in sorted(
        candidate for candidate in path.rglob("*") if candidate.is_file()
    ):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(child.read_bytes())
    return digest.hexdigest()


def write_certificate_artifacts(
    output_dir: Path,
    summary: Mapping[str, object],
    *,
    news_weights: pd.DataFrame,
    code_identity: str,
    parameter_identity: str,
    upstream_paths: tuple[Path, ...],
) -> None:
    """Write news weights, summary, and their governed manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "news_only_weights.parquet"
    summary_path = output_dir / "results.json"
    news_weights.to_parquet(weights_path, index=False)
    summary_path.write_text(
        json.dumps(dict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    outputs = [
        {
            "path": str(path),
            "kind": path.suffix.removeprefix("."),
            "sha256": sha256_path(path),
            "bytes": path.stat().st_size,
        }
        for path in (weights_path, summary_path)
    ]
    write_manifest(
        output_dir / "provenance.manifest.json",
        {
            "schema_version": "1.0",
            "stage_key": "p3_certificate",
            "lane": "paper3",
            "phase": "feasibility",
            "protocol_id": "paper3.information_certificate.v1",
            "code_identity": {"kind": "sha256", "value": code_identity},
            "parameter_identity": {
                "kind": "sha256",
                "value": parameter_identity,
            },
            "seed_policy": "deterministic-no-randomness",
            "upstream_artifacts": [
                {"kind": "path-sha256", "value": f"{path}:{sha256_path(path)}"}
                for path in upstream_paths
            ],
            "environment": {
                "compute_dtype": "float64",
                "evidence_status": "full_sample_feasibility",
            },
            "outputs": outputs,
            "risk_class": "moderate",
            "metadata": dict(summary),
        },
    )
