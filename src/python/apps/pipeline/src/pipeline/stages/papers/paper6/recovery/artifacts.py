"""Frozen inputs, semantic code identity and atomic recovery checkpoints."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
from hotelling.artifacts import reject_symlinks
from hotelling.dynamics.artifacts import (
    file_hash,
    read_arrays,
    repository_root,
    value_hash,
    write_arrays,
)
from hotelling.dynamics.recovery import FIRMS, PERIODS

from pipeline._kernels.paper6.recovery import HistoryMeasurement
from pipeline.io.semantic_identity import standalone_entrypoint_code_identity
from pipeline.stages.papers.paper6.artifacts import read, write

SUPPORT = "data/papers/paper6/historical-evidence/t79/article-support.npz"
INVENTORY = "data/papers/paper6/review-artifacts/t81/inventory.json"
FROZEN_SUPPORT_KEY = SUPPORT
COMMAND = "paper6-information-recovery"


def open_campaign(output: Path) -> Path:
    """Permit governed new outputs and resumptions, preserving historical evidence."""
    reject_symlinks(output.absolute())
    output = output.resolve()
    allowed = repository_root() / "data/papers/paper6/information_recovery"
    temporary = Path(tempfile.gettempdir()).resolve()
    if output == temporary or not (
        output.is_relative_to(allowed) or output.is_relative_to(temporary)
    ):
        msg = "output requires the information_recovery namespace or /tmp"
        raise ValueError(msg)
    if (
        output.exists()
        and any(output.iterdir())
        and not (output / "manifest.json").is_file()
    ):
        msg = "nonempty recovery directory has no resumable manifest"
        raise FileExistsError(msg)
    output.mkdir(parents=True, exist_ok=True)
    return output


def load_month_counts() -> tuple[
    np.ndarray[Any, np.dtype[np.int64]], dict[str, object]
]:
    """Verify frozen T81 identity and all 500 expanded firm-year counts."""
    root = repository_root()
    inventory = read(root / INVENTORY)
    if file_hash(root / SUPPORT) != inventory["input_sha256"][FROZEN_SUPPORT_KEY]:
        msg = "expanded article support differs from the frozen T81 inventory"
        raise ValueError(msg)
    with np.load(root / SUPPORT, allow_pickle=False) as saved:
        firms = saved["firm_ids"].tolist()
        dates = saved["dates"].astype("datetime64[M]").astype(np.int64) - np.datetime64(
            "2018-01", "M"
        ).astype(np.int64)
        indexes = saved["firm_index"]
    if (
        firms != inventory["firm_ids"]
        or len(firms) != FIRMS
        or np.any((dates < 0) | (dates >= PERIODS))
    ):
        msg = "expanded article dates or firm ordering differ from T81"
        raise ValueError(msg)
    counts = np.zeros((5, 12, 100), dtype=np.int64)
    np.add.at(counts, (dates // 12, dates % 12, indexes), 1)
    expected = {
        (row["year"], row["firm"]): row["expanded"]
        for row in inventory["annual_counts"]
    }
    for year in range(5):
        for firm, name in enumerate(firms):
            if int(counts[year, :, firm].sum()) != expected[(2018 + year, name)]:
                msg = f"expanded monthly count disagrees with T81: {name}/{2018 + year}"
                raise ValueError(msg)
    return counts, {
        "support": SUPPORT,
        "inventory": INVENTORY,
        "hashes": {name: file_hash(root / name) for name in (SUPPORT, INVENTORY)},
        "firm_ids": firms,
        "articles": int(counts.sum()),
    }


def load_stratified_month_counts() -> tuple[
    np.ndarray[Any, np.dtype[np.int64]], dict[str, object]
]:
    """Retain matched/additional month counts without changing the legacy loader."""
    expanded, inputs = load_month_counts()
    root = repository_root()
    inventory = read(root / INVENTORY)
    with np.load(root / SUPPORT, allow_pickle=False) as source:
        core = source["core"].astype(bool)
        months = source["dates"].astype("datetime64[M]").astype(
            np.int64
        ) - np.datetime64("2018-01", "M").astype(np.int64)
        firms = source["firm_index"]
    counts = np.zeros((2, *expanded.shape), dtype=np.int64)
    np.add.at(counts, ((~core).astype(np.int64), months // 12, months % 12, firms), 1)
    np.testing.assert_array_equal(counts.sum(axis=0), expanded)
    positions = {firm: i for i, firm in enumerate(inventory["firm_ids"])}
    for row in inventory["annual_counts"]:
        index = (row["year"] - 2018, positions[row["firm"]])
        core_total = int(counts[0, index[0], :, index[1]].sum())
        added_total = int(counts[1, index[0], :, index[1]].sum())
        if (
            core_total <= 0
            or core_total != row["matched"]
            or added_total != row["expanded"] - row["matched"]
        ):
            msg = f"stratified count differs from T81: {row['firm']}/{row['year']}"
            raise ValueError(msg)
    return counts, {
        **inputs,
        "strata": ["core", "additional"],
        "firm_years": len(inventory["annual_counts"]),
    }


def capture_provenance(output: Path, inputs: dict[str, object]) -> dict[str, object]:
    """Compute the standalone Semflow identity and retain frozen numerical inputs."""
    root = repository_root()
    for name in (SUPPORT, INVENTORY):
        target = output / "inputs" / Path(name).name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    return {
        "code_identity": standalone_entrypoint_code_identity(COMMAND),
        "inputs": inputs,
    }


def verify_provenance(
    output: Path, provenance: dict[str, Any], *, current: bool
) -> None:
    """Audit frozen evidence; require the same Semflow identity for resumption.

    Old source archives remain auditable, but proof/tooling byte changes do not
    define numerical compatibility. Legacy runs need an explicitly reviewed
    code-identity adoption before they can resume.
    """
    identities = provenance.get("files", {})
    if "files" in provenance and value_hash(identities) != provenance["source_hash"]:
        msg = "recovery source inventory hash mismatch"
        raise ValueError(msg)
    root = repository_root()
    for relative, digest in identities.items():
        path = Path(relative)
        if (
            path.is_absolute()
            or ".." in path.parts
            or file_hash(output / "sources" / path) != digest
        ):
            msg = f"recovery source snapshot mismatch: {relative}"
            raise ValueError(msg)
    for relative, digest in provenance["inputs"]["hashes"].items():
        if file_hash(output / "inputs" / Path(relative).name) != digest or (
            current and file_hash(root / relative) != digest
        ):
            msg = f"recovery frozen input mismatch: {relative}"
            raise ValueError(msg)
    if current and (
        provenance.get("code_identity") != standalone_entrypoint_code_identity(COMMAND)
    ):
        msg = (
            "recovery checkpoint code identity differs from the current Semflow "
            "entrypoint; preserve the saved campaign and explicitly review legacy "
            "identity adoption or start a fresh campaign for changed numerical code"
        )
        raise ValueError(msg)


def save_measurement(path: Path, result: HistoryMeasurement) -> str:
    """Persist undefined estimates using an explicit mask and finite stored arrays."""
    arrays = {
        field.name: np.asarray(getattr(result, field.name), dtype=np.float64)
        for field in fields(result)
    }
    arrays["defined"] = np.isfinite(result.estimates).astype(np.float64)
    arrays["estimates"] = np.where(arrays["defined"], result.estimates, 0)
    write_arrays(path, arrays)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return file_hash(path)


def load_measurement(path: Path, digest: str) -> HistoryMeasurement:
    """Reject changed checkpoints before loading deterministic statistics."""
    if file_hash(path) != digest:
        msg = f"recovery checkpoint hash mismatch: {path.name}"
        raise ValueError(msg)
    arrays = read_arrays(path)
    estimates = np.where(arrays["defined"], arrays["estimates"], np.nan)
    return HistoryMeasurement(
        float(arrays["beta0"]),
        float(arrays["affine_beta"]),
        arrays["population_beta"],
        arrays["reporting_only_beta"],
        estimates,
        float(arrays["maximum_solver_bound"]),
        float(arrays["minimum_capacity"]),
        float(arrays["minimum_baseline_price"]),
        float(arrays["minimum_opportunity"]),
        int(arrays["active_domain_violations"]),
    )


def checkpoint(output: Path, manifest: dict[str, Any]) -> None:
    """Publish the parent-owned ledger only after referenced arrays are durable."""
    write(output / "manifest.json", manifest)
