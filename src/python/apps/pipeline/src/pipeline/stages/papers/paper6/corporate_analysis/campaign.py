"""Bounded, resumable corporate measurement with frozen-input reconciliation."""

from __future__ import annotations

import logging
import multiprocessing
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml
from hotelling.artifacts import reject_symlinks
from hotelling.dynamics.artifacts import file_hash, repository_root, value_hash
from pydantic import BaseModel, ConfigDict, Field
from threadpoolctl import threadpool_limits

from pipeline._kernels.paper6.corporate_analysis import (
    DIMENSIONS,
    OBJECTIVE_TOLERANCE,
    SUPPORTS,
    YEARS,
    corporate_summary,
    point_masses,
    redraw_masses,
    return_readout,
)
from pipeline._kernels.paper6.reference_alignment import moment, summarize
from pipeline.io.semantic_identity import entrypoint_code_identity
from pipeline.stages.papers.paper6.artifacts import read, write
from pipeline.stages.papers.paper6.corporate_analysis.data import (
    BASELINE,
    financial,
    inventory,
    load_clouds,
)
from pipeline.stages.papers.paper6.corporate_analysis.geometry import (
    CostCache,
    build_cache,
    certificates,
    distances,
)

if TYPE_CHECKING:
    from pipeline._kernels.paper6.corporate_analysis import Cloud, CloudKey, FloatArray

COMMAND = "paper6-corporate-analysis"
CELLS = tuple(f"d{dim}_{support}" for dim in DIMENSIONS for support in SUPPORTS)
LEGACY_SEED = 20261026
RECONCILIATION_TOLERANCE = 1e-12
MAX_WORKERS = 3
SCRATCH = Path(tempfile.gettempdir()) / "paper6-corporate-analysis"
_LOG = logging.getLogger(__name__)
_WORKER: tuple[Any, ...] | None = None


class CorporateConfig(BaseModel):
    """Scientific configuration; execution resources do not change the design."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    draws: int = Field(default=199, ge=1)
    seed: int = Field(default=20261026, ge=0)


def read_config(path: Path) -> CorporateConfig:
    """Read the declared Paper 6 corporate-analysis scientific parameters."""
    params = yaml.safe_load(path.read_text(encoding="utf-8"))
    return CorporateConfig.model_validate(params["paper6"]["corporate_analysis"])


def _open_output(path: Path) -> Path:
    reject_symlinks(path.absolute())
    output = path.resolve()
    allowed = repository_root() / "data/papers/paper6/corporate_analysis"
    temporary = Path(tempfile.gettempdir()).resolve()
    if output == temporary or not (
        output.is_relative_to(allowed) or output.is_relative_to(temporary)
    ):
        msg = "corporate output requires its analysis namespace or a /tmp subdirectory"
        raise ValueError(msg)
    if (
        output.exists()
        and any(output.iterdir())
        and not (output / "manifest.json").is_file()
    ):
        msg = "nonempty corporate output has no resumable manifest"
        raise FileExistsError(msg)
    output.mkdir(parents=True, exist_ok=True)
    return output


def atomic_npz(path: Path, arrays: dict[str, Any]) -> None:
    """Admit a complete archive, including intentionally excluded diagonal NaNs."""
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _numeric_difference(actual: object, expected: object) -> float:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            msg = "baseline record type differs"
            raise TypeError(msg)
        return max(
            (
                _numeric_difference(actual[key], value)
                for key, value in expected.items()
            ),
            default=0,
        )
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            msg = "baseline record membership differs"
            raise ValueError(msg)
        return max(
            (_numeric_difference(a, b) for a, b in zip(actual, expected, strict=True)),
            default=0,
        )
    if isinstance(expected, (int, float)):
        if not isinstance(actual, (int, float)):
            msg = "baseline numerical field differs"
            raise TypeError(msg)
        return abs(float(actual) - float(expected))
    if actual != expected:
        msg = "baseline label or unavailable value differs"
        raise ValueError(msg)
    return 0


def reconcile_saved() -> dict[str, Any]:
    """Replay current moments, all-role deletions and redraws on identical arrays."""
    root = repository_root() / BASELINE
    saved = read(root / "results.json")
    results = {}
    with np.load(root / "measurement.npz", allow_pickle=False) as source:
        for name in CELLS:
            support = name.rsplit("_", 1)[1]
            rebuilt = summarize(
                source[f"{name}_D"],
                source[f"{support}_R"],
                source["firm_ids"].astype(str),
            )
            expected = {
                key: value
                for key, value in saved["cells"][name].items()
                if key in rebuilt
            }
            point_error = _numeric_difference(rebuilt, expected)
            draw_beta = np.asarray(
                [
                    moment(d, r)["beta"]
                    for d, r in zip(
                        source[f"draw_{name}_D"],
                        source[f"draw_{support}_R"],
                        strict=True,
                    )
                ],
                dtype=float,
            )
            draw_error = float(np.max(np.abs(draw_beta - source[f"draw_{name}_beta"])))
            quantiles = np.quantile(draw_beta, [0.025, 0.05, 0.975])
            expected_quantiles = [
                saved["cells"][name]["article_measurement_percentile95"][0],
                saved["cells"][name]["article_measurement_lower95"],
                saved["cells"][name]["article_measurement_percentile95"][1],
            ]
            quantile_error = float(np.max(np.abs(quantiles - expected_quantiles)))
            maximum = max(point_error, draw_error, quantile_error)
            if maximum > RECONCILIATION_TOLERANCE:
                msg = (
                    f"identical frozen-array reconciliation failed: {name}; "
                    f"maximum difference {maximum}"
                )
                raise ValueError(msg)
            results[name] = {
                "beta": rebuilt["beta"],
                "theta": rebuilt["theta"],
                "second_moment": rebuilt["second_moment"],
                "point_summary_max_abs_difference": point_error,
                "redraw_beta_max_abs_difference": draw_error,
                "quantile_max_abs_difference": quantile_error,
                "draws": len(draw_beta),
            }
    return {
        "status": "reconciled",
        "identical_array_absolute_tolerance": 1e-12,
        "cells": results,
    }


def _initialize(
    directory: Path,
    clouds: dict[CloudKey, Cloud],
    panels: dict[int, tuple[np.ndarray, FloatArray]],
    counts: np.ndarray,
    seed: int,
    identity: str,
    checkpoints: Path,
    threads: int,
) -> None:
    global _WORKER  # noqa: PLW0603
    _WORKER = (
        CostCache(directory, counts),
        clouds,
        panels,
        seed,
        identity,
        checkpoints,
        threadpool_limits(limits=threads),
    )


def _draw(number: int) -> int:
    if _WORKER is None:
        msg = "corporate worker has not been initialized"
        raise RuntimeError(msg)
    cache, clouds, panels, seed, identity, checkpoints, _ = _WORKER
    masses = redraw_masses(clouds, seed, number)
    financials = {
        support: return_readout(clouds, panels, masses[support], cache.n)
        for support in SUPPORTS
    }
    arrays = {"identity": np.asarray(identity), "draw": np.asarray(number)}
    offdiag = ~np.eye(cache.n, dtype=bool)
    with np.load(
        repository_root() / BASELINE / "measurement.npz", allow_pickle=False
    ) as baseline:
        historical_draws = baseline["draw_d2560_expanded_beta"].shape[0]
        for dimension in DIMENSIONS:
            for support in SUPPORTS:
                name = f"d{dimension}_{support}"
                distance = distances(cache, dimension, masses[support])
                stats = moment(distance, financials[support])
                for key, value in stats.items():
                    arrays[f"{name}_{key}"] = np.asarray(
                        np.nan if value is None else value
                    )
                arrays[f"{name}_defined"] = np.asarray(stats["beta"] is not None)
                if number < historical_draws and seed == LEGACY_SEED:
                    # Select the draw before excluding diagonal observations.
                    d_error = float(
                        np.max(
                            np.abs(
                                distance[:, offdiag]
                                - baseline[f"draw_{name}_D"][number][:, offdiag]
                            )
                        )
                    )
                    r_error = float(
                        np.max(
                            np.abs(
                                financials[support][:, offdiag]
                                - baseline[f"draw_{support}_R"][number][:, offdiag]
                            )
                        )
                    )
                    arrays[f"{name}_distance_replay_max_abs_difference"] = np.asarray(
                        d_error
                    )
                    arrays[f"{name}_financial_replay_max_abs_difference"] = np.asarray(
                        r_error
                    )
                    arrays[f"{name}_beta_replay_abs_difference"] = np.asarray(
                        abs(stats["beta"] - baseline[f"draw_{name}_beta"][number])
                    )
                    if (
                        d_error > OBJECTIVE_TOLERANCE
                        or r_error > RECONCILIATION_TOLERANCE
                    ):
                        msg = (
                            "raw redraw replay differs from frozen inputs: "
                            f"{number}/{name}"
                        )
                        raise ValueError(msg)
    atomic_npz(checkpoints / f"draw-{number:04d}.npz", arrays)
    return number


def validate_checkpoint(
    path: Path, identity: str, number: int, digest: str
) -> dict[str, np.ndarray]:
    """Require content integrity, design identity and every declared scalar cell."""
    if file_hash(path) != digest:
        msg = f"corporate checkpoint hash mismatch: {path.name}"
        raise ValueError(msg)
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    if str(arrays["identity"]) != identity or int(arrays["draw"]) != number:
        msg = f"corporate checkpoint design mismatch: {path.name}"
        raise ValueError(msg)
    for name in CELLS:
        for key in ("theta", "second_moment"):
            if arrays[f"{name}_{key}"].shape != () or not np.isfinite(
                arrays[f"{name}_{key}"]
            ):
                msg = f"invalid checkpoint moment: {path.name}/{name}/{key}"
                raise ValueError(msg)
        if bool(arrays[f"{name}_defined"]) and not np.isfinite(arrays[f"{name}_beta"]):
            msg = f"invalid defined checkpoint coefficient: {path.name}/{name}"
            raise ValueError(msg)
    return arrays


def _manifest(
    output: Path, config: CorporateConfig, inputs: dict[str, Any]
) -> dict[str, Any]:
    design = {
        "schema_version": 1,
        "configuration": config.model_dump(mode="json"),
        "cells": list(CELLS),
        "years": list(YEARS),
        "reference_year": YEARS[0],
        "target": "complete retained archive projection beta_archive",
        "sampling": (
            "paired canonical/additional-stratum article redraws; "
            "one shared origin draw per firm across all appearances"
        ),
        "financial_timing": (
            "article date strictly after return date; daily raw excess returns"
        ),
        "geometry": "squared W2; prefix then normalize; frozen 2018 reference laws",
        "weighting": (
            "equal target-year and equal nonself reference; center within target-year"
        ),
        "code_identity": entrypoint_code_identity(COMMAND),
        "software": {name: version(name) for name in ("numpy", "scipy", "POT")},
        "inputs": inputs,
    }
    identity = value_hash(design)
    path = output / "manifest.json"
    if path.exists():
        saved = read(path)
        if saved["identity"] != identity or saved["design"] != design:
            msg = "corporate manifest incompatible with current design, code or inputs"
            raise ValueError(msg)
        return saved
    manifest: dict[str, Any] = {
        "identity": identity,
        "design": design,
        "checkpoints": {},
        "artifacts": {},
    }
    write(path, manifest)
    return manifest


def _completed(output: Path, manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Validate completed evidence before permitting a cheap resumed read."""
    identity = manifest["identity"]
    result_path = output / "results.json"
    if result_path.exists():
        expected_artifacts = {"results.json", "reconciliation.json", "measurement.npz"}
        if set(manifest.get("artifacts", {})) != expected_artifacts or any(
            file_hash(output / name) != digest
            for name, digest in manifest["artifacts"].items()
        ):
            msg = "completed corporate artifact hash mismatch"
            raise ValueError(msg)
        result = read(result_path)
        draws = manifest["design"]["configuration"]["draws"]
        if (
            result["manifest_identity"] != identity
            or file_hash(output / "measurement.npz") != result["measurement_sha256"]
            or result["status"] != "complete"
            or result["draws"] != draws
            or set(manifest["checkpoints"]) != {str(number) for number in range(draws)}
        ):
            msg = "completed corporate analysis integrity mismatch"
            raise ValueError(msg)
        for number, digest in manifest["checkpoints"].items():
            validate_checkpoint(
                output / "checkpoints" / f"draw-{int(number):04d}.npz",
                identity,
                int(number),
                digest,
            )
        return result
    return None


def _point_analysis(
    cache: CostCache,
    clouds: dict[CloudKey, Cloud],
    panels: dict[int, tuple[np.ndarray, FloatArray]],
    firms: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    point: dict[str, Any] = {"firm_ids": np.asarray(firms), "years": np.asarray(YEARS)}
    cells = {}
    for support in SUPPORTS:
        masses = point_masses(clouds, support)
        point[f"{support}_R"] = return_readout(clouds, panels, masses, cache.n)
        for dimension in DIMENSIONS:
            name = f"d{dimension}_{support}"
            point[f"{name}_D"] = distances(cache, dimension, masses)
            cells[name] = corporate_summary(
                point[f"{name}_D"], point[f"{support}_R"], firms
            )
    return point, cells


def _save_completed_draw(output: Path, manifest: dict[str, Any], number: int) -> None:
    manifest["checkpoints"][str(number)] = file_hash(
        output / "checkpoints" / f"draw-{number:04d}.npz"
    )
    write(output / "manifest.json", manifest)
    _LOG.info("Corporate article redraws completed: %d", len(manifest["checkpoints"]))


def _execute_draws(
    output: Path,
    config: CorporateConfig,
    manifest: dict[str, Any],
    args: tuple[Any, ...],
    workers: int,
) -> None:
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    pending = []
    for number in range(config.draws):
        path = checkpoints / f"draw-{number:04d}.npz"
        if str(number) in manifest["checkpoints"]:
            validate_checkpoint(
                path, manifest["identity"], number, manifest["checkpoints"][str(number)]
            )
        else:
            pending.append(number)
    if workers == 1:
        _initialize(*args)
        for number in pending:
            _draw(number)
            _save_completed_draw(output, manifest, number)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize,
            initargs=args,
        ) as pool:
            futures = {pool.submit(_draw, number): number for number in pending}
            for future in as_completed(futures):
                _save_completed_draw(output, manifest, future.result())


def _collect_draws(
    output: Path, config: CorporateConfig, manifest: dict[str, Any]
) -> dict[str, np.ndarray]:
    checkpoints = output / "checkpoints"
    collected: dict[str, list[np.ndarray]] = {}
    for number in range(config.draws):
        arrays = validate_checkpoint(
            checkpoints / f"draw-{number:04d}.npz",
            manifest["identity"],
            number,
            manifest["checkpoints"][str(number)],
        )
        for key, value in arrays.items():
            if key not in {"identity", "draw"}:
                collected.setdefault(key, []).append(value)
    return {f"draw_{key}": np.stack(values) for key, values in collected.items()}


def _article_summaries(
    cells: dict[str, Any], arrays: dict[str, Any], draws: int
) -> None:
    for name, cell in cells.items():
        defined = arrays[f"draw_{name}_defined"].astype(bool)
        values = arrays[f"draw_{name}_beta"][defined]
        quantiles = (
            np.quantile(values, [0.025, 0.05, 0.975])
            if len(values)
            else (None, None, None)
        )
        cell.update(
            {
                f"article_sensitivity_{key}": None if value is None else float(value)
                for key, value in zip(("q025", "q05", "q975"), quantiles, strict=True)
            }
        )
        cell["article_measurement_percentile95"] = [
            cell["article_sensitivity_q025"],
            cell["article_sensitivity_q975"],
        ]
        cell["article_measurement_lower95"] = cell["article_sensitivity_q05"]
        cell["article_sensitivity_scope"] = (
            "Fixed archive, firm histories, returns and years; paired stratified "
            "article redraws, conditional on defined estimates. "
            "No population confidence claim."
        )
        cell["article_draw_attempts"] = draws
        cell["article_draw_defined"] = int(defined.sum())
        cell["article_draw_undefined"] = int((~defined).sum())


def run(
    output: Path,
    config: CorporateConfig,
    *,
    workers: int = 3,
    threads: int = 1,
    scratch: Path = SCRATCH,
) -> dict[str, Any]:
    """Replay four point cells and the declared paired article-selection draws."""
    if workers < 1 or workers > MAX_WORKERS or threads < 1:
        msg = "corporate replay requires one to three workers and positive threads"
        raise ValueError(msg)
    output = _open_output(output)
    reject_symlinks(scratch.absolute())
    saved, inputs = inventory()
    manifest = _manifest(output, config, inputs)
    if (completed := _completed(output, manifest)) is not None:
        return completed
    began = time.perf_counter()
    baseline = reconcile_saved()
    write(output / "reconciliation.json", baseline)
    identity = manifest["identity"]
    with threadpool_limits(limits=threads):
        clouds, sample_audit = load_clouds(saved)
        panels = financial(saved["firm_ids"])
        directory = scratch / identity
        cache, cache_record = build_cache(
            directory, identity, clouds, len(saved["firm_ids"])
        )
        qualification = certificates(cache, clouds)
        point, cells = _point_analysis(cache, clouds, panels, saved["firm_ids"])
        point_replay = _point_reconciliation(point)
        metadata = {
            key: {**cloud, "points": np.empty((0, 0))} for key, cloud in clouds.items()
        }
        del clouds
        args = (
            directory,
            metadata,
            panels,
            cache.counts,
            config.seed,
            identity,
            output / "checkpoints",
            threads,
        )
        _execute_draws(output, config, manifest, args, workers)
    point.update(_collect_draws(output, config, manifest))
    _article_summaries(cells, point, config.draws)
    atomic_npz(output / "measurement.npz", point)
    reconciliation = {
        "identical_frozen_arrays": baseline,
        "raw_point_replay": point_replay,
        "raw_article_replay": _article_reconciliation(point),
    }
    write(output / "reconciliation.json", reconciliation)
    result = {
        "schema_version": 1,
        "status": "complete",
        "scope": (
            "Complete retained archive descriptive projection; article-selection "
            "sensitivity conditions on firms, returns and years."
        ),
        "firms": len(saved["firm_ids"]),
        "firm_transitions": len(saved["firm_ids"]) * 4,
        "directional_coordinates_per_cell": len(saved["firm_ids"])
        * (len(saved["firm_ids"]) - 1)
        * 4,
        "draws": config.draws,
        "seed": config.seed,
        "cells": cells,
        "inventory": {
            key: saved[key]
            for key in ("width", "expanded_articles", "matched_articles")
        },
        "sample_audit": sample_audit,
        "reconciliation": reconciliation,
        "transport_qualification": qualification,
        "cost_cache": cache_record,
        "manifest_identity": identity,
        "measurement_sha256": file_hash(output / "measurement.npz"),
        "seconds": time.perf_counter() - began,
        "execution": {"workers": workers, "threads": threads},
        "population_inference": (
            "Unavailable: a calibrated corporate population "
            "bound has not been established."
        ),
    }
    write(output / "results.json", result)
    manifest["artifacts"] = {
        name: file_hash(output / name)
        for name in ("results.json", "reconciliation.json", "measurement.npz")
    }
    write(output / "manifest.json", manifest)
    return result


def _article_reconciliation(arrays: dict[str, Any]) -> dict[str, Any]:
    cells = {}
    for name in CELLS:
        fields = {
            key: float(np.max(arrays[f"draw_{name}_{key}"]))
            for key in (
                "distance_replay_max_abs_difference",
                "financial_replay_max_abs_difference",
                "beta_replay_abs_difference",
            )
            if f"draw_{name}_{key}" in arrays
        }
        cells[name] = {
            "status": "reconciled" if fields else "not_historical_stream",
            **fields,
        }
    return {
        "cells": cells,
        "transport_absolute_tolerance": OBJECTIVE_TOLERANCE,
        "financial_absolute_tolerance": RECONCILIATION_TOLERANCE,
    }


def _point_reconciliation(point: dict[str, Any]) -> dict[str, Any]:
    results = {}
    n = len(point["firm_ids"])
    offdiag = ~np.eye(n, dtype=bool)
    with np.load(
        repository_root() / BASELINE / "measurement.npz", allow_pickle=False
    ) as baseline:
        for name in CELLS:
            support = name.rsplit("_", 1)[1]
            distance_error = float(
                np.max(
                    np.abs(
                        point[f"{name}_D"][:, offdiag]
                        - baseline[f"{name}_D"][:, offdiag]
                    )
                )
            )
            financial_error = float(
                np.max(
                    np.abs(
                        point[f"{support}_R"][:, offdiag]
                        - baseline[f"{support}_R"][:, offdiag]
                    )
                )
            )
            if (
                distance_error > OBJECTIVE_TOLERANCE
                or financial_error > RECONCILIATION_TOLERANCE
            ):
                msg = f"raw corporate point replay mismatch: {name}"
                raise ValueError(msg)
            results[name] = {
                "distance_max_abs_difference": distance_error,
                "financial_max_abs_difference": financial_error,
            }
    return {
        "status": "reconciled",
        "transport_absolute_tolerance": OBJECTIVE_TOLERANCE,
        "identical_financial_absolute_tolerance": 1e-12,
        "cells": results,
    }
