"""Paper 3's shared representation ladder and paired return contrasts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Never, cast

import numpy as np
import pandas as pd
import yaml
from jcor.inference.block_length import optimal_block_length_numpy

from pipeline.io.paper3_certificate import sha256_path
from pipeline.io.params import load_params
from pipeline.stages.papers.paper3.certificate import (
    _MIN_REFERENCE_DRAWS,
    _news_only_benchmark,
    _reference_designs,
    _standardize_returns,
)
from pipeline.stages.papers.paper3.certificate_validation import _gmv
from pipeline.stages.papers.paper5._gate.bootstrap import stationary_bootstrap_indices
from pipeline.stages.shared.representation_ablation import (
    load_extended_representation_ablation_specs,
    load_representation_contrast_specs,
)
from pipeline.stages.substrate.panel import (
    load_aligned_return_panel,
    load_priced_distance_universe,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pipeline.stages.papers.paper3.certificate import _ReferenceDesign
    from pipeline.stages.shared.representation_ablation import (
        RepresentationAblationSpec,
        RepresentationContrastSpec,
    )


def _value_error(message: str) -> Never:
    raise ValueError(message)


def _type_error(message: str) -> Never:
    raise TypeError(message)


def _number(mapping: dict[str, object], key: str) -> float:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        _type_error(f"expected numeric {key!r} in Paper 3 artifact")
    return float(value)


@dataclass(frozen=True, slots=True)
class Paper3RepresentationAblationPaths:
    """One Paper 3 representation cell's inputs and outputs."""

    params_file: Path
    returns_dir: Path
    distance_artifact_dir: Path
    output_dir: Path
    canonical_distance_artifact_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class Paper3RepresentationMergePaths:
    """Inputs and output for the ten-cell Paper 3 merge."""

    params_file: Path
    cells_dir: Path
    canonical_certificate_summary: Path
    output_summary: Path
    returns_dir: Path | None = None
    contrast_draws_dir: Path | None = None


def _cell(params_file: Path, cell_id: str) -> RepresentationAblationSpec:
    matches = [
        spec
        for spec in load_extended_representation_ablation_specs(params_file)
        if spec.cell_id == cell_id
    ]
    if len(matches) != 1:
        message = f"representation ablation cell {cell_id!r} is not uniquely registered"
        _value_error(message)
    return matches[0]


def _reference_config(
    params_file: Path,
) -> tuple[tuple[_ReferenceDesign, ...], int, int, int, int]:
    params = load_params(params_file)
    paper3 = params.get("paper3")
    if not isinstance(paper3, dict):
        _type_error("params.yaml paper3 section must be a mapping")
    feasibility = paper3.get("certificate_feasibility")
    if not isinstance(feasibility, dict):
        _type_error("paper3.certificate_feasibility must be a mapping")
    reference = feasibility.get("news_reference")
    if not isinstance(reference, dict):
        _type_error("paper3 certificate news_reference must be a mapping")
    designs = _reference_designs(reference.get("designs"))
    draws = reference.get("draws")
    seed = reference.get("seed")
    validation_value = paper3.get("certificate_validation")
    validation: dict[str, object] = (
        cast("dict[str, object]", validation_value)
        if isinstance(validation_value, dict)
        else {}
    )
    return_draws = validation.get("return_block_replications", 2000)
    return_seed = validation.get("seed", 42)
    if (
        isinstance(draws, bool)
        or not isinstance(draws, int)
        or draws < _MIN_REFERENCE_DRAWS
    ):
        _value_error("paper3 news_reference.draws is invalid")
    if isinstance(seed, bool) or not isinstance(seed, int):
        _type_error("paper3 news_reference.seed is invalid")
    if (
        isinstance(return_draws, bool)
        or not isinstance(return_draws, int)
        or return_draws < 1
    ):
        _value_error(
            "paper3 certificate_validation.return_block_replications is invalid"
        )
    if isinstance(return_seed, bool) or not isinstance(return_seed, int):
        _type_error("paper3 certificate_validation.seed is invalid")
    return designs, draws, seed, return_draws, return_seed


def _return_bootstrap_draws(
    returns: np.ndarray,
    weights: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Re-standardize and recompute the sample-GMV benchmark per draw."""

    def _draw_value(date_indices: np.ndarray) -> float:
        sampled, _ = _standardize_returns(returns[date_indices])
        covariance = np.cov(sampled, rowvar=False, ddof=1)
        gmv_weights = _gmv(covariance)
        news_variance = float(np.var(sampled @ weights, ddof=1))
        gmv_variance = float(np.var(sampled @ gmv_weights, ddof=1))
        if (
            not np.isfinite(news_variance)
            or not np.isfinite(gmv_variance)
            or gmv_variance <= 0.0
        ):
            return float("nan")
        return 100.0 * news_variance / gmv_variance

    draws = np.full(len(indices), np.nan, dtype=np.float64)
    for draw_id, date_indices in enumerate(indices):
        try:
            draws[draw_id] = _draw_value(date_indices)
        except (ValueError, FloatingPointError):
            continue
    return draws, int(np.isfinite(draws).sum())


def run_paper3_representation_ablation(
    paths: Paper3RepresentationAblationPaths,
    cell_id: str,
) -> dict[str, object]:
    """Evaluate one return-free W2 allocation and its stationary return draws."""
    spec = _cell(paths.params_file, cell_id)
    designs, draws, seed, return_draws, return_seed = _reference_config(
        paths.params_file
    )
    tickers, _distance, squared_distance = load_priced_distance_universe(
        paths.distance_artifact_dir, paths.returns_dir
    )
    dates, returns = load_aligned_return_panel(paths.returns_dir, tickers)
    weights, benchmark = _news_only_benchmark(
        tickers,
        returns,
        squared_distance,
        designs,
        draws,
        seed,
        cap_policy="annotate",
    )
    news_weights = weights["weight"].to_numpy(dtype=np.float64)
    # The block length is a design choice, selected once from the canonical
    # portfolio and then held fixed across every representation cell.
    if spec.canonical:
        canonical_weights = news_weights
    elif paths.canonical_distance_artifact_dir is not None:
        canonical_tickers, _, canonical_squared = load_priced_distance_universe(
            paths.canonical_distance_artifact_dir, paths.returns_dir
        )
        if canonical_tickers != tickers:
            _value_error(
                "canonical and representation distance artifacts use different tickers"
            )
        canonical_frame, _canonical_benchmark = _news_only_benchmark(
            tickers,
            returns,
            canonical_squared,
            designs,
            draws,
            seed,
            cap_policy="annotate",
        )
        canonical_weights = canonical_frame["weight"].to_numpy(dtype=np.float64)
    else:
        canonical_weights = news_weights
    canonical_portfolio = returns @ canonical_weights
    block_length = max(1, round(float(optimal_block_length_numpy(canonical_portfolio))))
    indices = stationary_bootstrap_indices(
        len(returns),
        draws=return_draws,
        expected_block_length=block_length,
        seed=return_seed,
    )
    relative_draws, valid_draws = _return_bootstrap_draws(
        returns, news_weights, indices
    )
    summary: dict[str, object] = {
        "schema_version": "paper3_representation_ablation.v2",
        "cell": asdict(spec),
        "universe": {
            "tickers": tickers,
            "n_tickers": len(tickers),
            "n_return_observations": len(returns),
            "return_start": str(dates[0])[:10],
            "return_end": str(dates[-1])[:10],
        },
        "distance_artifact": str(paths.distance_artifact_dir),
        "news_only_benchmark": benchmark,
        "return_bootstrap": {
            "method": "stationary date bootstrap with fixed text-derived allocation",
            "requested_draws": return_draws,
            "valid_draws": valid_draws,
            "seed": return_seed,
            "block_length": block_length,
            "restandardize_each_draw": True,
            "relative_gmv_index": (
                "100 times news-only variance divided by draw-specific GMV variance"
            ),
        },
    }
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    weights.to_parquet(paths.output_dir / "news_only_weights.parquet", index=False)
    pd.DataFrame(
        {
            "draw_id": np.arange(return_draws, dtype=np.int32),
            "relative_gmv_index": relative_draws,
            "solve_ok": np.isfinite(relative_draws),
        }
    ).to_parquet(paths.output_dir / "return_bootstrap.parquet", index=False)
    (paths.output_dir / "summary.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False), encoding="utf-8"
    )
    provenance = {
        "schema_version": 1,
        "cell_id": cell_id,
        "inputs": {
            "params_file": sha256_path(paths.params_file),
            "returns_dir": sha256_path(paths.returns_dir),
            "distance_artifact_dir": sha256_path(paths.distance_artifact_dir),
        },
    }
    (paths.output_dir / "provenance.manifest.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _load_yaml(path: Path) -> dict[str, object]:
    if not path.exists():
        message = f"required Paper 3 representation artifact is missing: {path}"
        raise FileNotFoundError(message)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        message = f"Paper 3 representation artifact must be a mapping: {path}"
        _type_error(message)
    return cast("dict[str, object]", payload)


def _holm(pvalues: list[float | None]) -> list[float | None]:
    available = sorted(
        ((index, value) for index, value in enumerate(pvalues) if value is not None),
        key=lambda pair: pair[1],
    )
    result: list[float | None] = [None] * len(pvalues)
    running = 0.0
    for rank, (index, value) in enumerate(available):
        running = max(running, min(1.0, (len(available) - rank) * value))
        result[index] = running
    return result


def _paired_contrasts(
    rows: list[dict[str, object]],
    draw_frames: dict[str, pd.DataFrame],
    specs: tuple[RepresentationContrastSpec, ...],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for spec in specs:
        candidate = next(
            (
                row
                for row in rows
                if cast("dict[str, object]", row["cell"])["cell_id"]
                == spec.candidate_cell_id
            ),
            None,
        )
        reference = next(
            (
                row
                for row in rows
                if cast("dict[str, object]", row["cell"])["cell_id"]
                == spec.reference_cell_id
            ),
            None,
        )
        record: dict[str, object] = {
            "contrast_id": spec.contrast_id,
            "candidate_cell": spec.candidate_cell_id,
            "reference_cell": spec.reference_cell_id,
            "family": spec.family,
        }
        if candidate is None or reference is None:
            results.append(
                {
                    **record,
                    "status": "unavailable",
                    "reason": "one or both cells were not computed",
                }
            )
            continue
        candidate_draws = draw_frames.get(spec.candidate_cell_id)
        reference_draws = draw_frames.get(spec.reference_cell_id)
        if candidate_draws is None or reference_draws is None:
            results.append(
                {
                    **record,
                    "status": "unavailable",
                    "reason": "paired return draws are missing",
                }
            )
            continue
        paired = candidate_draws.merge(
            reference_draws,
            on="draw_id",
            suffixes=("_candidate", "_reference"),
            validate="one_to_one",
        )
        valid = paired.loc[
            paired["solve_ok_candidate"] & paired["solve_ok_reference"],
            "relative_gmv_index_candidate",
        ].to_numpy(dtype=np.float64) - paired.loc[
            paired["solve_ok_candidate"] & paired["solve_ok_reference"],
            "relative_gmv_index_reference",
        ].to_numpy(dtype=np.float64)
        valid = valid[np.isfinite(valid)]
        if valid.size == 0:
            results.append(
                {
                    **record,
                    "status": "unavailable",
                    "reason": "no common finite successful bootstrap draws",
                }
            )
            continue
        point_candidate = cast(
            "dict[str, object]",
            cast("dict[str, object]", candidate["news_only_benchmark"])["portfolios"],
        )["news_only"]
        point_reference = cast(
            "dict[str, object]",
            cast("dict[str, object]", reference["news_only_benchmark"])["portfolios"],
        )["news_only"]
        estimate = (
            _number(cast("dict[str, object]", point_candidate), "gmv_relative_variance")
            * 100.0
            - _number(
                cast("dict[str, object]", point_reference), "gmv_relative_variance"
            )
            * 100.0
        )
        ci_low, ci_high = np.quantile(valid, [0.025, 0.975])
        tail = min(
            (np.count_nonzero(valid <= 0.0) + 1.0) / (valid.size + 1.0),
            (np.count_nonzero(valid >= 0.0) + 1.0) / (valid.size + 1.0),
        )
        results.append(
            {
                **record,
                "status": "computed",
                "estimate": estimate,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "pvalue": float(min(1.0, 2.0 * tail)),
                "bootstrap_valid_draws": int(valid.size),
            }
        )
    for family in {spec.family for spec in specs}:
        indexes = [
            index for index, row in enumerate(results) if row["family"] == family
        ]
        for index, value in zip(
            indexes,
            _holm(
                [
                    cast("float | None", results[index].get("pvalue"))
                    if results[index].get("status") == "computed"
                    else None
                    for index in indexes
                ]
            ),
            strict=True,
        ):
            if value is not None:
                results[index]["holm_pvalue"] = float(value)
    return results


def merge_paper3_representation_ablation(
    paths: Paper3RepresentationMergePaths,
) -> dict[str, object]:
    """Validate and merge the ten Paper 3 cells plus paired return contrasts."""
    specs = load_extended_representation_ablation_specs(paths.params_file)
    rows = [
        _load_yaml(paths.cells_dir / spec.cell_id / "summary.yaml") for spec in specs
    ]
    expected_ids = [spec.cell_id for spec in specs]
    observed_ids = [
        str(cast("dict[str, object]", row["cell"])["cell_id"]) for row in rows
    ]
    if observed_ids != expected_ids:
        message = (
            "Paper 3 representation cells are out of order or incomplete: "
            f"{observed_ids}"
        )
        _value_error(message)
    universes = [row["universe"] for row in rows]
    if any(universe != universes[0] for universe in universes[1:]):
        _value_error("Paper 3 representation cells do not share one return universe")
    canonical = next(
        row for row in rows if cast("dict[str, object]", row["cell"])["canonical"]
    )
    publication = _load_yaml(paths.canonical_certificate_summary)
    if canonical["news_only_benchmark"] != publication.get("news_only_benchmark"):
        _value_error(
            "canonical Paper 3 representation cell does not reproduce the "
            "headline benchmark"
        )
    draw_root = paths.contrast_draws_dir or paths.cells_dir
    draw_frames: dict[str, pd.DataFrame] = {}
    for spec in specs:
        draw_path = draw_root / spec.cell_id / "return_bootstrap.parquet"
        if draw_path.exists():
            frame = pd.read_parquet(draw_path)
            required = {"draw_id", "relative_gmv_index", "solve_ok"}
            if set(frame.columns) != required:
                message = (
                    f"Paper 3 return-bootstrap schema is invalid for {spec.cell_id}"
                )
                _value_error(message)
            draw_frames[spec.cell_id] = frame.sort_values("draw_id", ignore_index=True)
    contrasts = _paired_contrasts(
        rows, draw_frames, load_representation_contrast_specs(paths.params_file)
    )
    calibration_id = "bge_large_minus_qwen8b_1024"
    calibration = next(
        (
            row
            for row in contrasts
            if row["contrast_id"] == calibration_id and row.get("status") == "computed"
        ),
        None,
    )
    calibration_payload: dict[str, object] = {
        "status": "unavailable",
        "reference_contrast": calibration_id,
        "threshold_interpretation": (
            "benchmark-calibrated, post-specified robustness threshold; not a "
            "decision-theoretic equivalence margin"
        ),
    }
    if calibration is not None:
        bound = max(
            abs(_number(calibration, "ci_low")), abs(_number(calibration, "ci_high"))
        )
        calibration_payload.update(
            {"status": "computed", "relative_gmv_equivalence_bound": bound}
        )
        for row in contrasts:
            if row["family"] == "vintage" and row.get("status") == "computed":
                row["equivalence_bound"] = bound
                row["equivalent"] = bool(
                    _number(row, "ci_low") > -bound and _number(row, "ci_high") < bound
                )
    merged: dict[str, object] = {
        "schema_version": "paper3_representation_ablation_merge.v3",
        "roster": expected_ids,
        "universe": universes[0],
        "cells": rows,
        "representation_contrasts": contrasts,
        "contrast_analysis": {
            "estimand": "candidate minus reference relative-GMV index points",
            "confidence_level": 0.95,
            "contrasts": contrasts,
            "calibration": calibration_payload,
        },
    }
    paths.output_summary.parent.mkdir(parents=True, exist_ok=True)
    paths.output_summary.write_text(
        yaml.safe_dump(merged, sort_keys=False), encoding="utf-8"
    )
    return merged


__all__ = [
    "Paper3RepresentationAblationPaths",
    "Paper3RepresentationMergePaths",
    "merge_paper3_representation_ablation",
    "run_paper3_representation_ablation",
]
