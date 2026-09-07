"""Merge Paper 5 pooled spatial fits across the governed representation ladder."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Never, cast

import numpy as np
import pandas as pd

from pipeline.stages.shared.representation_ablation import (
    load_extended_representation_ablation_specs,
    load_representation_contrast_specs,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pipeline.stages.shared.representation_ablation import (
        RepresentationAblationSpec,
        RepresentationContrastSpec,
    )


def _value_error(message: str) -> Never:
    raise ValueError(message)


def _number(mapping: dict[str, object], key: str) -> float:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        _value_error(f"expected numeric {key!r} in Paper 5 artifact")
    return float(value)


@dataclass(frozen=True, slots=True)
class Paper5RepresentationMergePaths:
    """Inputs and output for the Paper 5 representation merge."""

    params_file: Path
    cells_dir: Path
    canonical_results: Path
    output_summary: Path
    bootstrap_dir: Path | None = None
    contrast_draws_dir: Path | None = None


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        message = f"required Paper 5 representation artifact is missing: {path}"
        raise FileNotFoundError(message)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        message = f"Paper 5 representation artifact must be a mapping: {path}"
        _value_error(message)
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


def _paired_metric_contrasts(
    cells: list[dict[str, object]],
    frames: dict[str, pd.DataFrame],
    specs: tuple[RepresentationContrastSpec, ...],
    metric: str,
) -> list[dict[str, object]]:
    """Compute paired direct-rho or operator-gap contrasts."""
    results: list[dict[str, object]] = []
    for spec in specs:
        candidate = next(
            (
                row
                for row in cells
                if cast("dict[str, object]", row["cell"])["cell_id"]
                == spec.candidate_cell_id
            ),
            None,
        )
        reference = next(
            (
                row
                for row in cells
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
            "metric": metric,
        }
        if candidate is None or reference is None:
            results.append(
                {
                    **record,
                    "status": "unavailable",
                    "reason": "one or both cells are missing",
                }
            )
            continue
        candidate_frame = frames.get(spec.candidate_cell_id)
        reference_frame = frames.get(spec.reference_cell_id)
        if candidate_frame is None or reference_frame is None:
            results.append(
                {
                    **record,
                    "status": "unavailable",
                    "reason": "paired QMLE draws are missing",
                }
            )
            continue
        paired = candidate_frame.merge(
            reference_frame,
            on="draw_id",
            suffixes=("_candidate", "_reference"),
            validate="one_to_one",
        )
        valid_mask = paired["solve_ok_candidate"] & paired["solve_ok_reference"]
        difference = paired.loc[valid_mask, "value_candidate"].to_numpy(
            dtype=np.float64
        ) - paired.loc[valid_mask, "value_reference"].to_numpy(dtype=np.float64)
        difference = difference[np.isfinite(difference)]
        if difference.size == 0:
            results.append(
                {
                    **record,
                    "status": "unavailable",
                    "reason": "no common finite successful QMLE draws",
                }
            )
            continue
        candidate_pooled = cast(
            "dict[str, object]",
            cast("dict[str, object]", candidate["results"])["results"],
        )["pooled_2023_2026"]
        reference_pooled = cast(
            "dict[str, object]",
            cast("dict[str, object]", reference["results"])["results"],
        )["pooled_2023_2026"]
        candidate_flat = _number(
            cast(
                "dict[str, object]",
                cast("dict[str, object]", candidate_pooled)["w_flat"],
            ),
            "rho_hat",
        )
        reference_flat = _number(
            cast(
                "dict[str, object]",
                cast("dict[str, object]", reference_pooled)["w_flat"],
            ),
            "rho_hat",
        )
        if metric == "direct_rho":
            estimate = candidate_flat - reference_flat
        else:
            candidate_rbf = _number(
                cast(
                    "dict[str, object]",
                    cast("dict[str, object]", candidate_pooled)["w_h"],
                ),
                "rho_hat",
            )
            reference_rbf = _number(
                cast(
                    "dict[str, object]",
                    cast("dict[str, object]", reference_pooled)["w_h"],
                ),
                "rho_hat",
            )
            estimate = (candidate_flat - candidate_rbf) - (
                reference_flat - reference_rbf
            )
        ci_low, ci_high = np.quantile(difference, [0.025, 0.975])
        tail = min(
            (np.count_nonzero(difference <= 0.0) + 1.0) / (difference.size + 1.0),
            (np.count_nonzero(difference >= 0.0) + 1.0) / (difference.size + 1.0),
        )
        results.append(
            {
                **record,
                "status": "computed",
                "estimate": float(estimate),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "pvalue": float(min(1.0, 2.0 * tail)),
                "bootstrap_valid_draws": int(difference.size),
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


def _load_contrast_frames(
    root: Path, specs: tuple[RepresentationAblationSpec, ...]
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for spec in specs:
        path = root / str(getattr(spec, "cell_id", "")) / "bootstrap_draws.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        required = {"period", "w_variant", "draw_id", "rho_draw", "solve_ok"}
        if not required <= set(frame.columns):
            message = f"Paper 5 bootstrap-draw schema is invalid for {spec.cell_id}"
            _value_error(message)
        frame = frame.loc[
            (frame["period"] == "pooled_2023_2026")
            & frame["w_variant"].isin(["w_flat", "w_h"]),
            ["w_variant", "draw_id", "rho_draw", "solve_ok"],
        ].copy()
        if frame.duplicated(["draw_id", "w_variant"]).any():
            message = (
                "Paper 5 bootstrap draws duplicate draw/variant rows for "
                f"{spec.cell_id}"
            )
            _value_error(message)
        pivot = frame.pivot_table(
            index="draw_id",
            columns="w_variant",
            values=["rho_draw", "solve_ok"],
            aggfunc=lambda values: values.iloc[0],
        )
        if not {
            ("rho_draw", "w_flat"),
            ("rho_draw", "w_h"),
            ("solve_ok", "w_flat"),
            ("solve_ok", "w_h"),
        } <= set(pivot.columns):
            message = (
                f"Paper 5 pooled bootstrap draws are incomplete for {spec.cell_id}"
            )
            _value_error(message)
        # Preserve the common schedule and expose one metric-specific value later.
        pivot = pivot.reset_index()
        direct = pd.DataFrame(
            {
                "draw_id": pivot["draw_id"],
                "value": pivot[("rho_draw", "w_flat")],
                "solve_ok": pivot[("solve_ok", "w_flat")].astype(bool),
                "gap_value": pivot[("rho_draw", "w_flat")] - pivot[("rho_draw", "w_h")],
                "gap_solve_ok": pivot[("solve_ok", "w_flat")].astype(bool)
                & pivot[("solve_ok", "w_h")].astype(bool),
            }
        )
        frames[spec.cell_id] = direct
    return frames


def _metric_frames(
    frames: dict[str, pd.DataFrame], metric: str
) -> dict[str, pd.DataFrame]:
    if metric == "direct_rho":
        return frames
    return {
        cell_id: pd.DataFrame(
            {
                "draw_id": frame["draw_id"],
                "value": frame["gap_value"],
                "solve_ok": frame["gap_solve_ok"],
            }
        )
        for cell_id, frame in frames.items()
    }


def merge_paper5_representation_ablation(  # noqa: C901, PLR0915
    paths: Paper5RepresentationMergePaths,
) -> dict[str, object]:
    """Validate ten pooled cells, persistable draws, and paired contrasts."""
    specs = load_extended_representation_ablation_specs(paths.params_file)
    cells: list[dict[str, object]] = []
    period_contract: dict[str, object] | None = None
    inference_contract: object | None = None
    for spec in specs:
        payload = _load_json(paths.cells_dir / spec.cell_id / "results.json")
        identity = payload.get("identity")
        expected_identity = {
            "provider_id": spec.provider_id,
            "representation_id": spec.representation_id,
            "barycentre_arm_id": "wasserstein_w2_loo",
            "geometry_id": "wasserstein_w2",
        }
        if identity != expected_identity:
            message = f"Paper 5 representation identity mismatch for {spec.cell_id}"
            _value_error(message)
        results = payload.get("results")
        if not isinstance(results, dict) or "pooled_2023_2026" not in results:
            message = (
                "Paper 5 representation cell must contain pooled results: "
                f"{spec.cell_id}"
            )
            _value_error(message)
        pooled = results["pooled_2023_2026"]
        if not isinstance(pooled, dict) or not {"w_flat", "w_h"} <= set(pooled):
            message = f"Paper 5 representation cell {spec.cell_id} lacks focal matrices"
            _value_error(message)
        first = cast("dict[str, object]", pooled["w_flat"])
        contract = {key: first[key] for key in ("start", "end", "trading_days")}
        if period_contract is None:
            period_contract = contract
            inference_contract = payload.get("inference")
        elif (
            contract != period_contract
            or payload.get("inference") != inference_contract
        ):
            _value_error(
                "Paper 5 representation cells do not share dates/bootstrap settings"
            )
        cells.append({"cell": asdict(spec), "results": payload})

    canonical_results = _load_json(paths.canonical_results)
    canonical_pooled = cast(
        "dict[str, object]",
        cast("dict[str, object]", canonical_results["results"])["pooled_2023_2026"],
    )
    canonical_cell = next(
        cell
        for cell in cells
        if bool(cast("dict[str, object]", cell["cell"])["canonical"])
    )
    ablation_pooled = cast(
        "dict[str, object]",
        cast(
            "dict[str, object]",
            cast("dict[str, object]", canonical_cell["results"])["results"],
        )["pooled_2023_2026"],
    )
    for matrix_id in ("w_flat", "w_h"):
        if ablation_pooled[matrix_id] != canonical_pooled.get(matrix_id):
            message = f"canonical Paper 5 ablation does not reproduce {matrix_id}"
            _value_error(message)
    fixed = {
        matrix_id: canonical_pooled[matrix_id]
        for matrix_id in ("w_co_mentions", "equal_support")
        if matrix_id in canonical_pooled
    }
    contrast_specs = load_representation_contrast_specs(paths.params_file)
    draw_root = paths.contrast_draws_dir or paths.bootstrap_dir or paths.cells_dir
    frames = _load_contrast_frames(draw_root, specs)
    direct = _paired_metric_contrasts(
        cells, _metric_frames(frames, "direct_rho"), contrast_specs, "direct_rho"
    )
    gap = _paired_metric_contrasts(
        cells, _metric_frames(frames, "gap"), contrast_specs, "gap"
    )
    calibration_payload: dict[str, object] = {
        "status": "unavailable",
        "reference_contrast": "bge_large_minus_qwen8b_1024",
        "threshold_interpretation": (
            "benchmark-calibrated, post-specified robustness thresholds; not "
            "decision-theoretic equivalence margins"
        ),
    }
    direct_reference = next(
        (
            row
            for row in direct
            if row["contrast_id"] == "bge_large_minus_qwen8b_1024"
            and row.get("status") == "computed"
        ),
        None,
    )
    gap_reference = next(
        (
            row
            for row in gap
            if row["contrast_id"] == "bge_large_minus_qwen8b_1024"
            and row.get("status") == "computed"
        ),
        None,
    )
    if direct_reference is not None and gap_reference is not None:
        direct_bound = max(
            abs(_number(direct_reference, "ci_low")),
            abs(_number(direct_reference, "ci_high")),
        )
        gap_bound = max(
            abs(_number(gap_reference, "ci_low")),
            abs(_number(gap_reference, "ci_high")),
        )
        calibration_payload.update(
            {
                "status": "computed",
                "direct_rho_bound": direct_bound,
                "gap_bound": gap_bound,
            }
        )
        direct_vintage = {
            row["contrast_id"]: row for row in direct if row["family"] == "vintage"
        }
        gap_vintage = {
            row["contrast_id"]: row for row in gap if row["family"] == "vintage"
        }
        for contrast_id, row in direct_vintage.items():
            other = gap_vintage.get(contrast_id)
            if (
                row.get("status") == "computed"
                and other is not None
                and other.get("status") == "computed"
            ):
                row["equivalence_bound"] = direct_bound
                row["equivalent_direct_rho"] = bool(
                    _number(row, "ci_low") > -direct_bound
                    and _number(row, "ci_high") < direct_bound
                )
                other["equivalence_bound"] = gap_bound
                other["equivalent_gap"] = bool(
                    _number(other, "ci_low") > -gap_bound
                    and _number(other, "ci_high") < gap_bound
                )
                row["equivalent"] = bool(
                    row["equivalent_direct_rho"] and other["equivalent_gap"]
                )
                other["equivalent"] = row["equivalent"]
    merged: dict[str, object] = {
        "schema_version": "paper5_representation_ablation_merge.v3",
        "roster": [spec.cell_id for spec in specs],
        "period": period_contract,
        "inference": inference_contract,
        "cells": cells,
        "fixed_comparators": fixed,
        "representation_contrasts": {"direct_rho": direct, "gap": gap},
        "contrast_analysis": {
            "confidence_level": 0.95,
            "direct_rho": direct,
            "gap": gap,
            "barycentric_minus_rbf_rho_gap": gap,
            "calibration": calibration_payload,
        },
    }
    paths.output_summary.parent.mkdir(parents=True, exist_ok=True)
    paths.output_summary.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return merged


__all__ = ["Paper5RepresentationMergePaths", "merge_paper5_representation_ablation"]
