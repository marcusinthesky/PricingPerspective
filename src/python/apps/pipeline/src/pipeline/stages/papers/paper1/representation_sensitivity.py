"""Symmetric-firm-effect sensitivity across Paper 1 representations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import yaml
from jcor.model.dyadic import DyadicFit, DyadicInference, DyadicModel, DyadicSample
from scipy.stats import spearmanr

from pipeline.stages.papers.paper1._dyadic.adapters import legacy_node_count_schedule
from pipeline.stages.papers.paper1._dyadic.estimation import _fit_spec
from pipeline.stages.shared.representation_ablation import (
    RepresentationContrastSpec,
    load_encoder_vintage_specs,
    load_representation_ablation_specs,
    load_representation_contrast_specs,
)
from pipeline.stages.shared.typed_artifacts import read_typed_distance_artifact

if TYPE_CHECKING:
    from pathlib import Path

    from pipeline.stages.shared.representation_ablation import (
        RepresentationAblationSpec,
    )

_CANONICAL_MODEL = "qwen3-embedding-8b"
_CANONICAL_WIDTH = 128
_CANONICAL_DISTANCE = "wasserstein_w2"
_NATIVE_DIMENSIONS = {
    "bge-large-en-v1.5": 1024,
    "bge-m3": 1024,
    "qwen3-embedding-4b": 2560,
    "qwen3-embedding-8b": 4096,
}


@dataclass(frozen=True, slots=True)
class RepresentationSensitivityConfig:
    """Governed inputs and outputs for the representation grid."""

    pairs_file: Path
    node_counts_file: Path
    output_summary: Path
    output_draws: Path
    params_file: Path
    representation_dirs: tuple[Path, ...] = ()
    bootstrap_seed: int = 42
    vintage_dirs: tuple[Path, ...] = ()
    vintage_equivalence_bound: float = 0.012
    vintage_confidence_level: float = 0.95


def _first_multiplicity(values: pd.Series) -> int:
    """Return the sole per-draw/ticker bootstrap multiplicity."""
    return int(values.iloc[0])


def _node_counts(path: Path, tickers: list[str]) -> np.ndarray:
    frame = pd.read_parquet(path)
    observed = sorted(frame["ticker"].astype(str).unique())
    if observed != tickers:
        message = "node-count schedule and representation grid use different firms"
        raise ValueError(message)
    wide = frame.pivot_table(
        index="draw_id",
        columns="ticker",
        values="multiplicity",
        aggfunc=_first_multiplicity,
    )
    return wide.loc[:, tickers].sort_index().to_numpy(dtype=np.int64)


def _candidate_rows(
    pairs: pd.DataFrame,
    roster: tuple[RepresentationAblationSpec, ...],
    vintage_roster: tuple[RepresentationAblationSpec, ...] = (),
) -> list[tuple[str | None, str, str, int, int, str]]:
    available = {
        (str(model), int(articles), int(dimension))
        for model, articles, dimension in pairs[
            ["model", "m", "embedding_dimension"]
        ].itertuples(index=False, name=None)
    }
    encoder_rows = [
        (
            spec.cell_id,
            "encoder_dimension",
            spec.provider_id,
            _CANONICAL_WIDTH,
            spec.effective_dimension,
            _CANONICAL_DISTANCE,
        )
        for spec in roster
        if (spec.provider_id, _CANONICAL_WIDTH, spec.effective_dimension) in available
    ]
    vintage_rows = [
        (
            spec.cell_id,
            "encoder_vintage",
            spec.provider_id,
            _CANONICAL_WIDTH,
            spec.effective_dimension,
            _CANONICAL_DISTANCE,
        )
        for spec in vintage_roster
        if (spec.provider_id, _CANONICAL_WIDTH, spec.effective_dimension) in available
    ]
    distance_rows = [
        (
            None,
            "distance_family",
            _CANONICAL_MODEL,
            _CANONICAL_WIDTH,
            _NATIVE_DIMENSIONS[_CANONICAL_MODEL],
            distance,
        )
        for distance in ("wasserstein_w2", "wasserstein_w1", "energy_v")
    ]
    sampling_rows = [
        (
            None,
            "sample_size",
            _CANONICAL_MODEL,
            articles,
            _NATIVE_DIMENSIONS[_CANONICAL_MODEL],
            _CANONICAL_DISTANCE,
        )
        for articles in (32, 64, 128)
        if (
            _CANONICAL_MODEL,
            articles,
            _NATIVE_DIMENSIONS[_CANONICAL_MODEL],
        )
        in available
    ]
    return [*encoder_rows, *vintage_rows, *distance_rows, *sampling_rows]


def _aligned_cell(
    pairs: pd.DataFrame,
    model: str,
    articles: int,
    dimension: int,
    distance: str,
) -> pd.DataFrame:
    cell = pairs.loc[
        (pairs["model"] == model)
        & (pairs["m"] == articles)
        & (pairs["embedding_dimension"] == dimension),
        ["ticker_i", "ticker_j", "return_chord", distance],
    ].copy()
    cell = cell.rename(columns={distance: "candidate_distance"})
    return cell.sort_values(["ticker_i", "ticker_j"], ignore_index=True)


def _append_typed_cell(
    pairs: pd.DataFrame,
    artifact_dir: Path,
    *,
    model: str,
    dimension: int,
    expected_provider: str | None = None,
    expected_representation: str | None = None,
) -> pd.DataFrame:
    """Append one external W2 geometry to the comparator's dyad panel.

    The comparator's locked four-encoder panel is deliberately closed, so any
    further encoder enters here instead: its W2 matrix is joined to the same
    canonical dyads and return chords, leaving the headline panel untouched.
    Matryoshka prefixes reuse the canonical model label because they are that
    encoder at a narrower width; a separately trained encoder passes its own.
    """
    identity: dict[str, object] = {"distance_id": "wasserstein_w2"}
    if expected_provider is not None:
        identity["provider_id"] = expected_provider
    if expected_representation is not None:
        identity["representation_id"] = expected_representation
    frame, _ = read_typed_distance_artifact(artifact_dir, expected_identity=identity)
    typed = frame.rename(
        columns={
            "item_i": "ticker_i",
            "item_j": "ticker_j",
            "value": "wasserstein_w2",
        }
    )
    typed = typed.loc[typed["ticker_i"] < typed["ticker_j"]]
    returns = pairs.loc[
        (pairs["model"] == _CANONICAL_MODEL) & (pairs["m"] == _CANONICAL_WIDTH),
        ["ticker_i", "ticker_j", "return_chord"],
    ].drop_duplicates()
    typed = typed.merge(
        returns,
        on=["ticker_i", "ticker_j"],
        how="inner",
        validate="one_to_one",
    )
    typed["model"] = model
    typed["m"] = _CANONICAL_WIDTH
    typed["embedding_dimension"] = dimension
    typed["wasserstein_w1"] = np.nan
    typed["energy_v"] = np.nan
    for column in pairs.columns:
        if column not in typed:
            typed[column] = np.nan
    return pd.concat([pairs, typed[pairs.columns]], ignore_index=True)


def _append_registered_cells(
    pairs: pd.DataFrame,
    artifact_dirs: tuple[Path, ...],
    roster: tuple[RepresentationAblationSpec, ...],
) -> pd.DataFrame:
    """Append typed W2 cells after resolving each artifact to the shared roster.

    Both non-native Qwen representations and EttaX vintages use this path. The
    identity in the artifact, rather than its directory spelling, determines
    which governed cell receives the matrix.
    """
    by_identity = {(spec.provider_id, spec.representation_id): spec for spec in roster}
    seen: set[str] = set()
    for artifact_dir in artifact_dirs:
        summary = json.loads(
            (artifact_dir / "summary.json").read_text(encoding="utf-8")
        )
        provider_id = str(summary.get("provider_id", ""))
        representation_id = str(summary.get("representation_id", ""))
        spec = by_identity.get((provider_id, representation_id))
        if spec is None:
            message = (
                f"typed artifact {artifact_dir} declares ({provider_id!r}, "
                f"{representation_id!r}), which has no governed roster cell"
            )
            raise ValueError(message)
        if spec.cell_id in seen:
            message = f"duplicate representation artifact for {spec.cell_id}"
            raise ValueError(message)
        seen.add(spec.cell_id)
        pairs = _append_typed_cell(
            pairs,
            artifact_dir,
            model=provider_id,
            dimension=spec.effective_dimension,
            expected_provider=provider_id,
            expected_representation=representation_id,
        )
    return pairs


def _holm_adjust(pvalues: list[float | None]) -> list[float | None]:
    """Apply Holm's step-down adjustment while preserving input order."""
    available = sorted(
        ((index, value) for index, value in enumerate(pvalues) if value is not None),
        key=lambda pair: pair[1],
    )
    adjusted: list[float | None] = [None] * len(pvalues)
    running = 0.0
    for rank, (index, value) in enumerate(available):
        running = max(running, min(1.0, (len(available) - rank) * value))
        adjusted[index] = running
    return adjusted


def _paired_contrasts(
    cells: list[dict[str, object]],
    draws: pd.DataFrame,
    contrast_specs: tuple[RepresentationContrastSpec, ...],
    *,
    confidence_level: float,
    equivalence_bound: float | None = None,
) -> list[dict[str, object]]:
    """Compute governed candidate-minus-reference paired bootstrap contrasts."""
    if equivalence_bound is not None and equivalence_bound <= 0.0:
        message = "vintage equivalence bound must be positive"
        raise ValueError(message)
    if not 0.0 < confidence_level < 1.0:
        message = "vintage confidence level must lie strictly between zero and one"
        raise ValueError(message)

    computed = {
        str(cell["representation_cell"]): cell
        for cell in cells
        if cell.get("status") == "computed"
        and cell.get("representation_cell") is not None
    }
    alpha = 1.0 - confidence_level
    results: list[dict[str, object]] = []
    for contrast in contrast_specs:
        candidate_id = str(contrast.candidate_cell_id)
        reference_id = str(contrast.reference_cell_id)
        contrast_id = str(contrast.contrast_id)
        family = str(contrast.family)
        candidate_cell = computed.get(candidate_id)
        reference_cell = computed.get(reference_id)
        identity = {
            "contrast_id": contrast_id,
            "candidate_cell": candidate_id,
            "reference_cell": reference_id,
            "family": family,
        }
        if candidate_cell is None or reference_cell is None:
            results.append(
                {
                    **identity,
                    "status": "unavailable",
                    "reason": "one or both cells were not computed",
                }
            )
            continue

        candidate_draws = draws.loc[
            draws["cell_id"] == int(cast("int", candidate_cell["cell_id"])),
            ["draw_id", "standardized_effect", "solve_ok"],
        ].rename(
            columns={
                "standardized_effect": "candidate_effect",
                "solve_ok": "candidate_solve_ok",
            }
        )
        reference_draws = draws.loc[
            draws["cell_id"] == int(cast("int", reference_cell["cell_id"])),
            ["draw_id", "standardized_effect", "solve_ok"],
        ].rename(
            columns={
                "standardized_effect": "reference_effect",
                "solve_ok": "reference_solve_ok",
            }
        )
        paired = candidate_draws.merge(
            reference_draws,
            on="draw_id",
            how="inner",
            validate="one_to_one",
        )
        valid_mask = paired["candidate_solve_ok"].to_numpy(dtype=bool) & paired[
            "reference_solve_ok"
        ].to_numpy(dtype=bool)
        difference = paired["candidate_effect"].to_numpy(dtype=np.float64) - paired[
            "reference_effect"
        ].to_numpy(dtype=np.float64)
        valid = difference[valid_mask & np.isfinite(difference)]
        if valid.size == 0:
            results.append(
                {
                    **identity,
                    "status": "unavailable",
                    "reason": "no common finite successful bootstrap draws",
                }
            )
            continue

        ci_low, ci_high = np.quantile(
            valid,
            [alpha / 2.0, 1.0 - alpha / 2.0],
        )
        estimate = float(cast("float", candidate_cell["effect_one_sd"])) - float(
            cast("float", reference_cell["effect_one_sd"])
        )
        lower_tail = (np.count_nonzero(valid <= 0.0) + 1.0) / (valid.size + 1.0)
        upper_tail = (np.count_nonzero(valid >= 0.0) + 1.0) / (valid.size + 1.0)
        results.append(
            {
                **identity,
                "status": "computed",
                "estimate": estimate,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "pvalue": float(min(1.0, 2.0 * min(lower_tail, upper_tail))),
                **(
                    {
                        "equivalence_bound": equivalence_bound,
                        "equivalent": bool(
                            ci_low > -equivalence_bound and ci_high < equivalence_bound
                        ),
                    }
                    if family == "vintage" and equivalence_bound is not None
                    else {}
                ),
                "bootstrap_valid_draws": int(valid.size),
            }
        )
    for family in {str(contrast.family) for contrast in contrast_specs}:
        indexes = [
            index for index, row in enumerate(results) if row.get("family") == family
        ]
        adjusted = _holm_adjust(
            [
                cast("float | None", results[index].get("pvalue"))
                if results[index].get("status") == "computed"
                else None
                for index in indexes
            ]
        )
        for index, value in zip(indexes, adjusted, strict=True):
            if value is not None:
                results[index]["holm_pvalue"] = float(value)
    return results


def _paired_vintage_contrasts(
    cells: list[dict[str, object]],
    draws: pd.DataFrame,
    roster: tuple[RepresentationAblationSpec, ...],
    *,
    equivalence_bound: float,
    confidence_level: float,
) -> list[dict[str, object]]:
    """Backward-compatible wrapper for the vintage-only paired analysis."""
    pair_indexes = ((1, 0), (2, 0), (2, 1))
    vintage_contrasts = tuple(
        RepresentationContrastSpec(
            contrast_id=f"{roster[candidate].cell_id}_minus_{roster[reference].cell_id}",
            candidate_cell_id=roster[candidate].cell_id,
            reference_cell_id=roster[reference].cell_id,
            family="vintage",
            display_order=order,
        )
        for order, (candidate, reference) in enumerate(pair_indexes, start=1)
        if candidate < len(roster) and reference < len(roster)
    )
    return _paired_contrasts(
        cells,
        draws,
        vintage_contrasts,
        equivalence_bound=equivalence_bound,
        confidence_level=confidence_level,
    )


def _write_representation_outputs(
    config: RepresentationSensitivityConfig,
    *,
    requested_draws: int,
    cells: list[dict[str, object]],
    bootstrap_draws: pd.DataFrame,
    contrasts: list[dict[str, object]],
) -> dict[str, object]:
    """Persist the summary and common-schedule bootstrap draws."""
    summary: dict[str, object] = {
        "schema_version": 3,
        "estimand": (
            "change in return chord distance per full-sample dyadic standard "
            "deviation of the candidate distance, conditional on symmetric "
            "additive firm effects"
        ),
        "geometry_comparator": (
            "Spearman rank agreement with canonical Qwen3-Embedding-8B m=128 W2 "
            "on the common firm-dyad panel"
        ),
        "inference": {
            "method": "multinomial node bootstrap (dyad weight w_i*w_j)",
            "requested_draws": requested_draws,
            "seed": int(config.bootstrap_seed),
            "common_node_schedule": True,
        },
        "representation_contrasts": contrasts,
        "encoder_vintage_analysis": {
            "estimand": "later-vintage minus earlier-vintage standardized effect",
            "confidence_level": config.vintage_confidence_level,
            "equivalence_bound": config.vintage_equivalence_bound,
            "equivalence_rule": (
                "the paired percentile interval lies strictly inside the symmetric "
                "equivalence bound"
            ),
            "contrasts": [row for row in contrasts if row.get("family") == "vintage"],
        },
        "cells": cells,
    }
    config.output_summary.parent.mkdir(parents=True, exist_ok=True)
    config.output_summary.write_text(
        yaml.safe_dump(summary, sort_keys=False), encoding="utf-8"
    )
    bootstrap_draws.to_parquet(config.output_draws, index=False)
    return summary


def run_representation_sensitivity(
    config: RepresentationSensitivityConfig,
) -> dict[str, object]:
    """Fit a common symmetric-FE association for each governed distance cell."""
    pairs = pd.read_parquet(config.pairs_file)
    required = {
        "model",
        "m",
        "ticker_i",
        "ticker_j",
        "return_chord",
        "wasserstein_w1",
        "wasserstein_w2",
        "energy_v",
    }
    missing = sorted(required - set(pairs.columns))
    if missing:
        message = f"representation pair artifact is missing columns: {missing}"
        raise ValueError(message)
    pairs["embedding_dimension"] = pairs["model"].map(_NATIVE_DIMENSIONS)
    if pairs["embedding_dimension"].isna().any():
        message = "representation grid contains an encoder without a native dimension"
        raise ValueError(message)
    roster = load_representation_ablation_specs(config.params_file)
    vintage_roster = load_encoder_vintage_specs(config.params_file)
    pairs = _append_registered_cells(
        pairs,
        config.representation_dirs,
        roster,
    )
    pairs = _append_registered_cells(pairs, config.vintage_dirs, vintage_roster)

    canonical = _aligned_cell(
        pairs,
        _CANONICAL_MODEL,
        _CANONICAL_WIDTH,
        _NATIVE_DIMENSIONS[_CANONICAL_MODEL],
        _CANONICAL_DISTANCE,
    )
    tickers = sorted(
        set(canonical["ticker_i"].astype(str)) | set(canonical["ticker_j"].astype(str))
    )
    counts = _node_counts(config.node_counts_file, tickers)
    canonical_geometry = canonical[["ticker_i", "ticker_j", "candidate_distance"]]

    cells: list[dict[str, object]] = []
    draw_frames: list[pd.DataFrame] = []
    for cell_id, candidate_row in enumerate(
        _candidate_rows(pairs, roster, vintage_roster), start=1
    ):
        roster_cell, group, model, articles, dimension, distance = candidate_row
        cell = _aligned_cell(pairs, model, articles, dimension, distance)
        aligned = cell.merge(
            canonical_geometry.rename(
                columns={"candidate_distance": "canonical_w2_distance"}
            ),
            on=["ticker_i", "ticker_j"],
            how="inner",
            validate="one_to_one",
        )
        cell_tickers = sorted(
            set(aligned["ticker_i"].astype(str)) | set(aligned["ticker_j"].astype(str))
        )
        if cell_tickers != tickers or len(aligned) != len(canonical):
            cells.append(
                {
                    "cell_id": cell_id,
                    "representation_cell": roster_cell,
                    "group": group,
                    "model": model,
                    "articles": articles,
                    "embedding_dimension": dimension,
                    "distance": distance,
                    "status": "unavailable",
                    "reason": "candidate does not share the canonical firm-dyad panel",
                }
            )
            continue

        outcome = aligned["return_chord"].to_numpy(dtype=np.float64)
        candidate = aligned["candidate_distance"].to_numpy(dtype=np.float64)
        firm_i = aligned["ticker_i"].to_numpy(dtype=str)
        firm_j = aligned["ticker_j"].to_numpy(dtype=str)
        sample = DyadicSample[str](outcome, firm_i, firm_j)
        inference = DyadicInference[str](
            node_effects=True,
            effect_name_prefix="firm_fe",
            bootstrap_iters=counts.shape[0],
            bootstrap_seed=config.bootstrap_seed,
            node_schedule=legacy_node_count_schedule(sample, counts),
        )
        focal_name = (
            "w2_distance" if distance == "wasserstein_w2" else "energy_distance"
        )
        fit = _fit_spec(
            DyadicFit(
                model=DyadicModel([focal_name], {focal_name: candidate}, sample),
                inference=inference,
            ),
            spec_index=cell_id,
            include_draws=True,
        )
        coefficient = float(cast("float", fit[f"coef_{focal_name}"]))
        metric_sd = float(np.std(candidate, ddof=1))
        draws = cast("pd.DataFrame", fit["_bootstrap_draws"])
        beta_column = "beta_w2" if focal_name == "w2_distance" else "beta_energy"
        standardized = draws[beta_column].to_numpy(dtype=float) * metric_sd
        valid = standardized[np.isfinite(standardized) & draws["solve_ok"].to_numpy()]
        geometry_rank = float(
            spearmanr(
                aligned["candidate_distance"], aligned["canonical_w2_distance"]
            ).statistic
        )
        cells.append(
            {
                "cell_id": cell_id,
                "representation_cell": roster_cell,
                "group": group,
                "model": model,
                "articles": articles,
                "embedding_dimension": dimension,
                "distance": distance,
                "status": "computed",
                "canonical": bool(
                    model == _CANONICAL_MODEL
                    and articles == _CANONICAL_WIDTH
                    and dimension == _NATIVE_DIMENSIONS[_CANONICAL_MODEL]
                    and distance == _CANONICAL_DISTANCE
                ),
                "n_firms": len(tickers),
                "n_dyads": len(aligned),
                "geometry_spearman_vs_canonical_w2": geometry_rank,
                "coefficient": coefficient,
                "standard_error": float(cast("float", fit[f"se_{focal_name}"])),
                "pvalue": float(cast("float", fit[f"pvalue_{focal_name}"])),
                "ci_low": float(cast("float", fit[f"ci_low_{focal_name}"])),
                "ci_high": float(cast("float", fit[f"ci_high_{focal_name}"])),
                "effect_one_sd": coefficient * metric_sd,
                "effect_one_sd_ci_low": float(np.percentile(valid, 2.5)),
                "effect_one_sd_ci_high": float(np.percentile(valid, 97.5)),
                "partial_r2": float(cast("float", fit["within_r2"])),
                "bootstrap_valid_draws": int(valid.size),
                "bootstrap_requested_draws": int(counts.shape[0]),
            }
        )
        draw_frames.append(
            pd.DataFrame(
                {
                    "cell_id": cell_id,
                    "representation_cell": roster_cell,
                    "group": group,
                    "draw_id": draws["draw_id"].to_numpy(dtype=np.int32),
                    "standardized_effect": standardized,
                    "solve_ok": draws["solve_ok"].to_numpy(dtype=bool),
                }
            )
        )

    bootstrap_draws = pd.concat(draw_frames, ignore_index=True)
    contrast_specs = load_representation_contrast_specs(config.params_file)
    contrasts = _paired_contrasts(
        cells,
        bootstrap_draws,
        contrast_specs,
        confidence_level=config.vintage_confidence_level,
        equivalence_bound=config.vintage_equivalence_bound,
    )
    return _write_representation_outputs(
        config,
        requested_draws=int(counts.shape[0]),
        cells=cells,
        bootstrap_draws=bootstrap_draws,
        contrasts=contrasts,
    )
