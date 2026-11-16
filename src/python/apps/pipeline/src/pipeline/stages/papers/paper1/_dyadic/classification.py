"""Classification controls within Paper 1's symmetric firm-effects design."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from statadic.inference.dyadic import DyadicFitRequest
from statadic.law.dyadic import DyadicInputError, DyadicModel, DyadicSample
from statadic.numerics.linalg.dyadic import fwl_residualize

from pipeline.stages.papers.paper1._dyadic.adapters import dyadic_regression
from pipeline.stages.papers.paper1._dyadic.estimation import (
    _fit_spec,
    _return_chord_translation,
)

if TYPE_CHECKING:
    from pathlib import Path

    from statadic.inference.dyadic import DyadicInference


def classification_indicators(
    universe_csv: Path, sample: DyadicSample[str]
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Align complete snapshot labels; never treat missing labels as matches."""
    universe = pd.read_csv(universe_csv, dtype=str).set_index("Symbol")
    if not universe.index.is_unique:
        message = "classification roster has duplicate symbols"
        raise DyadicInputError(message)
    firms = sorted(
        set(np.asarray(sample.endpoint_i).tolist())
        | set(np.asarray(sample.endpoint_j).tolist())
    )
    labels = universe.reindex(firms)[["Sector", "Industry"]]
    if (
        labels.isna().any().any()
        or labels.apply(lambda col: col.str.strip().eq("")).any().any()
    ):
        message = "sector and industry labels must cover every sample firm"
        raise DyadicInputError(message)
    columns = {}
    for name in ("Sector", "Industry"):
        left = labels[name].reindex(sample.endpoint_i).to_numpy()
        right = labels[name].reindex(sample.endpoint_j).to_numpy()
        columns[f"same_{name.lower()}"] = (left == right).astype(float)
    return columns, {
        "source": str(universe_csv),
        "n_sectors": int(labels["Sector"].nunique()),
        "n_industries": int(labels["Industry"].nunique()),
        "point_in_time": False,
        "timing_scope": (
            "Undated roster snapshot, not historical classifications: these are "
            "retrospective conditional associations, not point-in-time controls."
        ),
    }


@dataclass(frozen=True, slots=True)
class EtnicControl:
    """Hoberg-Phillips ETNIC-All pair similarity aligned to the dyadic sample."""

    similarity: np.ndarray
    covered: np.ndarray
    metadata: dict[str, object]


def etnic_control(alignment_dir: Path, sample: DyadicSample[str]) -> EtnicControl:
    """Align the ETNIC alignment stage's pair similarities to the dyadic sample.

    ETNIC covers only firms filing 10-Ks with a curated gvkey, so the control
    defines a sub-roster rather than a full-panel column. Every pair among the
    roster's firms must carry a score: a partially covered roster would let a
    missing similarity silently shrink the sample it is compared against.
    """
    pairs = pd.read_parquet(
        alignment_dir / "pairs.parquet",
        columns=["ticker_i", "ticker_j", "etnic_all_similarity"],
    )
    summary = cast(
        "dict[str, Any]",
        json.loads((alignment_dir / "results.json").read_text(encoding="utf-8")),
    )
    lookup: dict[tuple[str, str], float] = {}
    for left, right, score in pairs.itertuples(index=False):
        lookup[left, right] = lookup[right, left] = float(score)
    endpoint_i = np.asarray(sample.endpoint_i)
    endpoint_j = np.asarray(sample.endpoint_j)
    similarity = np.array(
        [
            lookup.get((left, right), np.nan)
            for left, right in zip(endpoint_i, endpoint_j, strict=True)
        ]
    )
    covered = np.isfinite(similarity)
    roster = sorted(set(pairs.ticker_i) | set(pairs.ticker_j))
    in_roster = np.isin(endpoint_i, roster) & np.isin(endpoint_j, roster)
    if not np.array_equal(covered, in_roster):
        message = "ETNIC similarity must cover every pair among its roster firms"
        raise DyadicInputError(message)
    # The alignment stage names its matrix correlations after the Mantel test.
    correlations = summary["mantel"]["wasserstein_w2"]
    return EtnicControl(
        similarity=similarity,
        covered=covered,
        metadata={
            "source": str(alignment_dir),
            "n_firms": len(roster),
            "years": summary["roster"]["years"],
            "definition": (
                "mean Hoberg-Phillips ETNIC-All doc2vec similarity of 10-K "
                "business descriptions over co-observed fiscal years"
            ),
            "validation_matrix_correlation_w2": {
                "pearson": float(correlations["pearson"]["r"]),
                "spearman": float(correlations["spearman"]["r"]),
                # Descriptive: the alignment stage reports no permutation
                # p-value after sector residualization.
                "partial_sector": float(correlations["partial_sector"]["r"]),
                "pvalue_max": max(
                    float(correlations[method]["p"])
                    for method in ("pearson", "spearman")
                ),
                "permutations": int(summary["permutations"]),
            },
        },
    )


def _control_inference(
    model: DyadicModel[str], inference: DyadicInference[str], name: str
) -> dict[str, float]:
    """Bootstrap a non-focal control's coefficient on the row's own schedule.

    ``_fit_spec`` reports only the focal W2 coefficient, so the control's
    interval comes from a second fit of the same design and draws.
    """
    spec, data = dyadic_regression(model)
    fit = (
        spec.estimate(data)
        .infer(
            resampling=cast(
                "DyadicInference",
                replace(inference, required_bootstrap_columns=("w2_distance", name)),
            )
        )
        .core
    )
    uncertainty = fit.bootstrap_inference[name]
    return {
        "coefficient": float(fit.coefficients[name]),
        "se": uncertainty.standard_error,
        "pvalue": uncertainty.pvalue,
        "ci_low": uncertainty.ci_lower,
        "ci_high": uncertainty.ci_upper,
        "effect_one_sd": float(fit.effect_one_sd[name]),
    }


def focal_partial_r2(
    fit: dict[str, object], outcome: np.ndarray, focal: str = "w2_distance"
) -> dict[str, float]:
    """Measure one regressor's incremental fit after *all* other included ones."""
    names = cast("list[str]", fit["_point_design_names"])
    design = cast("np.ndarray", fit["_point_design"])
    # The residual of y on the design without the focal column is the
    # restricted-model residual; the full residual is the fit's own.
    residual = np.asarray(
        fwl_residualize(design, outcome, names.index(focal)).outcome_residual
    )
    full_residual = cast("np.ndarray", fit["_point_residual"])
    restricted_sse = float(residual @ residual)
    full_sse = float(full_residual @ full_residual)
    return {
        "restricted_sse": restricted_sse,
        "full_sse": full_sse,
        "partial_r2": 1.0 - full_sse / restricted_sse,
    }


def fit_classification_controls(
    universe_csv: Path,
    model: DyadicModel[str],
    inference: DyadicInference[str],
    etnic_alignment_dir: Path | None = None,
) -> tuple[dict[str, object], list[pd.DataFrame]]:
    """Extend the main FE design and fit its within-industry counterpart.

    Full-panel fits reuse the main node schedule. The within-industry graph
    resamples its own incident firms with the same draw count and seed; labels
    and its disconnected graph are held fixed, with rank handled by statadic.

    With ``etnic_alignment_dir``, two further rows share the ETNIC roster: the
    sector-adjusted design, then the same design with ETNIC similarity added.
    Both resample the same incident firms with the same seed, so they differ
    only by the control and W2's change is not a change of sample.
    """
    indicators, metadata = classification_indicators(universe_csv, model.sample)
    columns = {name: np.asarray(values) for name, values in model.columns.items()}
    columns.update(indicators)
    rows: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []
    specs: list[tuple[int, str, list[str], np.ndarray | None]] = [
        (10, "sector", ["same_sector"], None),
        (11, "industry", ["same_sector", "same_industry"], None),
        (12, "within-industry", [], indicators["same_industry"].astype(bool)),
    ]
    etnic = (
        None
        if etnic_alignment_dir is None
        else etnic_control(etnic_alignment_dir, model.sample)
    )
    if etnic is not None:
        columns["etnic_similarity"] = etnic.similarity
        specs.extend(
            [
                (13, "etnic-roster", ["same_sector"], etnic.covered),
                (14, "etnic", ["same_sector", "etnic_similarity"], etnic.covered),
            ]
        )
    for spec_index, key, controls, keep in specs:
        sample = model.sample
        local_columns = columns
        local_inference = inference
        if keep is not None:
            sample = DyadicSample[str](
                np.asarray(sample.outcome)[keep],
                np.asarray(sample.endpoint_i)[keep].tolist(),
                np.asarray(sample.endpoint_j)[keep].tolist(),
            )
            local_columns = {name: values[keep] for name, values in columns.items()}
            local_inference = replace(local_inference, node_schedule=None)
        fit = _fit_spec(
            DyadicFitRequest(
                DyadicModel(
                    ["w2_distance", *controls],
                    local_columns,
                    sample,
                    node_effects=True,
                    effect_name_prefix=model.effect_name_prefix,
                ),
                local_inference,
            ),
            spec_index=spec_index,
            include_draws=True,
        )
        frames.append(cast("pd.DataFrame", fit["_bootstrap_draws"]))
        effect = float(cast("float", fit["effect_one_sd_w2_distance"]))
        translation = _return_chord_translation(
            np.asarray(sample.outcome), effect, "w2"
        )
        median = cast("dict[str, dict[str, float]]", translation["anchors"])["q50"]
        nodes = sorted(
            set(np.asarray(sample.endpoint_i).tolist())
            | set(np.asarray(sample.endpoint_j).tolist())
        )
        rows.append(
            {
                "key": key,
                "spec_index": spec_index,
                "controls": controls,
                "n_firms": len(nodes),
                "n_dyads": fit["n_dyads"],
                "coefficient": fit["coef_w2_distance"],
                "se": fit["se_w2_distance"],
                "ci_low": fit["ci_low_w2_distance"],
                "ci_high": fit["ci_high_w2_distance"],
                "pvalue": fit["pvalue_w2_distance"],
                "effect_one_sd": effect,
                "regressor_sd": float(np.std(local_columns["w2_distance"], ddof=1)),
                "delta_correlation": median["delta_correlation"],
                "anchor_correlation": median["correlation"],
                **focal_partial_r2(fit, np.asarray(sample.outcome)),
                "restricted_model": ["intercept", "symmetric firm effects", *controls],
                "bootstrap_valid_draws": fit["bootstrap_valid_draws"],
                "bootstrap_requested_draws": fit["bootstrap_requested_draws"],
                "bootstrap_seed": fit["bootstrap_seed"],
                "bootstrap_common_node_counts": keep is None,
                "bootstrap_nodes": nodes,
                "design_rank": fit["design_rank"],
                "dropped_collinear_columns": fit["dropped_collinear_columns"],
            }
        )
        if "etnic_similarity" in controls:
            rows[-1]["etnic_similarity"] = {
                **_control_inference(
                    DyadicModel(
                        ["w2_distance", *controls],
                        local_columns,
                        sample,
                        node_effects=True,
                        effect_name_prefix=model.effect_name_prefix,
                    ),
                    local_inference,
                    "etnic_similarity",
                ),
                "partial_r2": focal_partial_r2(
                    fit, np.asarray(sample.outcome), "etnic_similarity"
                )["partial_r2"],
                "correlation_with_w2": float(
                    np.corrcoef(
                        local_columns["w2_distance"], local_columns["etnic_similarity"]
                    )[0, 1]
                ),
            }
    return {
        "classification": metadata,
        **({} if etnic is None else {"etnic": etnic.metadata}),
        "inference": "multinomial node bootstrap, percentile 95% intervals",
        "scope": (
            "Retrospective extensions of the primary symmetric-FE design. "
            "Partial R2 removes only W2 from each full model. The within-industry "
            "sample and standard deviation differ; its bootstrap resamples the "
            "listed incident firms, retaining within-industry edges only. The "
            "ETNIC rows share one roster and one resampling schedule."
        ),
        "rows": rows,
    }, frames
