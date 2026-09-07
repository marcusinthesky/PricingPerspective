"""Paper-1 model-ladder declaration, fitting, and selection."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from jcor.model.dyadic import (
    DyadicComputationError,
    DyadicFit,
    DyadicInference,
    DyadicModel,
)

from pipeline.stages.papers.paper1._dyadic.adapters import (
    draw_legacy_node_counts,
    legacy_node_count_schedule,
)
from pipeline.stages.papers.paper1._dyadic.contracts import (
    HISTORICAL_SPEC_INDEX,
    PRIMARY_SPEC_INDEX,
    LadderFits,
)
from pipeline.stages.papers.paper1._dyadic.estimation import _fit_spec

if TYPE_CHECKING:
    import pandas as pd
    from jcor.model.dyadic import DyadicSample

    from pipeline.stages.papers.paper1._dyadic.contracts import (
        DyadicConfoundConfig,
        PreparedDyadicInputs,
    )


def fit_model_ladder(
    specs: list[tuple[int, str, list[str]]],
    columns: dict[str, np.ndarray],
    sample: DyadicSample[str],
    inference: DyadicInference[str],
) -> LadderFits:
    """Fit the full specification ladder and validate its required rungs."""
    model_ladder: list[dict[str, object]] = []
    bootstrap_frames: list[pd.DataFrame] = []
    selected: dict[int, dict[str, object]] = {}
    for spec_index, label, regressor_names in specs:
        fit = _fit_spec(
            DyadicFit(
                model=DyadicModel(regressor_names, columns, sample),
                inference=DyadicInference(
                    node_effects=spec_index == PRIMARY_SPEC_INDEX,
                    effect_name_prefix="firm_fe",
                    bootstrap_iters=inference.bootstrap_iters,
                    bootstrap_seed=inference.bootstrap_seed,
                    node_schedule=inference.node_schedule,
                ),
            ),
            spec_index=spec_index,
            include_draws=True,
        )
        bootstrap_frames.append(cast("pd.DataFrame", fit["_bootstrap_draws"]))
        model_ladder.append(
            {
                "spec_index": spec_index,
                "label": label,
                **{key: value for key, value in fit.items() if not key.startswith("_")},
            }
        )
        if spec_index in {
            HISTORICAL_SPEC_INDEX,
            PRIMARY_SPEC_INDEX,
        }:
            selected[spec_index] = fit
    if HISTORICAL_SPEC_INDEX not in selected:
        message = "historical specification 3 was not fitted"
        raise DyadicComputationError(message)
    if PRIMARY_SPEC_INDEX not in selected:
        message = "primary symmetric-firm-effect specification was not fitted"
        raise DyadicComputationError(message)
    return LadderFits(
        model_ladder=model_ladder,
        bootstrap_frames=bootstrap_frames,
        historical=selected[HISTORICAL_SPEC_INDEX],
        primary=selected[PRIMARY_SPEC_INDEX],
    )


def configured_ladder(
    prepared: PreparedDyadicInputs,
    config: DyadicConfoundConfig,
    specs: list[tuple[int, str, list[str]]],
) -> tuple[np.ndarray, DyadicInference[str], LadderFits]:
    """Create the common bootstrap schedule and fit every declared rung."""
    node_counts = draw_legacy_node_counts(
        len(prepared.tickers), config.bootstrap_iters, config.bootstrap_seed
    )
    inference = DyadicInference(
        effect_name_prefix="firm_fe",
        bootstrap_iters=config.bootstrap_iters,
        bootstrap_seed=config.bootstrap_seed,
        node_schedule=legacy_node_count_schedule(prepared.sample, node_counts),
    )
    ladder = fit_model_ladder(
        specs,
        prepared.columns,
        prepared.sample,
        inference,
    )
    return node_counts, inference, ladder


def model_specification_metadata(
    columns: dict[str, np.ndarray],
    *,
    has_dispersions: bool,
) -> tuple[list[tuple[int, str, list[str]]], dict[str, object]]:
    """Declare the contiguous W2 specification ladder and its controls."""
    specs: list[tuple[int, str, list[str]]] = [
        (1, "W2 distance only", ["w2_distance"]),
        (2, "+ same-sector indicator", ["w2_distance", "same_sector"]),
        (
            3,
            "pooled controls [secondary]",
            ["w2_distance", "same_sector", "size_gap", "crisis"],
        ),
    ]
    diagnostics: dict[str, object] = {}
    specs.append(
        (
            4,
            "+ article-count controls",
            [
                "w2_distance",
                "same_sector",
                "size_gap",
                "crisis",
                "abs_log_n_diff",
                "min_log_n",
            ],
        )
    )
    if has_dispersions:
        specs.append(
            (
                5,
                "+ within-firm dispersion (B_i + B_j, |B_i - B_j|)",
                [
                    "w2_distance",
                    "same_sector",
                    "size_gap",
                    "crisis",
                    "abs_log_n_diff",
                    "min_log_n",
                    "disp_sum",
                    "disp_gap",
                ],
            )
        )
    specs.append(
        (6, "W2 distance + symmetric firm effects [canonical]", ["w2_distance"])
    )
    if not has_dispersions:
        return specs, diagnostics
    w2 = columns["w2_distance"]
    for name in ("disp_sum", "disp_gap"):
        correlation = float(np.corrcoef(w2, columns[name])[0, 1])
        squared_correlation = correlation * correlation
        diagnostics[f"corr_w2_{name}"] = correlation
        diagnostics[f"vif_w2_{name}"] = (
            float("inf")
            if squared_correlation >= 1.0
            else 1.0 / (1.0 - squared_correlation)
        )
    diagnostics["dispersion_control_definition"] = (
        "B_i = mean pairwise Euclidean distance among firm i's row-L2-normalised "
        "article embeddings (V-statistic normalisation, 1/N_i^2 over all ordered "
        "pairs including n = n'), matching the term the sample energy-distance "
        "estimator subtracts"
    )
    return specs, diagnostics
