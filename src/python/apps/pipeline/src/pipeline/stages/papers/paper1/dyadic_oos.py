"""Out-of-sample primary W2 dyadic association for Paper 1.

The Qwen3-Embedding 8B W2 matrix is frozen at its 2018--2022 estimate while
the return chord matrix and symmetric additive firm effects are re-estimated
on the 2023--2026 panel. Inference reuses the governed multinomial node
bootstrap convention from the in-sample specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import yaml
from jcor.model.dyadic import (
    DyadicComputationError,
    DyadicFit,
    DyadicInference,
    DyadicInputError,
    DyadicModel,
    DyadicSample,
)
from jcor.model.dyadic import (
    bootstrap_coefficient_inference as _bootstrap_inference,
)

from pipeline.stages.papers.paper1.dyadic_confound import (
    _fit_spec,
    _load_covariance_long,
    _load_w2_long,
    legacy_node_count_schedule,
)
from pipeline.stages.papers.paper1.dyadic_confound import (
    draw_node_counts as _draw_node_counts,
)

if TYPE_CHECKING:
    from pathlib import Path

_DEFAULT_BOOTSTRAP_ITERS = 1_999
_DEFAULT_BOOTSTRAP_SEED = 42
_MIN_VALID_DRAWS = 100
_W2_SPEC_INDEX = 7

_SCOPE = (
    "look-ahead-free associational estimate: the text-side matrix is frozen at "
    "its 2018-22 in-sample value and the return side is recomputed on 2023-2026 "
    "returns; this removes return-endogenous news coverage as an explanation of "
    "the association and does not make the exercise predictive, causal, or a "
    "test of the theorem's numerical floor"
)


@dataclass(frozen=True)
class DyadicOosConfig:
    """Artifact paths and bootstrap contract for the primary OOS W2 arm."""

    distance_artifact_dir: Path
    covariance_matrix: Path
    output_file: Path
    bootstrap_iters: int = _DEFAULT_BOOTSTRAP_ITERS
    bootstrap_seed: int = _DEFAULT_BOOTSTRAP_SEED


def _arm_columns(
    distance_artifact_dir: Path, covariance_matrix: Path
) -> tuple[
    list[str],
    list[str],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Build the surviving out-of-sample dyad panel for the W2 arm.

    Returns ``(text_universe, tickers, columns, y, firm_i, firm_j)`` where
    ``text_universe`` is the frozen text-side firm set before the out-of-sample
    return panel is intersected in, so attrition is measurable rather than
    assumed.
    """
    w2_long = _load_w2_long(distance_artifact_dir)
    energy_universe = sorted(
        str(ticker) for ticker in set(w2_long["ticker_i"]) | set(w2_long["ticker_j"])
    )
    merged = w2_long.merge(
        _load_covariance_long(covariance_matrix),
        on=["ticker_i", "ticker_j"],
        how="inner",
    )
    if merged.empty:
        message = (
            "no overlapping dyads between the frozen W2 matrix and the "
            "out-of-sample covariance matrix"
        )
        raise DyadicInputError(message)
    tickers = sorted(
        str(ticker) for ticker in set(merged["ticker_i"]) | set(merged["ticker_j"])
    )
    firm_i = merged["ticker_i"].to_numpy(dtype=str)
    firm_j = merged["ticker_j"].to_numpy(dtype=str)
    columns: dict[str, np.ndarray] = {
        "w2_distance": merged["w2_distance"].to_numpy(dtype=float),
    }
    correlation = np.clip(merged["correlation"].to_numpy(dtype=float), -1.0, 1.0)
    y = np.sqrt(2.0 * (1.0 - correlation))
    return energy_universe, tickers, columns, y, firm_i, firm_j


def _panel_block(
    energy_universe: list[str],
    tickers: list[str],
    n_dyads: int,
    design_columns: int,
) -> dict[str, object]:
    """Record the realized panel and state the degrees-of-freedom position.

    Requirement 3 of the task: the survivor count is what determines whether
    ``N-1`` firm effects estimated from ``N(N-1)/2`` dyads remains comfortable,
    so every count below is derived from the fitted panel rather than inherited
    from an upstream stage description.
    """
    n_firms = len(tickers)
    complete_dyads = n_firms * (n_firms - 1) // 2
    residual_dof = n_dyads - design_columns
    return {
        "n_firms": int(n_firms),
        "n_dyads": int(n_dyads),
        "complete_graph_dyads": int(complete_dyads),
        "graph_is_complete": bool(n_dyads == complete_dyads),
        "firm_effect_columns": int(n_firms - 1),
        "design_columns": int(design_columns),
        "residual_degrees_of_freedom": int(residual_dof),
        "dyads_per_estimated_parameter": float(n_dyads / design_columns),
        "degrees_of_freedom_statement": (
            f"{n_dyads} dyads support an intercept, one focal regressor, and "
            f"{n_firms - 1} symmetric firm-effect indicators ({design_columns} "
            f"estimated parameters), leaving {residual_dof} residual degrees of "
            "freedom; each firm effect is identified off that firm's "
            f"{n_firms - 1} incident dyads"
        ),
        "attrition": {
            "frozen_text_universe_firms": len(energy_universe),
            "surviving_firms": int(n_firms),
            "dropped_from_text_universe": sorted(set(energy_universe) - set(tickers)),
            "rule": (
                "surviving panel is the intersection of the frozen 2018-22 "
                "W2 dyads with the out-of-sample complete-case "
                "return covariance panel; attrition is handled upstream and is "
                "disclosed here as the realized firm-set difference"
            ),
        },
    }


def _arm_record(
    fit: dict[str, object],
    regressor: str,
    draw_column: str,
    regressor_values: np.ndarray,
    bootstrap_iters: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Summarize one arm with the in-sample inference convention.

    ``_fit_spec`` only enforces the valid-draw floor for its legacy energy
    regressor, so the same guard is applied explicitly to W2 here.
    """
    draws_frame = cast("pd.DataFrame", fit["_bootstrap_draws"])
    valid = draws_frame.loc[draws_frame["solve_ok"]]
    draws = valid[draw_column].dropna().to_numpy(dtype=float)
    floor = max(_MIN_VALID_DRAWS, bootstrap_iters // 2)
    if draws.size < floor:
        message = (
            f"only {draws.size}/{bootstrap_iters} valid node-bootstrap draws for "
            f"{regressor}"
        )
        raise DyadicComputationError(message)
    coefficients = cast("dict[str, dict[str, float]]", fit["coefficients"])
    coefficient = float(coefficients[regressor]["coef"])
    standard_error, t_statistic, p_value, ci_low, ci_high = _bootstrap_inference(
        draws, coefficient
    )
    regressor_sd = float(np.std(regressor_values, ddof=1))
    record: dict[str, object] = {
        "specification": (
            f"return chord distance on {regressor} with an intercept and "
            "symmetric additive firm effects"
        ),
        "regressor": regressor,
        "coef": coefficient,
        "se": standard_error,
        "t": float(t_statistic),
        "pvalue": float(p_value),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "interval_excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
        "regressor_sd": regressor_sd,
        "effect_one_sd": coefficient * regressor_sd,
        "within_r2": fit["within_r2"],
        "overall_r2": fit["overall_r2"],
        "sse_firm_effects_only": fit["sse_firm_effects_only"],
        "sse_firm_effects_plus_regressor": fit["sse_firm_effects_plus_regressors"],
        "n_dyads": fit["n_dyads"],
        "design_columns": fit["design_columns"],
        "design_rank": fit["design_rank"],
        "bootstrap_valid_draws": int(draws.size),
        "bootstrap_requested_draws": int(bootstrap_iters),
        "se_type": "multinomial node bootstrap (dyad weight w_i*w_j)",
        "interval_type": "percentile [2.5, 97.5]",
        "dropped_collinear_columns": fit["dropped_collinear_columns"],
    }
    return record, draws_frame


def _bootstrap_frame(w2_draws: pd.DataFrame) -> pd.DataFrame:
    """Label the primary W2 arm's per-draw records."""
    frame = w2_draws.assign(arm="wasserstein_w2").copy()
    frame["draw_id"] = frame["draw_id"].astype(np.int32)
    frame["spec_index"] = frame["spec_index"].astype(np.int8)
    for name in ("active_nodes", "effective_columns", "effective_rank"):
        frame[name] = frame[name].astype(np.int16)
    frame["active_dyads"] = frame["active_dyads"].astype(np.int32)
    return frame


def run_dyadic_oos(config: DyadicOosConfig) -> dict[str, object]:
    """Fit the primary out-of-sample W2 arm and write ``output_file`` (YAML).

    ``covariance_matrix`` must be the out-of-sample correlation matrix;
    W2 is the frozen in-sample text-side matrix. Returns the same
    dictionary that is written.
    """
    energy_universe, tickers, columns, y, firm_i, firm_j = _arm_columns(
        config.distance_artifact_dir, config.covariance_matrix
    )
    sample = DyadicSample[str](y, firm_i, firm_j)
    node_counts = _draw_node_counts(
        len(tickers), config.bootstrap_iters, config.bootstrap_seed
    )
    node_schedule = legacy_node_count_schedule(sample, node_counts)

    w2_fit = _fit_spec(
        DyadicFit[str](
            model=DyadicModel[str](["w2_distance"], columns, sample),
            inference=DyadicInference[str](
                node_effects=True,
                effect_name_prefix="firm_fe",
                bootstrap_iters=config.bootstrap_iters,
                bootstrap_seed=config.bootstrap_seed,
                node_schedule=node_schedule,
            ),
        ),
        spec_index=_W2_SPEC_INDEX,
        include_draws=True,
    )
    w2_record, w2_draws = _arm_record(
        w2_fit,
        "w2_distance",
        "beta_w2",
        columns["w2_distance"],
        config.bootstrap_iters,
    )

    result: dict[str, object] = {
        "estimand": (
            "out-of-sample primary dyadic association: return chord distance on "
            "the frozen W2 metric with symmetric additive firm effects"
        ),
        "text_side": (
            "FROZEN 2018-22 in-sample Qwen3-Embedding 8B W2 matrix; the news archive "
            "ends 2023-01-02 so every article predates the evaluation window by "
            "construction; no re-embedding and no encoder sweep"
        ),
        "return_side": (
            "chord distance sqrt(2(1-rho)) from the out-of-sample correlation "
            "matrix; intercept and symmetric firm effects re-estimated on the "
            "surviving panel"
        ),
        "primary_metric": "wasserstein_w2",
        "inference_scope": _SCOPE,
        "bootstrap_common_node_counts": True,
        "bootstrap_seed": int(config.bootstrap_seed),
        "bootstrap_requested_draws": int(config.bootstrap_iters),
        "panel": _panel_block(
            energy_universe,
            tickers,
            len(y),
            int(cast("int", w2_fit["design_columns"])),
        ),
        "w2_arm": w2_record,
        "prespecification": (
            "convention fixed in advance by the in-sample design: same node "
            "bootstrap, same iteration count, same seed, same 2.5/97.5 "
            "percentile interval, same symmetric-firm-effect form; no "
            "specification was adjusted after any estimate was seen, and a null "
            "or adjudication-reversing result ships unchanged"
        ),
    }

    config.output_file.parent.mkdir(parents=True, exist_ok=True)
    _bootstrap_frame(w2_draws).to_parquet(
        config.output_file.parent / "bootstrap_draws.parquet", index=False
    )
    pd.DataFrame(
        {
            "draw_id": np.repeat(
                np.arange(config.bootstrap_iters, dtype=np.int32), len(tickers)
            ),
            "ticker": np.tile(
                np.asarray(tickers, dtype=object), config.bootstrap_iters
            ),
            "multiplicity": node_counts.reshape(-1).astype(np.int16),
        }
    ).to_parquet(
        config.output_file.parent / "node_bootstrap_counts.parquet", index=False
    )
    config.output_file.write_text(
        yaml.safe_dump(result, sort_keys=False), encoding="utf-8"
    )
    return result
