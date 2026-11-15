"""Paper-1 adapters around the general array-only dyadic estimator."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from scipy.stats import skew
from statadic.estimation.least_squares.dyadic import DyadicComputationError
from statadic.inference.dyadic import DyadicBootstrapDesign
from statadic.numerics.linalg.dyadic import fwl_residualize, symmetric_dyadic_effects

from pipeline.stages.papers.paper1._dyadic.adapters import (
    dyadic_regression,
)

if TYPE_CHECKING:
    from statadic.inference.dyadic import (
        DyadicBootstrapResult,
        DyadicFitRequest,
        DyadicInference,
    )

_BootstrapDesign = DyadicBootstrapDesign
_symmetric_firm_effects = symmetric_dyadic_effects

#: Draws below this count make a third moment meaningless, not merely noisy.
_MIN_SKEWNESS_DRAWS = 3


def _sample_skewness(draws: np.ndarray) -> float:
    """Return SciPy's bias-uncorrected Fisher-Pearson coefficient on finite draws.

    Fewer than three draws, constant samples, and numerically unresolved
    near-constant samples have no skewness diagnostic and return NaN.
    """
    values = np.asarray(draws, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < _MIN_SKEWNESS_DRAWS or np.ptp(values) <= 0.0:
        return float("nan")
    return float(skew(values, bias=True))


def _fwl_display(
    x_kept: np.ndarray,
    kept_names: list[str],
    y: np.ndarray,
    omitted_firm: str,
    focal_name: str = "energy_distance",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return a both-sides residualized FWL display of the focal slope."""
    focal_pos = kept_names.index(focal_name)
    residuals = fwl_residualize(x_kept, y, focal_pos)
    focal_residual = np.asarray(residuals.focal_residual)
    outcome_residual = np.asarray(residuals.outcome_residual)
    slope = float(residuals.slope)
    prefix = "energy" if focal_name == "energy_distance" else "w2"
    frame = pd.DataFrame(
        {
            f"{prefix}_fwl_residual": focal_residual,
            "outcome_fwl_residual": outcome_residual,
            "fwl_fitted": slope * focal_residual,
        }
    )
    summary: dict[str, object] = {
        "slope": slope,
        "focal_regressor": focal_name,
        "identity": (
            "X_tilde = M_Z X; D_tilde = M_Z D; "
            "beta = (X_tilde' D_tilde)/(X_tilde' X_tilde)"
        ),
        "nuisance_design": (
            "intercept plus symmetric additive firm effects a_i + a_j, one "
            "indicator per non-omitted firm, taken from the rank-cleaned "
            "primary point design"
        ),
        "omitted_firm_indicator": omitted_firm,
        "nuisance_columns": int(x_kept.shape[1] - 1),
        "n_dyads": len(y),
        "focal_residual_sd": float(np.std(focal_residual, ddof=1)),
        "outcome_residual_sd": float(np.std(outcome_residual, ddof=1)),
        "fitted_line_basis": (
            "least squares through all dyads; any equal-count bin means shown "
            "alongside are descriptive overlays and do not enter the fit"
        ),
        "scope": (
            "algebraic re-expression of the reported conditional coefficient; "
            "not an identification or functional-form test"
        ),
    }
    return frame, summary


def _partial_r2_block(
    restricted_sse: float | None,
    full_sse: float | None,
    within_r2: float | None,
    focal_label: str = "energy metric",
) -> dict[str, object]:
    """Return formula-defined conditional goodness of fit for the primary rung."""
    focal_symbol = "E" if focal_label == "energy metric" else "X"
    result: dict[str, object] = {
        "definition": f"partial_r2 = 1 - SSE_(FE+{focal_symbol})/SSE_FE",
        "restricted_model": "intercept plus symmetric additive firm effects",
        "full_model": f"restricted model plus the {focal_label}",
        "sse_firm_effects_only": restricted_sse,
        "sse_firm_effects_plus_focal": full_sse,
        "partial_r2": within_r2,
        "scope": (
            "defined only where the restricted fixed-effect model exists; "
            "undefined for pooled rungs and not comparable with full-model "
            "R-squared across blocks"
        ),
    }
    if focal_label == "energy metric":
        result["sse_firm_effects_plus_energy"] = full_sse
    return result


def _return_chord_translation(
    y: np.ndarray,
    effect_one_sd: float,
    focal_key: str = "energy",
) -> dict[str, object]:
    """Re-express a fitted chord-distance effect on the correlation scale."""
    quantile_levels = (25.0, 50.0, 75.0)
    quantiles = {
        f"q{int(level)}": float(np.percentile(y, level)) for level in quantile_levels
    }
    anchors: dict[str, object] = {}
    for key, chord in quantiles.items():
        chord_after = chord + effect_one_sd
        rho = 1.0 - chord * chord / 2.0
        rho_after = 1.0 - chord_after * chord_after / 2.0
        anchors[key] = {
            "chord_distance": chord,
            "correlation": rho,
            "chord_distance_after_one_sd": chord_after,
            "correlation_after_one_sd": rho_after,
            "delta_correlation": rho_after - rho,
        }
    return {
        "transformation": "rho = 1 - D^2/2; d rho = -D dD - (dD)^2/2",
        "quantile_levels_percent": [float(level) for level in quantile_levels],
        "return_chord_distance_quantiles": quantiles,
        "return_chord_distance_min": float(np.min(y)),
        "return_chord_distance_max": float(np.max(y)),
        f"delta_chord_one_sd_{focal_key}": float(effect_one_sd),
        "anchors": anchors,
        "scope": (
            "deterministic given the fitted coefficient; same reduced-form, "
            "same-window scope as the coefficient itself, and not evidence "
            "about a transmission constant, exposure norms, systematic "
            "variance shares, or the numerical covariance floor"
        ),
    }


def _bootstrap_result_frame(
    result: DyadicBootstrapResult,
    *,
    spec_index: int,
) -> pd.DataFrame:
    """Adapt the array-only statadic result to Paper 1's stable artifact schema.

    t65 moved every ``DyadicBootstrapResult`` field to ``jax.Array``; this
    stage owns the host conversion at its artifact edge, so the parquet
    schema written downstream is unchanged.
    """
    return pd.DataFrame(
        {
            "draw_id": np.asarray(result.draw_id),
            "spec_index": np.full(result.draw_id.shape, spec_index, dtype=np.int64),
            "beta_energy": np.asarray(result.coefficient("energy_distance")),
            "beta_w2": np.asarray(result.coefficient("w2_distance")),
            "beta_energy_squared": np.asarray(
                result.coefficient("energy_distance_centered_squared")
            ),
            "sse": np.asarray(result.sse),
            "overall_r2": np.asarray(result.overall_r2),
            "within_r2": np.asarray(result.within_r2),
            "active_nodes": np.asarray(result.active_nodes),
            "active_dyads": np.asarray(result.active_dyads),
            "effective_columns": np.asarray(result.effective_columns),
            "effective_rank": np.asarray(result.effective_rank),
            "solve_ok": np.asarray(result.solve_ok),
        }
    )


def _fit_spec(
    request: DyadicFitRequest[str],
    *,
    spec_index: int = -1,
    include_draws: bool = False,
) -> dict[str, object]:
    """Adapt a general statadic dyadic fit to Paper 1's artifact contract."""
    model = request.model
    inference = replace(
        request.inference,
        required_bootstrap_columns=tuple(
            name
            for name in (
                "energy_distance",
                "w2_distance",
                "energy_distance_centered_squared",
            )
            if name in model.regressor_names
        ),
    )
    spec, data = dyadic_regression(model)
    point = spec.estimate(data)
    # The facade erases endpoint types; the aligned schedule still uses these strings.
    fit = point.infer(resampling=cast("DyadicInference", inference)).core
    boot_frame = _bootstrap_result_frame(fit.bootstrap_draws, spec_index=spec_index)
    valid_frame = boot_frame.loc[boot_frame["solve_ok"]]
    focal_name = (
        "w2_distance"
        if "w2_distance" in model.regressor_names
        and "energy_distance" not in model.regressor_names
        else "energy_distance"
    )
    focal_column = "beta_w2" if focal_name == "w2_distance" else "beta_energy"
    boot = valid_frame[focal_column].dropna().to_numpy(dtype=float)
    if focal_name in model.regressor_names and boot.size < max(
        100, inference.bootstrap_iters // 2
    ):
        message = (
            f"only {boot.size}/{inference.bootstrap_iters} valid node-bootstrap draws"
        )
        raise DyadicComputationError(message)
    coefficient = float(fit.coefficients.get(focal_name, np.nan))
    uncertainty = fit.bootstrap_inference.get(focal_name)
    result: dict[str, object] = {
        "n_dyads": fit.n_dyads,
        "regressors": model.regressor_names,
        f"coef_{focal_name}": coefficient,
        f"effect_one_sd_{focal_name}": fit.effect_one_sd.get(focal_name, float("nan")),
        f"se_{focal_name}": uncertainty.standard_error if uncertainty else float("nan"),
        f"t_{focal_name}": uncertainty.t_statistic if uncertainty else float("nan"),
        f"pvalue_{focal_name}": uncertainty.pvalue if uncertainty else float("nan"),
        f"ci_low_{focal_name}": uncertainty.ci_lower if uncertainty else float("nan"),
        f"ci_high_{focal_name}": uncertainty.ci_upper if uncertainty else float("nan"),
        "bootstrap_valid_draws": fit.bootstrap_valid_draws,
        "bootstrap_requested_draws": fit.bootstrap_requested_draws,
        "bootstrap_seed": fit.bootstrap_seed,
        # Fisher-Pearson g1 on the valid energy-coefficient draws. Recorded
        # because the reported interval is the percentile one, whose accuracy
        # degrades with skewness in exactly this distribution: the appendix
        # discloses the measured value rather than asserting symmetry. NaN
        # when the draws are degenerate, which the zero-variance guard below
        # keeps out of the ratio.
        f"bootstrap_skew_{focal_name}": _sample_skewness(boot),
        "se_type": "multinomial node bootstrap (dyad weight w_i*w_j)",
        "firm_effects": model.node_effects,
        "overall_r2": fit.overall_r2,
        "within_r2": fit.within_r2,
        "sse_firm_effects_only": fit.sse_node_effects_only,
        "sse_firm_effects_plus_regressors": (
            fit.sse_full if model.node_effects else None
        ),
        "design_rank": fit.design_rank,
        "design_columns": fit.design_columns,
        f"fwl_{focal_name}": fit.fwl_coefficients.get(focal_name),
        "coefficients": {
            name: {"coef": float(value)} for name, value in fit.coefficients.items()
        },
        "dropped_collinear_columns": list(fit.dropped_collinear_columns),
    }
    if include_draws:
        # statadic returns JAX arrays; this stage is the reporting boundary, so it
        # materializes them once here. The FWL numerical core remains in JAX
        # under the stage's caller-owned float64 policy, and its display adapter
        # likewise converts residual arrays before assembling DataFrames.
        result["_bootstrap_draws"] = boot_frame
        result["_point_design"] = np.asarray(fit.point_design)
        result["_point_parameters"] = np.asarray(point.core.parameters)
        result["_point_design_names"] = list(fit.point_design_names)
        result["_point_fitted"] = np.asarray(fit.point_fitted)
        result["_point_residual"] = np.asarray(fit.point_residual)
    return result
