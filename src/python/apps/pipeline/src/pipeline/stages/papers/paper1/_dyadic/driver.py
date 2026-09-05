"""Top-level Paper-1 dyadic-stage orchestration."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from jcor.model.dyadic import (
    DyadicFit,
    DyadicModel,
)
from jcor.operators.design import symmetric_dyadic_effects

from pipeline.stages.papers.paper1._dyadic.artifacts import write_dyadic_artifacts
from pipeline.stages.papers.paper1._dyadic.contracts import (
    PRIMARY_SPEC_INDEX,
    DyadicArtifacts,
)
from pipeline.stages.papers.paper1._dyadic.diagnostics import (
    _assemble_primary_fit_frame,
    _design_diagnostic_row,
    _influence_diagnostics,
    _leave_one_firm_out,
    _residual_scale_diagnostic,
)
from pipeline.stages.papers.paper1._dyadic.estimation import (
    _fit_spec,
    _partial_r2_block,
    _return_chord_translation,
)
from pipeline.stages.papers.paper1._dyadic.inputs import prepare_dyadic_inputs
from pipeline.stages.papers.paper1._dyadic.ladder import (
    configured_ladder,
    model_specification_metadata,
)
from pipeline.stages.papers.paper1._dyadic.random_exposure import (
    RandomExposureInputs,
    fixed_w2_random_exposure_diagnostic,
)

if TYPE_CHECKING:
    from pipeline.stages.papers.paper1._dyadic.contracts import (
        DyadicConfoundConfig,
    )


def _w2_curvature_display(
    w2: np.ndarray,
    outcome: np.ndarray,
    firm_i: np.ndarray,
    firm_j: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return a descriptive quadratic check after symmetric-FE residualization."""
    effects, _ = symmetric_dyadic_effects(firm_i, firm_j)
    nuisance = np.column_stack([np.ones(len(outcome)), np.asarray(effects)])
    outcome_residual = (
        outcome - nuisance @ np.linalg.lstsq(nuisance, outcome, rcond=None)[0]
    )
    centered = w2 - float(np.mean(w2))
    terms = np.column_stack([w2, centered**2])
    residual_terms = terms - nuisance @ np.linalg.lstsq(nuisance, terms, rcond=None)[0]
    coefficients = np.linalg.lstsq(residual_terms, outcome_residual, rcond=None)[0]
    return (
        pd.DataFrame(
            {
                "w2_distance": w2,
                "w2_fe_residual": residual_terms[:, 0],
                "outcome_fe_residual": outcome_residual,
                "quadratic_fe_partial_fit": residual_terms @ coefficients,
            }
        ),
        {
            "linear_coefficient": float(coefficients[0]),
            "quadratic_coefficient": float(coefficients[1]),
            "scope": (
                "descriptive curvature check after the primary symmetric-firm-"
                "effect residualization; not a separate hypothesis test"
            ),
        },
    )


def run_dyadic_confound(
    config: DyadicConfoundConfig,
) -> dict[str, object]:
    """Fit the Paper-1 dyadic ladder and write its governed artifact family."""
    prepared = prepare_dyadic_inputs(config)
    sample = prepared.sample

    specs, regressor_diagnostics = model_specification_metadata(
        prepared.columns, has_dispersions=prepared.dispersions is not None
    )
    common_node_counts, inference, ladder = configured_ladder(prepared, config, specs)
    primary_fit = ladder.primary

    design_rows = [
        _design_diagnostic_row(
            spec_index,
            DyadicModel(regressor_names, prepared.columns, sample),
            firm_effects=spec_index == PRIMARY_SPEC_INDEX,
            focal_name="w2_distance",
        )
        for spec_index, _label, regressor_names in specs
    ]
    design_diagnostics = pd.DataFrame(design_rows)

    functional_display, functional_form_sensitivity = _w2_curvature_display(
        prepared.columns["w2_distance"],
        np.asarray(sample.outcome),
        np.asarray(sample.endpoint_i),
        np.asarray(sample.endpoint_j),
    )

    primary_design = cast("np.ndarray", primary_fit["_point_design"])
    # jcor's dyadic carriers are backend-neutral (`ArrayView`); the
    # reporting helpers below are NumPy, so materialize at this boundary.
    primary_fit_frame, fwl_diagnostics = _assemble_primary_fit_frame(
        primary_fit,
        np.asarray(sample.outcome),
        np.asarray(sample.endpoint_i),
        np.asarray(sample.endpoint_j),
        functional_display,
        focal_name="w2_distance",
    )
    leave_one_out = _leave_one_firm_out(
        DyadicModel(["w2_distance"], prepared.columns, sample),
        float(cast("float | int | str", primary_fit["coef_w2_distance"])),
        float(cast("float | int | str", primary_fit["se_w2_distance"])),
        focal_name="w2_distance",
    )
    residual_scale, residual_scale_null = _residual_scale_diagnostic(
        primary_fit_frame,
        np.asarray(sample.endpoint_i),
        np.asarray(sample.endpoint_j),
    )
    influence_diagnostics = _influence_diagnostics(
        primary_fit_frame, leave_one_out, primary_design.shape[1]
    )

    firm_effect_columns, _ = symmetric_dyadic_effects(
        sample.endpoint_i, sample.endpoint_j
    )
    firm_effect_design = np.column_stack(
        [np.ones(len(sample.outcome)), firm_effect_columns]
    )
    w2_metric = prepared.columns["w2_distance"]
    fitted_outcome = (
        firm_effect_design
        @ np.linalg.lstsq(firm_effect_design, np.asarray(sample.outcome), rcond=None)[0]
    )
    fitted_w2 = (
        firm_effect_design
        @ np.linalg.lstsq(firm_effect_design, w2_metric, rcond=None)[0]
    )
    residual_outcome = sample.outcome - fitted_outcome
    residual_w2 = w2_metric - fitted_w2

    def variance_share(observed: np.ndarray, residual: np.ndarray) -> float:
        total = float(np.sum((observed - observed.mean()) ** 2))
        return 1.0 - float(residual @ residual) / total

    firm_margin_diagnostics = {
        "w2_metric_variance_share": variance_share(w2_metric, residual_w2),
        "return_distance_variance_share": variance_share(
            np.asarray(sample.outcome), residual_outcome
        ),
        "within_residual_correlation": float(
            np.corrcoef(residual_w2, residual_outcome)[0, 1]
        ),
        "between_fitted_correlation": float(
            np.corrcoef(fitted_w2, fitted_outcome)[0, 1]
        ),
    }

    coefficients = cast(
        "dict[str, dict[str, float]]", ladder.historical["coefficients"]
    )
    random_exposure, random_exposure_frame = fixed_w2_random_exposure_diagnostic(
        RandomExposureInputs(
            w2=prepared.columns["w2_distance"],
            correlation=prepared.columns["return_correlation"],
            firm_i=np.asarray(sample.endpoint_i),
            firm_j=np.asarray(sample.endpoint_j),
            tickers=prepared.tickers,
            node_counts=common_node_counts,
        )
    )
    energy_fit = _fit_spec(
        DyadicFit(
            model=DyadicModel(["energy_distance"], prepared.columns, sample),
            inference=replace(inference, node_effects=True),
        ),
        spec_index=7,
        include_draws=True,
    )
    ladder.bootstrap_frames.append(cast("pd.DataFrame", energy_fit["_bootstrap_draws"]))
    energy_comparator = {
        "regressor": "energy_v",
        "role": "single theory-aligned lower-cost comparison row",
        "coefficient": energy_fit["coef_energy_distance"],
        "effect_one_sd": energy_fit["effect_one_sd_energy_distance"],
        "se": energy_fit["se_energy_distance"],
        "t": energy_fit["t_energy_distance"],
        "pvalue": energy_fit["pvalue_energy_distance"],
        "ci_low": energy_fit["ci_low_energy_distance"],
        "ci_high": energy_fit["ci_high_energy_distance"],
        "within_r2": energy_fit["within_r2"],
        "overall_r2": energy_fit["overall_r2"],
    }
    result: dict[str, object] = {
        "n_dyads": primary_fit["n_dyads"],
        "n_firms": len(prepared.tickers),
        "primary_metric": "wasserstein_w2",
        "primary_regressor": "w2_distance",
        "primary_spec_index": PRIMARY_SPEC_INDEX,
        "coef_w2_distance": primary_fit["coef_w2_distance"],
        "effect_one_sd_w2_distance": primary_fit["effect_one_sd_w2_distance"],
        "se_w2_distance": primary_fit["se_w2_distance"],
        "t_w2_distance": primary_fit["t_w2_distance"],
        "pvalue_w2_distance": primary_fit["pvalue_w2_distance"],
        "ci_low_w2_distance": primary_fit["ci_low_w2_distance"],
        "ci_high_w2_distance": primary_fit["ci_high_w2_distance"],
        "overall_r2": primary_fit["overall_r2"],
        "within_r2": primary_fit["within_r2"],
        "energy_dyadic_comparator": energy_comparator,
        "coef_same_sector": coefficients["same_sector"]["coef"],
        "coef_size_gap": coefficients["size_gap"]["coef"],
        "coef_crisis": coefficients["crisis"]["coef"],
        "coef_intercept": coefficients["intercept"]["coef"],
        "dropped_collinear_columns": ladder.historical["dropped_collinear_columns"],
        "se_type": "multinomial node bootstrap (dyad weight w_i*w_j)",
        "w2_scale": "rooted balanced quadratic Wasserstein distance",
        "inference_scope": "associational; symmetric additive firm margins",
        "fwl_diagnostics": fwl_diagnostics,
        "partial_r2_diagnostics": _partial_r2_block(
            cast("float | None", primary_fit["sse_firm_effects_only"]),
            cast("float | None", primary_fit["sse_firm_effects_plus_regressors"]),
            cast("float | None", primary_fit["within_r2"]),
            focal_label="W2 distance",
        ),
        "correlation_scale_translation": _return_chord_translation(
            np.asarray(sample.outcome),
            float(cast("float | int | str", primary_fit["effect_one_sd_w2_distance"])),
            focal_key="w2",
        ),
        "firm_margin_diagnostics": firm_margin_diagnostics,
        "functional_form_sensitivity": functional_form_sensitivity,
        "residual_scale_diagnostic": residual_scale,
        "influence_diagnostics": influence_diagnostics,
        "bootstrap_common_node_counts": True,
        "bootstrap_seed": int(config.bootstrap_seed),
        "model_ladder": ladder.model_ladder,
        "regressor_diagnostics": regressor_diagnostics,
        "random_exposure_diagnostic": random_exposure,
        "w2_primary_regression": {
            "status": "computed",
            "regressor": "wasserstein_w2",
            "coefficient": primary_fit["coef_w2_distance"],
            "effect_one_sd": primary_fit["effect_one_sd_w2_distance"],
            "se": primary_fit["se_w2_distance"],
            "t": primary_fit["t_w2_distance"],
            "pvalue": primary_fit["pvalue_w2_distance"],
            "ci_low": primary_fit["ci_low_w2_distance"],
            "ci_high": primary_fit["ci_high_w2_distance"],
            "within_r2": primary_fit["within_r2"],
            "overall_r2": primary_fit["overall_r2"],
            "bootstrap_valid_draws": primary_fit["bootstrap_valid_draws"],
            "bootstrap_requested_draws": primary_fit["bootstrap_requested_draws"],
        },
    }
    if prepared.dispersion_note is not None:
        result["dispersion_note"] = prepared.dispersion_note
    if prepared.u_sensitivity is not None:
        result["u_statistic_sensitivity"] = prepared.u_sensitivity

    write_dyadic_artifacts(
        DyadicArtifacts(
            output_file=config.output_file,
            result=result,
            tickers=prepared.tickers,
            node_counts=common_node_counts,
            bootstrap_frames=ladder.bootstrap_frames,
            design_diagnostics=design_diagnostics,
            primary_fit=primary_fit_frame,
            leave_one_out=leave_one_out,
            residual_scale_null=residual_scale_null,
            random_exposure_diagnostic=random_exposure_frame,
        )
    )
    return result
