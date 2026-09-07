"""Paper 5 H3 robustness-bracket and H4 encompassing evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
from jcor.model.covariance import sample_covariance as _sample_cov
from jcor.operators.covariance import sigma_dist as sigma_dist_fn

from pipeline.stages.papers.paper3.empirical import calibrate_kappa
from pipeline.stages.papers.paper5 import hypotheses as hyp
from pipeline.stages.papers.paper5._run.rho import _resolve_rho_se

if TYPE_CHECKING:
    import pandas as pd

    from pipeline.stages.papers.paper5._run.contracts import _H3EvaluationInputs


type H3VariantFamilies = dict[str, dict[str, object]]


def _evaluate_h3(
    inputs: _H3EvaluationInputs,
) -> dict[str, H3VariantFamilies]:
    """Evaluate the primary H3 bracket families for both spatial weights."""
    results: dict[str, H3VariantFamilies] = {}
    train_returns = inputs.fold_data["eval2021"]["train"]
    test_returns = inputs.fold_data["eval2021"]["test"]
    diagonal_variance = np.var(train_returns, axis=0, ddof=1)
    for variant_name in ("w_flat", "w_h"):
        if variant_name == "w_flat" and not bool(inputs.flat_bound["certified"]):
            unavailable: H3VariantFamilies = {
                "certification": {
                    "status": str(inputs.flat_bound["certification_status"]),
                    "available": False,
                }
            }
            results[variant_name] = unavailable
            continue
        row = inputs.rho_df[
            (inputs.rho_df["w_variant"] == variant_name)
            & (inputs.rho_df["method"] == "qmle")
            & (inputs.rho_df["fold"] == "eval2021")
        ].iloc[0]
        perturbation_bound = (
            float(inputs.flat_bound["bound_row_median"])
            if variant_name == "w_flat"
            else inputs.delta_w_h["row_norm_median"]
        )
        bracket = hyp.sar_robust_bracket(
            inputs.w_variants[variant_name],
            perturbation_bound,
            float(row["rho_hat"]),
            _resolve_rho_se(variant_name, "eval2021", row),
            diagonal_variance,
        )
        results[variant_name] = hyp.h3_bracket_families(
            bracket,
            np.asarray(_sample_cov(test_returns, ddof=1)),
            inputs.coverage[0],
            inputs.coverage[1],
            inputs.lw_band,
        )
    return results


def _evaluate_h4(
    folds: dict[str, tuple[str, str]],
    fold_data: dict[str, dict[str, np.ndarray]],
    rho_df: pd.DataFrame,
    w_variants: dict[str, np.ndarray],
    d2: np.ndarray,
) -> dict[str, object]:
    """Evaluate H4 encompassing tests for each fold and spatial-weight family."""
    results: dict[str, object] = {}
    for variant_name in ("w_flat", "w_h"):
        rows = {}
        for fold_name in folds:
            train_returns = fold_data[fold_name]["train"]
            test_returns = fold_data[fold_name]["test"]
            row = rho_df[
                (rho_df["w_variant"] == variant_name)
                & (rho_df["method"] == "qmle")
                & (rho_df["fold"] == fold_name)
            ].iloc[0]
            diagonal_variance = np.var(train_returns, axis=0, ddof=1)
            kappa_hat, _ = calibrate_kappa(train_returns, d2)
            sigma_sq = np.diag(np.cov(train_returns, rowvar=False, ddof=1))
            # Materialize the JAX covariance for the NumPy hypothesis record.
            with jax.enable_x64(new_val=True):
                sigma_dist = np.asarray(sigma_dist_fn(d2, sigma_sq, kappa=kappa_hat))
            rows[fold_name] = hyp.h4_encompassing(
                hyp.EncompassingInputs(
                    train_returns,
                    test_returns,
                    w_variants[variant_name],
                    float(row["rho_hat"]),
                    diagonal_variance,
                    sigma_dist,
                )
            )
        results[variant_name] = rows
    return results


def _evaluate_h3_per_fold(
    folds: dict[str, tuple[str, str]],
    inputs: _H3EvaluationInputs,
) -> dict[str, dict[str, H3VariantFamilies]]:
    """Evaluate H3 bracket families across every configured fold."""
    results: dict[str, dict[str, H3VariantFamilies]] = {}
    for fold_name in folds:
        results[fold_name] = {}
        train_returns = inputs.fold_data[fold_name]["train"]
        test_returns = inputs.fold_data[fold_name]["test"]
        for variant_name in ("w_flat", "w_h"):
            if variant_name == "w_flat" and not bool(inputs.flat_bound["certified"]):
                unavailable: H3VariantFamilies = {
                    "certification": {
                        "status": str(inputs.flat_bound["certification_status"]),
                        "available": False,
                    }
                }
                results[fold_name][variant_name] = unavailable
                continue
            row = inputs.rho_df[
                (inputs.rho_df["w_variant"] == variant_name)
                & (inputs.rho_df["method"] == "qmle")
                & (inputs.rho_df["fold"] == fold_name)
            ].iloc[0]
            perturbation_bound = (
                float(inputs.flat_bound["bound_row_median"])
                if variant_name == "w_flat"
                else inputs.delta_w_h["row_norm_median"]
            )
            bracket = hyp.sar_robust_bracket(
                inputs.w_variants[variant_name],
                perturbation_bound,
                float(row["rho_hat"]),
                _resolve_rho_se(variant_name, fold_name, row),
                np.var(train_returns, axis=0, ddof=1),
            )
            results[fold_name][variant_name] = hyp.h3_bracket_families(
                bracket,
                np.asarray(_sample_cov(test_returns, ddof=1)),
                inputs.coverage[0],
                inputs.coverage[1],
                inputs.lw_band,
            )
    return results
