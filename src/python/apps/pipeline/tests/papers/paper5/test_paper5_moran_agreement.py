"""Tests for Paper 5's focused post-corpus number catalog."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from pipeline.figures.paper5.numbers import Paper5NumbersPaths, generate_numbers

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, payload: object) -> None:
    """Write one minimal JSON fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _numbers_paths(
    root: Path, geometry: Path, params: Path, bridge: Path, output: Path
) -> Paper5NumbersPaths:
    """Name every catalog input under one fixture root, as dvc.yaml does."""
    return Paper5NumbersPaths(
        post_corpus_qmle=root / "post_corpus_qmle" / "results.json",
        transport_order_sensitivity=root
        / "transport_order_sensitivity"
        / "results.json",
        spatial_value_comparison=root / "spatial_value_comparison" / "results.json",
        economic_value=root / "economic_value" / "results.json",
        point_in_time_encoder=root / "point_in_time_encoder" / "results.json",
        transport_dimension_ladder=root / "transport_dimension_ladder" / "results.json",
        two_field_qmle=root / "two_field_qmle" / "results.json",
        network_certificate=root / "network_certificate" / "results.json",
        geometry_summary=geometry,
        universe_horizon_bridge=bridge,
        params_file=params,
        output_file=output,
    )


def _boundary(rho: float, lower: float, upper: float) -> dict[str, float]:
    """Build one pinned single-field boundary with its resampled interval."""
    return {
        "rho_b": rho,
        "rho": rho,
        "rho_ci_lower": lower,
        "rho_ci_upper": upper,
        "lambda": rho / (1.0 - rho),
        "lambda_ci_lower": lower / (1.0 - lower),
        "lambda_ci_upper": upper / (1.0 - upper),
    }


def _two_field_period() -> dict[str, object]:
    """Build one minimal joint two-field period with both nested boundaries."""
    rho_b, rho_n = 0.53, 0.23
    total = rho_b + rho_n
    return {
        "trading_days": 815,
        "joint": {
            "rho_b": rho_b,
            "rho_n": rho_n,
            "rho_total_fitted": total,
            "theta": rho_b / total,
            "lambda_b": rho_b / (1.0 - total),
            "lambda_n": rho_n / (1.0 - total),
            "loglik": 111.0,
        },
        "boundaries": {
            "w_flat": _boundary(0.78, 0.75, 0.81),
            "w_co_mentions": _boundary(0.61, 0.56, 0.66),
        },
        "loglik_gain_vs_w_flat": 1.0,
        "loglik_gain_vs_co_mentions": 2.0,
        "bootstrap": {
            "rho_b_ci_lower": 0.50,
            "rho_b_ci_upper": 0.57,
            "rho_n_ci_lower": 0.18,
            "rho_n_ci_upper": 0.27,
            "rho_total_ci_lower": 0.73,
            "rho_total_ci_upper": 0.79,
            "theta_ci_lower": 0.65,
            "theta_ci_upper": 0.75,
            "lambda_b_ci_lower": 1.92,
            "lambda_b_ci_upper": 2.63,
            "lambda_n_ci_lower": 0.70,
            "lambda_n_ci_upper": 1.28,
            "rho_channel_correlation": -0.70,
            "field_b_boundary_share": 0.0,
            "field_n_boundary_share": 0.0,
        },
    }


def _bridge_cell(
    cell_id: str,
    rho: float,
    *,
    universe_size: int,
    horizon: str,
    end: str,
    trading_days: int,
) -> dict[str, object]:
    """Build one observed universe-by-horizon bridge cell."""
    lower, upper = rho - 0.03, rho + 0.03
    return {
        "cell_id": cell_id,
        "universe_size": universe_size,
        "horizon": horizon,
        "status": "observed",
        "window_start": "2023-01-03",
        "window_end": end,
        "trading_days": trading_days,
        "rho_hat": rho,
        "rho_ci_lower": lower,
        "rho_ci_upper": upper,
        "lambda_hat": rho / (1.0 - rho),
        "lambda_ci_lower": lower / (1.0 - lower),
        "lambda_ci_upper": upper / (1.0 - upper),
    }


def _write_inputs(results_dir: Path) -> tuple[Path, Path, Path]:
    """Write the mandatory producer artifacts at minimal scale."""
    result = {
        "rho_hat": 0.78,
        "rho_ci_lower": 0.74,
        "rho_ci_upper": 0.81,
        "loglik": 110.0,
        "trading_days": 815,
        "start": "2023-01-03",
        "end": "2026-04-06",
    }
    _write_json(
        results_dir / "post_corpus_qmle" / "results.json",
        {
            "primary_period": "pooled_2023_2026",
            "results": {"pooled_2023_2026": {"w_flat": result}},
        },
    )
    transport_result = {
        "rho_hat": 0.68,
        "rho_ci_lower": 0.65,
        "rho_ci_upper": 0.71,
        "loglik": 209_000.0,
        "loglik_per_firm_day": 2.56,
        "unadjusted_raw_return_rho": 0.84,
        "factor_conditioning_rho_change": -0.16,
        "n_factors": 6,
    }
    factor_comparison_results = {}
    for model_id, n_factors, w2_rho in (
        ("ff3", 3, 0.70),
        ("ff3_mom", 4, 0.69),
        ("ff5", 5, 0.695),
        ("ff5_mom", 6, 0.68),
    ):
        w1_rho = w2_rho - 0.005
        factor_comparison_results[model_id] = {
            "post_corpus_results": {
                "w1": {
                    **transport_result,
                    "rho_hat": w1_rho,
                    "rho_ci_lower": w1_rho - 0.03,
                    "rho_ci_upper": w1_rho + 0.03,
                    "n_factors": n_factors,
                },
                "w2": {
                    **transport_result,
                    "rho_hat": w2_rho,
                    "rho_ci_lower": w2_rho - 0.03,
                    "rho_ci_upper": w2_rho + 0.03,
                    "n_factors": n_factors,
                },
            },
            "paired_contrast_w2_minus_w1": {"rho": 0.005},
            "predictive_comparison": {
                "w1_score": 2.52,
                "w2_score": 2.5199,
                "w2_minus_w1": -0.0001,
            },
        }
    _write_json(
        results_dir / "transport_order_sensitivity" / "results.json",
        {
            "claim_label": "transport-order sensitivity",
            "claim_boundary": {"paper6_structural_no_go": "unchanged"},
            "sample": {
                "n_assets": 100,
                "calibration": {
                    "start": "2018-03-19",
                    "end": "2022-12-30",
                    "factor_complete_dates": 1207,
                },
                "evaluation": {
                    "start": "2023-01-03",
                    "end": "2026-04-06",
                    "factor_complete_dates": 815,
                },
            },
            "post_corpus_factor_qmle": {
                "primary": True,
                "results": {
                    "w1": {**transport_result, "rho_hat": 0.675},
                    "w2": transport_result,
                },
                "paired_contrast_w2_minus_w1": {
                    "rho": 0.005,
                    "rho_ci_lower": 0.002,
                    "rho_ci_upper": 0.008,
                    "loglik": -12.0,
                    "loglik_ci_lower": -20.0,
                    "loglik_ci_upper": -3.0,
                },
            },
            "predictive_comparison": {
                "w1_score": 2.5261,
                "w2_score": 2.5260,
                "w2_minus_w1": -0.0001,
                "w2_minus_w1_ci_lower": -0.0003,
                "w2_minus_w1_ci_upper": 0.0001,
                "equivalence": {
                    "margin": 0.0005,
                    "interval_lower": -0.00028,
                    "interval_upper": 0.00007,
                    "status": "equivalent",
                },
            },
            "post_corpus_factor_joint_qmle": {
                "fields_entered_unchanged": True,
                "literal_matrix_orthogonalization": False,
                "joint": {
                    "rho_w1": 0.680,
                    "rho_w2": 0.000,
                    "rho_total": 0.680,
                    "theta_w1": 1.0,
                    "loglik": -1.2345e5,
                    "loglik_per_firm_day": -1.61,
                },
                "loglik_gain_vs_w1_only": 0.0,
                "loglik_gain_vs_w2_only": 11.5,
                "bootstrap": {
                    "rho_w1_ci_lower": 0.640,
                    "rho_w1_ci_upper": 0.712,
                    "rho_w2_ci_lower": 0.000,
                    "rho_w2_ci_upper": 0.031,
                    "rho_total_ci_lower": 0.645,
                    "rho_total_ci_upper": 0.715,
                    "theta_w1_ci_lower": 0.952,
                    "theta_w1_ci_upper": 1.0,
                    "w1_only_boundary_share": 0.612,
                    "w2_only_boundary_share": 0.004,
                    "rho_channel_correlation": -0.874,
                    "rho_channel_correlation_status": "estimated",
                },
                "quasi_lr_tests": {
                    "w2_given_w1": {
                        "statistic": 0.4,
                        "bootstrap_p_value": 0.71,
                        "bootstrap_exceedances": 710,
                        "decision": "not_rejected",
                    },
                    "w1_given_w2": {
                        "statistic": 23.0,
                        "bootstrap_p_value": 0.001,
                        "bootstrap_exceedances": 1,
                        "decision": "rejected",
                    },
                },
                "incremental_geometry_conclusion": "w1_dominates_at_optimum",
                "follow_up_status": "results_aware",
            },
            "operator_separation": {
                "off_diagonal_weight_correlation": 0.998,
                "active_support_jaccard": 0.938,
                "induced_signal_correlation": 0.9998,
                "relative_signal_difference_sd": 0.022,
                "factor_residualized_signal_correlation": 0.9987,
                "factor_residualized_relative_signal_difference_sd": 0.018,
                "row_correlation_minimum": 0.994,
                "weakly_separated": True,
            },
            "factor_model_comparison": {
                "model_order": ["ff3", "ff3_mom", "ff5", "ff5_mom"],
                "primary_model": "ff5_mom",
                "identical_return_dates_across_models": True,
                "results": factor_comparison_results,
            },
        },
    )
    _write_json(
        results_dir / "two_field_qmle" / "results.json",
        {
            "profile": {"theta_points": 401, "rho_points": 1000},
            "overlap": {
                "support": {
                    "jaccard_mean": 0.52,
                    "jaccard_median": 0.51,
                    "directed_share_mean": 0.58,
                    "rank_matched_share_mean": 0.64,
                    "focal_support_size_mean": 39.3,
                    "comparator_support_size_mean": 27.1,
                    "random_null_directed_share_mean": 0.53,
                    "random_null_p_value": 0.001,
                    "n_random": 1000,
                },
                "weights": {
                    "union_pearson_mean": 0.62,
                    "union_cosine_mean": 0.71,
                    "common_edge_spearman_mean": 0.47,
                    "common_edge_count_mean": 22.8,
                    "offdiagonal_pearson": 0.65,
                },
                "induced_field": {
                    "pooled_pearson": 0.89,
                    "pooled_r_squared": 0.80,
                    "per_asset_pearson_mean": 0.90,
                    "per_asset_pearson_min": 0.56,
                    "per_asset_pearson_max": 0.97,
                    "random_null_pooled_pearson_mean": 0.70,
                    "random_null_p_value": 0.001,
                },
            },
            "factor_conditioned": {
                "dates": {"trading_days": 815},
                "joint": {
                    "rho_b": 0.53,
                    "rho_n": 0.23,
                    "rho_total": 0.76,
                    "theta": 0.70,
                    "lambda_b": 2.21,
                    "lambda_n": 0.96,
                    "loglik": 111.0,
                },
                "loglik_gain_vs_w_flat": 1.0,
                "loglik_gain_vs_co_mentions": 2.0,
                "bootstrap": {
                    "rho_b_ci_lower": 0.50,
                    "rho_b_ci_upper": 0.57,
                    "rho_n_ci_lower": 0.18,
                    "rho_n_ci_upper": 0.27,
                    "rho_total_ci_lower": 0.73,
                    "rho_total_ci_upper": 0.79,
                    "lambda_b_ci_lower": 1.92,
                    "lambda_b_ci_upper": 2.63,
                    "lambda_n_ci_lower": 0.70,
                    "lambda_n_ci_upper": 1.28,
                    "theta_ci_lower": 0.65,
                    "theta_ci_upper": 0.75,
                    "rho_channel_correlation": -0.70,
                    "field_b_boundary_share": 0.0,
                    "field_n_boundary_share": 0.0,
                },
                "quasi_lr_tests": {
                    "news_given_barycentric": {
                        "statistic": 5.0,
                        "bootstrap_p_value": 0.01,
                        "bootstrap_exceedances": 20,
                    },
                    "barycentric_given_news": {
                        "statistic": 6.0,
                        "bootstrap_p_value": 0.01,
                        "bootstrap_exceedances": 20,
                    },
                },
                # Paired-refit held-out block, shaped as two_field_stage.py
                # writes it (``factor_conditioned["held_out"]``).
                "held_out": {
                    "models": {
                        "factor_only": {"held_out_score": 2.4689},
                        "w_flat": {"held_out_score": 2.5260},
                        "w_co_mentions": {"held_out_score": 2.5193},
                        "joint": {"held_out_score": 2.5302},
                    },
                    "contrasts": {
                        "joint_minus_factor_only": {
                            "estimate": 0.0613,
                            "ci_lower": 0.0530,
                            "ci_upper": 0.0707,
                        },
                        "joint_minus_barycentric": {
                            "estimate": 0.0042,
                            "ci_lower": 0.0024,
                            "ci_upper": 0.0059,
                        },
                        "joint_minus_news": {
                            "estimate": 0.0109,
                            "ci_lower": 0.0093,
                            "ci_upper": 0.0127,
                        },
                    },
                },
            },
            "results": {"pooled_2023_2026": _two_field_period()},
        },
    )
    _write_json(
        results_dir / "spatial_value_comparison" / "results.json",
        {
            "schema_version": 1,
            "primary_specification": {
                "factor_model": "ff5_mom",
                "variance_model": "spherical",
            },
            "model_rows": [
                {
                    "scope": "factor_ladder",
                    "variance_model": "spherical",
                    "model": "ff5_mom",
                    "n_factors": 6,
                    "parameter_count": 10,
                    "loglik": -100.0,
                    "aic": 220.0,
                    "hqc_dates": 232.0,
                    "hqc_firm_days": 238.0,
                    "bic_dates": 240.0,
                    "bic_firm_days": 250.0,
                }
            ],
            "contrast_rows": [],
            "reduced_form_identity": {
                "diagnostics": {
                    variance: {
                        order: {
                            "max_abs_coefficient_error": 0.0,
                            "max_abs_fitted_mean_error": 0.0,
                        }
                        for order in ("w1", "w2")
                    }
                    for variance in ("spherical", "diagonal")
                }
            },
        },
    )
    _write_json(
        results_dir / "network_certificate" / "results.json",
        {
            "n_assets": 100,
            "structural": {
                "row_sum_residual_max": 1e-15,
                "minimum_w_squared_weight": 0.002,
            },
            "perron": {
                "dobrushin_coefficient": 0.82,
                "stationary_distribution_minimum": 0.01,
                "stationary_l1_residual": 1e-13,
            },
            "finite_rho": {
                "w_flat": {
                    "rho": 0.78,
                    "normalized_resolvent_error_linf": 0.66,
                    "dobrushin_upper_bound_linf": 1.24,
                    "lean_neumann_upper_bound_linf": 2.64,
                    "complement_norm": 0.5,
                    "lean_neumann_region": True,
                    "resolvent_common_mode_mean": 4.5,
                    "resolvent_common_mode_identity_error": 1e-14,
                    "normalized_common_mode_mean": 1.0,
                    "normalized_common_mode_identity_error": 1e-15,
                    "own_exposure_weight_mean": 0.2,
                    "own_exposure_weight_minimum": 0.1,
                    "own_exposure_weight_maximum": 0.3,
                    "other_exposure_weight_mean": 0.8,
                    "other_exposure_weight_minimum": 0.7,
                    "other_exposure_weight_maximum": 0.9,
                },
                "factor_conditioned_w2": {
                    "rho": 0.68,
                    "resolvent_common_mode_mean": 3.125,
                    "resolvent_common_mode_identity_error": 1e-14,
                    "normalized_common_mode_mean": 1.0,
                    "normalized_common_mode_identity_error": 1e-15,
                    "own_exposure_weight_mean": 0.32,
                    "own_exposure_weight_minimum": 0.3,
                    "own_exposure_weight_maximum": 0.4,
                    "other_exposure_weight_mean": 0.68,
                    "other_exposure_weight_minimum": 0.6,
                    "other_exposure_weight_maximum": 0.7,
                },
            },
        },
    )
    _write_json(
        results_dir / "economic_value" / "results.json",
        {
            "rbf_tuning": {
                "selected_multiplier": 1.0,
                "median_bandwidth": 2.0,
            },
            "model_rows": [
                {
                    "model": "factor_only",
                    "rho_hat": 0.0,
                    "loglik": -100.0,
                    "held_out_score": 1.0,
                    "held_out_score_gain_vs_factor_only": 0.0,
                    "held_out_gain_ci_lower": 0.0,
                    "held_out_gain_ci_upper": 0.0,
                },
                {
                    "model": "w2",
                    "rho_hat": 0.68,
                    "loglik": -90.0,
                    "held_out_score": 1.1,
                    "held_out_score_gain_vs_factor_only": 0.1,
                    "held_out_gain_ci_lower": 0.08,
                    "held_out_gain_ci_upper": 0.12,
                },
                {
                    "model": "centroid",
                    "rho_hat": 0.47,
                    "loglik": -91.0,
                    "held_out_score": 1.08,
                    "held_out_score_gain_vs_factor_only": 0.08,
                    "held_out_gain_ci_lower": 0.06,
                    "held_out_gain_ci_upper": 0.10,
                },
            ],
            "paired_contrasts": {
                "w2_minus_centroid": {
                    "estimate": 0.02,
                    "ci_lower": 0.01,
                    "ci_upper": 0.03,
                    "positive_share": 1.0,
                },
                "ledoit_wolf_minus_w2": {
                    "estimate": -0.01,
                    "ci_lower": -0.02,
                    "ci_upper": 0.0,
                    "positive_share": 0.05,
                },
            },
            "peer_panel_rows": [
                {
                    "target": "AMD",
                    "candidate": "NVDA",
                    "weight": 0.25,
                    "distance_rank": 1,
                    "same_sector": True,
                    "same_industry": True,
                    "co_mention": True,
                    "effective_source_count": 4.0,
                    "incoming_mass": 0.9,
                }
            ],
        },
    )
    _write_json(
        results_dir / "transport_dimension_ladder" / "results.json",
        {
            "cells": [
                {
                    "effective_dimension": dimension,
                    "switched_atom_total": 62612,
                    "matched_chord_max_w2_minus_w1": -0.0035,
                    "matched_chord_p95_w2_minus_w1": -0.0022,
                    "matched_chord_sd_w2_minus_w1": -0.0011,
                    "matched_tail_count_w2_minus_w1": -0.076,
                    "switched_matched_chord_w2_minus_w1": 0.0007,
                    "max_reduction_pair_share": 0.215,
                    "max_increase_pair_share": 0.041,
                    "max_reduction_given_lower": -0.018,
                    "max_increase_given_higher": 0.009,
                    "tail_reduction_pair_share": 0.084,
                    "tail_trade_pattern": True,
                }
                for dimension in (4096, 1024, 256, 64)
            ]
        },
    )
    geometry_summary = results_dir / "geometry-summary.json"
    _write_json(geometry_summary, {"metadata": {"sample_size": 128}})
    params_path = results_dir / "params.yaml"
    params_path.write_text("annual_raw_article_floor: 47\n", encoding="utf-8")
    bridge_summary = results_dir / "universe-horizon-summary.json"
    reference_short = _bridge_cell(
        "reference_52_short",
        0.77,
        universe_size=52,
        horizon="short",
        end="2026-04-06",
        trading_days=816,
    )
    reference_long = _bridge_cell(
        "reference_52_long",
        0.78,
        universe_size=52,
        horizon="long",
        end="2026-07-15",
        trading_days=885,
    )
    current_short = _bridge_cell(
        "current_100_short",
        0.84,
        universe_size=100,
        horizon="short",
        end="2026-04-06",
        trading_days=815,
    )
    _write_json(
        bridge_summary,
        {
            "cells": {
                "reference_52_short": reference_short,
                "reference_52_long": reference_long,
                "current_100_short": current_short,
            },
            "common_return_check": {
                "status": "identical_within_tolerance",
                "n_dates": 815,
                "max_abs_difference": 0.0,
            },
            "contrasts": {
                "horizon_long_minus_short_at_52": {
                    "status": "descriptive_only",
                    "rho_difference": 0.01,
                    "lambda_difference": 0.2,
                },
                "universe_100_minus_52_at_short": {
                    "status": "descriptive_only",
                    "rho_difference": 0.07,
                    "lambda_difference": 1.3,
                },
            },
        },
    )
    _write_json(
        results_dir / "point_in_time_encoder" / "results.json",
        {
            "identity": {
                "provider_id": "chronobert-2017",
                "representation_id": "chronobert-2017-unit",
                "factor_model": "ff5_mom",
            },
            "design": {
                "encoder_training_cutoff": "2017-12-31",
                "article_window_start": "2018-01-01",
                "calibration_start": "2018-01-01",
                "representation_is_pre_period": True,
            },
            "model_rows": [
                {
                    "model": "w2",
                    "held_out_score_gain_vs_factor_only": 0.028367,
                }
            ],
            "paired_contrasts": {
                "w2_minus_rbf": {
                    "estimate": 0.018875,
                    "ci_lower": 0.016209,
                    "ci_upper": 0.021831,
                }
            },
        },
    )
    return geometry_summary, params_path, bridge_summary


def test_generate_numbers_binds_only_post_corpus_catalog(tmp_path: Path) -> None:
    """The recut catalog binds QMLE and theory-witness values without gate data."""
    geometry_summary, params_path, bridge_summary = _write_inputs(tmp_path)
    out = tmp_path / "numbers.tex"
    generate_numbers(
        _numbers_paths(tmp_path, geometry_summary, params_path, bridge_summary, out)
    )
    text = out.read_text(encoding="utf-8")
    assert "\\ppDeclareValue{post-pooled-2023-2026-w-flat-rho}{0.780000}" in text
    assert "\\ppDeclareValue{transport-w2-rho}{0.680000}" in text
    held_out = "post-pooled-2023-2026-factor-two-field-held-out"
    assert f"\\ppDeclareValue{{{held_out}-score-joint}}{{2.530200}}" in text
    assert (
        f"\\ppDeclareValue{{{held_out}-joint-minus-news-ci-lower}}{{0.009300}}" in text
    )
    assert "\\ppDeclareValue{economic-ledoit-wolf-minus-w2-estimate}{-0.010000}" in text
    assert "\\ppDeclareValue{transport-delta-rho}{0.005000}" in text
    assert "\\ppDeclareValue{transport-equivalence-status}{equivalent}" in text
    assert "\\ppDeclareValue{transport-signal-correlation}{0.999800}" in text
    assert (
        "\\ppDeclareValue{transport-factor-residualized-signal-correlation}"
        "{0.998700}" in text
    )
    assert (
        "\\ppDeclareValue{transport-factor-residualized-relative-signal-difference-sd}"
        "{0.018000}" in text
    )
    assert "\\ppDeclareValue{transport-factor-ff3-w2-rho}{0.700000}" in text
    assert "\\ppDeclareValue{transport-factor-ff3-mom-n-factors}{4}" in text
    assert (
        "\\ppDeclareValue{spatial-value-factor-ff5-mom-hqc-dates}{232.000000}" in text
    )
    assert (
        "\\ppDeclareValue{spatial-value-factor-ff5-mom-hqc-firm-days}{238.000000}"
        in text
    )
    assert "\\ppDeclareValue{post-pooled-2023-2026-w-flat-trading-days}{815}" in text
    assert "\\ppDeclareValue{post-evaluation-date-start}{2023-01-03}" in text
    assert "\\ppDeclareValue{post-evaluation-date-end}{2026-04-06}" in text
    assert "\\ppDeclareValue{network-dobrushin}{0.820000}" in text
    assert "\\ppDeclareValue{paper5-balanced-cloud-size}{128}" in text
    assert "\\ppDeclareValue{annual-raw-article-floor}{47}" in text
    assert "\\ppDeclareValue{bridge-reference-52-short-rho}{0.770000}" in text
    assert "\\ppDeclareValue{bridge-horizon-rho-difference}{0.010000}" in text
    assert "\\ppDeclareValue{bridge-universe-rho-difference}{0.070000}" in text
    assert "\\ppDeclareValue{post-pooled-2023-2026-two-field-rho-b}{0.530000}" in text
    assert "\\ppDeclareValue{overlap-induced-pearson}{0.890000}" in text
    assert "{gate-" not in text
    assert "h2-alpha-hat" not in text


def test_generate_numbers_rejects_inconsistent_primary_windows(tmp_path: Path) -> None:
    """Primary variants cannot publish one date label for different windows."""
    geometry_summary, params_path, bridge_summary = _write_inputs(tmp_path)
    qmle_path = tmp_path / "post_corpus_qmle" / "results.json"
    qmle = json.loads(qmle_path.read_text(encoding="utf-8"))
    primary = qmle["results"][qmle["primary_period"]]
    primary["w_co_mentions"] = {
        **primary["w_flat"],
        "start": "2023-02-01",
    }
    _write_json(qmle_path, qmle)

    with pytest.raises(
        ValueError,
        match="primary QMLE variants do not share one evaluation window",
    ):
        generate_numbers(
            _numbers_paths(
                tmp_path,
                geometry_summary,
                params_path,
                bridge_summary,
                tmp_path / "numbers.tex",
            )
        )


def test_generate_numbers_requires_every_governed_artifact(tmp_path: Path) -> None:
    """Missing post-corpus evidence fails instead of emitting placeholders."""
    with pytest.raises(FileNotFoundError, match="required Paper 5 numbers input"):
        generate_numbers(
            _numbers_paths(
                tmp_path,
                tmp_path / "geometry-summary.json",
                tmp_path / "params.yaml",
                tmp_path / "universe-horizon-summary.json",
                tmp_path / "numbers.tex",
            )
        )
