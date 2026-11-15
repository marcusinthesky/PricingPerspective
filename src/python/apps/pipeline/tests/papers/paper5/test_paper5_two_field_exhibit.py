"""Tests for Paper 5's two-field number bindings and joint-region exhibit."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from pipeline.figures.paper5.figures import (
    TwoFieldRegion,
    render_two_field_region,
)
from pipeline.figures.paper5.numbers import (
    _factor_conditioned_two_field_records,
    _overlap_records,
    _two_field_records,
)
from pipeline.io.values import PublicationValue
from pipeline.stages.papers.paper5.two_field_stage import _refit_score_contrast

if TYPE_CHECKING:
    from pathlib import Path

    from pipeline.io.values import Record

_SEED = 20260823
_TWO_BOOTSTRAP_DRAWS = 2


def _boundary(rho: float, lower: float, upper: float) -> dict[str, float]:
    """Build one pinned single-field boundary with its resampled interval.

    Only one channel is active at a boundary, so its coefficient is the total
    and the intensity is the same monotone transform used everywhere else.
    """
    return {
        "rho_b": rho,
        "rho": rho,
        "rho_ci_lower": lower,
        "rho_ci_upper": upper,
        "lambda": rho / (1.0 - rho),
        "lambda_ci_lower": lower / (1.0 - lower),
        "lambda_ci_upper": upper / (1.0 - upper),
    }


def _payload() -> dict[str, Any]:
    """Build a minimal two-field stage payload with one evaluation period.

    The channel intensities are derived from the coefficients rather than typed
    out, so the fixture cannot drift from the closure it stands in for.
    """
    rho_b, rho_n = 0.535206, 0.227318
    total = rho_b + rho_n
    return {
        "profile": {"theta_points": 401, "rho_points": 1000},
        "overlap": {
            "support": {
                "jaccard_mean": 0.51,
                "jaccard_median": 0.50,
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
        "results": {
            "pooled_2023_2026": {
                "trading_days": 885,
                "joint": {
                    "rho_b": rho_b,
                    "rho_n": rho_n,
                    "rho_total_fitted": total,
                    "theta": rho_b / total,
                    "lambda_b": rho_b / (1.0 - total),
                    "lambda_n": rho_n / (1.0 - total),
                    "loglik": 111128.85,
                },
                "boundaries": {
                    "w_flat": _boundary(0.779, 0.746, 0.809),
                    "w_co_mentions": _boundary(0.606, 0.557, 0.657),
                },
                "loglik_gain_vs_w_flat": 219.4,
                "loglik_gain_vs_co_mentions": 850.4,
                "quasi_lr_tests": {
                    "news_given_distributional": {
                        "null": "rho_n = 0",
                        "tested_channel": "news",
                        "statistic": 438.8,
                        "bootstrap_p_value": 0.0005,
                        "bootstrap_exceedances": 0,
                    },
                    "distributional_given_news": {
                        "null": "rho_b = 0",
                        "tested_channel": "distributional",
                        "statistic": 1700.8,
                        "bootstrap_p_value": 0.0005,
                        "bootstrap_exceedances": 0,
                    },
                },
                "bootstrap": {
                    "rho_b_ci_lower": 0.498,
                    "rho_b_ci_upper": 0.568,
                    "rho_n_ci_lower": 0.185,
                    "rho_n_ci_upper": 0.273,
                    "rho_total_ci_lower": 0.730,
                    "rho_total_ci_upper": 0.793,
                    "theta_ci_lower": 0.648,
                    "theta_ci_upper": 0.753,
                    "lambda_b_ci_lower": 1.925,
                    "lambda_b_ci_upper": 2.633,
                    "lambda_n_ci_lower": 0.698,
                    "lambda_n_ci_upper": 1.284,
                    "rho_channel_correlation": -0.697,
                    "field_b_boundary_share": 0.0,
                    "field_n_boundary_share": 0.0,
                },
            }
        },
    }


def _keys(records: list[Record]) -> set[str]:
    """Collect the publication keys a record list declares."""
    return {record.key for record in records if isinstance(record, PublicationValue)}


def _numbers(records: list[Record]) -> dict[str, float]:
    """Collect the numeric publication values a record list declares."""
    return {
        record.key: float(record.value)
        for record in records
        if isinstance(record, PublicationValue)
        and isinstance(record.value, (int, float))
    }


def test_two_field_records_expose_both_channels_and_their_intervals() -> None:
    """Every channel coefficient, intensity, and interval reaches the manuscript."""
    keys = _keys(_two_field_records(_payload()))
    prefix = "post-pooled-2023-2026-two-field"

    for suffix in ("rho-b", "rho-n", "lambda-b", "lambda-n", "theta", "rho-total"):
        assert f"{prefix}-{suffix}" in keys
        assert f"{prefix}-{suffix}-ci-lower" in keys
        assert f"{prefix}-{suffix}-ci-upper" in keys
    assert f"{prefix}-loglik-gain-w-flat" in keys
    assert f"{prefix}-loglik-gain-co-mentions" in keys
    assert f"{prefix}-channel-correlation" in keys
    assert f"{prefix}-qlr-news-given-distributional-statistic" in keys
    assert f"{prefix}-qlr-news-given-distributional-pvalue" in keys
    assert f"{prefix}-qlr-distributional-given-news-statistic" in keys
    assert f"{prefix}-qlr-distributional-given-news-pvalue" in keys
    assert {"two-field-theta-points", "two-field-rho-points"} <= keys


def test_factor_conditioned_records_expose_the_primary_joint_model() -> None:
    """The factor-conditioned fit publishes the primary channels and tests."""
    payload = {
        "factor_conditioned": {
            "dates": {"trading_days": 800},
            "joint": {
                "rho_b": 0.40,
                "rho_n": 0.20,
                "rho_total": 0.60,
                "theta": 2.0 / 3.0,
                "lambda_b": 1.0,
                "lambda_n": 0.5,
                "loglik": 123.0,
            },
            "bootstrap": {
                "rho_b_ci_lower": 0.35,
                "rho_b_ci_upper": 0.45,
                "rho_n_ci_lower": 0.15,
                "rho_n_ci_upper": 0.25,
                "rho_total_ci_lower": 0.55,
                "rho_total_ci_upper": 0.65,
                "theta_ci_lower": 0.60,
                "theta_ci_upper": 0.72,
                "lambda_b_ci_lower": 0.8,
                "lambda_b_ci_upper": 1.2,
                "lambda_n_ci_lower": 0.3,
                "lambda_n_ci_upper": 0.7,
                "rho_channel_correlation": -0.5,
                "field_b_boundary_share": 0.0,
                "field_n_boundary_share": 0.0,
            },
            "loglik_gain_vs_w_flat": 10.0,
            "loglik_gain_vs_co_mentions": 20.0,
            "quasi_lr_tests": {
                "news_given_barycentric": {
                    "statistic": 12.0,
                    "bootstrap_p_value": 0.01,
                    "bootstrap_exceedances": 19,
                },
                "barycentric_given_news": {
                    "statistic": 22.0,
                    "bootstrap_p_value": 0.01,
                    "bootstrap_exceedances": 19,
                },
            },
            "held_out": {
                "models": {
                    "factor_only": {"held_out_score": 2.4},
                    "w_flat": {"held_out_score": 2.5},
                    "w_co_mentions": {"held_out_score": 2.49},
                    "joint": {"held_out_score": 2.52},
                },
                "contrasts": {
                    "joint_minus_factor_only": {
                        "estimate": 0.12,
                        "ci_lower": 0.10,
                        "ci_upper": 0.14,
                    },
                    "joint_minus_barycentric": {
                        "estimate": 0.02,
                        "ci_lower": 0.01,
                        "ci_upper": 0.03,
                    },
                    "joint_minus_news": {
                        "estimate": 0.03,
                        "ci_lower": 0.02,
                        "ci_upper": 0.04,
                    },
                },
            },
        }
    }
    values = _numbers(_factor_conditioned_two_field_records(payload))
    prefix = "post-pooled-2023-2026-factor-two-field"

    assert values[f"{prefix}-rho-b"] + values[f"{prefix}-rho-n"] == pytest.approx(
        values[f"{prefix}-rho-total"]
    )
    assert values[f"{prefix}-lambda-b"] == pytest.approx(
        values[f"{prefix}-rho-b"] / (1.0 - values[f"{prefix}-rho-total"])
    )
    assert f"{prefix}-qlr-news-given-barycentric-pvalue" in values
    assert f"{prefix}-qlr-barycentric-given-news-pvalue" in values
    assert f"{prefix}-held-out-score-joint" in values
    assert f"{prefix}-held-out-joint-minus-barycentric-estimate" in values
    assert f"{prefix}-held-out-joint-minus-news-ci-upper" in values


def test_refit_score_contrast_reports_paired_interval() -> None:
    """Held-out contrasts summarize per-draw refit gains and expose direction."""
    draws = np.array([0.05, 0.10], dtype=np.float64)

    contrast = _refit_score_contrast(draws, estimate=0.075)

    assert contrast["estimate"] == pytest.approx(0.075)
    assert contrast["bootstrap_valid_draws"] == _TWO_BOOTSTRAP_DRAWS
    assert contrast["positive_share"] == pytest.approx(1.0)
    assert contrast["ci_lower"] <= contrast["bootstrap_mean"] <= contrast["ci_upper"]


def test_overlap_records_report_each_stage_against_its_null() -> None:
    """Support and induced-field agreement are both published with their nulls."""
    records = _overlap_records(_payload())
    keys = _keys(records)
    values = _numbers(records)

    assert {
        "overlap-directed-share",
        "overlap-support-null-share",
        "overlap-support-excess-pct",
    } <= keys
    assert {"overlap-induced-pearson", "overlap-induced-null-pearson"} <= keys
    assert {"overlap-support-null-pvalue", "overlap-induced-null-pvalue"} <= keys
    assert values["overlap-support-excess-pct"] == pytest.approx(
        100.0
        * (values["overlap-directed-share"] - values["overlap-support-null-share"])
    )


def test_published_channel_values_reconstruct_the_closure() -> None:
    """The published coefficients and intensities agree with the closure's map."""
    values = _numbers(_two_field_records(_payload()))
    prefix = "post-pooled-2023-2026-two-field"
    rho_b = values[f"{prefix}-rho-b"]
    rho_n = values[f"{prefix}-rho-n"]
    total = values[f"{prefix}-rho-total"]

    assert rho_b + rho_n == pytest.approx(total, abs=1e-3)
    assert total < 1.0
    assert values[f"{prefix}-lambda-b"] == pytest.approx(
        rho_b / (1.0 - total), abs=1e-3
    )
    assert values[f"{prefix}-lambda-n"] == pytest.approx(
        rho_n / (1.0 - total), abs=1e-3
    )
    assert all(math.isfinite(value) for value in values.values())


def test_joint_likelihood_gains_over_both_boundaries_are_nonnegative() -> None:
    """The joint model nests both single-field fits, so neither gain is negative."""
    values = _numbers(_two_field_records(_payload()))
    prefix = "post-pooled-2023-2026-two-field"

    assert values[f"{prefix}-loglik-gain-w-flat"] >= 0.0
    assert values[f"{prefix}-loglik-gain-co-mentions"] >= 0.0


def test_joint_region_exhibit_renders_a_pgf(tmp_path: Path) -> None:
    """The joint-region figure renders from a scattered surface and a cloud."""
    rng = np.random.default_rng(_SEED)
    theta = np.linspace(0.0, 1.0, 41)
    rho = np.linspace(0.0, 0.99, 41)
    grid_theta, grid_rho = np.meshgrid(theta, rho, indexing="ij")
    rho_b = (grid_rho * grid_theta).reshape(-1)
    rho_n = (grid_rho * (1.0 - grid_theta)).reshape(-1)
    loglik = -((rho_b - 0.53) ** 2 + (rho_n - 0.23) ** 2) * 1e4
    out = tmp_path / "two_field_region.pgf"

    region = TwoFieldRegion(
        surface_rho_b=rho_b,
        surface_rho_n=rho_n,
        surface_loglik=loglik,
        bootstrap_rho_b=0.53 + 0.02 * rng.normal(size=200),
        bootstrap_rho_n=0.23 + 0.02 * rng.normal(size=200),
        joint_rho_b=0.53,
        joint_rho_n=0.23,
        field_b_boundary=0.78,
        field_n_boundary=0.61,
    )
    render_two_field_region(region, out)
    assert out.exists()
    assert out.stat().st_size > 0
    rendered = out.read_text(encoding="utf-8")
    assert "quasi-log-likelihood" in rendered
    assert "contour (drop 2.9957)" in rendered
    assert "95\\% joint region" not in rendered
