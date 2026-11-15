"""Emit Paper 5's post-corpus QMLE and network-certificate values."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import yaml

from pipeline.figures.paper5.point_in_time import validate_point_in_time_metadata
from pipeline.io.values import (
    MISSING,
    Blank,
    Comment,
    PublicationValue,
    write_generated_numbers,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pipeline.io.values import Record


def _load_json(path: Path) -> dict[str, Any]:
    """Load one mandatory Paper 5 JSON artifact."""
    if not path.exists():
        message = f"required Paper 5 numbers input is missing: {path}"
        raise FileNotFoundError(message)
    return json.loads(path.read_text(encoding="utf-8"))


def _adjustment_intensity(rho: float) -> float:
    """Map a spatial feedback parameter to its quadratic adjustment intensity.

    The spatial closure sets ``rho = lambda / (1 + lambda)``, which inverts to
    ``lambda = rho / (1 - rho)`` and is strictly increasing on ``(-1, 1)``.
    """
    if not -1.0 < rho < 1.0:
        message = (
            "spatial feedback outside the stationarity region has no "
            f"adjustment-intensity representation: rho={rho!r}"
        )
        raise ValueError(message)
    return rho / (1.0 - rho)


def _qmle_records(qmle: dict[str, Any]) -> list[Record]:
    """Build pooled and annual post-corpus SAR QMLE publication values."""
    primary_period = str(qmle["primary_period"])
    primary_variants = qmle["results"][primary_period]
    starts = {str(result["start"]) for result in primary_variants.values()}
    ends = {str(result["end"]) for result in primary_variants.values()}
    if len(starts) != 1 or len(ends) != 1:
        message = (
            "Paper 5 primary QMLE variants do not share one evaluation window: "
            f"starts={sorted(starts)}, ends={sorted(ends)}"
        )
        raise ValueError(message)
    records: list[Record] = [
        Comment("--- Post-corpus pooled and annual SAR QMLE ---"),
        PublicationValue("post-evaluation-date-start", starts.pop()),
        PublicationValue("post-evaluation-date-end", ends.pop()),
    ]
    for period, variants in qmle["results"].items():
        period_key = period.replace("_", "-")
        for variant, result in variants.items():
            variant_key = variant.replace("_", "-")
            prefix = f"post-{period_key}-{variant_key}"
            rho = float(result["rho_hat"])
            rho_lower = float(result["rho_ci_lower"])
            rho_upper = float(result["rho_ci_upper"])
            records.extend(
                [
                    PublicationValue(f"{prefix}-rho", rho),
                    PublicationValue(f"{prefix}-rho-ci-lower", rho_lower),
                    PublicationValue(f"{prefix}-rho-ci-upper", rho_upper),
                    PublicationValue(f"{prefix}-loglik", float(result["loglik"])),
                    PublicationValue(
                        f"{prefix}-trading-days", int(result["trading_days"])
                    ),
                    PublicationValue(f"{prefix}-lambda", _adjustment_intensity(rho)),
                    PublicationValue(
                        f"{prefix}-lambda-ci-lower",
                        _adjustment_intensity(rho_lower),
                    ),
                    PublicationValue(
                        f"{prefix}-lambda-ci-upper",
                        _adjustment_intensity(rho_upper),
                    ),
                ]
            )
    records.append(Blank())
    return records


def _factor_joint_transport_records(joint: dict[str, Any]) -> list[Record]:
    """Build publication values for unchanged-field joint W1/W2 inference."""
    if joint.get("fields_entered_unchanged") is not True:
        message = "Paper 5 joint W1/W2 model must retain both supplied fields"
        raise ValueError(message)
    if joint.get("literal_matrix_orthogonalization") is not False:
        message = "Paper 5 joint W1/W2 model cannot be labelled orthogonalization"
        raise ValueError(message)
    fit = joint["joint"]
    bootstrap = joint["bootstrap"]
    tests = joint["quasi_lr_tests"]
    channel_correlation = bootstrap["rho_channel_correlation"]
    channel_correlation_value: float | str = (
        "undefined" if channel_correlation is None else float(channel_correlation)
    )
    records: list[Record] = [
        Blank(),
        Comment("--- Factor-conditioned joint W1/W2 incremental tests ---"),
        PublicationValue("transport-joint-w1-rho", float(fit["rho_w1"])),
        PublicationValue("transport-joint-w2-rho", float(fit["rho_w2"])),
        PublicationValue("transport-joint-rho-total", float(fit["rho_total"])),
        PublicationValue("transport-joint-theta-w1", float(fit["theta_w1"])),
        PublicationValue("transport-joint-loglik", float(fit["loglik"])),
        PublicationValue(
            "transport-joint-loglik-per-firm-day",
            float(fit["loglik_per_firm_day"]),
        ),
        PublicationValue(
            "transport-joint-loglik-gain-w1",
            float(joint["loglik_gain_vs_w1_only"]),
        ),
        PublicationValue(
            "transport-joint-loglik-gain-w2",
            float(joint["loglik_gain_vs_w2_only"]),
        ),
        PublicationValue(
            "transport-joint-channel-correlation",
            channel_correlation_value,
        ),
        PublicationValue(
            "transport-joint-channel-correlation-status",
            str(bootstrap["rho_channel_correlation_status"]),
        ),
        PublicationValue(
            "transport-joint-w1-only-boundary-share",
            float(bootstrap["w1_only_boundary_share"]),
        ),
        PublicationValue(
            "transport-joint-w2-only-boundary-share",
            float(bootstrap["w2_only_boundary_share"]),
        ),
    ]
    for coefficient in ("rho_w1", "rho_w2", "rho_total", "theta_w1"):
        key = coefficient.replace("_", "-")
        records.extend(
            [
                PublicationValue(
                    f"transport-joint-{key}-ci-lower",
                    float(bootstrap[f"{coefficient}_ci_lower"]),
                ),
                PublicationValue(
                    f"transport-joint-{key}-ci-upper",
                    float(bootstrap[f"{coefficient}_ci_upper"]),
                ),
            ]
        )
    for label in ("w2_given_w1", "w1_given_w2"):
        test = tests[label]
        key = label.replace("_", "-")
        records.extend(
            [
                PublicationValue(
                    f"transport-joint-qlr-{key}-statistic",
                    float(test["statistic"]),
                ),
                PublicationValue(
                    f"transport-joint-qlr-{key}-pvalue",
                    float(test["bootstrap_p_value"]),
                ),
                PublicationValue(
                    f"transport-joint-qlr-{key}-exceedances",
                    int(test["bootstrap_exceedances"]),
                ),
                PublicationValue(
                    f"transport-joint-qlr-{key}-decision",
                    str(test["decision"]),
                ),
            ]
        )
    records.extend(
        [
            PublicationValue(
                "transport-joint-incremental-conclusion",
                str(joint["incremental_geometry_conclusion"]),
            ),
            PublicationValue(
                "transport-joint-follow-up-status",
                str(joint["follow_up_status"]),
            ),
        ]
    )
    return records


def _transport_order_records(transport: dict[str, Any]) -> list[Record]:
    """Build factor-conditioned W1/W2 transport-order publication values."""
    if transport.get("claim_label") != "transport-order sensitivity":
        message = "Paper 5 W1/W2 artifact must retain its transport-order claim label"
        raise ValueError(message)
    if transport["claim_boundary"].get("paper6_structural_no_go") != "unchanged":
        message = "Paper 5 W1/W2 artifact cannot reopen Paper 6's structural no-go"
        raise ValueError(message)

    fits = transport["post_corpus_factor_qmle"]
    if fits.get("primary") is not True:
        message = "Paper 5 factor-conditioned W1/W2 fit must be primary"
        raise ValueError(message)
    results = fits["results"]
    contrast = fits["paired_contrast_w2_minus_w1"]
    predictive = transport["predictive_comparison"]
    equivalence = predictive["equivalence"]
    separation = transport["operator_separation"]
    calibration = transport["sample"]["calibration"]
    evaluation = transport["sample"]["evaluation"]
    factor_comparison = transport["factor_model_comparison"]
    expected_models = ["ff3", "ff3_mom", "ff5", "ff5_mom"]
    if factor_comparison.get("model_order") != expected_models:
        message = "Paper 5 nested factor comparison has a non-governed model order"
        raise ValueError(message)
    if factor_comparison.get("primary_model") != "ff5_mom":
        message = "Paper 5 nested factor comparison must retain FF5+MOM as primary"
        raise ValueError(message)
    if factor_comparison.get("identical_return_dates_across_models") is not True:
        message = "Paper 5 nested factor models must share one return panel"
        raise ValueError(message)

    records: list[Record] = [
        Blank(),
        Comment("--- Factor-conditioned W1/W2 transport-order sensitivity ---"),
        PublicationValue("transport-calibration-date-start", str(calibration["start"])),
        PublicationValue("transport-calibration-date-end", str(calibration["end"])),
        PublicationValue(
            "transport-calibration-trading-days",
            int(calibration["factor_complete_dates"]),
        ),
        PublicationValue("transport-evaluation-date-start", str(evaluation["start"])),
        PublicationValue("transport-evaluation-date-end", str(evaluation["end"])),
        PublicationValue(
            "transport-evaluation-trading-days",
            int(evaluation["factor_complete_dates"]),
        ),
        PublicationValue("transport-n-assets", int(transport["sample"]["n_assets"])),
    ]
    for order in ("w1", "w2"):
        result = results[order]
        rho = float(result["rho_hat"])
        rho_lower = float(result["rho_ci_lower"])
        rho_upper = float(result["rho_ci_upper"])
        prefix = f"transport-{order}"
        records.extend(
            [
                PublicationValue(f"{prefix}-rho", rho),
                PublicationValue(f"{prefix}-rho-ci-lower", rho_lower),
                PublicationValue(f"{prefix}-rho-ci-upper", rho_upper),
                PublicationValue(f"{prefix}-lambda", _adjustment_intensity(rho)),
                PublicationValue(
                    f"{prefix}-lambda-ci-lower",
                    _adjustment_intensity(rho_lower),
                ),
                PublicationValue(
                    f"{prefix}-lambda-ci-upper",
                    _adjustment_intensity(rho_upper),
                ),
                PublicationValue(f"{prefix}-loglik", float(result["loglik"])),
                PublicationValue(
                    f"{prefix}-loglik-per-firm-day",
                    float(result["loglik_per_firm_day"]),
                ),
                PublicationValue(
                    f"{prefix}-unadjusted-rho",
                    float(result["unadjusted_raw_return_rho"]),
                ),
                PublicationValue(
                    f"{prefix}-factor-conditioning-rho-change",
                    float(result["factor_conditioning_rho_change"]),
                ),
            ]
        )
    records.extend(
        [
            PublicationValue("transport-n-factors", int(results["w2"]["n_factors"])),
            PublicationValue("transport-delta-rho", float(contrast["rho"])),
            PublicationValue(
                "transport-delta-rho-ci-lower", float(contrast["rho_ci_lower"])
            ),
            PublicationValue(
                "transport-delta-rho-ci-upper", float(contrast["rho_ci_upper"])
            ),
            PublicationValue("transport-delta-loglik", float(contrast["loglik"])),
            PublicationValue(
                "transport-delta-loglik-ci-lower",
                float(contrast["loglik_ci_lower"]),
            ),
            PublicationValue(
                "transport-delta-loglik-ci-upper",
                float(contrast["loglik_ci_upper"]),
            ),
            PublicationValue(
                "transport-w1-predictive-score", float(predictive["w1_score"])
            ),
            PublicationValue(
                "transport-w2-predictive-score", float(predictive["w2_score"])
            ),
            PublicationValue(
                "transport-delta-predictive-score",
                float(predictive["w2_minus_w1"]),
            ),
            PublicationValue(
                "transport-delta-predictive-score-ci-lower",
                float(predictive["w2_minus_w1_ci_lower"]),
            ),
            PublicationValue(
                "transport-delta-predictive-score-ci-upper",
                float(predictive["w2_minus_w1_ci_upper"]),
            ),
            PublicationValue(
                "transport-equivalence-margin", float(equivalence["margin"])
            ),
            PublicationValue(
                "transport-equivalence-ci-lower",
                float(equivalence["interval_lower"]),
            ),
            PublicationValue(
                "transport-equivalence-ci-upper",
                float(equivalence["interval_upper"]),
            ),
            PublicationValue(
                "transport-equivalence-status", str(equivalence["status"])
            ),
            PublicationValue(
                "transport-weight-correlation",
                float(separation["off_diagonal_weight_correlation"]),
            ),
            PublicationValue(
                "transport-active-support-jaccard",
                float(separation["active_support_jaccard"]),
            ),
            PublicationValue(
                "transport-signal-correlation",
                float(separation["induced_signal_correlation"]),
            ),
            PublicationValue(
                "transport-relative-signal-difference-sd",
                float(separation["relative_signal_difference_sd"]),
            ),
            PublicationValue(
                "transport-factor-residualized-signal-correlation",
                float(separation["factor_residualized_signal_correlation"]),
            ),
            PublicationValue(
                "transport-factor-residualized-relative-signal-difference-sd",
                float(separation["factor_residualized_relative_signal_difference_sd"]),
            ),
            PublicationValue(
                "transport-minimum-row-correlation",
                float(separation["row_correlation_minimum"]),
            ),
            PublicationValue(
                "transport-weakly-separated", bool(separation["weakly_separated"])
            ),
        ]
    )
    records.extend(
        _factor_joint_transport_records(transport["post_corpus_factor_joint_qmle"])
    )
    records.extend([Blank(), Comment("--- Nested common-factor sensitivity ---")])
    comparison_results = factor_comparison["results"]
    for model_id in expected_models:
        model = comparison_results[model_id]
        model_key = model_id.replace("_", "-")
        model_results = model["post_corpus_results"]
        model_predictive = model["predictive_comparison"]
        records.append(
            PublicationValue(
                f"transport-factor-{model_key}-n-factors",
                int(model_results["w2"]["n_factors"]),
            )
        )
        for order in ("w1", "w2"):
            result = model_results[order]
            prefix = f"transport-factor-{model_key}-{order}"
            records.extend(
                [
                    PublicationValue(f"{prefix}-rho", float(result["rho_hat"])),
                    PublicationValue(
                        f"{prefix}-rho-ci-lower", float(result["rho_ci_lower"])
                    ),
                    PublicationValue(
                        f"{prefix}-rho-ci-upper", float(result["rho_ci_upper"])
                    ),
                    PublicationValue(f"{prefix}-loglik", float(result["loglik"])),
                    PublicationValue(
                        f"{prefix}-loglik-per-firm-day",
                        float(result["loglik_per_firm_day"]),
                    ),
                    PublicationValue(
                        f"{prefix}-predictive-score",
                        float(model_predictive[f"{order}_score"]),
                    ),
                ]
            )
        contrast = model["paired_contrast_w2_minus_w1"]
        records.extend(
            [
                PublicationValue(
                    f"transport-factor-{model_key}-delta-rho",
                    float(contrast["rho"]),
                ),
                PublicationValue(
                    f"transport-factor-{model_key}-delta-predictive-score",
                    float(model_predictive["w2_minus_w1"]),
                ),
            ]
        )
    return records


def _spatial_value_model_records(model_rows: list[dict[str, Any]]) -> list[Record]:
    records: list[Record] = []
    for row in model_rows:
        scope = str(row["scope"])
        model = str(row["model"]).replace("_", "-")
        if scope == "factor_ladder":
            prefix = f"spatial-value-factor-{model}"
        else:
            variance = str(row["variance_model"])
            prefix = f"spatial-value-{variance}-{model}"
        for suffix, source in (
            ("n-factors", "n_factors"),
            ("parameter-count", "parameter_count"),
            ("loglik", "loglik"),
            ("aic", "aic"),
            ("hqc-dates", "hqc_dates"),
            ("hqc-firm-days", "hqc_firm_days"),
            ("bic-dates", "bic_dates"),
            ("bic-firm-days", "bic_firm_days"),
        ):
            value = row[source]
            records.append(PublicationValue(f"{prefix}-{suffix}", value))
        if scope == "spatial_value":
            for suffix, source in (
                ("rho", "rho_hat"),
                ("rho-ci-lower", "rho_ci_lower"),
                ("rho-ci-upper", "rho_ci_upper"),
                ("held-out-score", "held_out_score"),
            ):
                records.append(
                    PublicationValue(f"{prefix}-{suffix}", float(row[source]))
                )
    return records


def _spatial_value_contrast_records(
    contrast_rows: list[dict[str, Any]],
) -> list[Record]:
    records: list[Record] = []
    for contrast in contrast_rows:
        prefix = (
            f"spatial-value-{contrast['variance_model']}-{contrast['operator']}"
            "-versus-none"
        )
        for suffix, source in (
            ("loglik-gain", "loglik_gain"),
            ("qlr-statistic", "statistic"),
            ("qlr-pvalue", "bootstrap_p_value"),
            ("held-out-score-gain", "held_out_score_gain"),
            ("held-out-score-gain-ci-lower", "held_out_score_gain_ci_lower"),
            ("held-out-score-gain-ci-upper", "held_out_score_gain_ci_upper"),
        ):
            records.append(
                PublicationValue(f"{prefix}-{suffix}", float(contrast[source]))
            )
        records.extend(
            [
                PublicationValue(f"{prefix}-qlr-decision", str(contrast["decision"])),
                PublicationValue(
                    f"{prefix}-held-out-decision",
                    str(contrast["held_out_decision"]),
                ),
            ]
        )
    return records


def _spatial_value_identity_records(
    diagnostics: dict[str, dict[str, dict[str, float]]],
) -> list[Record]:
    records: list[Record] = []
    for variance in ("spherical", "diagonal"):
        for order in ("w1", "w2"):
            prefix = f"spatial-value-identity-{variance}-{order}"
            result = diagnostics[variance][order]
            records.extend(
                [
                    PublicationValue(
                        f"{prefix}-coefficient-error",
                        float(result["max_abs_coefficient_error"]),
                        precision=12,
                    ),
                    PublicationValue(
                        f"{prefix}-fitted-mean-error",
                        float(result["max_abs_fitted_mean_error"]),
                        precision=12,
                    ),
                ]
            )
    return records


def _spatial_value_records(spatial_value: dict[str, Any]) -> list[Record]:
    """Build publication values for factor-only versus spatial comparisons."""
    if spatial_value.get("schema_version") != 1:
        message = "Paper 5 spatial-value artifact must use schema version 1"
        raise ValueError(message)
    primary = spatial_value["primary_specification"]
    if (
        primary.get("factor_model") != "ff5_mom"
        or primary.get("variance_model") != "spherical"
    ):
        message = (
            "Paper 5 spatial-value primary specification must be spherical FF5+MOM"
        )
        raise ValueError(message)
    records: list[Record] = [
        Blank(),
        Comment("--- Factor-only versus spatial model value ---"),
    ]
    records.extend(_spatial_value_model_records(spatial_value["model_rows"]))
    records.extend(_spatial_value_contrast_records(spatial_value["contrast_rows"]))
    records.extend(
        _spatial_value_identity_records(
            spatial_value["reduced_form_identity"]["diagnostics"]
        )
    )
    return records


def _two_field_period_records(prefix: str, period: dict[str, Any]) -> list[Record]:
    """Build one evaluation period's joint estimates and nested comparisons."""
    joint = period["joint"]
    bootstrap = period["bootstrap"]
    boundaries = period["boundaries"]
    records: list[Record] = [
        PublicationValue(f"{prefix}-rho-b", float(joint["rho_b"])),
        PublicationValue(f"{prefix}-rho-n", float(joint["rho_n"])),
        PublicationValue(f"{prefix}-rho-total", float(joint["rho_total_fitted"])),
        PublicationValue(f"{prefix}-theta", float(joint["theta"])),
        PublicationValue(f"{prefix}-lambda-b", float(joint["lambda_b"])),
        PublicationValue(f"{prefix}-lambda-n", float(joint["lambda_n"])),
        PublicationValue(f"{prefix}-loglik", float(joint["loglik"])),
        PublicationValue(f"{prefix}-trading-days", int(period["trading_days"])),
        PublicationValue(
            f"{prefix}-loglik-gain-w-flat", float(period["loglik_gain_vs_w_flat"])
        ),
        PublicationValue(
            f"{prefix}-loglik-gain-co-mentions",
            float(period["loglik_gain_vs_co_mentions"]),
        ),
    ]
    for label, boundary in (
        ("w-flat", boundaries["w_flat"]),
        ("co-mentions", boundaries["w_co_mentions"]),
    ):
        records.extend(
            PublicationValue(
                f"{prefix}-boundary-{label}-{name}", float(boundary[source])
            )
            for name, source in (
                ("rho", "rho"),
                ("rho-ci-lower", "rho_ci_lower"),
                ("rho-ci-upper", "rho_ci_upper"),
                ("lambda", "lambda"),
                ("lambda-ci-lower", "lambda_ci_lower"),
                ("lambda-ci-upper", "lambda_ci_upper"),
            )
        )
    records.extend(
        [
            PublicationValue(
                f"{prefix}-channel-correlation",
                float(bootstrap["rho_channel_correlation"]),
            ),
            PublicationValue(
                f"{prefix}-boundary-share-b",
                float(bootstrap["field_b_boundary_share"]),
            ),
            PublicationValue(
                f"{prefix}-boundary-share-n",
                float(bootstrap["field_n_boundary_share"]),
            ),
        ]
    )
    interval_keys = (
        "rho-b",
        "rho-n",
        "rho-total",
        "theta",
        "lambda-b",
        "lambda-n",
    )
    records.extend(
        PublicationValue(
            f"{prefix}-{name}-ci-{bound}",
            float(bootstrap[f"{name.replace('-', '_')}_ci_{bound}"]),
        )
        for name in interval_keys
        for bound in ("lower", "upper")
    )
    tests = period.get("quasi_lr_tests")
    if isinstance(tests, dict):
        for label, source in (
            ("news-given-distributional", "news_given_distributional"),
            ("distributional-given-news", "distributional_given_news"),
        ):
            test = tests[source]
            records.extend(
                [
                    PublicationValue(
                        f"{prefix}-qlr-{label}-statistic",
                        float(test["statistic"]),
                    ),
                    PublicationValue(
                        f"{prefix}-qlr-{label}-pvalue",
                        float(test["bootstrap_p_value"]),
                    ),
                ]
            )
    return records


def _two_field_records(two_field: dict[str, Any]) -> list[Record]:
    """Build the joint two-field publication values for every evaluation period."""
    records: list[Record] = [
        Blank(),
        Comment("--- Joint two-field spatial estimates and nested comparisons ---"),
    ]
    for period, result in two_field["results"].items():
        prefix = f"post-{period.replace('_', '-')}-two-field"
        records.extend(_two_field_period_records(prefix, result))
    records.append(
        PublicationValue(
            "two-field-theta-points", int(two_field["profile"]["theta_points"])
        )
    )
    records.append(
        PublicationValue(
            "two-field-rho-points", int(two_field["profile"]["rho_points"])
        )
    )
    return records


def _factor_conditioned_two_field_records(two_field: dict[str, Any]) -> list[Record]:
    """Build the primary factor-conditioned barycentric/news bindings."""
    result = two_field.get("factor_conditioned")
    if not isinstance(result, dict):
        message = "Paper 5 two-field artifact lacks factor-conditioned results"
        raise TypeError(message)
    joint = result["joint"]
    bootstrap = result["bootstrap"]
    prefix = "post-pooled-2023-2026-factor-two-field"
    records: list[Record] = [
        Blank(),
        Comment("--- Primary factor-conditioned barycentric/news joint model ---"),
        PublicationValue(f"{prefix}-rho-b", float(joint["rho_b"])),
        PublicationValue(f"{prefix}-rho-n", float(joint["rho_n"])),
        PublicationValue(f"{prefix}-rho-total", float(joint["rho_total"])),
        PublicationValue(f"{prefix}-theta", float(joint["theta"])),
        PublicationValue(f"{prefix}-lambda-b", float(joint["lambda_b"])),
        PublicationValue(f"{prefix}-lambda-n", float(joint["lambda_n"])),
        PublicationValue(f"{prefix}-loglik", float(joint["loglik"])),
        PublicationValue(
            f"{prefix}-trading-days", int(result["dates"]["trading_days"])
        ),
        PublicationValue(
            f"{prefix}-loglik-gain-w-flat",
            float(result["loglik_gain_vs_w_flat"]),
        ),
        PublicationValue(
            f"{prefix}-loglik-gain-co-mentions",
            float(result["loglik_gain_vs_co_mentions"]),
        ),
        PublicationValue(
            f"{prefix}-channel-correlation",
            float(bootstrap["rho_channel_correlation"]),
        ),
        PublicationValue(
            f"{prefix}-boundary-share-b",
            float(bootstrap["field_b_boundary_share"]),
        ),
        PublicationValue(
            f"{prefix}-boundary-share-n",
            float(bootstrap["field_n_boundary_share"]),
        ),
    ]
    for name in ("rho-b", "rho-n", "rho-total"):
        source = name.replace("-", "_")
        for bound in ("lower", "upper"):
            records.extend(
                [
                    PublicationValue(
                        f"{prefix}-{name}-ci-{bound}",
                        float(bootstrap[f"{source}_ci_{bound}"]),
                    )
                ]
            )
    for name in ("lambda-b", "lambda-n", "theta"):
        source = name.replace("-", "_")
        for bound in ("lower", "upper"):
            records.extend(
                [
                    PublicationValue(
                        f"{prefix}-{name}-ci-{bound}",
                        float(bootstrap[f"{source}_ci_{bound}"]),
                    )
                ]
            )
    tests = result["quasi_lr_tests"]
    for label, source in (
        ("news-given-barycentric", "news_given_barycentric"),
        ("barycentric-given-news", "barycentric_given_news"),
    ):
        test = tests[source]
        records.extend(
            [
                PublicationValue(
                    f"{prefix}-qlr-{label}-statistic", float(test["statistic"])
                ),
                PublicationValue(
                    f"{prefix}-qlr-{label}-pvalue",
                    float(test["bootstrap_p_value"]),
                ),
                PublicationValue(
                    f"{prefix}-qlr-{label}-exceedances",
                    int(test["bootstrap_exceedances"]),
                ),
            ]
        )
    records.extend(_factor_conditioned_held_out_records(prefix, result))
    return records


def _factor_conditioned_held_out_records(
    prefix: str, result: dict[str, Any]
) -> list[Record]:
    """Build the paired-refit held-out joint-score bindings."""
    held_out = result.get("held_out")
    if not isinstance(held_out, dict):
        message = "Paper 5 factor-conditioned results lack held-out joint scores"
        raise TypeError(message)
    models = held_out["models"]
    contrasts = held_out["contrasts"]
    records: list[Record] = []
    for label, source in (
        ("factor-only", "factor_only"),
        ("barycentric", "w_flat"),
        ("news", "w_co_mentions"),
        ("joint", "joint"),
    ):
        records.append(
            PublicationValue(
                f"{prefix}-held-out-score-{label}",
                float(models[source]["held_out_score"]),
            )
        )
    for label in (
        "joint-minus-factor-only",
        "joint-minus-barycentric",
        "joint-minus-news",
    ):
        contrast = contrasts[label.replace("-", "_")]
        records.extend(
            PublicationValue(
                f"{prefix}-held-out-{label}-{field.replace('_', '-')}",
                float(contrast[field]),
            )
            for field in ("estimate", "ci_lower", "ci_upper")
        )
    return records


def _overlap_records(two_field: dict[str, Any]) -> list[Record]:
    """Build the interaction-field overlap publication values."""
    overlap = two_field["overlap"]
    support = overlap["support"]
    weights = overlap["weights"]
    induced = overlap["induced_field"]
    directed_share = float(support["directed_share_mean"])
    null_share = float(support["random_null_directed_share_mean"])
    return [
        Blank(),
        Comment("--- Interaction-field overlap: peers, weights, induced field ---"),
        PublicationValue("overlap-jaccard-mean", float(support["jaccard_mean"])),
        PublicationValue("overlap-jaccard-median", float(support["jaccard_median"])),
        PublicationValue("overlap-directed-share", directed_share),
        PublicationValue(
            "overlap-rank-matched-share", float(support["rank_matched_share_mean"])
        ),
        PublicationValue("overlap-support-null-share", null_share),
        PublicationValue(
            "overlap-support-excess-pct", 100.0 * (directed_share - null_share)
        ),
        PublicationValue(
            "overlap-support-null-pvalue",
            float(support["random_null_p_value"]),
            precision=3,
        ),
        PublicationValue(
            "overlap-w-flat-support-size", float(support["focal_support_size_mean"])
        ),
        PublicationValue(
            "overlap-co-mentions-support-size",
            float(support["comparator_support_size_mean"]),
        ),
        PublicationValue("overlap-support-draws", int(support["n_random"])),
        PublicationValue(
            "overlap-weight-union-pearson", float(weights["union_pearson_mean"])
        ),
        PublicationValue(
            "overlap-weight-union-cosine", float(weights["union_cosine_mean"])
        ),
        PublicationValue(
            "overlap-weight-common-spearman",
            float(weights["common_edge_spearman_mean"]),
        ),
        PublicationValue(
            "overlap-weight-common-count", float(weights["common_edge_count_mean"])
        ),
        PublicationValue(
            "overlap-weight-offdiagonal-pearson",
            float(weights["offdiagonal_pearson"]),
        ),
        PublicationValue("overlap-induced-pearson", float(induced["pooled_pearson"])),
        PublicationValue(
            "overlap-induced-r-squared", float(induced["pooled_r_squared"])
        ),
        PublicationValue(
            "overlap-induced-null-pearson",
            float(induced["random_null_pooled_pearson_mean"]),
        ),
        PublicationValue(
            "overlap-induced-null-pvalue",
            float(induced["random_null_p_value"]),
            precision=3,
        ),
        PublicationValue(
            "overlap-induced-asset-min", float(induced["per_asset_pearson_min"])
        ),
        PublicationValue(
            "overlap-induced-asset-max", float(induced["per_asset_pearson_max"])
        ),
    ]


def _network_records(network: dict[str, Any]) -> list[Record]:
    """Build frozen-operator contraction and finite-rho publication values."""
    structural = network["structural"]
    perron = network["perron"]
    finite = network["finite_rho"]["w_flat"]
    factor_finite = network["finite_rho"].get("factor_conditioned_w2")
    if not isinstance(factor_finite, dict):
        message = (
            "Paper 5 network certificate must contain the primary "
            "factor-conditioned W2 witness"
        )
        raise TypeError(message)

    def exposure_records(prefix: str, record: dict[str, Any]) -> list[Record]:
        """Emit common-mode and exposure bindings for one fitted coefficient."""
        return [
            PublicationValue(f"{prefix}-rho", float(record["rho"])),
            PublicationValue(
                f"{prefix}-g-common-mode-mean",
                float(record["resolvent_common_mode_mean"]),
            ),
            PublicationValue(
                f"{prefix}-g-common-mode-error",
                float(record["resolvent_common_mode_identity_error"]),
                precision=6,
                kind="e",
            ),
            PublicationValue(
                f"{prefix}-h-common-mode-mean",
                float(record["normalized_common_mode_mean"]),
            ),
            PublicationValue(
                f"{prefix}-h-common-mode-error",
                float(record["normalized_common_mode_identity_error"]),
                precision=6,
                kind="e",
            ),
            PublicationValue(
                f"{prefix}-own-exposure-mean",
                float(record["own_exposure_weight_mean"]),
            ),
            PublicationValue(
                f"{prefix}-own-exposure-minimum",
                float(record["own_exposure_weight_minimum"]),
            ),
            PublicationValue(
                f"{prefix}-own-exposure-maximum",
                float(record["own_exposure_weight_maximum"]),
            ),
            PublicationValue(
                f"{prefix}-other-exposure-mean",
                float(record["other_exposure_weight_mean"]),
            ),
            PublicationValue(
                f"{prefix}-other-exposure-minimum",
                float(record["other_exposure_weight_minimum"]),
            ),
            PublicationValue(
                f"{prefix}-other-exposure-maximum",
                float(record["other_exposure_weight_maximum"]),
            ),
        ]

    records: list[Record] = [
        Comment("--- Frozen-operator Perron and finite-rho witnesses ---"),
        PublicationValue("n-tickers-priced", int(network["n_assets"])),
        PublicationValue(
            "network-row-sum-residual",
            float(structural["row_sum_residual_max"]),
            precision=8,
        ),
        PublicationValue(
            "network-w2-minimum-weight",
            float(structural["minimum_w_squared_weight"]),
        ),
        PublicationValue("network-dobrushin", float(perron["dobrushin_coefficient"])),
        PublicationValue(
            "network-stationary-minimum",
            float(perron["stationary_distribution_minimum"]),
        ),
        PublicationValue(
            "network-stationary-residual",
            float(perron["stationary_l1_residual"]),
            precision=6,
            kind="e",
        ),
        PublicationValue(
            "network-w-flat-resolvent-error",
            float(finite["normalized_resolvent_error_linf"]),
        ),
        PublicationValue(
            "network-w-flat-resolvent-bound",
            float(finite["dobrushin_upper_bound_linf"]),
        ),
        # The Neumann envelope exists only inside its region of convergence:
        # {1 - ||rho N||}^-1 is a bound only when ||rho N|| < 1. On the 100-firm
        # panel every fitted variant sits OUTSIDE that region (w_flat at 1.277),
        # so there is no bound to report and the certificate emits null. Binding
        # MISSING renders an em dash rather than asserting a number that does
        # not exist; the Dobrushin envelope above is unaffected and still holds.
        PublicationValue(
            "network-w-flat-lean-neumann-bound",
            (
                float(finite["lean_neumann_upper_bound_linf"])
                if finite.get("lean_neumann_upper_bound_linf") is not None
                else MISSING
            ),
        ),
        PublicationValue(
            "network-w-flat-complement-norm",
            float(finite["complement_norm"]),
        ),
        PublicationValue(
            "network-w-flat-in-neumann-region",
            bool(finite["lean_neumann_region"]),
        ),
        PublicationValue(
            "network-w-flat-g-common-mode-mean",
            float(finite["resolvent_common_mode_mean"]),
        ),
        PublicationValue(
            "network-w-flat-g-common-mode-error",
            float(finite["resolvent_common_mode_identity_error"]),
            precision=6,
            kind="e",
        ),
        PublicationValue(
            "network-w-flat-h-common-mode-mean",
            float(finite["normalized_common_mode_mean"]),
        ),
        PublicationValue(
            "network-w-flat-h-common-mode-error",
            float(finite["normalized_common_mode_identity_error"]),
            precision=6,
            kind="e",
        ),
        PublicationValue(
            "network-w-flat-own-exposure-mean",
            float(finite["own_exposure_weight_mean"]),
        ),
        PublicationValue(
            "network-w-flat-own-exposure-minimum",
            float(finite["own_exposure_weight_minimum"]),
        ),
        PublicationValue(
            "network-w-flat-own-exposure-maximum",
            float(finite["own_exposure_weight_maximum"]),
        ),
        PublicationValue(
            "network-w-flat-other-exposure-mean",
            float(finite["other_exposure_weight_mean"]),
        ),
        PublicationValue(
            "network-w-flat-other-exposure-minimum",
            float(finite["other_exposure_weight_minimum"]),
        ),
        PublicationValue(
            "network-w-flat-other-exposure-maximum",
            float(finite["other_exposure_weight_maximum"]),
        ),
    ]
    records.extend(
        [
            Blank(),
            Comment("--- Primary factor-conditioned W2 multiplier ---"),
            *exposure_records("network-factor-w2", factor_finite),
        ]
    )
    return records


def _economic_value_records(economic: dict[str, Any]) -> list[Record]:
    """Build factor-conditioned comparator and peer-panel bindings."""
    rows = economic.get("model_rows")
    if not isinstance(rows, list) or not rows:
        message = "Paper 5 economic-value artifact must contain model rows"
        raise ValueError(message)
    records: list[Record] = [
        Blank(),
        Comment("--- Factor-conditioned interaction-field comparisons ---"),
        PublicationValue(
            "economic-rbf-selected-multiplier",
            float(economic["rbf_tuning"]["selected_multiplier"]),
        ),
        PublicationValue(
            "economic-rbf-median-bandwidth",
            float(economic["rbf_tuning"]["median_bandwidth"]),
        ),
    ]
    for raw in rows:
        if not isinstance(raw, dict):
            message = "Paper 5 economic-value model row must be a mapping"
            raise TypeError(message)
        model = str(raw["model"]).replace("_", "-")
        prefix = f"economic-{model}"
        rho_hat = raw["rho_hat"]
        loglik = raw["loglik"]
        records.extend(
            [
                PublicationValue(
                    f"{prefix}-rho",
                    MISSING if rho_hat is None else float(rho_hat),
                ),
                PublicationValue(
                    f"{prefix}-loglik",
                    MISSING if loglik is None else float(loglik),
                ),
                PublicationValue(
                    f"{prefix}-held-out-score", float(raw["held_out_score"])
                ),
                PublicationValue(
                    f"{prefix}-held-out-gain",
                    float(raw["held_out_score_gain_vs_factor_only"]),
                ),
                PublicationValue(
                    f"{prefix}-held-out-gain-ci-lower",
                    float(raw["held_out_gain_ci_lower"]),
                ),
                PublicationValue(
                    f"{prefix}-held-out-gain-ci-upper",
                    float(raw["held_out_gain_ci_upper"]),
                ),
            ]
        )
    row_by_model = {str(raw["model"]): raw for raw in rows if isinstance(raw, dict)}
    w2_gain = float(row_by_model["w2"]["held_out_score_gain_vs_factor_only"])
    centroid_gain = float(
        row_by_model["centroid"]["held_out_score_gain_vs_factor_only"]
    )
    if w2_gain <= 0.0:
        message = "Paper 5 W2 held-out gain must be positive"
        raise ValueError(message)
    records.append(
        PublicationValue("economic-centroid-w2-gain-share", centroid_gain / w2_gain)
    )
    paired_contrasts = economic.get("paired_contrasts")
    if not isinstance(paired_contrasts, dict):
        message = "Paper 5 economic-value artifact must contain paired contrasts"
        raise TypeError(message)
    centroid_contrast = paired_contrasts.get("w2_minus_centroid")
    if not isinstance(centroid_contrast, dict):
        message = "Paper 5 economic-value artifact lacks the W2-centroid contrast"
        raise TypeError(message)
    records.extend(
        [
            PublicationValue(
                "economic-w2-minus-centroid-estimate",
                float(centroid_contrast["estimate"]),
            ),
            PublicationValue(
                "economic-w2-minus-centroid-ci-lower",
                float(centroid_contrast["ci_lower"]),
            ),
            PublicationValue(
                "economic-w2-minus-centroid-ci-upper",
                float(centroid_contrast["ci_upper"]),
            ),
            PublicationValue(
                "economic-w2-minus-centroid-positive-share",
                float(centroid_contrast["positive_share"]),
            ),
        ]
    )
    ledoit_wolf_contrast = paired_contrasts.get("ledoit_wolf_minus_w2")
    if not isinstance(ledoit_wolf_contrast, dict):
        message = "Paper 5 economic-value artifact lacks the Ledoit-Wolf-W2 contrast"
        raise TypeError(message)
    records.extend(
        [
            PublicationValue(
                "economic-ledoit-wolf-minus-w2-estimate",
                float(ledoit_wolf_contrast["estimate"]),
            ),
            PublicationValue(
                "economic-ledoit-wolf-minus-w2-ci-lower",
                float(ledoit_wolf_contrast["ci_lower"]),
            ),
            PublicationValue(
                "economic-ledoit-wolf-minus-w2-ci-upper",
                float(ledoit_wolf_contrast["ci_upper"]),
            ),
            PublicationValue(
                "economic-ledoit-wolf-minus-w2-positive-share",
                float(ledoit_wolf_contrast["positive_share"]),
            ),
        ]
    )
    peer_rows = economic.get("peer_panel_rows")
    if not isinstance(peer_rows, list) or not peer_rows:
        message = "Paper 5 economic-value artifact must contain peer rows"
        raise ValueError(message)
    amd_rows = [
        row for row in peer_rows if isinstance(row, dict) and row.get("target") == "AMD"
    ]
    if not amd_rows:
        message = "Paper 5 peer panel must include the declared AMD target"
        raise ValueError(message)
    top_amd = min(
        amd_rows,
        key=lambda row: (-float(row["weight"]), str(row["candidate"])),
    )
    records.extend(
        [
            Blank(),
            Comment("--- Descriptive W2 peer-panel bindings ---"),
            PublicationValue(
                "economic-peer-amd-top-candidate", str(top_amd["candidate"])
            ),
            PublicationValue("economic-peer-amd-top-weight", float(top_amd["weight"])),
            PublicationValue(
                "economic-peer-amd-top-distance-rank", int(top_amd["distance_rank"])
            ),
            PublicationValue(
                "economic-peer-amd-top-same-sector", bool(top_amd["same_sector"])
            ),
            PublicationValue(
                "economic-peer-amd-top-same-industry", bool(top_amd["same_industry"])
            ),
            PublicationValue(
                "economic-peer-amd-top-co-mention", bool(top_amd["co_mention"])
            ),
            PublicationValue(
                "economic-peer-amd-effective-source-count",
                float(top_amd["effective_source_count"]),
            ),
            PublicationValue(
                "economic-peer-amd-incoming-mass", float(top_amd["incoming_mass"])
            ),
        ]
    )
    return records


def _point_in_time_records(point_in_time: dict[str, Any]) -> list[Record]:
    """Build bindings for the pre-period ChronoBERT primary-design check."""
    validate_point_in_time_metadata(point_in_time)
    rows = point_in_time.get("model_rows")
    if not isinstance(rows, list):
        message = "point-in-time artifact must contain model rows"
        raise TypeError(message)
    row_by_model = {str(row["model"]): row for row in rows if isinstance(row, dict)}
    w2 = row_by_model.get("w2")
    if not isinstance(w2, dict):
        message = "point-in-time artifact lacks the W2 row"
        raise TypeError(message)
    contrasts = point_in_time.get("paired_contrasts")
    if not isinstance(contrasts, dict):
        message = "point-in-time artifact lacks paired contrasts"
        raise TypeError(message)
    w2_rbf = contrasts.get("w2_minus_rbf")
    if not isinstance(w2_rbf, dict):
        message = "point-in-time artifact lacks the W2-RBF contrast"
        raise TypeError(message)
    return [
        Comment("--- Point-in-time representation sensitivity ---"),
        PublicationValue(
            "point-in-time-w2-held-out-gain",
            float(w2["held_out_score_gain_vs_factor_only"]),
        ),
        PublicationValue(
            "point-in-time-w2-minus-rbf-estimate", float(w2_rbf["estimate"])
        ),
        PublicationValue(
            "point-in-time-w2-minus-rbf-ci-lower", float(w2_rbf["ci_lower"])
        ),
        PublicationValue(
            "point-in-time-w2-minus-rbf-ci-upper", float(w2_rbf["ci_upper"])
        ),
    ]


def _geometry_records(geometry: dict[str, Any]) -> list[Record]:
    """Build publication values for the balanced article-cloud construction."""
    metadata = geometry.get("metadata")
    sample_size = metadata.get("sample_size") if isinstance(metadata, dict) else None
    if not isinstance(sample_size, int) or sample_size < 1:
        message = "Paper 5 geometry summary must declare a positive sample_size"
        raise ValueError(message)
    return [
        Blank(),
        Comment("--- Frozen text-geometry construction ---"),
        PublicationValue("paper5-balanced-cloud-size", sample_size),
    ]


def _switch_tail_records(ladder: dict[str, Any]) -> list[Record]:
    """Bind what the W1-versus-W2 reassignment does to matched chord distances.

    Only the two ladder endpoints are published: the full-dimension cell carries
    the operator Paper 5 estimates, and the 64-dimension cell is the endpoint at
    which assignments separate most.
    """
    cells = {int(cell["effective_dimension"]): cell for cell in ladder["cells"]}
    records: list[Record] = [
        Blank(),
        Comment("--- Switched-assignment matched-distance tails (W2 minus W1) ---"),
    ]
    for dimension in (4096, 64):
        cell = cells.get(dimension)
        if cell is None:
            message = (
                "Paper 5 switched-assignment values require the "
                f"{dimension}-dimension ladder cell"
            )
            raise ValueError(message)
        prefix = f"transport-switch-d{dimension}"
        records.extend(
            [
                PublicationValue(
                    f"{prefix}-switched-atoms", int(cell["switched_atom_total"])
                ),
                PublicationValue(
                    f"{prefix}-max-difference",
                    float(cell["matched_chord_max_w2_minus_w1"]),
                    precision=6,
                ),
                PublicationValue(
                    f"{prefix}-p95-difference",
                    float(cell["matched_chord_p95_w2_minus_w1"]),
                    precision=6,
                ),
                PublicationValue(
                    f"{prefix}-sd-difference",
                    float(cell["matched_chord_sd_w2_minus_w1"]),
                    precision=6,
                ),
                PublicationValue(
                    f"{prefix}-tail-count-difference",
                    float(cell["matched_tail_count_w2_minus_w1"]),
                    precision=4,
                ),
                PublicationValue(
                    f"{prefix}-switched-difference",
                    float(cell["switched_matched_chord_w2_minus_w1"]),
                    precision=6,
                ),
                PublicationValue(
                    f"{prefix}-max-reduction-share",
                    float(cell["max_reduction_pair_share"]),
                ),
                PublicationValue(
                    f"{prefix}-max-increase-share",
                    float(cell["max_increase_pair_share"]),
                ),
                PublicationValue(
                    f"{prefix}-max-reduction-given-lower",
                    float(cell["max_reduction_given_lower"]),
                    precision=6,
                ),
                PublicationValue(
                    f"{prefix}-max-increase-given-higher",
                    float(cell["max_increase_given_higher"]),
                    precision=6,
                ),
                PublicationValue(
                    f"{prefix}-tail-reduction-share",
                    float(cell["tail_reduction_pair_share"]),
                ),
                PublicationValue(
                    f"{prefix}-tail-trade-pattern", bool(cell["tail_trade_pattern"])
                ),
            ]
        )
    return records


def _configuration_records(params: dict[str, Any]) -> list[Record]:
    """Build manuscript bindings for governed shared sample thresholds."""
    annual_floor = params.get("annual_raw_article_floor")
    if not isinstance(annual_floor, int) or isinstance(annual_floor, bool):
        message = "params.yaml must declare an integer annual_raw_article_floor"
        raise TypeError(message)
    if annual_floor < 1:
        message = "annual_raw_article_floor must be positive"
        raise ValueError(message)
    return [
        Blank(),
        Comment("--- Governed sample configuration ---"),
        PublicationValue("annual-raw-article-floor", annual_floor),
    ]


def _universe_horizon_bridge_records(bridge: dict[str, Any]) -> list[Record]:
    """Build bindings for the observed universe-by-horizon bridge cells."""
    cells = bridge["cells"]
    cell_specs = (
        ("reference_52_short", "bridge-reference-52-short"),
        ("reference_52_long", "bridge-reference-52-long"),
        ("current_100_short", "bridge-current-100-short"),
    )
    records: list[Record] = [
        Blank(),
        Comment("--- Descriptive universe-by-horizon bridge ---"),
    ]
    for cell_id, prefix in cell_specs:
        cell = cells[cell_id]
        if cell.get("status") != "observed":
            message = f"Paper 5 bridge cell {cell_id!r} must be observed"
            raise ValueError(message)
        records.extend(
            [
                PublicationValue(f"{prefix}-rho", float(cell["rho_hat"])),
                PublicationValue(f"{prefix}-rho-ci-lower", float(cell["rho_ci_lower"])),
                PublicationValue(f"{prefix}-rho-ci-upper", float(cell["rho_ci_upper"])),
                PublicationValue(f"{prefix}-lambda", float(cell["lambda_hat"])),
                PublicationValue(
                    f"{prefix}-lambda-ci-lower", float(cell["lambda_ci_lower"])
                ),
                PublicationValue(
                    f"{prefix}-lambda-ci-upper", float(cell["lambda_ci_upper"])
                ),
                PublicationValue(f"{prefix}-trading-days", int(cell["trading_days"])),
                PublicationValue(f"{prefix}-date-start", str(cell["window_start"])),
                PublicationValue(f"{prefix}-date-end", str(cell["window_end"])),
            ]
        )

    common_check = bridge["common_return_check"]
    if common_check.get("status") != "identical_within_tolerance":
        message = "Paper 5 bridge shared-return identity check did not pass"
        raise ValueError(message)
    contrasts = bridge["contrasts"]
    for contrast_id, prefix in (
        ("horizon_long_minus_short_at_52", "bridge-horizon"),
        ("universe_100_minus_52_at_short", "bridge-universe"),
    ):
        contrast = contrasts[contrast_id]
        if contrast.get("status") != "descriptive_only":
            message = f"Paper 5 bridge contrast {contrast_id!r} must be descriptive"
            raise ValueError(message)
        records.extend(
            [
                PublicationValue(
                    f"{prefix}-rho-difference", float(contrast["rho_difference"])
                ),
                PublicationValue(
                    f"{prefix}-lambda-difference",
                    float(contrast["lambda_difference"]),
                ),
            ]
        )
    records.extend(
        [
            PublicationValue(
                "bridge-common-return-dates", int(common_check["n_dates"])
            ),
            PublicationValue(
                "bridge-common-return-max-difference",
                float(common_check["max_abs_difference"]),
                precision=8,
            ),
        ]
    )
    return records


@dataclass(frozen=True, slots=True)
class Paper5NumbersPaths:
    """Every saved artifact the Paper 5 value catalog reads, and its output."""

    post_corpus_qmle: Path
    transport_order_sensitivity: Path
    spatial_value_comparison: Path
    economic_value: Path
    point_in_time_encoder: Path
    transport_dimension_ladder: Path
    two_field_qmle: Path
    network_certificate: Path
    geometry_summary: Path
    universe_horizon_bridge: Path
    params_file: Path
    output_file: Path


def generate_numbers(paths: Paper5NumbersPaths) -> None:
    """Write the focused Paper 5 value catalog from governed model artifacts."""
    qmle = _load_json(paths.post_corpus_qmle)
    transport = _load_json(paths.transport_order_sensitivity)
    spatial_value = _load_json(paths.spatial_value_comparison)
    economic_value = _load_json(paths.economic_value)
    point_in_time = _load_json(paths.point_in_time_encoder)
    two_field = _load_json(paths.two_field_qmle)
    network = _load_json(paths.network_certificate)
    ladder = _load_json(paths.transport_dimension_ladder)
    geometry = _load_json(paths.geometry_summary)
    if not paths.params_file.exists():
        message = f"required Paper 5 numbers input is missing: {paths.params_file}"
        raise FileNotFoundError(message)
    params = yaml.safe_load(paths.params_file.read_text(encoding="utf-8"))
    if not isinstance(params, dict):
        message = "params.yaml must contain a mapping"
        raise TypeError(message)
    bridge = _load_json(paths.universe_horizon_bridge)
    records: list[Record] = [
        Comment(
            "AUTO-GENERATED by pipeline render-paper5-numbers — do not edit by hand."
        ),
        Comment("Source: data/papers/paper5 post-corpus and factor QMLE,"),
        Comment("two-field QMLE, network certificate, and text geometry"),
        Blank(),
    ]
    records.extend(_qmle_records(qmle))
    records.extend(_transport_order_records(transport))
    records.extend(_spatial_value_records(spatial_value))
    records.extend(_factor_conditioned_two_field_records(two_field))
    records.extend(_two_field_records(two_field))
    records.extend(_overlap_records(two_field))
    records.extend(_network_records(network))
    records.extend(_economic_value_records(economic_value))
    records.extend(_point_in_time_records(point_in_time))
    records.extend(_geometry_records(geometry))
    records.extend(_switch_tail_records(ladder))
    records.extend(_configuration_records(params))
    records.extend(_universe_horizon_bridge_records(bridge))
    write_generated_numbers(paths.output_file, records)


__all__ = ["Paper5NumbersPaths", "generate_numbers"]
