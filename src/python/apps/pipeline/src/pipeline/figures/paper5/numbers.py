"""Emit Paper 5's post-corpus QMLE and network-certificate values."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import yaml

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
    return [
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


def generate_numbers(
    results_dir: Path,
    geometry_summary_path: Path,
    params_path: Path,
    universe_horizon_summary_path: Path,
    out_path: Path,
) -> None:
    """Write the focused Paper 5 value catalog from governed model artifacts."""
    qmle = _load_json(results_dir / "post_corpus_qmle" / "results.json")
    two_field = _load_json(results_dir / "two_field_qmle" / "results.json")
    network = _load_json(results_dir / "network_certificate.json")
    geometry = _load_json(geometry_summary_path)
    if not params_path.exists():
        message = f"required Paper 5 numbers input is missing: {params_path}"
        raise FileNotFoundError(message)
    params = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    if not isinstance(params, dict):
        message = "params.yaml must contain a mapping"
        raise TypeError(message)
    bridge = _load_json(universe_horizon_summary_path)
    records: list[Record] = [
        Comment("AUTO-GENERATED by pipeline paper5-numbers — do not edit by hand."),
        Comment("Source: data/papers/paper5 post-corpus QMLE, two-field QMLE,"),
        Comment("network certificate, and typed text-geometry summary"),
        Blank(),
    ]
    records.extend(_qmle_records(qmle))
    records.extend(_two_field_records(two_field))
    records.extend(_overlap_records(two_field))
    records.extend(_network_records(network))
    records.extend(_geometry_records(geometry))
    records.extend(_configuration_records(params))
    records.extend(_universe_horizon_bridge_records(bridge))
    write_generated_numbers(out_path, records)


__all__ = ["generate_numbers"]
