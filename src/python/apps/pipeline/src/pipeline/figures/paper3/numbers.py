"""Route-B-only generated number bindings for Paper 3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from pipeline.figures._common import _load_yaml
from pipeline.io.values import Blank, Comment, PublicationValue, write_generated_numbers

if TYPE_CHECKING:
    from pipeline.io.values import Record

_MIN_ANATOMY_VINTAGES = 2
_EARLIER_TRANSITION_END = 2019
_LATER_TRANSITION_END = 2020


@dataclass(frozen=True, slots=True)
class Paper3NumberInputs:
    """Inputs retained for compatibility with the legacy combined renderer."""

    summary: dict[str, Any]
    results: pd.DataFrame
    params: dict[str, Any] | None = None
    mc_validation: dict[str, Any] | None = None
    hedging_validation: dict[str, Any] | None = None
    cross_model_summary: dict[str, Any] | None = None
    certificate_summary: dict[str, Any] | None = None
    validation_summary: dict[str, Any] | None = None
    anatomy_summary: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class Paper3NumberPaths:
    """Artifacts consumed by the Paper 3 Route-B number leaf."""

    paper3_dir: Path
    numbers_path: Path
    params_path: Path = Path("params.yaml")
    mc_validation_dir: Path = Path("data/mc/monte_carlo_validation")
    hedging_validation_dir: Path = Path("data/mc/hedging_error_validation")
    pricefree_dir: Path = Path("data/mc/pricefree_comparison")
    cross_model_summary: Path = Path(
        "data/shared/ablations/paper3_model_robustness/summary.yaml"
    )
    size_power_path: Path = Path("data/mc/gmm_wolak_size_power/summary.yaml")
    certificate_dir: Path = Path("data/papers/paper3/certificate")
    validation_dir: Path = Path("data/papers/paper3/certificate_validation")
    anatomy_dir: Path = Path("data/papers/paper3/portfolio_anatomy")


def _append_certificate_summary(
    records: list[Record], summary: dict[str, Any] | None
) -> None:
    if not summary:
        return
    portfolios = summary["portfolio_credit"]
    records.extend(
        [
            PublicationValue(
                f"certificate-{alias}-{endpoint.replace('_', '-')}-pct",
                100.0 * float(portfolios[portfolio][endpoint]),
            )
            for portfolio, alias in (
                ("equal_weight", "equal-weight"),
                ("inverse_volatility", "inverse-volatility"),
            )
            for endpoint in ("least_conservative", "most_conservative")
        ]
    )


def _append_validation_summary(
    records: list[Record], summary: dict[str, Any] | None
) -> None:
    if not summary:
        return
    records.extend(
        [
            Blank(),
            Comment("--- Frozen chronological certificate validation ---"),
        ]
    )
    for mode in ("operational", "geometry_only"):
        alias = mode.replace("_", "-")
        block = summary["validity"][mode]
        records.extend(
            [
                PublicationValue(
                    f"validation-{alias}-coverage-pct",
                    100.0 * float(block["coverage"]),
                ),
                PublicationValue(
                    f"validation-{alias}-coverage-lower-pct",
                    100.0 * float(block["coverage_lower"]),
                ),
            ]
        )
    informative = summary["informativeness"]
    records.extend(
        [
            PublicationValue(
                "validation-mean-lower-credit-pct",
                100.0 * float(informative["mean_lower_credit_inverse_volatility"]),
            ),
            PublicationValue(
                "validation-floor-activation-pct",
                100.0 * float(informative["floor_activation_rate"]),
            ),
        ]
    )
    decision = summary["decision_value"]
    records.extend(
        [
            PublicationValue(
                "validation-variance-certificate",
                float(decision["mean_future_variance"]["certificate"]),
                kind="e",
            ),
            PublicationValue(
                "validation-variance-inverse-volatility",
                float(decision["mean_future_variance"]["inverse_volatility"]),
                kind="e",
            ),
            PublicationValue(
                "validation-variance-difference",
                float(decision["certificate_minus_inverse_volatility"]),
                kind="e",
            ),
            PublicationValue(
                "validation-required-delta",
                float(decision["mean_required_delta"]),
            ),
        ]
    )


def _append_calibration_boundary(
    records: list[Record], summary: dict[str, Any] | None
) -> None:
    """Append the diagnostic that locates why calibration failed.

    The headline result is that no cell passed. Without these values a reader
    cannot tell whether the declared grid spanned a region the data rejected or
    simply stopped before the rule was reachable, which are different claims.
    """
    if not summary:
        return
    cells = summary.get("calibration_cells")
    if not isinstance(cells, list) or not cells:
        message = "validation summary is missing calibration cells"
        raise TypeError(message)
    best = max(cells, key=lambda cell: float(cell["coverage_lower"]))
    records.extend(
        [
            Blank(),
            Comment("--- Calibration grid boundary diagnostic ---"),
            PublicationValue("calibration-cells-total", len(cells)),
            PublicationValue(
                "calibration-cells-margin-pass",
                sum(1 for cell in cells if float(cell["mean_margin_lower"]) >= 0.0),
            ),
            PublicationValue(
                "calibration-cells-passing",
                sum(1 for cell in cells if bool(cell["passes"])),
            ),
            PublicationValue(
                "calibration-grid-maximum-scale",
                max(float(cell["L"]) for cell in cells),
                precision=2,
            ),
            PublicationValue(
                "calibration-grid-maximum-slack",
                max(float(cell["tau"]) for cell in cells),
                precision=3,
            ),
            PublicationValue(
                "calibration-best-coverage-scale", float(best["L"]), precision=2
            ),
            PublicationValue(
                "calibration-best-coverage-slack", float(best["tau"]), precision=3
            ),
            PublicationValue(
                "calibration-best-coverage-pct", 100.0 * float(best["coverage"])
            ),
            PublicationValue(
                "calibration-best-coverage-lower-pct",
                100.0 * float(best["coverage_lower"]),
            ),
            PublicationValue(
                "calibration-best-coverage-credit-pct",
                100.0 * float(best["mean_lower_credit"]),
            ),
        ]
    )


def _append_reference_percentiles(
    records: list[Record], benchmark: dict[str, Any]
) -> None:
    """Append the in-sample rank of the news allocation under each reference law."""
    reference = benchmark.get("reference_population")
    if not isinstance(reference, dict):
        message = "news-only benchmark summary is missing its reference population"
        raise TypeError(message)
    designs = reference.get("designs")
    if not isinstance(designs, list) or not designs:
        message = "news-only reference population declares no designs"
        raise TypeError(message)
    records.append(Blank())
    records.append(Comment("--- News-only reference-population percentiles ---"))
    for design in designs:
        alias = str(design["label"]).replace("_", "-")
        percentile = design["percentile"]
        records.extend(
            [
                PublicationValue(
                    f"news-only-percentile-{alias}-pct",
                    100.0 * float(percentile["news_only"]),
                ),
                PublicationValue(
                    f"equal-percentile-{alias}-pct",
                    100.0 * float(percentile["equal_weight"]),
                ),
                PublicationValue(
                    f"reference-effective-n-{alias}",
                    float(design["mean_effective_number_assets"]),
                    precision=2,
                ),
            ]
        )
    records.append(
        PublicationValue("reference-draws", int(designs[0]["retained_draws"]))
    )
    records.append(
        PublicationValue(
            "reference-tight-cap-pct",
            100.0 * float(designs[0]["maximum_weight"]),
        )
    )


def _append_news_benchmark_summary(
    records: list[Record], summary: dict[str, Any] | None
) -> None:
    """Append the full-sample descriptive news-only benchmark values."""
    if not summary:
        return
    benchmark = summary.get("news_only_benchmark")
    if not isinstance(benchmark, dict):
        return
    portfolios = benchmark.get("portfolios")
    objective = benchmark.get("objective")
    if not isinstance(portfolios, dict) or not isinstance(objective, dict):
        message = "news-only benchmark summary is missing portfolio diagnostics"
        raise TypeError(message)
    news = portfolios["news_only"]
    equal = portfolios["equal_weight"]
    sample_gmv = portfolios["sample_gmv"]
    records.extend(
        [
            Blank(),
            Comment("--- Full-sample descriptive news-only variance benchmark ---"),
            PublicationValue("news-only-variance", float(news["variance"])),
            PublicationValue(
                "news-only-gmv-relative-variance-pct",
                100.0 * float(news["gmv_relative_variance"]),
            ),
            PublicationValue(
                "news-only-gmv-excess-variance-pct",
                100.0 * (float(news["gmv_relative_variance"]) - 1.0),
            ),
            PublicationValue("equal-standardized-variance", float(equal["variance"])),
            PublicationValue(
                "news-only-equal-variance-reduction-pct",
                100.0 * (1.0 - float(news["variance"]) / float(equal["variance"])),
            ),
            PublicationValue(
                "equal-gmv-relative-variance-pct",
                100.0 * float(equal["gmv_relative_variance"]),
            ),
            PublicationValue("sample-gmv-variance", float(sample_gmv["variance"])),
            PublicationValue(
                "news-only-certificate-credit-pct",
                100.0 * float(news["certificate_credit"]),
            ),
            PublicationValue(
                "news-only-maximum-weight-pct",
                100.0 * float(news["maximum_weight"]),
            ),
            PublicationValue(
                "news-only-effective-number-assets",
                float(news["effective_number_assets"]),
                precision=2,
            ),
            PublicationValue(
                "news-only-equal-credit-gap-pct",
                100.0
                * float(
                    objective["certificate_credit"]
                    - objective["equal_weight_certificate_credit"]
                ),
            ),
        ]
    )
    _append_reference_percentiles(records, benchmark)


def _append_portfolio_anatomy_summary(
    records: list[Record], summary: dict[str, Any] | None
) -> None:
    """Append allocation-anatomy and expanding-cutoff diagnostics."""
    if not summary:
        return
    portfolio = summary.get("portfolio")
    certificate = summary.get("certificate")
    vintages = summary.get("vintages")
    identities = summary.get("identity_checks")
    if (
        not isinstance(portfolio, dict)
        or not isinstance(certificate, dict)
        or not isinstance(vintages, list)
        or not isinstance(identities, dict)
    ):
        message = "portfolio anatomy summary is missing required diagnostics"
        raise TypeError(message)
    vintage_rows = [row for row in vintages if isinstance(row, dict)]
    if len(vintage_rows) != len(vintages) or len(vintage_rows) < _MIN_ANATOMY_VINTAGES:
        message = "portfolio anatomy requires at least two valid vintage rows"
        raise TypeError(message)
    by_year = {int(row["year"]): row for row in vintage_rows}
    if _EARLIER_TRANSITION_END not in by_year or _LATER_TRANSITION_END not in by_year:
        message = "portfolio anatomy requires the 2019 and 2020 cutoffs"
        raise ValueError(message)
    turnovers = [
        float(row["one_way_turnover"])
        for row in vintage_rows
        if row.get("one_way_turnover") is not None
    ]
    effective_numbers = [float(row["effective_number_assets"]) for row in vintage_rows]
    maximum_weights = [float(row["maximum_weight"]) for row in vintage_rows]
    records.extend(
        [
            Blank(),
            Comment("--- Portfolio anatomy and expanding information cutoffs ---"),
            PublicationValue("anatomy-active-firms", int(portfolio["active_firms"])),
            PublicationValue(
                "anatomy-within-sector-credit-pct",
                100.0 * float(certificate["within_sector_share"]),
            ),
            PublicationValue(
                "anatomy-cross-sector-credit-pct",
                100.0 * float(certificate["cross_sector_share"]),
            ),
            PublicationValue(
                "anatomy-vintage-effective-n-min",
                min(effective_numbers),
                precision=2,
            ),
            PublicationValue(
                "anatomy-vintage-effective-n-max",
                max(effective_numbers),
                precision=2,
            ),
            PublicationValue(
                "anatomy-vintage-maximum-weight-min-pct",
                100.0 * min(maximum_weights),
            ),
            PublicationValue(
                "anatomy-vintage-maximum-weight-max-pct",
                100.0 * max(maximum_weights),
            ),
            PublicationValue("anatomy-turnover-min-pct", 100.0 * min(turnovers)),
            PublicationValue("anatomy-turnover-max-pct", 100.0 * max(turnovers)),
            PublicationValue(
                "anatomy-turnover-2018-2019-pct",
                100.0 * float(by_year[_EARLIER_TRANSITION_END]["one_way_turnover"]),
            ),
            PublicationValue(
                "anatomy-turnover-2019-2020-pct",
                100.0 * float(by_year[_LATER_TRANSITION_END]["one_way_turnover"]),
            ),
            PublicationValue(
                "anatomy-kkt-firm-share-max-gap",
                float(identities["firm_share_max_abs_gap"]),
                precision=2,
                kind="e",
            ),
            PublicationValue(
                "anatomy-final-vintage-weight-max-gap",
                float(identities["final_vintage_weight_max_abs_gap"]),
                precision=2,
                kind="e",
            ),
        ]
    )


def _write_paper3_numbers(path: Path, inputs: Paper3NumberInputs) -> None:
    """Write only values cited by the Route B manuscript."""
    certificate = inputs.certificate_summary or {}
    universe = certificate.get("universe", {})
    records: list[Record] = [
        Comment("AUTO-GENERATED by pipeline paper3-numbers — do not edit by hand."),
        Comment("--- Paper 3 Route B publication values ---"),
        PublicationValue("n-tickers", int(universe.get("n_tickers", 0))),
        PublicationValue("n-obs", int(universe.get("n_return_observations", 0))),
    ]
    _append_certificate_summary(records, inputs.certificate_summary)
    _append_news_benchmark_summary(records, inputs.certificate_summary)
    _append_portfolio_anatomy_summary(records, inputs.anatomy_summary)
    _append_validation_summary(records, inputs.validation_summary)
    _append_calibration_boundary(records, inputs.validation_summary)
    write_generated_numbers(path, records)


def render_paper3_numbers(paths: Paper3NumberPaths) -> None:
    """Emit the isolated Route B number bindings."""
    certificate = (
        _load_yaml(paths.certificate_dir / "summary.yaml")
        if (paths.certificate_dir / "summary.yaml").exists()
        else {}
    )
    validation = (
        _load_yaml(paths.validation_dir / "summary.yaml")
        if (paths.validation_dir / "summary.yaml").exists()
        else {}
    )
    anatomy = (
        _load_yaml(paths.anatomy_dir / "summary.yaml")
        if (paths.anatomy_dir / "summary.yaml").exists()
        else {}
    )
    _write_paper3_numbers(
        paths.numbers_path,
        Paper3NumberInputs(
            summary={},
            results=pd.DataFrame(),
            certificate_summary=certificate,
            validation_summary=validation,
            anatomy_summary=anatomy,
        ),
    )


__all__ = [
    "Paper3NumberInputs",
    "Paper3NumberPaths",
    "render_paper3_numbers",
]
