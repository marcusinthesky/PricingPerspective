"""Dyadic confound-aware regression bindings.

``_append_dyadic_records`` is the single entry point and fixes the emission
order of the whole dyadic block; the per-family helpers below and in
:mod:`pipeline.figures.paper1._numbers.sensitivity` must not be called directly
or reordered, because record order is the order of the generated LaTeX file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pipeline.figures.paper1._numbers.common import _opt_float
from pipeline.figures.paper1._numbers.sensitivity import (
    _append_correlation_scale_records,
    _append_fwl_records,
)
from pipeline.io.values import Blank, Comment, PublicationValue
from pipeline.stages.papers.paper1._dyadic.random_exposure import (
    SCENARIO_LIPSCHITZ,
    SCENARIO_SLACK,
)

if TYPE_CHECKING:
    from pipeline.io.values import Record


def _append_dyadic_records(
    records: list[Record], dyadic_confound: dict[str, Any] | None
) -> None:
    if not dyadic_confound:
        return
    _append_dyadic_primary_records(records, dyadic_confound)
    _append_random_exposure_records(records, dyadic_confound)
    _append_dyadic_margin_records(records, dyadic_confound)
    _append_dyadic_postfit_records(records, dyadic_confound)
    _append_fwl_records(records, dyadic_confound)
    _append_correlation_scale_records(records, dyadic_confound)


def _append_dyadic_primary_records(
    records: list[Record], dyadic_confound: dict[str, Any]
) -> None:
    """Bind the primary specification's coefficient, interval, and fit."""
    primary = cast("dict[str, Any]", dyadic_confound["w2_primary_regression"])
    records += [
        Blank(),
        Comment(
            "--- Dyadic confound-aware robustness check "
            "(data/papers/paper1/dyadic_confound/summary.yaml) ---"
        ),
        PublicationValue("n-firms", int(dyadic_confound.get("n_firms", 0))),
        PublicationValue("n-pairs", int(dyadic_confound.get("n_dyads", 0))),
        PublicationValue(
            "confound-w2-coef",
            float(primary["coefficient"]),
        ),
        PublicationValue("confound-w2-se", float(primary["se"])),
        PublicationValue(
            "confound-w2-p",
            float(primary["pvalue"]),
        ),
        PublicationValue("confound-n-dyads", int(dyadic_confound.get("n_dyads", 0))),
        PublicationValue(
            "confound-w2-ci-low",
            _opt_float(primary.get("ci_low")),
        ),
        PublicationValue(
            "confound-w2-ci-high",
            _opt_float(primary.get("ci_high")),
        ),
        PublicationValue(
            "confound-overall-r2", _opt_float(dyadic_confound.get("overall_r2"))
        ),
        PublicationValue(
            "confound-within-r2", _opt_float(dyadic_confound.get("within_r2"))
        ),
        PublicationValue(
            "confound-w2-effect-one-sd",
            _opt_float(primary.get("effect_one_sd")),
        ),
        PublicationValue(
            "confound-bootstrap-valid",
            int(primary.get("bootstrap_valid_draws", 0)),
        ),
    ]


def _append_random_exposure_records(
    records: list[Record], dyadic_confound: dict[str, Any]
) -> None:
    """Bind the complete optional-W2 catalog, using MISSING when unavailable."""
    diagnostic = dyadic_confound.get("random_exposure_diagnostic", {}) or {}
    w2_fit = dyadic_confound.get("w2_primary_regression", {}) or {}
    scenarios: list[dict[str, Any]] = (
        cast("list[dict[str, Any]]", diagnostic.get("scenarios", []))
        if isinstance(diagnostic, dict)
        else []
    )
    scenario_by_key = {
        (float(row.get("L")), float(row.get("tau"))): row
        for row in scenarios
        if isinstance(row, dict)
        and row.get("L") is not None
        and row.get("tau") is not None
    }
    records += [
        Blank(),
        Comment(
            "--- Fixed-W2 random-exposure envelope diagnostic "
            "(data/papers/paper1/dyadic_confound/summary.yaml) ---"
        ),
        PublicationValue(
            "confound-w2-within-r2",
            _opt_float(w2_fit.get("within_r2") if isinstance(w2_fit, dict) else None),
        ),
    ]
    for lipschitz in SCENARIO_LIPSCHITZ:
        for slack in SCENARIO_SLACK:
            row = scenario_by_key.get((lipschitz, slack), {})
            key_prefix = f"confound-w2-l{lipschitz:g}-tau{slack:g}".replace(".", "p")
            records.extend(
                [
                    PublicationValue(
                        f"{key_prefix}-coverage",
                        _opt_float(row.get("coverage_share")),
                    ),
                    PublicationValue(
                        f"{key_prefix}-violation",
                        _opt_float(row.get("violation_share")),
                    ),
                    PublicationValue(
                        f"{key_prefix}-unresolved",
                        _opt_float(row.get("unresolved_share")),
                    ),
                ]
            )


def _append_dyadic_margin_records(
    records: list[Record], dyadic_confound: dict[str, Any]
) -> None:
    """Bind the firm-margin variance-share decomposition."""
    margin_diagnostics = dyadic_confound.get("firm_margin_diagnostics", {}) or {}
    if margin_diagnostics:
        records += [
            PublicationValue(
                "confound-firm-share-w2",
                _opt_float(margin_diagnostics.get("w2_metric_variance_share")),
            ),
            PublicationValue(
                "confound-firm-share-return",
                _opt_float(margin_diagnostics.get("return_distance_variance_share")),
            ),
            PublicationValue(
                "confound-within-correlation",
                _opt_float(margin_diagnostics.get("within_residual_correlation")),
            ),
            PublicationValue(
                "confound-between-correlation",
                _opt_float(margin_diagnostics.get("between_fitted_correlation")),
            ),
        ]


def _append_dyadic_postfit_records(
    records: list[Record], dyadic_confound: dict[str, Any]
) -> None:
    """Bind influence and residual-scale diagnostics for the canonical rung."""
    influence = dyadic_confound.get("influence_diagnostics", {}) or {}
    residual_scale = dyadic_confound.get("residual_scale_diagnostic", {}) or {}
    if influence or residual_scale:
        records += [
            PublicationValue(
                "confound-lofo-max-se",
                _opt_float(influence.get("max_abs_lofo_delta_in_bootstrap_se")),
            ),
            PublicationValue(
                "confound-lofo-sign-flips",
                int(influence.get("lofo_sign_flip_count", 0)),
            ),
            PublicationValue(
                "confound-max-leverage-dyad",
                str(influence.get("max_leverage_dyad", "unavailable")),
            ),
            PublicationValue(
                "confound-max-leverage",
                _opt_float(influence.get("max_leverage")),
            ),
            PublicationValue(
                "confound-leverage-threshold",
                _opt_float(influence.get("leverage_flag_threshold_2p_over_n")),
            ),
            PublicationValue(
                "confound-max-studentized-residual",
                _opt_float(influence.get("most_extreme_studentized_residual")),
            ),
            PublicationValue(
                "confound-residual-scale-stat",
                _opt_float(residual_scale.get("statistic")),
            ),
            PublicationValue(
                "confound-residual-scale-p",
                _opt_float(residual_scale.get("permutation_pvalue")),
            ),
        ]
