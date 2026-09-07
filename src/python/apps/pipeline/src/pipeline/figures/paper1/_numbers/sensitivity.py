"""FWL and correlation-scale bindings for the dyadic fit.

Emission order is fixed by ``_append_dyadic_records`` in
:mod:`pipeline.figures.paper1._numbers.dyadic`: FWL, then the correlation-scale
translation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pipeline.figures.paper1._numbers.common import _opt_float
from pipeline.io.values import Blank, Comment, PublicationValue

if TYPE_CHECKING:
    from pipeline.io.values import Record


def _append_fwl_records(records: list[Record], dyadic_confound: dict[str, Any]) -> None:
    """Bind the FWL slope check and the formula-defined partial R-squared."""
    fwl = dyadic_confound.get("fwl_diagnostics", {}) or {}
    partial = dyadic_confound.get("partial_r2_diagnostics", {}) or {}
    if fwl:
        records += [
            Blank(),
            Comment(
                "--- True FWL display: both sides residualized on the primary "
                "point design's intercept + symmetric firm effects ---"
            ),
            PublicationValue("confound-fwl-slope", _opt_float(fwl.get("slope"))),
            PublicationValue(
                "confound-fwl-slope-abs-error",
                _opt_float(fwl.get("abs_slope_reproduction_error")),
            ),
            PublicationValue(
                "confound-fwl-nuisance-columns",
                int(fwl.get("nuisance_columns", 0)),
            ),
            PublicationValue(
                "confound-fwl-omitted-firm",
                str(fwl.get("omitted_firm_indicator", "unavailable")),
            ),
        ]
    if partial:
        records += [
            PublicationValue(
                "confound-partial-r2", _opt_float(partial.get("partial_r2"))
            ),
            PublicationValue(
                "confound-sse-firm-effects",
                _opt_float(partial.get("sse_firm_effects_only")),
            ),
            PublicationValue(
                "confound-sse-firm-effects-w2",
                _opt_float(partial.get("sse_firm_effects_plus_focal")),
            ),
        ]


def _append_correlation_scale_records(
    records: list[Record], dyadic_confound: dict[str, Any]
) -> None:
    """Bind return-chord quantiles and their correlation-scale translation.

    Deterministic arithmetic on the already-governed coefficient: ``rho = 1 -
    D^2/2``. Signed correlations are emitted at each anchor, before and after a
    one-standard-deviation energy increase, so no consumer has to infer a sign.
    """
    translation = dyadic_confound.get("correlation_scale_translation", {}) or {}
    if not translation:
        return
    quantiles = translation.get("return_chord_distance_quantiles", {}) or {}
    anchors = translation.get("anchors", {}) or {}
    records += [
        Blank(),
        Comment(
            "--- Return-chord quantiles and the deterministic correlation-scale "
            "re-expression rho = 1 - D^2/2 (t34.6.3.1); not a new estimand ---"
        ),
        PublicationValue(
            "chord-delta-one-sd",
            _opt_float(translation.get("delta_chord_one_sd_w2")),
        ),
    ]
    for key, label in (("q25", "q25"), ("q50", "median"), ("q75", "q75")):
        anchor = anchors.get(key, {}) or {}
        records += [
            PublicationValue(f"chord-{label}", _opt_float(quantiles.get(key))),
            PublicationValue(
                f"chord-rho-{label}", _opt_float(anchor.get("correlation"))
            ),
            PublicationValue(
                f"chord-rho-after-{label}",
                _opt_float(anchor.get("correlation_after_one_sd")),
            ),
            PublicationValue(
                f"chord-delta-rho-{label}",
                _opt_float(anchor.get("delta_correlation")),
            ),
        ]
