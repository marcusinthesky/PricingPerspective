"""Render the Paper 5 point-in-time encoder sensitivity table."""

from __future__ import annotations

import json
from datetime import date
from typing import TYPE_CHECKING, NoReturn, cast

if TYPE_CHECKING:
    from pathlib import Path


class Paper5PointInTimeTableError(ValueError):
    """Raised when the point-in-time table artifact is invalid."""


def _invalid(message: str) -> NoReturn:
    """Raise the table renderer's contract error."""
    raise Paper5PointInTimeTableError(message)


def _number(row: dict[str, object], key: str) -> float:
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        _invalid(f"point-in-time table field {key!r} must be numeric")
    return float(value)


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        _invalid("point-in-time results must be a JSON object")
    return cast("dict[str, object]", payload)


def validate_point_in_time_metadata(payload: dict[str, object]) -> None:
    """Require the governed 2017 identity and evidence of a pre-period cutoff."""
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        _invalid("point-in-time results lack identity metadata")
    if (
        identity.get("provider_id") != "chronobert-2017"
        or identity.get("representation_id") != "chronobert-2017-unit"
    ):
        _invalid("point-in-time table requires the pre-period ChronoBERT 2017 encoder")
    if identity.get("factor_model") != "ff5_mom":
        _invalid("point-in-time table requires the primary factor model")
    design = payload.get("design")
    if not isinstance(design, dict):
        _invalid("point-in-time results lack pre-period design metadata")
    try:
        cutoff = date.fromisoformat(design["encoder_training_cutoff"])
        article_start = date.fromisoformat(design["article_window_start"])
        calibration_start = date.fromisoformat(design["calibration_start"])
    except (KeyError, TypeError, ValueError) as exc:
        message = (
            "point-in-time results require parseable pre-period cutoff/window dates"
        )
        raise Paper5PointInTimeTableError(message) from exc
    if (
        cutoff != date(2017, 12, 31)
        or cutoff >= article_start
        or cutoff >= calibration_start
        or design.get("representation_is_pre_period") is not True
    ):
        _invalid("point-in-time results must contain truthful pre-period metadata")


def render_paper5_point_in_time_table(results_path: Path, output_path: Path) -> None:
    """Render the factor-conditioned ChronoBERT 2017 score comparison."""
    payload = _load(results_path)
    validate_point_in_time_metadata(payload)
    raw_rows = payload.get("model_rows")
    if not isinstance(raw_rows, list):
        _invalid("point-in-time results lack model rows")
    rows = cast("list[dict[str, object]]", raw_rows)
    expected = {"factor_only", "w2", "rbf_median"}
    if {str(row.get("model")) for row in rows} != expected:
        _invalid("point-in-time table has an unexpected model set")
    rendered_rows: list[str] = []
    for row in rows:
        model = str(row["model"])
        interval = (
            "--"
            if model == "factor_only"
            else (
                f"[{_number(row, 'held_out_gain_ci_lower'):.4f}, "
                f"{_number(row, 'held_out_gain_ci_upper'):.4f}]"
            )
        )
        rendered_rows.append(
            " & ".join(
                [
                    str(row["label"]),
                    f"{_number(row, 'rho_hat'):.3f}",
                    f"{_number(row, 'held_out_score'):.6f}",
                    f"{_number(row, 'held_out_score_gain_vs_factor_only'):.6f}",
                    interval,
                ]
            )
            + r" \\"
        )
    contrasts = payload.get("paired_contrasts")
    if not isinstance(contrasts, dict):
        _invalid("point-in-time results lack paired contrasts")
    contrast = contrasts.get("w2_minus_rbf")
    if not isinstance(contrast, dict):
        _invalid("point-in-time results lack the W2-RBF contrast")
    contrast_row = (
        "W2 minus median RBF"
        " & -- & -- & "
        f"{_number(contrast, 'estimate'):.6f} & "
        f"[{_number(contrast, 'ci_lower'):.4f}, "
        f"{_number(contrast, 'ci_upper'):.4f}] \\\\"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                "% AUTO-GENERATED -- do not edit by hand.",
                r"\begin{table}[H]",
                r"\centering",
                r"\scriptsize",
                r"\setlength{\tabcolsep}{3.2pt}",
                r"\begin{tabular}{lrrrr}",
                r"\toprule",
                (
                    "Interaction field & $\\hat\\rho$ & Held-out score & Gain & "
                    "95\\% CI "
                    r"\\"
                ),
                r"\midrule",
                *rendered_rows,
                r"\midrule",
                contrast_row,
                r"\bottomrule",
                r"\end{tabular}",
                (
                    r"\caption{Point-in-time representation sensitivity under the "
                    r"primary FF5-plus-momentum specification. ChronoBERT 2017 has a "
                    r"reported training-information cutoff of 2017-12-31, pre-period "
                    r"relative to the 2018--2022 article and calibration windows. "
                    r"This is not a claim of historical checkpoint availability. "
                    r"Fields are held "
                    r"fixed throughout the 2023--2026 evaluation. The score is the "
                    r"mean Gaussian quasi-log score per firm-day; intervals use paired "
                    r"stationary-block bootstrap draws. The RBF field uses the median "
                    r"off-diagonal "
                    r"squared W2 distance as its bandwidth.}"
                ),
                r"\label{tab:p5-point-in-time-encoder}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


__all__ = [
    "Paper5PointInTimeTableError",
    "render_paper5_point_in_time_table",
]
