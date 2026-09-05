"""Timing-separated primary dyadic bindings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pipeline.io.values import Blank, Comment, PublicationValue

if TYPE_CHECKING:
    from pipeline.io.values import Record


def _append_oos_dyadic_records(
    records: list[Record], oos_dyadic: dict[str, Any] | None
) -> None:
    """Emit the t37 out-of-sample primary W2 dyadic bindings.

    This is the look-ahead-free counterpart to the manuscript's headline
    specification, not to the demoted matrix statistic: the text-side matrix is
    frozen at its 2018-22 in-sample estimate and only the return chord distance
    is recomputed, with intercept and symmetric additive firm effects
    re-estimated on the surviving panel.

    Every value is emitted verbatim at the shared six-digit precision whatever
    its sign or size; no significance threshold changes the rendered result.
    """
    if not oos_dyadic:
        return
    panel = oos_dyadic["panel"]
    attrition = panel["attrition"]
    w2 = oos_dyadic["w2_arm"]
    records += [
        Blank(),
        Comment(
            "--- t37 out-of-sample PRIMARY dyadic association: FIXED 2018-22 "
            "Qwen3-Embedding 8B W2 matrix, 2023-2026 return chord distance ---"
        ),
        Comment(
            "Source: data/papers/paper1/dyadic_oos/summary.yaml. Symmetric "
            "additive firm effects re-estimated on the surviving panel;"
        ),
        Comment(
            "multinomial node bootstrap (dyad weight w_i*w_j), seed 42, "
            "percentile [2.5, 97.5] -- the in-sample convention, fixed in advance."
        ),
        PublicationValue("oos-dyadic-n-firms", int(panel["n_firms"])),
        PublicationValue("oos-dyadic-n-dyads", int(panel["n_dyads"])),
        PublicationValue("oos-dyadic-design-columns", int(panel["design_columns"])),
        PublicationValue(
            "oos-dyadic-residual-dof", int(panel["residual_degrees_of_freedom"])
        ),
        PublicationValue(
            "oos-dyadic-text-universe-firms",
            int(attrition["frozen_text_universe_firms"]),
        ),
        PublicationValue(
            "oos-dyadic-attrition-firms",
            len(attrition["dropped_from_text_universe"]),
        ),
        Comment("W2 primary specification carried out of sample"),
        PublicationValue("oos-dyadic-w2-coef", float(w2["coef"])),
        PublicationValue("oos-dyadic-w2-se", float(w2["se"])),
        PublicationValue("oos-dyadic-w2-p", float(w2["pvalue"])),
        PublicationValue("oos-dyadic-w2-ci-low", float(w2["ci_low"])),
        PublicationValue("oos-dyadic-w2-ci-high", float(w2["ci_high"])),
        PublicationValue("oos-dyadic-w2-effect-one-sd", float(w2["effect_one_sd"])),
        PublicationValue("oos-dyadic-w2-within-r2", float(w2["within_r2"])),
    ]
