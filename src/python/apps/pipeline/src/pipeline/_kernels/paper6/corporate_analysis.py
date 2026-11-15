"""Frozen-corporate weighted transport and article-selection measurement adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import ot

from pipeline._kernels.paper6.reference_alignment import strict_tail_readout, summarize

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

YEARS = (2018, 2019, 2020, 2021, 2022)
DIMENSIONS = (2560, 64)
SUPPORTS = ("matched", "expanded")
MARGINAL_TOLERANCE = 1e-10
OBJECTIVE_TOLERANCE = 1e-8
MATRIX_AXES = 2
FloatArray = np.ndarray[Any, np.dtype[np.float64]]
CloudKey = tuple[int, int]


class Cloud(TypedDict):
    """Canonical article coordinates, dates and original-sample membership."""

    points: FloatArray
    dates: NDArray[np.datetime64]
    core: NDArray[np.bool_]


def normalized(points: FloatArray, dimension: int) -> FloatArray:
    """Prefix then normalize, preserving the original representation convention."""
    selected = np.asarray(points[:, :dimension], dtype=np.float64)
    norms = np.linalg.norm(selected, axis=1)
    if not len(selected) or not np.isfinite(selected).all() or np.any(norms <= 0):
        msg = "invalid article representation"
        raise ValueError(msg)
    return selected / norms[:, None]


def exact_transport(
    cost: ArrayLike,
    source: ArrayLike | None = None,
    destination: ArrayLike | None = None,
    *,
    certify: bool = False,
) -> float:
    """Certify unregularized squared-cost OT on unequal weighted finite supports."""
    matrix = np.asarray(cost, dtype=np.float64, order="C")
    if matrix.ndim != MATRIX_AXES or not all(matrix.shape):
        msg = "transport requires a nonempty cost matrix"
        raise ValueError(msg)
    n, m = matrix.shape
    left = np.full(n, 1 / n) if source is None else np.asarray(source, dtype=float)
    right = (
        np.full(m, 1 / m)
        if destination is None
        else np.asarray(destination, dtype=float)
    )
    if (
        not np.isfinite(matrix).all()
        or np.min(matrix) < -MARGINAL_TOLERANCE
        or left.shape != (n,)
        or right.shape != (m,)
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or np.any(left < 0)
        or np.any(right < 0)
        or abs(left.sum() - 1) > MARGINAL_TOLERANCE
        or abs(right.sum() - 1) > MARGINAL_TOLERANCE
    ):
        msg = "invalid finite transport problem"
        raise ValueError(msg)
    plan, log = ot.emd(
        left,
        right,
        matrix,
        log=True,
        numThreads=1,
        numItermax=1_000_000,
        check_marginals=True,
    )
    if log["warning"] is not None or log["result_code"] != 1:
        msg = f"transport solver failure: {log['warning']}"
        raise ValueError(msg)
    np.testing.assert_allclose(plan.sum(1), left, rtol=0, atol=MARGINAL_TOLERANCE)
    np.testing.assert_allclose(plan.sum(0), right, rtol=0, atol=MARGINAL_TOLERANCE)
    value = float(np.sum(plan * matrix))
    dual = float(left @ log["u"] + right @ log["v"])
    if abs(value - dual) > OBJECTIVE_TOLERANCE * max(1, abs(value)):
        msg = "transport primal/dual objectives disagree"
        raise ValueError(msg)
    if certify:
        reduced = matrix - log["u"][:, None] - log["v"][None, :]
        active = (left > 0)[:, None] & (right > 0)[None, :]
        if np.min(reduced[active]) < -OBJECTIVE_TOLERANCE:
            msg = "transport dual infeasibility on positive support"
            raise ValueError(msg)
    return value


def point_masses(
    clouds: dict[CloudKey, Cloud], support: str
) -> dict[CloudKey, FloatArray]:
    """Put matched and expanded laws on one canonical article index."""
    if support not in SUPPORTS:
        msg = "unknown frozen article support"
        raise ValueError(msg)
    result = {}
    for key, cloud in clouds.items():
        flags = cloud["core"]
        mass = flags.astype(float) if support == "matched" else np.ones(len(flags))
        if not mass.sum():
            msg = "empty article support"
            raise ValueError(msg)
        result[key] = mass / mass.sum()
    return result


def redraw_masses(
    clouds: dict[CloudKey, Cloud], seed: int, draw: int
) -> dict[str, dict[CloudKey, FloatArray]]:
    """Keep legacy streams and paired canonical/additional stratum redraws."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, draw]))
    result: dict[str, dict[CloudKey, FloatArray]] = {key: {} for key in SUPPORTS}
    for key, cloud in sorted(clouds.items()):
        core = np.flatnonzero(cloud["core"])
        added = np.flatnonzero(~cloud["core"])
        if not len(core):
            msg = "redraw requires a nonempty canonical stratum"
            raise ValueError(msg)
        counts = np.bincount(
            rng.choice(core, size=len(core)), minlength=len(cloud["core"])
        )
        result["matched"][key] = counts / len(core)
        if len(added):
            counts += np.bincount(
                rng.choice(added, size=len(added)), minlength=len(counts)
            )
        result["expanded"][key] = counts / len(counts)
    return result


def return_readout(
    clouds: dict[CloudKey, Cloud],
    panels: dict[int, tuple[NDArray[np.datetime64], FloatArray]],
    masses: dict[CloudKey, FloatArray],
    n: int,
) -> FloatArray:
    """Pair financial tails with exactly the articles used for target geometry."""
    result = np.full((len(YEARS) - 1, n, n), np.nan)
    for t, year in enumerate(YEARS[1:]):
        dates, returns = panels[year]
        for firm in range(n):
            result[t, firm] = strict_tail_readout(
                clouds[firm, year]["dates"].astype("datetime64[D]").astype(np.float64),
                masses[firm, year][:, None],
                dates.astype("datetime64[D]").astype(np.float64),
                returns,
            )[0]
            result[t, firm, firm] = np.nan
    return result


def corporate_summary(
    distances: FloatArray, financial: FloatArray, firms: list[str]
) -> dict[str, Any]:
    """Add pooled descriptive scale and year concentration to shared summaries."""
    summary: dict[str, Any] = summarize(distances, financial, firms)
    beta = summary["beta"]
    scales = summary["scale"]
    return_rms = scales["centered_return_rms"]
    approach_rms = scales["reference_centered_adjustment_rms"]
    alignment = (
        None if beta is None or approach_rms == 0 else beta * return_rms / approach_rms
    )
    annual_q = summary["annual_theta_contributions"]
    annual_d = summary["annual_second_moment_contributions"]
    denominator = summary["second_moment"]
    year_shares = (
        [
            {"label": row["label"], "share": row["contribution"] / denominator}
            for row in annual_d
        ]
        if denominator > 0
        else [{"label": row["label"], "share": 0.0} for row in annual_d]
    )
    return {
        **summary,
        "target": "beta_archive",
        "numerator": summary["theta"] * summary["observations"],
        "denominator": denominator * summary["observations"],
        "annual_components": [
            {
                "year": int(q["label"]),
                "theta_contribution": q["contribution"],
                "second_moment_contribution": d["contribution"],
                "beta": None
                if d["contribution"] == 0
                else q["contribution"] / d["contribution"],
            }
            for q, d in zip(annual_q, annual_d, strict=True)
        ],
        "annual_beta": [
            {
                "label": q["label"],
                "beta": None
                if d["contribution"] == 0
                else q["contribution"] / d["contribution"],
            }
            for q, d in zip(annual_q, annual_d, strict=True)
        ],
        "year_denominator": {
            "shares": year_shares,
            "herfindahl": sum(row["share"] ** 2 for row in year_shares),
        },
        "normalized_alignment": alignment,
        "within_projection_fit": None if alignment is None else alignment**2,
        "standardized_effect": {
            "approach_units_per_return_sd": None if beta is None else beta * return_rms,
            "approach_sd_per_return_sd": alignment,
            "interpretation": (
                "Descriptive within-target-year one-regressor projection; "
                "no predictive interpretation."
            ),
        },
        "inference": {
            "status": "unavailable",
            "reason": (
                "Article redraws condition on the retained archive, firms, "
                "returns and years; no calibrated corporate population "
                "interval is established."
            ),
        },
    }
