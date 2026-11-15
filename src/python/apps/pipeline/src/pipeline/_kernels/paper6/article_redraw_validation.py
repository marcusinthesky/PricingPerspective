"""Nested validation of the corporate stratified article-redraw procedure.

Only joint stratum/month/atom counts survive within one outer archive. The
original recovery and corporate random streams are never consumed or changed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from hotelling.dynamics.recovery import (
    FIRMS,
    NEWS_TICKS,
    histories,
    reporting_laws,
    solve_history,
)
from hotelling.dynamics.transport import axis_point_w2
from hotelling.orthant import SOLVE_TARGET
from pydantic import BaseModel, ConfigDict

from pipeline._kernels.paper6.directional_validation import panel_arrays
from pipeline._kernels.paper6.recovery import MONTHS, YEARS
from pipeline._kernels.paper6.reference_alignment import (
    annual_moments,
    pool_dated_laws,
    strict_tail_readout,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]

SCENARIOS = ("A", "B", "C")
HISTORIES = 16
STRATA = 2
SEED = 2026102601
ARCHIVE_AXES = 6
COUNT_AXES = 4
LAW_AXES = 5
MIN_REPLICATIONS = 2
INNER_BATCH = 8


class ArticleRedrawValidationConfig(BaseModel):
    """The fixed observed-coverage expanded-support nested design."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    outer_replications: Literal[160] = 160
    inner_redraws: Literal[199] = 199
    pilot_replications: Literal[4] = 4
    seed: Literal[2026102601] = SEED


def validate_counts(counts: IntArray) -> None:
    """Admit nonnegative core/additional counts and nonempty annual core laws."""
    if (
        counts.ndim != COUNT_AXES
        or counts.shape[:3] != (STRATA, YEARS, MONTHS)
        or not np.issubdtype(counts.dtype, np.integer)
        or np.any(counts < 0)
        or np.any(counts[0].sum(axis=1) <= 0)
    ):
        msg = "stratified counts require [2,5,12,firm] with positive annual core totals"
        raise ValueError(msg)


def economic_design(
    index: int, counts: IntArray
) -> tuple[dict[str, FloatArray], dict[str, Any]]:
    """Solve one economy once and reuse it for the three observation mechanisms."""
    validate_counts(counts)
    if counts.shape[-1] != FIRMS:
        msg = "economic validation requires the fixed one-hundred-firm design"
        raise ValueError(msg)
    if not 0 <= index < HISTORIES:
        msg = "history must belong to the exhaustive sixteen-history design"
        raise ValueError(msg)
    history = histories()[index]
    solved = solve_history(history)
    capacity = np.asarray(solved.operating.capacity)
    fixed = np.broadcast_to(capacity[0], capacity.shape)
    laws = np.stack(
        [
            np.asarray(reporting_laws(capacities, history.cumulative, amplitude))[1:]
            for capacities, amplitude in ((capacity, 0), (fixed, 0), (fixed, 0.005))
        ]
    ).reshape(len(SCENARIOS), YEARS, MONTHS, counts.shape[-1], -1)
    news = np.asarray(solved.financial_news)
    expanded = counts.sum(axis=0)
    pooled = np.stack([pool_dated_laws(law, expanded) for law in laws])
    targets = measure_regimes(pooled, expanded, news)
    # The stationary common-reporting control is exactly zero by construction.
    arithmetic_q = targets["q"].copy()
    arithmetic_beta = targets["beta"].copy()
    targets["q"][1] = 0
    targets["beta"][1] = 0
    shares = capacity[..., 1] / capacity.sum(axis=-1)
    first_news = NEWS_TICKS[0]
    response = np.abs(shares[first_news + 1] - shares[first_news])
    premises: dict[str, Any] = {
        "maximum_solver_bound": float(np.max(solved.operating.bounds)),
        "minimum_capacity": float(capacity.min()),
        "minimum_affine_capacity": solved.minimum_affine_capacity,
        "minimum_baseline_price": solved.minimum_baseline_price,
        "minimum_opportunity": solved.minimum_opportunity,
        "active_domain_violations": solved.active_domain_violations,
        "signal_scale": {
            "activity": 1,
            "news_source_tick": first_news,
            "first_response_tick": first_news + 1,
            "first_response_abs_share_change_min": float(response.min()),
            "first_response_abs_share_change_max": float(response.max()),
            "maximum_abs_share_change_from_origin": float(
                np.max(np.abs(shares - shares[0]))
            ),
            "units": "fractions of each firm's total activity capacity",
        },
    }
    premises["premises_held"] = bool(
        premises["maximum_solver_bound"] <= SOLVE_TARGET
        and premises["minimum_capacity"] > 0
        and premises["minimum_affine_capacity"] > 0
        and premises["minimum_baseline_price"] > 0
        and premises["minimum_opportunity"] > 0
        and premises["active_domain_violations"] == 0
    )
    return {
        "laws": laws,
        "news": news,
        **{f"target_{name}": value for name, value in targets.items()},
        "target_arithmetic_q": arithmetic_q,
        "target_arithmetic_beta": arithmetic_beta,
    }, {"history": index, **premises}


def _rng(
    history: int,
    outer: int,
    role: int,
    stratum: int,
    *,
    regime: int = 0,
    inner: int = 0,
    pilot: bool = False,
) -> np.random.Generator:
    key = [SEED, history, outer, role, stratum, regime, inner]
    return np.random.default_rng(np.random.SeedSequence([*key, 99] if pilot else key))


def _month_draw(rng: np.random.Generator, months: IntArray) -> IntArray:
    """Draw dates within a stratum, including valid absent additional strata."""
    totals = months.sum(axis=1)
    probabilities = np.moveaxis(months, 1, -1).astype(float)
    probabilities = np.divide(
        probabilities,
        totals[..., None],
        out=np.zeros_like(probabilities),
        where=totals[..., None] > 0,
    )
    probabilities[..., 0] += totals == 0
    return np.moveaxis(rng.multinomial(totals, probabilities), -1, 1)


def outer_archive(
    laws: FloatArray, counts: IntArray, history: int, outer: int, *, pilot: bool = False
) -> IntArray:
    """Generate fixed-size strata with shared dates and regime-specific atoms."""
    validate_counts(counts)
    if (
        laws.ndim != LAW_AXES
        or laws.shape[1:4] != counts.shape[1:]
        or not np.isfinite(laws).all()
        or np.any(laws < 0)
        or not np.allclose(laws.sum(axis=-1), 1, rtol=1e-12, atol=1e-12)
    ):
        msg = "monthly laws must match the stratified count design"
        raise ValueError(msg)
    archive = np.zeros((laws.shape[0], STRATA, *laws.shape[1:]), dtype=np.int64)
    for stratum in range(STRATA):
        dates = _month_draw(
            _rng(history, outer, 0, stratum, pilot=pilot), counts[stratum]
        )
        for regime in range(len(laws)):
            archive[regime, stratum] = _rng(
                history, outer, 1, stratum, regime=regime, pilot=pilot
            ).multinomial(dates, laws[regime])
    return archive


def bootstrap_archive(
    archive: IntArray, history: int, outer: int, inner: int, *, pilot: bool = False
) -> IntArray:
    """Exactly resample empirical date/atom pairs within each corporate stratum."""
    if (
        archive.ndim != ARCHIVE_AXES
        or archive.shape[1:4] != (STRATA, YEARS, MONTHS)
        or not np.issubdtype(archive.dtype, np.integer)
        or np.any(archive < 0)
    ):
        msg = "joint archive requires nonnegative [regime,2,5,12,firm,atom] counts"
        raise ValueError(msg)
    dates = archive.sum(axis=-1)
    if not np.all(dates == dates[0:1]) or np.any(dates[0, 0].sum(axis=1) <= 0):
        msg = "regimes must share nonempty core date counts"
        raise ValueError(msg)
    result = np.zeros_like(archive)
    for stratum in range(STRATA):
        sampled_dates = _month_draw(
            _rng(history, outer, 2, stratum, inner=inner, pilot=pilot),
            dates[0, stratum],
        )
        for regime in range(len(archive)):
            joint = archive[regime, stratum]
            marginal = dates[regime, stratum]
            conditional = np.divide(
                joint,
                marginal[..., None],
                out=np.zeros_like(joint, dtype=float),
                where=marginal[..., None] > 0,
            )
            conditional[..., 0] += marginal == 0
            result[regime, stratum] = _rng(
                history, outer, 3, stratum, regime=regime, inner=inner, pilot=pilot
            ).multinomial(sampled_dates, conditional)
    return result


def archive_panel(archive: IntArray) -> tuple[FloatArray, IntArray]:
    """Pool expanded laws and their paired dates without retaining article IDs."""
    combined = archive.sum(axis=1)
    dates = combined[0].sum(axis=-1)
    totals = dates.sum(axis=1)
    if np.any(totals <= 0):
        msg = "expanded annual archive is empty"
        raise ValueError(msg)
    pooled = combined.sum(axis=2) / totals[None, ..., None]
    return pooled, dates


def measure_regimes(
    pooled: FloatArray, dates: IntArray, news: FloatArray
) -> dict[str, FloatArray]:
    """Reuse the centered annual estimator and one financial readout per panel."""
    financial, first_approach = panel_arrays(pooled[0], dates, news)
    beta, q, d, defined = (np.zeros(len(pooled)) for _ in range(4))
    for regime, law in enumerate(pooled):
        if regime == 0:
            approach = first_approach
        else:
            distance = axis_point_w2(law[:, :, None, :], law[0][None, None, :, :])
            approach = distance[:-1] - distance[1:]
        moments = annual_moments(financial, approach)
        q[regime], d[regime] = moments.numerator.sum(), moments.denominator.sum()
        if moments.beta is not None:
            beta[regime], defined[regime] = moments.beta, 1
    return {"beta": beta, "q": q, "d": d, "defined": defined}


def sensitivity_quantiles(beta: FloatArray, defined: FloatArray) -> dict[str, Any]:
    """Match corporate conditional-on-defined linear percentiles and disclose losses."""
    mask = defined.astype(bool)
    if beta.ndim != 1 or mask.shape != beta.shape or not np.isfinite(beta[mask]).all():
        msg = "quantiles require matching scalar estimates and their defined mask"
        raise ValueError(msg)
    values = beta[mask]
    quantiles = (
        np.quantile(values, [0.025, 0.05, 0.975], method="linear")
        if len(values)
        else (None, None, None)
    )
    return {
        **{
            f"article_sensitivity_{name}": None if value is None else float(value)
            for name, value in zip(("q025", "q05", "q975"), quantiles, strict=True)
        },
        "inner_attempts": len(beta),
        "inner_defined": len(values),
        "inner_undefined": int((~mask).sum()),
        "quantile_method": "linear",
        "quantile_status": "unavailable"
        if not mask.any()
        else "complete"
        if mask.all()
        else "conditional_on_defined",
        "quantile_reason": "no_defined_inner_estimates"
        if not mask.any()
        else None
        if mask.all()
        else "undefined_inner_estimates",
    }


def measure_regimes_batch(
    pooled: FloatArray, dates: IntArray, news: FloatArray
) -> dict[str, FloatArray]:
    """Batch the existing strict readout and annual moments, preserving their axes."""
    batch, regimes, years, firms, _ = pooled.shape
    if years != YEARS or dates.shape != (batch, YEARS, MONTHS, firms):
        msg = "batched panels require matching year/month/firm dates and annual laws"
        raise ValueError(msg)
    financial = np.full((batch, YEARS - 1, firms, firms), np.nan)
    diagonal = np.arange(firms)
    for year in range(1, YEARS):
        mass = np.moveaxis(dates[:, year], 1, 0).reshape(MONTHS, batch * firms)
        mass = mass / mass.sum(axis=0, keepdims=True)
        financial[:, year - 1] = strict_tail_readout(
            np.arange(year * MONTHS + 1, (year + 1) * MONTHS + 1, dtype=float),
            mass,
            np.arange(year * MONTHS, (year + 1) * MONTHS, dtype=float),
            news[year * MONTHS : (year + 1) * MONTHS],
        ).reshape(batch, firms, firms)
    financial[:, :, diagonal, diagonal] = np.nan
    distance = axis_point_w2(
        pooled[:, :, :, :, None, :], pooled[:, :, 0, None, None, :, :]
    )
    approach = distance[:, :, :-1] - distance[:, :, 1:]
    flat_financial = np.broadcast_to(
        financial[:, None], (batch, regimes, YEARS - 1, firms, firms)
    ).reshape(-1, firms, firms)
    moments = annual_moments(flat_financial, approach.reshape(-1, firms, firms))
    q = moments.numerator.reshape(batch, regimes, YEARS - 1).sum(axis=-1)
    d = moments.denominator.reshape(batch, regimes, YEARS - 1).sum(axis=-1)
    return {
        "beta": np.divide(q, d, out=np.zeros_like(q), where=d > 0),
        "q": q,
        "d": d,
        "defined": (d > 0).astype(float),
    }


def _inner_batch(
    archive: IntArray,
    history: int,
    outer: int,
    indices: range,
    news: FloatArray,
    *,
    pilot: bool,
) -> tuple[dict[str, FloatArray] | None, str | None]:
    pooled: list[FloatArray] = []
    dates: list[IntArray] = []
    try:
        for replicate in indices:
            sampled = bootstrap_archive(archive, history, outer, replicate, pilot=pilot)
            laws, months = archive_panel(sampled)
            pooled.append(laws)
            dates.append(months)
        return measure_regimes_batch(np.stack(pooled), np.stack(dates), news), None
    except (ValueError, ArithmeticError, RuntimeError) as error:
        return None, f"{type(error).__name__}: {error}"


def measure_outer(
    design: dict[str, FloatArray],
    counts: IntArray,
    history: int,
    outer: int,
    *,
    inner_redraws: int = 199,
    pilot: bool = False,
) -> tuple[dict[str, FloatArray], list[dict[str, Any]]]:
    """Consume one archive and retain only scalar inner and outer evidence."""
    if inner_redraws < MIN_REPLICATIONS:
        msg = "nested validation requires at least two inner redraws"
        raise ValueError(msg)
    archive = outer_archive(design["laws"], counts, history, outer, pilot=pilot)
    pooled, dates = archive_panel(archive)
    point = measure_regimes(pooled, dates, design["news"])
    exact = np.stack([pool_dated_laws(law, dates) for law in design["laws"]])
    date_target = measure_regimes(exact, dates, design["news"])
    date_target["beta"][1], date_target["q"][1] = 0, 0
    inner = {
        key: np.zeros((len(SCENARIOS), inner_redraws))
        for key in ("beta", "q", "d", "defined", "execution_succeeded")
    }
    inner_errors = []
    for start in range(0, inner_redraws, INNER_BATCH):
        indices = range(start, min(start + INNER_BATCH, inner_redraws))
        result, error = _inner_batch(
            archive, history, outer, indices, design["news"], pilot=pilot
        )
        if result is not None:
            for key, value in result.items():
                inner[key][:, indices.start : indices.stop] = value.T
            inner["execution_succeeded"][:, indices.start : indices.stop] = 1
            continue
        # Isolate failures rather than discarding the other members of a batch.
        for replicate in indices:
            single, single_error = _inner_batch(
                archive,
                history,
                outer,
                range(replicate, replicate + 1),
                design["news"],
                pilot=pilot,
            )
            if single is None:
                inner_errors.append(single_error or error or "inner execution failed")
                continue
            for key, value in single.items():
                inner[key][:, replicate] = value[0]
            inner["execution_succeeded"][:, replicate] = 1
    records = []
    for regime, scenario in enumerate(SCENARIOS):
        quantiles = sensitivity_quantiles(
            inner["beta"][regime], inner["defined"][regime]
        )
        records.append(
            {
                "scenario_id": scenario,
                "history": history,
                "outer_replicate": outer,
                "budget": 1,
                "support": "expanded",
                "execution_succeeded": True,
                "inner_execution_attempts": inner_redraws,
                "inner_execution_failed": len(inner_errors),
                "inner_execution_error": inner_errors[0] if inner_errors else None,
                "target": "beta_lambda_h",
                "oracle_constructed_zero": scenario == "B",
                **{
                    f"oracle_{key}": float(design[f"target_{key}"][regime])
                    for key in ("beta", "q", "d", "defined")
                },
                "oracle_arithmetic_q": float(design["target_arithmetic_q"][regime]),
                "oracle_arithmetic_beta": float(
                    design["target_arithmetic_beta"][regime]
                ),
                **{
                    f"operating_{key}": float(
                        design[f"target_{key}"][0 if scenario == "A" else 1]
                    )
                    for key in ("beta", "q", "d", "defined")
                },
                **{
                    f"outer_date_oracle_{key}": float(date_target[key][regime])
                    for key in ("beta", "q", "d", "defined")
                },
                **{
                    f"archive_{key}": float(point[key][regime])
                    for key in ("beta", "q", "d", "defined")
                },
                **quantiles,
            }
        )
    return inner, records


def _rate(
    events: np.ndarray,
    available: np.ndarray,
    *,
    unavailable_histories: tuple[int, ...] = (),
) -> dict[str, Any]:
    attempts = int(events.size)
    count = int(events.sum())
    available_count = int(available.sum())
    history_rates = events.mean(axis=1)
    rate = float(history_rates.mean())
    mcse = float(
        np.sqrt(
            np.var(events.astype(float), axis=1, ddof=1).sum()
            / (events.shape[0] ** 2 * events.shape[1])
        )
    )
    return {
        "equal_history_rate": None if unavailable_histories else rate,
        "mcse": None if unavailable_histories else mcse,
        "count": count,
        "attempts": attempts,
        "available_attempts": available_count,
        "unavailable_attempts": attempts - available_count,
        "available_rate": count / available_count
        if available_count and not unavailable_histories
        else None,
        "unavailable_histories": list(unavailable_histories),
        "unavailable_reason": "missing_declared_history_oracle"
        if unavailable_histories
        else None,
    }


def summarize_outer(
    rows: list[dict[str, Any]], outer_replications: int
) -> dict[str, Any]:
    """Use outer archives as replications and histories as exhaustive equal strata."""
    expected = {
        (scenario, history, outer)
        for scenario in SCENARIOS
        for history in range(HISTORIES)
        for outer in range(outer_replications)
    }
    keys = [
        (row["scenario_id"], row["history"], row["outer_replicate"]) for row in rows
    ]
    if (
        len(keys) != len(expected)
        or set(keys) != expected
        or outer_replications < MIN_REPLICATIONS
    ):
        msg = (
            "nested summaries require complete unique scenario/history/outer membership"
        )
        raise ValueError(msg)
    scenarios: list[dict[str, Any]] = []
    per_history: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        members = sorted(
            (row for row in rows if row["scenario_id"] == scenario),
            key=lambda row: (row["history"], row["outer_replicate"]),
        )
        targets: dict[int, float] = {}
        for row in members:
            if row["oracle_defined"] and row["oracle_beta"] is not None:
                history = row["history"]
                target = float(row["oracle_beta"])
                if history in targets and targets[history] != target:
                    msg = "fixed-history reported-law oracle changed across archives"
                    raise ValueError(msg)
                targets[history] = target
        missing_targets = tuple(
            history for history in range(HISTORIES) if history not in targets
        )
        quantile_available = np.asarray(
            [
                bool(
                    row["execution_succeeded"]
                    and row["article_sensitivity_q05"] is not None
                )
                for row in members
            ]
        ).reshape(HISTORIES, outer_replications)
        available = (
            quantile_available
            & np.asarray([history in targets for history in range(HISTORIES)])[:, None]
        )
        events = {
            key: np.zeros_like(available)
            for key in ("central_containment", "q05_containment", "positive_q05")
        }
        for index, row in enumerate(members):
            h, outer = divmod(index, outer_replications)
            if quantile_available[h, outer]:
                events["positive_q05"][h, outer] = row["article_sensitivity_q05"] > 0
            if available[h, outer]:
                low, high, q05, target = (
                    row["article_sensitivity_q025"],
                    row["article_sensitivity_q975"],
                    row["article_sensitivity_q05"],
                    targets[h],
                )
                events["central_containment"][h, outer] = low <= target <= high
                events["q05_containment"][h, outer] = q05 <= target
        header = {
            "scenario_id": scenario,
            "budget": 1,
            "support": "expanded",
            "target": "beta_lambda_h",
            "outer_attempts": len(members),
            "available_outer_attempts": int(quantile_available.sum()),
            "missing_oracle_histories": list(missing_targets),
            "sampling_target": (
                "Fixed economy and history; reported population law "
                "with prescribed core/additional monthly margins."
            ),
            "lower_bound_method": (
                "Conditional-on-defined article redraw percentile "
                "q05; no normal approximation."
            ),
            "quantile_method": "linear",
            "status": "simulation_calibration_only",
            "reason": (
                "Generated monthly-timing design; "
                "no unconditional corporate confidence claim."
            ),
            "positive_q05_interpretation": "Conditional zero-alignment rejection."
            if scenario == "B"
            else (
                "Positive-q05 frequency; nonzero history targets "
                "are not null rejection errors."
            ),
            "inner_undefined": sum(row["inner_undefined"] for row in members),
            "inner_execution_failed": sum(
                row.get("inner_execution_failed", 0) for row in members
            ),
            "outer_failed_execution": sum(
                not row["execution_succeeded"] for row in members
            ),
            "failed_premises": sum(
                not row.get("premises_held", False) for row in members
            ),
        }
        scenarios.append(
            {
                **header,
                **{
                    key: _rate(
                        value,
                        quantile_available if key == "positive_q05" else available,
                        unavailable_histories=()
                        if key == "positive_q05"
                        else missing_targets,
                    )
                    for key, value in events.items()
                },
            }
        )
        per_history.extend(
            {
                "scenario_id": scenario,
                "history": history,
                "outer_attempts": outer_replications,
                **{
                    key: _rate(
                        value[history : history + 1],
                        (quantile_available if key == "positive_q05" else available)[
                            history : history + 1
                        ],
                        unavailable_histories=(history,)
                        if history in missing_targets and key != "positive_q05"
                        else (),
                    )
                    for key, value in events.items()
                },
            }
            for history in range(HISTORIES)
        )
    return {"scenarios": scenarios, "per_history": per_history}
