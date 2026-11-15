"""Bounded multinomial measurement panels for the fixed recovery design."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np
from hotelling.dynamics.recovery import (
    REPORTING_AMPLITUDES,
    histories,
    reporting_laws,
    solve_dimensional_history,
    solve_history,
)
from hotelling.dynamics.transport import axis_point_w2
from numpy.typing import NDArray

from pipeline._kernels.paper6.reference_alignment import (
    financial_tails,
    pool_dated_laws,
    pooled_moments,
    recovery_summary,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from jax import Array

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]
MONTHS = 12
YEARS = 5
LAW_AXES = 4
BUDGETS = (1, 4, 16)
REDRAWS = 128
SEED = 20261024


@dataclass(frozen=True)
class HistoryMeasurement:
    """Separate measurement error, reporting distortion and affine approximation."""

    beta0: float
    affine_beta: float
    population_beta: FloatArray
    reporting_only_beta: FloatArray
    estimates: FloatArray
    maximum_solver_bound: float
    minimum_capacity: float
    minimum_baseline_price: float
    minimum_opportunity: float
    active_domain_violations: int


def measure_history(
    index: int,
    empirical_counts: IntArray,
    *,
    redraws: int = REDRAWS,
    dimension: int | None = None,
) -> HistoryMeasurement:
    """Reuse one completed Mesa path across every article panel and reporting law."""
    history = histories()[index]
    solved = (
        solve_history(history)
        if dimension is None
        else solve_dimensional_history(history, dimension)
    )
    capacity = solved.operating.capacity
    firms = capacity.shape[1]
    financial_news = np.asarray(solved.financial_news)

    def oracle(laws: Array | FloatArray) -> float:
        # Hotelling composes reporting laws on the JAX graph. Pooling and the
        # fixed NumPy article redraw schedule own the host statistical boundary.
        monthly = np.asarray(laws)[1:].reshape(YEARS, MONTHS, firms, laws.shape[-1])
        pooled = pool_dated_laws(monthly, empirical_counts)
        return estimate_panel(pooled, empirical_counts, financial_news)

    common = reporting_laws(capacity, history.cumulative, 0)
    changing = (
        reporting_laws(capacity, history.cumulative, 0.005)
        if dimension is None
        else common
    )
    fixed = np.broadcast_to(capacity[0], capacity.shape)
    amplitudes = REPORTING_AMPLITUDES if dimension is None else (0.0,)
    population = np.array(
        [
            oracle(reporting_laws(capacity, history.cumulative, amplitude))
            for amplitude in amplitudes
        ]
    )
    control = np.array(
        [
            oracle(reporting_laws(fixed, history.cumulative, amplitude))
            for amplitude in amplitudes
        ]
    )
    regimes = (common[1:], changing[1:]) if dimension is None else (common[1:],)
    laws = np.stack(regimes).reshape(
        len(regimes), YEARS, MONTHS, firms, common.shape[-1]
    )
    estimates = article_redraws(
        laws,
        empirical_counts,
        financial_news,
        index,
        redraws=redraws,
        budgets=BUDGETS if dimension is None else (1, 16),
    )
    return HistoryMeasurement(
        oracle(common),
        oracle(reporting_laws(solved.affine_shares, history.cumulative, 0)),
        population,
        control,
        estimates,
        float(solved.operating.bounds.max()),
        float(capacity.min()),
        solved.minimum_baseline_price,
        solved.minimum_opportunity,
        solved.active_domain_violations,
    )


def estimate_panel(
    pooled_laws: FloatArray, dated_counts: IntArray, financial_news: FloatArray
) -> float:
    """Reuse each sampled origin law across every reference and annual approach."""
    firms = pooled_laws.shape[1]
    if (
        pooled_laws.ndim != LAW_AXES - 1
        or len(pooled_laws) != YEARS
        or dated_counts.shape != (YEARS, MONTHS, firms)
        or financial_news.shape != (YEARS * MONTHS, firms)
    ):
        msg = "panel requires five annual laws, monthly counts and dated financial news"
        raise ValueError(msg)
    distances = axis_point_w2(
        pooled_laws[:, :, None, :], pooled_laws[0][None, None, :, :]
    )
    financial = np.stack(
        [
            financial_tails(
                np.arange(year * MONTHS + 1, (year + 1) * MONTHS + 1, dtype=np.float64),
                dated_counts[year],
                np.arange(year * MONTHS, (year + 1) * MONTHS, dtype=np.float64),
                financial_news[year * MONTHS : (year + 1) * MONTHS],
            )
            for year in range(1, YEARS)
        ]
    )
    beta = pooled_moments(distances, financial).beta
    return np.nan if beta is None else beta


def article_redraws(
    monthly_laws: FloatArray,
    empirical_counts: IntArray,
    financial_news: FloatArray,
    history: int,
    *,
    redraws: int = REDRAWS,
    budgets: tuple[int, ...] = BUDGETS,
) -> FloatArray:
    """Nested count increments, shared dates across regimes and independent panels.

    Laws have [regime,year,month,firm,atom] axes. Only regime-by-budget-by-panel
    statistics are returned; pairwise tensors do not survive each calculation.
    Stream identities depend on history and replicate, independent of execution
    order, worker count and resumption. Budgets and the seed are fixed constants.
    """
    estimates = np.empty((len(monthly_laws), len(budgets), redraws))
    for regime, budget, replicate, pooled, dates in article_panels(
        monthly_laws, empirical_counts, history, redraws=redraws, budgets=budgets
    ):
        estimates[regime, budget, replicate] = estimate_panel(
            pooled, dates, financial_news
        )
    return estimates


def article_panels(
    monthly_laws: FloatArray,
    empirical_counts: IntArray,
    history: int,
    *,
    redraws: int = REDRAWS,
    budgets: tuple[int, ...] = BUDGETS,
    stream: tuple[int, ...] = (),
) -> Iterator[tuple[int, int, int, FloatArray, IntArray]]:
    """Yield the retained paired panel design, with optional new scenario streams.

    Empty ``stream`` reproduces historical date and atom streams exactly. New
    scenarios append their identity without changing that historical schedule.
    Returned panel arrays are ephemeral; callers persist scalar statistics only.
    """
    if (
        monthly_laws.ndim != LAW_AXES + 1
        or monthly_laws.shape[1:4] != empirical_counts.shape
        or empirical_counts.shape[:2] != (YEARS, MONTHS)
        or not np.issubdtype(empirical_counts.dtype, np.integer)
        or np.any(empirical_counts < 0)
        or np.any(empirical_counts.sum(axis=1) <= 0)
        or not np.isfinite(monthly_laws).all()
        or np.any(monthly_laws < 0)
        or not np.allclose(monthly_laws.sum(axis=-1), 1, rtol=1e-12, atol=1e-12)
        or redraws <= 0
        or history < 0
        or not budgets
        or budgets[0] <= 0
        or any(first >= second for first, second in pairwise(budgets))
        or any(value < 0 for value in stream)
    ):
        msg = (
            "redraws require monthly probability laws and nonempty empirical firm-years"
        )
        raise ValueError(msg)
    regimes, _, _, firms, atoms = monthly_laws.shape
    totals = empirical_counts.sum(axis=1)
    date_probabilities = np.moveaxis(empirical_counts / totals[:, None, :], 1, -1)
    for replicate in range(redraws):
        date_rng = np.random.default_rng(
            np.random.SeedSequence([SEED, history, replicate, 0, *stream])
        )
        atom_rng = [
            np.random.default_rng(
                np.random.SeedSequence([SEED, history, replicate, 1, regime, *stream])
            )
            for regime in range(regimes)
        ]
        dates = np.zeros_like(empirical_counts)
        sampled = np.zeros((regimes, YEARS, MONTHS, firms, atoms), dtype=np.int64)
        previous = 0
        for budget_index, budget in enumerate(budgets):
            increments = date_rng.multinomial(
                (budget - previous) * totals, date_probabilities
            )
            increments = np.moveaxis(increments, -1, 1)
            dates += increments
            for regime in range(regimes):
                sampled[regime] += atom_rng[regime].multinomial(
                    increments, monthly_laws[regime]
                )
                pooled = sampled[regime].sum(axis=1) / (budget * totals[..., None])
                yield regime, budget_index, replicate, pooled, dates
            previous = budget


def summarize_measurements(
    measurements: list[HistoryMeasurement | None], *, appendix: bool = False
) -> dict[str, Any]:
    """Keep all enumerated histories in recovery and certify declared premises."""
    budgets = (1, 16) if appendix else BUDGETS
    regimes = 1 if appendix else 2
    missing = sum(item is None for item in measurements)
    if missing:
        return {
            "recovery": None,
            "finite_affine_departure": None,
            "reporting_floor": None,
            "reporting_only_floor": None,
            "smallest_recovered_budget": [None] * regimes,
            "diagnostics": {"failed_histories": missing},
        }
    complete = [item for item in measurements if item is not None]
    oracle = np.asarray([item.beta0 for item in complete])
    estimates = np.stack([item.estimates for item in complete])
    scale = float(np.sqrt(np.mean(oracle**2)))
    if not np.isfinite(oracle).all():
        msg = "enumerated population coefficients are undefined"
        raise ArithmeticError(msg)
    summaries = [
        [
            asdict(recovery_summary(estimates[:, regime, budget], oracle))
            for budget in range(len(budgets))
        ]
        for regime in range(regimes)
    ]
    population = np.stack([item.population_beta for item in complete])
    control = np.stack([item.reporting_only_beta for item in complete])
    affine = np.asarray([item.affine_beta for item in complete])
    return {
        "recovery": summaries[0] if appendix else summaries,
        "finite_affine_departure": None
        if scale == 0
        else float(np.sqrt(np.mean((affine - oracle) ** 2)) / scale),
        "reporting_floor": [None] * population.shape[1]
        if scale == 0
        else (
            np.sqrt(np.mean((population - oracle[:, None]) ** 2, axis=0)) / scale
        ).tolist(),
        "reporting_only_floor": [None] * control.shape[1]
        if scale == 0
        else (np.sqrt(np.mean(control**2, axis=0)) / scale).tolist(),
        "normalization_reason": "zero_oracle_scale" if scale == 0 else None,
        "smallest_recovered_budget": [
            next(
                (
                    budget
                    for budget, result in zip(budgets, row, strict=True)
                    if result["recovered"]
                ),
                None,
            )
            for row in summaries
        ],
        "diagnostics": {
            "maximum_solver_bound": max(item.maximum_solver_bound for item in complete),
            "minimum_capacity": min(item.minimum_capacity for item in complete),
            "minimum_baseline_price": min(
                item.minimum_baseline_price for item in complete
            ),
            "minimum_opportunity": min(item.minimum_opportunity for item in complete),
            "active_domain_violations": sum(
                item.active_domain_violations for item in complete
            ),
        },
    }
