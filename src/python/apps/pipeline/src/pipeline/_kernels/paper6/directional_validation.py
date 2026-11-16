"""Known-target directional diagnostics for the retained article estimator.

Histories are exhaustive strata, never iid dyads. Scalar records retain every
attempt, including undefined statistics and failed scientific premises.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from hotelling.dynamics.recovery import (
    REPORTING_AMPLITUDES,
    histories,
    reporting_laws,
    solve_history,
)
from hotelling.dynamics.transport import axis_point_w2
from hotelling.orthant import SOLVE_TARGET
from pydantic import BaseModel, ConfigDict, Field

from pipeline._kernels.paper6.inference_audit import pooled_candidate
from pipeline._kernels.paper6.recovery import (
    BUDGETS,
    MONTHS,
    REDRAWS,
    YEARS,
    article_panels,
)
from pipeline._kernels.paper6.reference_alignment import (
    MATRIX_AXES,
    MIN_REDRAWS,
    annual_moments,
    centered_references,
    financial_tails,
    pool_dated_laws,
    recovery_summary,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from pipeline._kernels.paper6.recovery import HistoryMeasurement

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]
HISTORIES = 16
RESOLUTION = 1e-12


class DirectionalValidationConfig(BaseModel):
    """Cohesive typed configuration; the scientific design is deliberately fixed."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    redraws: Literal[160] = 160
    pilot_redraws: Literal[4] = 4
    numerical_resolution: float = Field(default=RESOLUTION, gt=0)


@dataclass(frozen=True)
class Scenario:
    """One declared observation law over one retained economic path."""

    scenario_id: str
    regimes: tuple[str, ...]
    amplitudes: tuple[float, ...]
    budgets: tuple[int, ...]
    shock_multiplier: float
    stationary_capacity: bool
    stream: tuple[int, ...]


SCENARIOS = (
    Scenario(
        "A",
        ("common", "changing"),
        (0, 0.005),
        BUDGETS,
        1,
        stationary_capacity=False,
        stream=(),
    ),
    Scenario("B", ("common",), (0,), BUDGETS, 1, stationary_capacity=True, stream=(2,)),
    Scenario(
        "C",
        tuple(f"reporting_{a:g}" for a in REPORTING_AMPLITUDES),
        REPORTING_AMPLITUDES,
        BUDGETS,
        1,
        stationary_capacity=True,
        stream=(3,),
    ),
    Scenario(
        "D_half", ("common",), (0,), (1,), 0.5, stationary_capacity=False, stream=(4, 0)
    ),
    Scenario(
        "D_double", ("common",), (0,), (1,), 2, stationary_capacity=False, stream=(4, 1)
    ),
)


def failed_scenario(
    scenario: Scenario, history: int, redraws: int, error: Exception
) -> list[dict[str, Any]]:
    """Keep every declared attempt when a single operating path is unavailable."""
    return [
        {
            "scenario_id": scenario.scenario_id,
            "history": history,
            "regime": regime,
            "reporting_amplitude": amplitude,
            "shock_multiplier": scenario.shock_multiplier,
            "budget": budget,
            "replicate": replicate,
            "beta": None,
            "q": None,
            "d": None,
            "sign": None,
            "defined": False,
            "execution_succeeded": False,
            "premises_held": False,
            "oracle_beta": None,
            "oracle_sign": None,
            "operating_beta": None,
            "operating_sign": None,
            "candidate_se": None,
            "candidate_status": "execution_failed",
            "exact_population_candidate_se": None,
            "moment_unavailability_reason": "economic_path_or_domain_failure",
            "execution_error": f"{type(error).__name__}: {error}",
        }
        for regime, amplitude in zip(scenario.regimes, scenario.amplitudes, strict=True)
        for budget in scenario.budgets
        for replicate in range(redraws)
    ]


def panel_arrays(
    pooled: FloatArray, counts: IntArray, news: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Use identical frozen origins, article dates and strict financial tails."""
    distances = axis_point_w2(pooled[:, :, None, :], pooled[0][None, None, :, :])
    financial = np.stack(
        [
            financial_tails(
                np.arange(year * MONTHS + 1, (year + 1) * MONTHS + 1, dtype=np.float64),
                counts[year],
                np.arange(year * MONTHS, (year + 1) * MONTHS, dtype=np.float64),
                news[year * MONTHS : (year + 1) * MONTHS],
            )
            for year in range(1, YEARS)
        ]
    )
    return financial, distances[:-1] - distances[1:]


def panel_diagnostics(
    pooled: FloatArray,
    counts: IntArray,
    news: FloatArray,
    *,
    resolution: float = RESOLUTION,
) -> dict[str, Any]:
    """Persist moments, scale and an arithmetic sign bound, without pair tensors."""
    financial, approach = panel_arrays(pooled, counts, news)
    moments = annual_moments(financial, approach)
    rc, ac = centered_references(financial), centered_references(approach)
    included = ~np.eye(rc.shape[-1], dtype=bool)
    r, a = rc[:, included], ac[:, included]
    q, d = float(moments.numerator.sum()), float(moments.denominator.sum())
    delta_a = 16 * resolution
    delta_r = 4 * resolution * float(np.max(np.abs(news).sum(axis=0)))
    delta_q = float(
        np.sum(
            delta_r * np.mean(np.abs(a), axis=1)
            + delta_a * np.mean(np.abs(r), axis=1)
            + delta_r * delta_a
            + resolution * np.mean(np.abs(r * a), axis=1)
        )
    )
    rrms, arms = float(np.sqrt(np.mean(r**2))), float(np.sqrt(np.mean(a**2)))
    beta = moments.beta
    candidate = pooled_candidate(financial, approach)
    return {
        "beta": beta,
        "q": q,
        "d": d,
        "annual_q": moments.numerator.tolist(),
        "annual_d": moments.denominator.tolist(),
        "q_resolution": delta_q,
        "sign": sign(q, delta_q) if beta is not None else None,
        "financial_rms": rrms,
        "approach_rms": arms,
        "normalized_alignment": None
        if beta is None or arms == 0
        else beta * rrms / arms,
        "defined": beta is not None,
        "denominator_status": "zero"
        if d == 0
        else "near_zero"
        if d <= resolution
        else "positive",
        "candidate_se": candidate["candidate_se"],
        "candidate_status": candidate["status"],
    }


def sign(value: float, tolerance: float) -> int:
    """Zero is numerically unresolved, never a correct positive direction."""
    return int(value > tolerance) - int(value < -tolerance)


def _population_target(
    diagnostics: dict[str, Any], *, constructed_zero: bool
) -> dict[str, Any]:
    """Use exact structural zeros while retaining their arithmetic residuals."""
    if not constructed_zero:
        return diagnostics
    return {
        **diagnostics,
        "arithmetic_beta": diagnostics["beta"],
        "arithmetic_q": diagnostics["q"],
        "arithmetic_approach_rms": diagnostics["approach_rms"],
        "beta": 0.0,
        "q": 0.0,
        "sign": 0,
        "approach_rms": 0.0,
    }


def measure_validation_history(
    index: int,
    counts: IntArray,
    *,
    redraws: int = 160,
    pilot: bool = False,
    resolution: float = RESOLUTION,
) -> list[dict[str, Any]]:
    """Reuse each solved path across all its observation laws and redraws."""
    rows: list[dict[str, Any]] = []
    solved_paths = {}
    for scenario in SCENARIOS:
        multiplier = scenario.shock_multiplier
        history = histories(shock_multiplier=multiplier)[index]
        if multiplier not in solved_paths:
            try:
                solved_paths[multiplier] = solve_history(history)
            except (ArithmeticError, TimeoutError, RuntimeError) as error:
                rows.extend(failed_scenario(scenario, index, redraws, error))
                continue
        solved = solved_paths[multiplier]
        capacity = np.asarray(solved.operating.capacity)
        news = np.asarray(solved.financial_news)
        baseline = np.broadcast_to(capacity[0], capacity.shape)
        observed = baseline if scenario.stationary_capacity else capacity
        laws = np.stack(
            [
                np.asarray(reporting_laws(observed, history.cumulative, amplitude))[1:]
                for amplitude in scenario.amplitudes
            ]
        ).reshape(len(scenario.regimes), YEARS, MONTHS, counts.shape[-1], -1)

        cumulative = np.asarray(history.cumulative)

        def oracle(
            capacities: FloatArray,
            amplitude: float,
            cumulative: FloatArray = cumulative,
            financial_news: FloatArray = news,
            *,
            constructed_zero: bool = scenario.stationary_capacity,
        ) -> dict[str, Any]:
            monthly = np.asarray(reporting_laws(capacities, cumulative, amplitude))[1:]
            pooled = pool_dated_laws(
                monthly.reshape(YEARS, MONTHS, counts.shape[-1], -1), counts
            )
            diagnostics = panel_diagnostics(
                pooled, counts, financial_news, resolution=resolution
            )
            return _population_target(
                diagnostics, constructed_zero=constructed_zero and amplitude == 0
            )

        operating = oracle(observed, 0)
        tangent = oracle(
            observed
            if scenario.stationary_capacity
            else np.asarray(solved.affine_shares),
            0,
        )
        oracles = [oracle(observed, amplitude) for amplitude in scenario.amplitudes]
        premise = {
            "maximum_solver_bound": float(np.max(solved.operating.bounds)),
            "minimum_capacity": float(capacity.min()),
            "minimum_affine_capacity": solved.minimum_affine_capacity,
            "minimum_baseline_price": solved.minimum_baseline_price,
            "minimum_opportunity": solved.minimum_opportunity,
            "active_domain_violations": solved.active_domain_violations,
        }
        premises_held = bool(
            premise["maximum_solver_bound"] <= SOLVE_TARGET
            and premise["minimum_capacity"] > 0
            and premise["minimum_affine_capacity"] > 0
            and premise["minimum_baseline_price"] > 0
            and premise["minimum_opportunity"] > 0
            and premise["active_domain_violations"] == 0
        )
        stream = (*scenario.stream, 99) if pilot else scenario.stream
        for regime, budget, replicate, pooled, dates in article_panels(
            laws,
            counts,
            index,
            redraws=redraws,
            budgets=scenario.budgets,
            stream=stream,
        ):
            reference = oracles[regime]
            detail = panel_diagnostics(pooled, dates, news, resolution=resolution)
            rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "history": index,
                    "regime": scenario.regimes[regime],
                    "reporting_amplitude": scenario.amplitudes[regime],
                    "budget": scenario.budgets[budget],
                    "replicate": replicate,
                    "stream": "pilot"
                    if pilot
                    else "historical_prefix"
                    if scenario.scenario_id == "A" and replicate < REDRAWS
                    else "production",
                    "shock_multiplier": multiplier,
                    "execution_succeeded": True,
                    "premises_held": premises_held,
                    "execution_error": None,
                    **premise,
                    **detail,
                    "oracle_beta": reference["beta"],
                    "oracle_q": reference["q"],
                    "oracle_d": reference["d"],
                    "oracle_sign": reference["sign"],
                    "oracle_arithmetic_beta": reference.get(
                        "arithmetic_beta", reference["beta"]
                    ),
                    "oracle_arithmetic_q": reference.get(
                        "arithmetic_q", reference["q"]
                    ),
                    "oracle_q_resolution": reference["q_resolution"],
                    "oracle_financial_rms": reference["financial_rms"],
                    "oracle_approach_rms": reference["approach_rms"],
                    "exact_population_candidate_se": reference["candidate_se"],
                    "oracle_constructed_zero": bool(
                        scenario.stationary_capacity
                        and scenario.amplitudes[regime] == 0
                    ),
                    "operating_beta": operating["beta"],
                    "operating_q": operating["q"],
                    "operating_d": operating["d"],
                    "operating_sign": operating["sign"],
                    "operating_constructed_zero": scenario.stationary_capacity,
                    "operating_arithmetic_beta": operating.get(
                        "arithmetic_beta", operating["beta"]
                    ),
                    "operating_arithmetic_q": operating.get(
                        "arithmetic_q", operating["q"]
                    ),
                    "tangent_beta": tangent["beta"],
                    "tangent_q": tangent["q"],
                    "tangent_d": tangent["d"],
                    "tangent_sign": tangent["sign"],
                    "tangent_constructed_zero": scenario.stationary_capacity,
                    "reporting_distortion": None
                    if reference["beta"] is None or operating["beta"] is None
                    else reference["beta"] - operating["beta"],
                    "measurement_error": None
                    if detail["beta"] is None or reference["beta"] is None
                    else detail["beta"] - reference["beta"],
                    "operating_error": None
                    if detail["beta"] is None or operating["beta"] is None
                    else detail["beta"] - operating["beta"],
                }
            )
    return rows


def _rate(sample: FloatArray) -> dict[str, Any]:
    """Equal-stratum mean with MCSE from independent within-history draws."""
    return {
        "rate": float(np.mean(sample)),
        "mcse": float(
            np.sqrt(
                np.var(sample, axis=1, ddof=1).sum()
                / (len(sample) ** 2 * sample.shape[1])
            )
        ),
        "numerator": int(sample.sum()),
        "denominator": int(sample.size),
    }


def performance(
    estimates: FloatArray,
    oracle: FloatArray,
    *,
    estimate_signs: FloatArray | None = None,
    oracle_signs: FloatArray | None = None,
) -> dict[str, Any]:
    """Absolute and directional performance with explicit attempt denominators."""
    if (
        estimates.ndim != MATRIX_AXES
        or oracle.shape != (len(estimates),)
        or estimates.shape[1] < MIN_REDRAWS
    ):
        msg = "performance requires repeated panels and one oracle per stratum"
        raise ValueError(msg)
    defined = np.isfinite(estimates)
    target_defined = np.isfinite(oracle)
    signs = (
        np.where(defined, np.sign(estimates) * (np.abs(estimates) > RESOLUTION), 0)
        if estimate_signs is None
        else estimate_signs
    )
    oracle_signs = (
        np.sign(oracle) * (np.abs(oracle) > RESOLUTION)
        if oracle_signs is None
        else oracle_signs
    )
    eligible = target_defined & (oracle_signs != 0)
    error = estimates - oracle[:, None]
    full = bool(defined.all() and target_defined.all())
    legacy = recovery_summary(estimates, oracle) if target_defined.all() else None
    legacy_recovered = (
        None if legacy is None or legacy.normalized_rmse is None else legacy.recovered
    )
    magnitude_reason = (
        "undefined_target" if legacy is None else legacy.normalization_reason
    )
    return {
        "total": int(estimates.size),
        "defined": int(defined.sum()),
        "undefined": int((~defined).sum()),
        "target_undefined_histories": int((~target_defined).sum()),
        "above_zero": _rate((defined & (signs > 0)).astype(np.float64)),
        "below_zero": _rate((defined & (signs < 0)).astype(np.float64)),
        "unresolved": _rate((defined & (signs == 0)).astype(np.float64)),
        "undefined_rate": _rate((~defined).astype(np.float64)),
        "correct_sign": None
        if not eligible.any()
        else _rate(
            (
                defined[eligible] & (signs[eligible] == oracle_signs[eligible, None])
            ).astype(np.float64)
        ),
        "wrong_sign": None
        if not eligible.any()
        else _rate(
            (
                defined[eligible] & (signs[eligible] == -oracle_signs[eligible, None])
            ).astype(np.float64)
        ),
        "sign_ineligible_attempts": int((~eligible).sum() * estimates.shape[1]),
        "sign_ineligibility_reason": "zero_or_unresolved_oracle"
        if not eligible.any()
        else None,
        "bias": float(np.mean(error)) if full else None,
        "absolute_bias": abs(float(np.mean(error))) if full else None,
        "bias_mcse": None if legacy is None else legacy.bias_mcse,
        "empirical_sd": float(np.sqrt(np.mean(np.var(estimates, axis=1, ddof=1))))
        if full
        else None,
        "mae": float(np.mean(np.abs(error))) if full else None,
        "rmse": float(np.sqrt(np.mean(error**2))) if full else None,
        "normalized_rmse": None if legacy is None else legacy.normalized_rmse,
        "normalized_rmse_mcse": None if legacy is None else legacy.rmse_mcse,
        "normalization_reason": magnitude_reason,
        "legacy_recovered": legacy_recovered,
        "legacy_magnitude_status": "unavailable"
        if legacy_recovered is None
        else "pass"
        if legacy_recovered
        else "fail",
        "legacy_magnitude_reason": magnitude_reason
        if legacy_recovered is None
        else None,
        "magnitude_scope": "Complete-attempt metrics; unavailable if any "
        "attempted statistic or target is undefined.",
        "empirical_sd_scope": "Root mean within-history sampling variance; "
        "between-history dispersion is not Monte Carlo variation.",
    }


def _paired_values(
    records: list[dict[str, Any]], metric_name: str, redraws: int
) -> FloatArray:
    if metric_name == "positive":
        return np.array(
            [
                float(item["beta"] is not None and (item.get("sign") or 0) > 0)
                for item in records
            ]
        ).reshape(HISTORIES, redraws)
    if metric_name == "correct_sign":
        return np.array(
            [
                np.nan
                if not item.get("oracle_sign")
                else float(
                    item["beta"] is not None and item.get("sign") == item["oracle_sign"]
                )
                for item in records
            ]
        ).reshape(HISTORIES, redraws)
    return np.array(
        [
            np.nan
            if item["beta"] is None or item.get("oracle_beta") is None
            else abs(item["beta"] - item["oracle_beta"])
            for item in records
        ]
    ).reshape(HISTORIES, redraws)


def _targets(scenario: str) -> tuple[tuple[str, str, str], ...]:
    common = (
        ("beta_lambda_h", "oracle_beta", "oracle_sign"),
        ("beta0_h", "operating_beta", "operating_sign"),
    )
    return (
        (*common, ("tangent_beta0_h", "tangent_beta", "tangent_sign"))
        if scenario.startswith("D_")
        else common
    )


def _domain_departure(row: dict[str, Any]) -> bool:
    return bool(
        (row.get("active_domain_violations") or 0) > 0
        or any(
            row.get(field) is not None and row[field] <= 0
            for field in (
                "minimum_capacity",
                "minimum_affine_capacity",
                "minimum_opportunity",
                "minimum_baseline_price",
            )
        )
    )


def _locality(
    groups: dict[tuple[str, str, int], list[dict[str, Any]]], redraws: int
) -> list[dict[str, Any]]:
    """Expose finite/tangent sign differences even when local premises depart."""
    result = []
    for (scenario, regime, budget), records in sorted(groups.items()):
        if not scenario.startswith("D_"):
            continue
        strata = [records[h * redraws] for h in range(HISTORIES)]
        known = [
            row
            for row in strata
            if row.get("operating_sign") is not None
            and row.get("tangent_sign") is not None
        ]
        result.append(
            {
                "scenario_id": scenario,
                "regime": regime,
                "budget": budget,
                "histories": HISTORIES,
                "defined_oracle_pairs": len(known),
                "finite_tangent_sign_disagreements": sum(
                    row["operating_sign"] != row["tangent_sign"] for row in known
                ),
                "finite_oracle_sign_counts": {
                    str(value): sum(row["operating_sign"] == value for row in known)
                    for value in (-1, 0, 1)
                },
                "tangent_oracle_sign_counts": {
                    str(value): sum(row["tangent_sign"] == value for row in known)
                    for value in (-1, 0, 1)
                },
                "domain_departures": sum(_domain_departure(row) for row in strata),
                "per_history": [
                    {
                        key: row.get(key)
                        for key in (
                            "history",
                            "operating_beta",
                            "operating_q",
                            "operating_d",
                            "operating_sign",
                            "tangent_beta",
                            "tangent_q",
                            "tangent_d",
                            "tangent_sign",
                            "active_domain_violations",
                            "premises_held",
                        )
                    }
                    for row in strata
                ],
            }
        )
    return result


def summarize_rows(rows: Sequence[dict[str, Any]], *, redraws: int) -> dict[str, Any]:
    """Require complete exhaustive strata before producing balanced summaries."""
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (row["scenario_id"], row["regime"], row["budget"]), []
        ).append(row)
    aggregate: list[dict[str, Any]] = []
    per_history: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    for key, records in sorted(groups.items()):
        records.sort(key=lambda item: (item["history"], item["replicate"]))
        membership = [(item["history"], item["replicate"]) for item in records]
        if membership != [(h, r) for h in range(HISTORIES) for r in range(redraws)]:
            msg = f"incomplete or duplicated directional strata: {key}"
            raise ValueError(msg)
        estimates = np.array(
            [np.nan if item["beta"] is None else item["beta"] for item in records]
        ).reshape(HISTORIES, redraws)
        signs = np.array(
            [0 if item.get("sign") is None else item["sign"] for item in records]
        ).reshape(HISTORIES, redraws)
        for target, beta_field, sign_field in _targets(key[0]):
            oracle = np.array(
                [
                    np.nan
                    if records[h * redraws].get(beta_field) is None
                    else records[h * redraws][beta_field]
                    for h in range(HISTORIES)
                ]
            )
            oracle_signs = np.array(
                [records[h * redraws].get(sign_field, 0) or 0 for h in range(HISTORIES)]
            )
            stats = performance(
                estimates, oracle, estimate_signs=signs, oracle_signs=oracle_signs
            )
            header = {
                "scenario_id": key[0],
                "scenario": key[0],
                "regime": key[1],
                "budget": key[2],
                "target": target,
            }
            aggregate.append(
                {
                    **header,
                    **stats,
                    "correct_sign_rate": None
                    if stats["correct_sign"] is None
                    else stats["correct_sign"]["rate"],
                    "correct_sign_mcse": None
                    if stats["correct_sign"] is None
                    else stats["correct_sign"]["mcse"],
                    "correct_sign_denominator": None
                    if stats["correct_sign"] is None
                    else stats["correct_sign"]["denominator"],
                    "wrong_sign_rate": None
                    if stats["wrong_sign"] is None
                    else stats["wrong_sign"]["rate"],
                    "positive_rate": stats["above_zero"]["rate"],
                    "positive_mcse": stats["above_zero"]["mcse"],
                    "positive_denominator": stats["above_zero"]["denominator"],
                    "domain_departures": sum(
                        _domain_departure(records[h * redraws])
                        for h in range(HISTORIES)
                    ),
                    "failed_premises": sum(
                        not item["premises_held"] for item in records
                    ),
                    "failed_execution": sum(
                        not item["execution_succeeded"] for item in records
                    ),
                    "mean_oracle_ratio": float(np.mean(oracle))
                    if np.isfinite(oracle).all()
                    else None,
                    "ratio_of_mean_oracle_moments": _oracle_ratio(
                        records, redraws, target
                    ),
                    **_oracle_quantities(records, redraws, target),
                }
            )
            per_history.extend(
                {
                    **header,
                    "history": h,
                    **performance(
                        estimates[h : h + 1],
                        oracle[h : h + 1],
                        estimate_signs=signs[h : h + 1],
                        oracle_signs=oracle_signs[h : h + 1],
                    ),
                }
                for h in range(HISTORIES)
            )
    for scenario, regime in sorted({key[:2] for key in groups}):
        budgets = sorted(key[2] for key in groups if key[:2] == (scenario, regime))
        for first, second in pairwise(budgets):
            left, right = (
                groups[(scenario, regime, first)],
                groups[(scenario, regime, second)],
            )
            for metric in ("positive", "correct_sign", "absolute_error"):
                difference = _paired_values(right, metric, redraws) - _paired_values(
                    left, metric, redraws
                )
                complete = bool(np.isfinite(difference).all())
                contrasts.append(
                    {
                        "scenario_id": scenario,
                        "regime": regime,
                        "budget_from": first,
                        "budget_to": second,
                        "metric": metric,
                        "difference": float(difference.mean()) if complete else None,
                        "mcse": float(
                            np.sqrt(
                                np.var(difference, axis=1, ddof=1).sum()
                                / (HISTORIES**2 * redraws)
                            )
                        )
                        if complete
                        else None,
                        "paired_attempts": int(difference.size),
                        "undefined_pairs": int((~np.isfinite(difference)).sum()),
                    }
                )
    return {
        "aggregate": aggregate,
        "per_history": per_history,
        "paired_budget_contrasts": contrasts,
        "locality": _locality(groups, redraws),
    }


def _oracle_ratio(
    records: list[dict[str, Any]], redraws: int, target: str
) -> float | None:
    prefix = {
        "beta_lambda_h": "oracle",
        "beta0_h": "operating",
        "tangent_beta0_h": "tangent",
    }[target]
    values = [records[h * redraws] for h in range(HISTORIES)]
    if any(
        item.get(f"{prefix}_q") is None or item.get(f"{prefix}_d") is None
        for item in values
    ):
        return None
    q = sum(item[f"{prefix}_q"] for item in values)
    d = sum(item[f"{prefix}_d"] for item in values)
    return None if d == 0 else q / d


def _oracle_quantities(
    records: list[dict[str, Any]], redraws: int, target: str
) -> dict[str, Any]:
    """Keep E[q/d] and E[q]/E[d] separate under the stated exhaustive design."""
    prefix = {
        "beta_lambda_h": "oracle",
        "beta0_h": "operating",
        "tangent_beta0_h": "tangent",
    }[target]
    strata = [records[h * redraws] for h in range(HISTORIES)]
    known_beta = all(row.get(f"{prefix}_beta") is not None for row in strata)
    known_moments = all(
        row.get(f"{prefix}_q") is not None and row.get(f"{prefix}_d") is not None
        for row in strata
    )
    q = (
        float(np.mean([row[f"{prefix}_q"] for row in strata]))
        if known_moments
        else None
    )
    d = (
        float(np.mean([row[f"{prefix}_d"] for row in strata]))
        if known_moments
        else None
    )
    return {
        "equal_history_mean_ratio": float(
            np.mean([row[f"{prefix}_beta"] for row in strata])
        )
        if known_beta
        else None,
        "equal_history_mean_ratio_reason": None
        if known_beta
        else "missing_history_oracle",
        "equal_history_mean_q": q,
        "equal_history_mean_d": d,
        "ratio_of_equal_history_moments": None
        if q is None or d is None or d == 0
        else q / d,
        "ratio_of_equal_history_moments_reason": "missing_history_oracle_moments"
        if not known_moments
        else "zero_equal_history_denominator"
        if d == 0
        else None,
        "oracle_weighting_law": "Equal weight 1/16 over the exhaustive sixteen "
        "sign histories; no missing-history renormalization.",
    }


def legacy_rows(measurements: Sequence[HistoryMeasurement]) -> list[dict[str, Any]]:
    """Postprocess original checkpoints; never invent unsaved q/d or precision."""
    if len(measurements) != HISTORIES:
        msg = "legacy analysis requires all sixteen canonical history checkpoints"
        raise ValueError(msg)
    rows = []
    for h, measurement in enumerate(measurements):
        if measurement.estimates.shape != (2, 3, 128):
            msg = "legacy analysis requires the unchanged two-regime 128-redraw design"
            raise ValueError(msg)
        held = bool(
            measurement.active_domain_violations == 0
            and measurement.maximum_solver_bound <= SOLVE_TARGET
            and measurement.minimum_capacity > 0
            and measurement.minimum_baseline_price > 0
            and measurement.minimum_opportunity > 0
        )
        for regime, name in enumerate(("common", "changing")):
            oracle = measurement.population_beta[0 if regime == 0 else 3]
            for b, budget in enumerate(BUDGETS):
                for replicate, estimate in enumerate(measurement.estimates[regime, b]):
                    defined = bool(np.isfinite(estimate))
                    rows.append(
                        {
                            "scenario_id": "legacy_A",
                            "history": h,
                            "regime": name,
                            "budget": budget,
                            "replicate": replicate,
                            "beta": float(estimate) if defined else None,
                            "defined": defined,
                            "sign": sign(float(estimate), RESOLUTION)
                            if defined
                            else None,
                            "oracle_beta": float(oracle),
                            "oracle_sign": sign(float(oracle), RESOLUTION),
                            "operating_beta": measurement.beta0,
                            "operating_sign": sign(measurement.beta0, RESOLUTION),
                            "tangent_beta": measurement.affine_beta,
                            "measurement_error": float(estimate - oracle)
                            if defined
                            else None,
                            "operating_error": float(estimate - measurement.beta0)
                            if defined
                            else None,
                            "reporting_distortion": float(oracle - measurement.beta0),
                            "q": None,
                            "d": None,
                            "oracle_q": None,
                            "oracle_d": None,
                            "moment_unavailability_reason": (
                                "not_saved_in_historical_checkpoint"
                            ),
                            "sign_resolution_scope": "Legacy beta absolute 1e-12; "
                            "numerator error cannot be reconstructed.",
                            "execution_succeeded": True,
                            "premises_held": held,
                        }
                    )
    return rows


def legacy_directional(measurements: Sequence[HistoryMeasurement]) -> dict[str, Any]:
    """Preserve historical literal-positive diagnostic independently of new signs."""
    array = np.stack([item.estimates for item in measurements])
    cells = []
    for regime in range(2):
        row = []
        for budget in range(3):
            sample = array[:, regime, budget]
            positive = np.isfinite(sample) & (sample > 0)
            stats = _rate(positive.astype(np.float64))
            row.append(
                {
                    "positive": stats["numerator"],
                    "total": stats["denominator"],
                    "frequency": stats["rate"],
                    "mcse": stats["mcse"],
                    "undefined": int((~np.isfinite(sample)).sum()),
                }
            )
        cells.append(row)
    return {
        "cells": cells,
        "reporting_only_positive_histories": sum(
            bool(item.reporting_only_beta[3] > 0) for item in measurements
        ),
    }
