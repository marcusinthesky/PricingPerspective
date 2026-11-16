"""Audit the retained iid-firm first-projection candidate under stated targets.

The contraction below is algebraically the existing complete-triple projection,
not another covariance estimator. Conditional article resampling does not make
the firms iid or turn this candidate into article-selection uncertainty.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import TYPE_CHECKING, TypedDict

import numpy as np

from pipeline._kernels.paper6.reference_alignment import (
    annual_moments,
    centered_references,
    conditional_covariance,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

type FloatArray = NDArray[np.float64]

ONE_SIDED_Z = NormalDist().inv_cdf(0.95)
FIXTURE_SEED = 620260000
FIXTURE_REPLICATIONS = 2500
FIXTURE_FIRMS = 16
FIXTURE_VARIANCE_FACTOR = 1.0125
MIN_FIRMS = 4
MIN_REPLICATIONS = 2


class Candidate(TypedDict):
    """Pooled ratio and the unvalidated iid-firm covariance candidate."""

    beta: float | None
    numerator: float
    denominator: float
    candidate_se: float | None
    candidate_lower_bound: float | None
    status: str
    reason: str | None


class Rate(TypedDict):
    """Independent-replication rate with the full attempted denominator."""

    count: int
    attempts: int
    rate: float
    mcse: float


class BoundDiagnostics(TypedDict):
    """Numerical diagnostics whose sampling interpretation is supplied by the caller."""

    target: float
    attempts: int
    undefined_estimates: int
    unavailable_candidate_se: int
    degenerate_or_negative_candidate_se: int
    usable_bounds: int
    empirical_dispersion: float | None
    average_candidate_se: float | None
    candidate_se_to_dispersion: float | None
    target_containment: Rate
    positive_bound: Rate
    average_lower_margin: float | None
    interval_length: None
    interval_length_reason: str
    one_sided_z: float
    rate_denominator: str


def _contracted_projection(product: FloatArray) -> FloatArray:
    """Contract all incident symmetric triples without materializing them.

    For centered rows, the incident kernel sum is
    ((N-2)*row_sum + (N-1)*column_sum + total_sum)/3. Substitution
    into vector_projection's 3*c_N*(incident/choose(N-1,2)-U) gives
    this expression. Both roles and the entire annual vector are retained.
    """
    n = product.shape[1]
    rows = product.sum(axis=2).T
    columns = product.sum(axis=1).T
    totals = product.sum(axis=(1, 2))
    return ((n - 2) * rows + (n - 1) * columns + totals) / (n - 1) ** 2 - 3 * totals / (
        n * (n - 1)
    )


def pooled_candidate(financial: ArrayLike, approach: ArrayLike) -> Candidate:
    """Reuse the pooled-ratio derivative and conditional covariance in O(TN²).

    A zero annual denominator is allowed if the pooled denominator is positive.
    A degenerate first projection is recorded separately from an undefined slope.
    No normal-approximation or empirical eligibility claim is made here.
    """
    financial = np.asarray(financial, dtype=np.float64)
    approach = np.asarray(approach, dtype=np.float64)
    moments = annual_moments(financial, approach)
    if financial.shape[1] < MIN_FIRMS:
        msg = "candidate projection requires at least four firm histories"
        raise ValueError(msg)
    numerator = float(moments.numerator.sum())
    denominator = float(moments.denominator.sum())
    result: Candidate = {
        "beta": moments.beta,
        "numerator": numerator,
        "denominator": denominator,
        "candidate_se": None,
        "candidate_lower_bound": None,
        "status": "zero_denominator",
        "reason": "The pooled financial denominator is zero.",
    }
    if moments.beta is None:
        return result
    rc, ac = centered_references(financial), centered_references(approach)
    phi_q, phi_d = _contracted_projection(rc * ac), _contracted_projection(rc * rc)
    influence = (phi_q - moments.beta * phi_d).sum(axis=1) / denominator
    covariance = conditional_covariance(influence[:, None])["matrix"]
    standard_error = float(np.sqrt(covariance[0, 0]))
    result["candidate_se"] = standard_error
    if standard_error == 0:
        result["status"] = "degenerate_projection"
        result["reason"] = "The empirical first projection has zero dispersion."
        return result
    result["candidate_lower_bound"] = moments.beta - ONE_SIDED_Z * standard_error
    result["status"] = "defined"
    result["reason"] = None
    return result


def iid_fixture(
    x: ArrayLike, error: ArrayLike, target_effect: ArrayLike, *, beta: float
) -> tuple[FloatArray, FloatArray]:
    """Known-target four-year iid histories with persistent cross-role dependence.

    X and E are independent standard normals and U is an independent Rademacher
    variable per firm. R[t,i,j]=c[t]*X[j], and
    A=beta*R+g[t]*(E[j]+0.5*E[i]*X[j])+t*U[i]. The last term is
    annihilated by reference centering. c=(1,2,1,2), g=(1,1,2,2).
    The population ratio is beta; its first influence is .9*E*(X+.5),
    with asymptotic variance 1.0125/N. This is a mathematical fixture,
    not an equilibrium or a model of the corporate sampling process.
    """
    x, error, target_effect = (
        np.asarray(value, dtype=np.float64) for value in (x, error, target_effect)
    )
    if (
        x.ndim != 1
        or len(x) < MIN_FIRMS
        or error.shape != x.shape
        or target_effect.shape != x.shape
        or not np.isfinite([x, error, target_effect]).all()
        or not np.isfinite(beta)
    ):
        msg = "fixture requires finite matching firm vectors and a finite target"
        raise ValueError(msg)
    c = np.array([1.0, 2.0, 1.0, 2.0])[:, None, None]
    g = np.array([1.0, 1.0, 2.0, 2.0])[:, None, None]
    financial = np.broadcast_to(c * x[None, None, :], (4, len(x), len(x))).copy()
    approach = (
        beta * financial
        + g * (error[None, None, :] + 0.5 * error[None, :, None] * x[None, None, :])
        + np.arange(4)[:, None, None] * target_effect[None, :, None]
    )
    diagonal = np.arange(len(x))
    financial[:, diagonal, diagonal] = np.nan
    approach[:, diagonal, diagonal] = np.nan
    return financial, approach


def iid_replications(
    *,
    replications: int = FIXTURE_REPLICATIONS,
    firms: int = FIXTURE_FIRMS,
    seed: int = FIXTURE_SEED,
) -> dict[str, FloatArray]:
    """Generate fixed-count independent histories, pairing zero/positive targets."""
    if replications < MIN_REPLICATIONS or firms < MIN_FIRMS:
        msg = "audit requires at least two replications and four firms"
        raise ValueError(msg)
    estimates = np.full((2, replications), np.nan)
    standard_errors = np.full_like(estimates, np.nan)
    lower_bounds = np.full_like(estimates, np.nan)
    numerators = np.full_like(estimates, np.nan)
    denominators = np.full_like(estimates, np.nan)
    analytic_influence_sd = np.full(replications, np.nan)
    for replicate in range(replications):
        rng = np.random.default_rng(np.random.SeedSequence([seed, replicate]))
        x, error = rng.normal(size=(2, firms))
        target_effect = rng.choice([-1.0, 1.0], size=firms)
        analytic_influence_sd[replicate] = np.std(
            0.9 * error * (x + 0.5), ddof=1
        ) / np.sqrt(firms)
        for index, beta in enumerate((0.0, 0.5)):
            financial, approach = iid_fixture(x, error, target_effect, beta=beta)
            candidate = pooled_candidate(financial, approach)
            for destination, key in (
                (estimates, "beta"),
                (standard_errors, "candidate_se"),
                (lower_bounds, "candidate_lower_bound"),
                (numerators, "numerator"),
                (denominators, "denominator"),
            ):
                value = candidate[key]
                if isinstance(value, float):
                    destination[index, replicate] = value
    return {
        "targets": np.array([0.0, 0.5]),
        "estimates": estimates,
        "candidate_se": standard_errors,
        "candidate_lower_bound": lower_bounds,
        "numerator": numerators,
        "denominator": denominators,
        "analytic_influence_sample_se": analytic_influence_sd,
    }


def bernoulli_rate(events: ArrayLike, *, attempts: int | None = None) -> Rate:
    """Give counts, full-attempt denominator and iid Monte Carlo uncertainty."""
    events = np.asarray(events, dtype=bool)
    total = len(events) if attempts is None else attempts
    if events.ndim != 1 or total < len(events) or total < MIN_REPLICATIONS:
        msg = "rate requires one-dimensional events and at least two total attempts"
        raise ValueError(msg)
    count = int(events.sum())
    rate = count / total
    return {
        "count": count,
        "attempts": total,
        "rate": rate,
        "mcse": float(np.sqrt(rate * (1 - rate) / (total - 1))),
    }


def bound_diagnostics(
    estimates: ArrayLike, standard_errors: ArrayLike, target: float
) -> BoundDiagnostics:
    """Summarize a candidate bound without granting its sampling interpretation.

    Event rates count unavailable/degenerate bounds as non-events and expose
    their counts, so a failed draw cannot improve the reported rate denominator.
    The upper endpoint is infinity; lower-margin is a finite measure of its
    distance below the estimate, rather than a fictitious finite interval width.
    """
    estimates = np.asarray(estimates, dtype=np.float64)
    standard_errors = np.asarray(standard_errors, dtype=np.float64)
    if (
        estimates.ndim != 1
        or standard_errors.shape != estimates.shape
        or len(estimates) < MIN_REPLICATIONS
        or not np.isfinite(target)
    ):
        msg = "bound diagnostics require matching redraw vectors and finite target"
        raise ValueError(msg)
    defined = np.isfinite(estimates)
    usable = defined & np.isfinite(standard_errors) & (standard_errors > 0)
    lower = estimates - ONE_SIDED_Z * standard_errors
    dispersion = (
        float(np.std(estimates[defined], ddof=1)) if defined.sum() > 1 else None
    )
    average_se = float(np.mean(standard_errors[usable])) if usable.any() else None
    return {
        "target": target,
        "attempts": len(estimates),
        "undefined_estimates": int((~defined).sum()),
        "unavailable_candidate_se": int((~np.isfinite(standard_errors)).sum()),
        "degenerate_or_negative_candidate_se": int(
            (np.isfinite(standard_errors) & (standard_errors <= 0)).sum()
        ),
        "usable_bounds": int(usable.sum()),
        "empirical_dispersion": dispersion,
        "average_candidate_se": average_se,
        "candidate_se_to_dispersion": (
            average_se / dispersion
            if average_se is not None and dispersion is not None and dispersion > 0
            else None
        ),
        "target_containment": bernoulli_rate(usable & (lower <= target)),
        "positive_bound": bernoulli_rate(usable & (lower > 0)),
        "average_lower_margin": (
            float(ONE_SIDED_Z * average_se) if average_se is not None else None
        ),
        "interval_length": None,
        "interval_length_reason": "The candidate interval is [lower, +infinity).",
        "one_sided_z": ONE_SIDED_Z,
        "rate_denominator": "All attempted replications, including unavailable bounds.",
    }
