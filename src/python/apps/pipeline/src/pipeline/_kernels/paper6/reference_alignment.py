"""Frozen-reference directional moments and dated article measurement.

The calendar is fixed. Complete-triple first projections are conditional iid
firm-history benchmarks; covariance consistency and nondegeneracy remain
obligations, and no population confidence coverage is asserted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from numpy.typing import ArrayLike

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]
CUBE_AXES = 3
MATRIX_AXES = 2
MIN_FIRMS = 2
MIN_REDRAWS = 2
RECOVERY_THRESHOLD = 0.25
MIN_TRIPLE_FIRMS = 4


def centered_references(values: FloatArray) -> FloatArray:
    """Center finite off-diagonal coordinates within each target-year vector."""
    values = np.asarray(values, dtype=np.float64)
    if (
        values.ndim != CUBE_AXES
        or values.shape[1] != values.shape[2]
        or values.shape[1] < MIN_FIRMS
        or not values.shape[0]
    ):
        msg = "expected nonempty [year,firm,reference] with at least two firms"
        raise ValueError(msg)
    mask = ~np.eye(values.shape[1], dtype=bool)
    if not np.isfinite(values[:, mask]).all():
        msg = "off-diagonal coordinates must be finite"
        raise ValueError(msg)
    included = np.where(mask, values, 0)
    mean = included.sum(axis=-1, keepdims=True) / (values.shape[1] - 1)
    return np.where(mask, included - mean, 0)


@dataclass(frozen=True)
class DirectionalMoments:
    """Equal-year moments; the pooled slope weights years by financial dispersion."""

    numerator: FloatArray
    denominator: FloatArray

    @property
    def beta(self) -> float | None:
        """An absent pooled denominator produces an explicit undefined estimate."""
        denominator = float(self.denominator.sum())
        return None if denominator == 0 else float(self.numerator.sum() / denominator)


def annual_moments(financial: FloatArray, approach: FloatArray) -> DirectionalMoments:
    """Compute T81's centered moments without treating pair coordinates as iid."""
    if financial.shape != approach.shape:
        msg = "financial and approach coordinates must have matching shapes"
        raise ValueError(msg)
    rc, ac = centered_references(financial), centered_references(approach)
    mask = ~np.eye(rc.shape[1], dtype=bool)
    return DirectionalMoments(
        np.mean((rc * ac)[:, mask], axis=1),
        np.mean((rc * rc)[:, mask], axis=1),
    )


def pooled_moments(distances: FloatArray, financial: FloatArray) -> DirectionalMoments:
    """Distances use the SAME origin reference law at every annual date."""
    if distances.shape != (financial.shape[0] + 1, *financial.shape[1:]):
        msg = "expected D[T+1,N,N] and R[T,N,N]"
        raise ValueError(msg)
    return annual_moments(financial, distances[:-1] - distances[1:])


def pool_dated_laws(laws: FloatArray, counts: IntArray) -> FloatArray:
    """Pool monthly laws using the exact date counts also passed to financial tails.

    Axes are [year,month,firm,atom] and [year,month,firm]. Each firm-year
    must have positive total article count. No independent date redraw occurs.
    """
    if (
        laws.ndim != CUBE_AXES + 1
        or counts.shape != laws.shape[:-1]
        or not np.issubdtype(counts.dtype, np.integer)
        or np.any(counts < 0)
        or not np.isfinite(laws).all()
        or np.any(laws < 0)
        or not np.allclose(laws.sum(axis=-1), 1, rtol=1e-12, atol=1e-12)
        or np.any(counts.sum(axis=1) <= 0)
    ):
        msg = "pooling requires monthly probability laws and positive annual counts"
        raise ValueError(msg)
    return np.sum(laws * counts[..., None], axis=1) / counts.sum(axis=1)[..., None]


def financial_tails(
    article_dates: FloatArray,
    counts: IntArray,
    news_dates: FloatArray,
    financial_news: FloatArray,
) -> FloatArray:
    """Strict news-before-article readout with target-specific empirical date mass.

    For one year, counts are [date,target] and news is [news_date,reference].
    Zero target tails are valid and retained; self coordinates are NaN.
    """
    if (
        financial_news.ndim != MATRIX_AXES
        or article_dates.ndim != 1
        or news_dates.ndim != 1
        or counts.shape != (len(article_dates), financial_news.shape[1])
        or financial_news.shape[0] != len(news_dates)
        or not np.issubdtype(counts.dtype, np.integer)
        or np.any(counts < 0)
        or np.any(counts.sum(axis=0) <= 0)
        or not np.isfinite(article_dates).all()
        or not np.isfinite(news_dates).all()
        or not np.isfinite(financial_news).all()
    ):
        msg = "financial tails require finite dated news and positive target counts"
        raise ValueError(msg)
    mass = counts / counts.sum(axis=0, keepdims=True)
    result = strict_tail_readout(article_dates, mass, news_dates, financial_news)
    np.fill_diagonal(result, np.nan)
    return result


def strict_tail_readout(
    article_dates: ArrayLike,
    article_mass: ArrayLike,
    news_dates: ArrayLike,
    financial_news: ArrayLike,
) -> FloatArray:
    """Read target-specific probability masses with strict news-before-article timing.

    Mass has [article date,target] axes. This also admits fractional redraw
    weights from the historical empirical adapters. Date inputs must use the
    same numerical calendar units; adapters convert calendar days explicitly.
    """
    dates = finite(article_dates, "article dates", 1)
    news = np.asarray(news_dates, dtype=np.float64)
    mass = finite(article_mass, "article mass", 2)
    financial = np.asarray(financial_news, dtype=np.float64)
    if (
        news.ndim != 1
        or financial.ndim != MATRIX_AXES
        or not np.isfinite(news).all()
        or not np.isfinite(financial).all()
        or mass.shape[0] != len(dates)
        or financial.shape[0] != len(news)
        or np.any(mass < 0)
        or not np.allclose(mass.sum(axis=0), 1, rtol=1e-12, atol=1e-12)
    ):
        msg = "financial readout requires dated news and target probability mass"
        raise ValueError(msg)
    tails = (dates[:, None] > news[None, :]).T @ mass
    return tails.T @ financial


@dataclass(frozen=True)
class TripleProjection:
    """Finite complete-triple identity, all-role deletions and first projections."""

    moments: DirectionalMoments
    numerator_influence: FloatArray
    denominator_influence: FloatArray
    deletion_numerator: FloatArray
    deletion_denominator: FloatArray


def triple_projection(financial: FloatArray, approach: FloatArray) -> TripleProjection:
    """Evaluate every unordered triple with every firm in the target role."""
    moments = annual_moments(financial, approach)
    members, kq, kd = complete_triple_moments(financial, approach)
    projection = vector_projection(members, kq, kd, financial.shape[1])
    return TripleProjection(
        moments,
        projection["phi_q"],
        projection["phi_d"],
        projection["deletion_q"],
        projection["deletion_d"],
    )


@dataclass(frozen=True)
class RecoverySummary:
    """Balanced-history Monte Carlo error against the common-reporting oracle."""

    bias: float | None
    bias_mcse: float | None
    normalized_rmse: float | None
    rmse_mcse: float | None
    undefined: int
    recovered: bool
    rmse: float | None = None
    mae: float | None = None
    normalization_reason: str | None = None


def recovery_summary(estimates: FloatArray, oracle: FloatArray) -> RecoverySummary:
    """Use a stratified delta-method MCSE; undefined draws never disappear.

    Rows are enumerated histories and columns independent redraws. If any draw
    is undefined, recovery is undefined and cannot pass the declared convention.
    """
    if (
        estimates.ndim != MATRIX_AXES
        or oracle.shape != (len(estimates),)
        or estimates.shape[1] < MIN_REDRAWS
        or not np.isfinite(oracle).all()
    ):
        msg = "recovery requires finite history oracles and repeated panels"
        raise ValueError(msg)
    undefined = int(np.count_nonzero(~np.isfinite(estimates)))
    if undefined:
        return RecoverySummary(
            None,
            None,
            None,
            None,
            undefined,
            recovered=False,
            normalization_reason="undefined_estimates",
        )
    error = estimates - oracle[:, None]
    squared = error**2
    scale = float(np.sqrt(np.mean(oracle**2)))
    rmse = float(np.sqrt(np.mean(squared)))
    variance = float(
        np.sum(np.var(squared, axis=1, ddof=1))
        / (len(oracle) ** 2 * estimates.shape[1])
    )
    mcse = (
        None
        if scale == 0
        else 0.0
        if rmse == 0
        else float(np.sqrt(variance) / (2 * rmse * scale))
    )
    normalized = None if scale == 0 else rmse / scale
    bias_mcse = float(
        np.sqrt(
            np.sum(np.var(error, axis=1, ddof=1))
            / (len(oracle) ** 2 * estimates.shape[1])
        )
    )
    return RecoverySummary(
        float(np.mean(error)),
        bias_mcse,
        normalized,
        mcse,
        0,
        recovered=bool(
            normalized is not None
            and mcse is not None
            and normalized + 2 * mcse <= RECOVERY_THRESHOLD
        ),
        rmse=rmse,
        mae=float(np.mean(np.abs(error))),
        normalization_reason="zero_oracle_scale" if scale == 0 else None,
    )


class FixedLinearGmm(TypedDict):
    """Numerical fixed-weight minimizer and rank diagnostics."""

    theta: FloatArray
    moment_residual: FloatArray
    criterion: float
    rank: int
    condition_number: float
    weight_condition_number: float


class CovarianceCandidate(TypedDict):
    """Conditional candidate requiring iid histories and covariance consistency."""

    matrix: FloatArray
    eigenvalues: FloatArray
    max_abs_influence_mean: float
    scope: str


class KernelProjection(TypedDict):
    """All-role finite kernel projections and recentered deletion moments."""

    c_n: float
    q: FloatArray
    d: FloatArray
    phi_q: FloatArray
    phi_d: FloatArray
    deletion_q: FloatArray
    deletion_d: FloatArray


class FiniteMoment(TypedDict):
    """Pooled moments with explicit zero-denominator status."""

    beta: float | None
    theta: float
    second_moment: float


class Contribution(TypedDict):
    """One finite numerator contribution."""

    label: str
    contribution: float


class Share(TypedDict):
    """One finite financial dispersion share."""

    label: str
    share: float


class ShareSummary(TypedDict):
    """Descriptive concentration of financial dispersion."""

    shares: list[Share]
    herfindahl: float


def finite(value: ArrayLike, name: str, ndim: int) -> FloatArray:
    """Admit finite, nonempty arrays without changing global precision settings."""
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim or not result.size or not np.isfinite(result).all():
        msg = f"{name} must be a finite nonempty {ndim}-D array"
        raise ValueError(msg)
    return result


def fixed_linear_gmm(
    q: ArrayLike, design: ArrayLike, weight: ArrayLike
) -> FixedLinearGmm:
    """Minimize (q-B theta)' W (q-B theta) for supplied fixed SPD W.

    Cholesky whitening and a QR solve avoid explicit inverses and normal-equation
    squaring of conditioning. The rank and condition number refer to the whitened
    design. A weighting matrix estimated from these data is outside this API.
    """
    q, design, weight = (
        finite(q, "q", 1),
        finite(design, "B", 2),
        finite(weight, "W", 2),
    )
    k, p = design.shape
    if q.shape != (k,) or weight.shape != (k, k):
        msg = "GMM moment/design/weight dimensions disagree"
        raise ValueError(msg)
    if not np.allclose(weight, weight.T, rtol=1e-10, atol=1e-12):
        msg = "fixed GMM weight must be symmetric positive definite"
        raise ValueError(msg)
    try:
        lower = np.linalg.cholesky(weight)
    except np.linalg.LinAlgError as error:
        msg = "fixed GMM weight must be symmetric positive definite"
        raise ValueError(msg) from error
    transformed_design, transformed_q = lower.T @ design, lower.T @ q
    if (
        not np.isfinite(transformed_design).all()
        or not np.isfinite(transformed_q).all()
    ):
        msg = "overflow in fixed-weight GMM whitening"
        raise ValueError(msg)
    singular = np.linalg.svd(transformed_design, compute_uv=False)
    rank = int(np.linalg.matrix_rank(transformed_design))
    if rank != p:
        msg = "rank-deficient GMM design: unrestricted theta not identified"
        raise ValueError(msg)
    orthogonal, upper = np.linalg.qr(transformed_design, mode="reduced")
    theta = np.linalg.solve(upper, orthogonal.T @ transformed_q)
    residual = q - design @ theta
    weighted_residual = lower.T @ residual
    if not np.isfinite(theta).all() or not np.isfinite(residual).all():
        msg = "nonfinite fixed-weight GMM solution"
        raise ValueError(msg)
    return {
        "theta": theta,
        "moment_residual": residual,
        "criterion": float(weighted_residual @ weighted_residual),
        "rank": rank,
        "condition_number": float(singular[0] / singular[-1]),
        "weight_condition_number": float(np.linalg.cond(weight)),
    }


def reference_arrays(
    financial: ArrayLike, approach: ArrayLike
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    """Exclude only self coordinates; refuse any missing off-diagonal value."""
    financial, approach = (
        np.asarray(financial, dtype=np.float64),
        np.asarray(approach, dtype=np.float64),
    )
    if (
        financial.ndim != CUBE_AXES
        or not financial.shape[0]
        or financial.shape != approach.shape
        or financial.shape[1] != financial.shape[2]
        or financial.shape[1] < MIN_TRIPLE_FIRMS
    ):
        msg = "expected matching R/A[T,N,N], with N >= 4"
        raise ValueError(msg)
    offdiag = ~np.eye(financial.shape[1], dtype=bool)
    if (
        not np.isfinite(financial[:, offdiag]).all()
        or not np.isfinite(approach[:, offdiag]).all()
    ):
        msg = "missing off-diagonal financial or approach coordinate"
        raise ValueError(msg)
    # The diagonal is never a reference; its NaNs must not contaminate products.
    return (
        np.where(offdiag[None], financial, 0.0),
        np.where(offdiag[None], approach, 0.0),
        offdiag,
    )


def complete_triple_moments(
    financial: ArrayLike, approach: ArrayLike
) -> tuple[IntArray, FloatArray, FloatArray]:
    """Return all C(N,3) symmetric kernels, retaining the annual vector."""
    financial, approach, _ = reference_arrays(financial, approach)
    t, n, _ = financial.shape
    members = np.asarray(list(combinations(range(n), 3)), dtype=np.int64)
    kq, kd = np.zeros((len(members), t)), np.zeros((len(members), t))
    for focal, first, second in ((0, 1, 2), (1, 0, 2), (2, 0, 1)):
        i, j, k = members[:, focal], members[:, first], members[:, second]
        dr = (financial[:, i, j] - financial[:, i, k]).T
        da = (approach[:, i, j] - approach[:, i, k]).T
        kq += dr * da / 3
        kd += dr * dr / 3
    return members, kq, kd


def conditional_covariance(influence: ArrayLike) -> CovarianceCandidate:
    """Cov(IF)/N for a specified, unverified conditional iid-history benchmark."""
    influence = finite(influence, "firm-history influence", 2)
    n = len(influence)
    if n < MIN_FIRMS:
        msg = "two firm histories are required for a covariance candidate"
        raise ValueError(msg)
    centered = influence - influence.mean(axis=0)
    covariance = centered.T @ centered / (n * (n - 1))
    return {
        "matrix": covariance,
        "eigenvalues": np.asarray(np.linalg.eigvalsh(covariance), dtype=np.float64),
        "max_abs_influence_mean": float(np.max(np.abs(influence.mean(axis=0)))),
        "scope": (
            "Unverified conditional iid firm histories; fixed calendar environment; "
            "nondegenerate square-integrable first projection and consistent "
            "covariance estimation required. No coverage claim."
        ),
    }


def vector_projection(
    members: IntArray, kq: ArrayLike, kd: ArrayLike, n: int
) -> KernelProjection:
    """Finite plug-in order-three projections and exact induced deletions."""
    if n < MIN_TRIPLE_FIRMS:
        msg = "at least four firm histories are required for role deletions"
        raise ValueError(msg)
    kq, kd = finite(kq, "q kernels", 2), finite(kd, "d kernels", 2)
    expected = np.asarray(list(combinations(range(n), 3)), dtype=np.int64)
    if (
        not np.array_equal(members, expected)
        or kq.shape != kd.shape
        or len(kq) != len(expected)
    ):
        msg = "requires every sorted unordered triple exactly once"
        raise ValueError(msg)
    if np.any(kd < 0):
        msg = "squared financial-contrast kernels must be nonnegative"
        raise ValueError(msg)
    uq, ud = kq.mean(axis=0), kd.mean(axis=0)
    incidence_q, incidence_d = np.zeros((n, kq.shape[1])), np.zeros((n, kd.shape[1]))
    for role in range(3):
        np.add.at(incidence_q, members[:, role], kq)
        np.add.at(incidence_d, members[:, role], kd)
    c_n = (n - 2) / (2 * (n - 1))
    incident_count = comb(n - 1, 2)
    phi_q = 3 * c_n * (incidence_q / incident_count - uq)
    phi_d = 3 * c_n * (incidence_d / incident_count - ud)
    remaining_count = comb(n - 1, 3)
    c_remaining = (n - 3) / (2 * (n - 2))
    deletion_q = c_remaining * (kq.sum(axis=0) - incidence_q) / remaining_count
    deletion_d = c_remaining * (kd.sum(axis=0) - incidence_d) / remaining_count
    return {
        "c_n": c_n,
        "q": c_n * uq,
        "d": c_n * ud,
        "phi_q": phi_q,
        "phi_d": phi_d,
        "deletion_q": deletion_q,
        "deletion_d": deletion_d,
    }


def fixed_common_slope_influence(
    q: ArrayLike, d: ArrayLike, phi_q: ArrayLike, phi_d: ArrayLike, weight: ArrayLike
) -> tuple[float, FloatArray, FloatArray]:
    """Full fixed-W delta derivative of a common-slope moment-ratio minimizer.

    Under heterogeneous annual slopes, the nonzero residual multiplies changes
    in d. Omitting this term silently assumes correct specification. W is fixed;
    the influence of an estimated weighting matrix is deliberately not supplied.
    """
    q, d = finite(q, "annual q", 1), finite(d, "annual d", 1)
    phi_q, phi_d = finite(phi_q, "q influence", 2), finite(phi_d, "d influence", 2)
    weight = finite(weight, "fixed weight", 2)
    fit = fixed_linear_gmm(q, d[:, None], weight)
    if phi_q.shape != phi_d.shape or phi_q.shape[1] != len(q):
        msg = "annual moment influence shapes disagree"
        raise ValueError(msg)
    beta = float(fit["theta"][0])
    denominator = float(d @ weight @ d)
    full = (phi_q @ weight @ d + phi_d @ weight @ (q - 2 * beta * d)) / denominator
    correct_specification_only = (phi_q - beta * phi_d) @ weight @ d / denominator
    return beta, full, correct_specification_only


def analyze(
    financial: FloatArray, approach: FloatArray
) -> tuple[dict[str, object], dict[str, FloatArray | IntArray]]:
    """Compute finite identities and explicitly scoped covariance candidates."""
    moments = annual_moments(financial, approach)
    q, d = moments.numerator, moments.denominator
    if np.any(d <= 0):
        msg = "each annual coefficient requires positive financial dispersion"
        raise ValueError(msg)
    n = np.asarray(financial).shape[1]
    members, kq, kd = complete_triple_moments(financial, approach)
    projection = vector_projection(members, kq, kd, n)
    np.testing.assert_allclose(projection["q"], q, rtol=0, atol=1e-12)
    np.testing.assert_allclose(projection["d"], d, rtol=0, atol=1e-12)
    phi_q, phi_d = projection["phi_q"], projection["phi_d"]
    annual_beta = q / d
    pooled = float(q.sum() / d.sum())
    annual_if = (phi_q - annual_beta * phi_d) / d
    pooled_if = (phi_q - pooled * phi_d).sum(axis=1) / d.sum()
    weight = np.eye(len(q))
    fitted = fixed_linear_gmm(q, d[:, None], weight)
    common, common_if, restricted_if = fixed_common_slope_influence(
        q, d, phi_q, phi_d, weight
    )
    all_influence = np.column_stack([annual_if, pooled_if, common_if])
    covariance = conditional_covariance(all_influence)
    moment_covariance = conditional_covariance(np.column_stack([phi_q, phi_d]))
    effective_weights = d * d / (d @ d)
    deletion_denominator = projection["deletion_d"].sum(axis=1)
    deletion_beta = np.divide(
        projection["deletion_q"].sum(axis=1),
        deletion_denominator,
        out=np.full(n, np.nan),
        where=deletion_denominator != 0,
    )
    identity_error = {
        "maximum_annual_q_absolute_error": float(np.max(np.abs(projection["q"] - q))),
        "maximum_annual_d_absolute_error": float(np.max(np.abs(projection["d"] - d))),
        "maximum_phi_q_mean_absolute": float(np.max(np.abs(phi_q.mean(axis=0)))),
        "maximum_phi_d_mean_absolute": float(np.max(np.abs(phi_d.mean(axis=0)))),
    }
    summary = {
        "annual_q": q.tolist(),
        "annual_d": d.tolist(),
        "annual_beta": annual_beta.tolist(),
        "original_pooled_beta": pooled,
        "original_pooled_effective_year_weights": (d / d.sum()).tolist(),
        "fixed_identity_common_slope": {
            "beta": common,
            "fixed_weight": weight.tolist(),
            "moment_residual": fitted["moment_residual"].tolist(),
            "criterion": fitted["criterion"],
            "criterion_scope": (
                "Descriptive fixed-I moment discrepancy; "
                "no Hansen J reference or p-value."
            ),
            "effective_year_weights": effective_weights.tolist(),
            "design_rank": fitted["rank"],
            "whitened_design_condition_number": fitted["condition_number"],
            "full_if_minus_correct_specification_only_rms": float(
                np.sqrt(np.mean((common_if - restricted_if) ** 2))
            ),
            "full_if_minus_correct_specification_only_maximum_absolute": float(
                np.max(np.abs(common_if - restricted_if))
            ),
        },
        "complete_triple_count": len(members),
        "kernel_order": 3,
        "normalization_c_n": projection["c_n"],
        "identities": identity_error,
        "conditional_candidate_coefficient_covariance": {
            **covariance,
            "matrix": covariance["matrix"].tolist(),
            "eigenvalues": covariance["eigenvalues"].tolist(),
            "order": [f"annual_{i}" for i in range(len(q))]
            + ["original_pooled", "fixed_identity_common"],
        },
        "conditional_candidate_moment_covariance": {
            **moment_covariance,
            "matrix": moment_covariance["matrix"].tolist(),
            "eigenvalues": moment_covariance["eigenvalues"].tolist(),
            "order": [f"q_{i}" for i in range(len(q))]
            + [f"d_{i}" for i in range(len(q))],
        },
        "first_projection_status": (
            "Nonzero empirical projections do not establish population "
            "nondegeneracy, conditional iid histories or frequentist coverage."
        ),
        "inference_scope": (
            "Fixed-calendar conditional-iid-history Hajek candidate only; "
            "no normal p-values or confidence bounds. Article histories, "
            "cross-role spillovers and common omitted shocks can invalidate "
            "these assumptions."
        ),
    }
    arrays: dict[str, FloatArray | IntArray] = {
        "members": members,
        "Kq": kq,
        "Kd": kd,
        "q": q,
        "d": d,
        "phi_q": phi_q,
        "phi_d": phi_d,
        "annual_beta_if": annual_if,
        "pooled_beta_if": pooled_if,
        "common_beta_full_if": common_if,
        "common_beta_correct_specification_only_if": restricted_if,
        "deletion_q": projection["deletion_q"],
        "deletion_d": projection["deletion_d"],
        "deletion_beta": deletion_beta,
    }
    return summary, arrays


def _arrays(
    distances: ArrayLike, returns: ArrayLike
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    distance = np.asarray(distances, dtype=np.float64)
    financial = np.asarray(returns, dtype=np.float64)
    if (
        distance.ndim != CUBE_AXES
        or financial.ndim != CUBE_AXES
        or not financial.shape[0]
        or distance.shape[0] != financial.shape[0] + 1
        or distance.shape[1] != distance.shape[2]
        or financial.shape[1:] != distance.shape[1:]
        or distance.shape[1] < MIN_FIRMS
    ):
        msg = "expected D[T+1,N,N] and R[T,N,N] with N >= 2"
        raise ValueError(msg)
    n = distance.shape[1]
    offdiag = ~np.eye(n, dtype=bool)
    if (
        not np.isfinite(distance[:, offdiag]).all()
        or not np.isfinite(financial[:, offdiag]).all()
    ):
        msg = "all off-diagonal observations must be finite"
        raise ValueError(msg)
    return distance, financial, offdiag


def reference_components(
    distances: ArrayLike, returns: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, NDArray[np.bool_]]:
    """Target-year centering shared by all active finite estimator adapters."""
    distance, financial, offdiag = _arrays(distances, returns)
    centered = centered_references(financial)
    adjustment = centered_references(distance[:-1] - distance[1:])
    return centered, adjustment, centered * adjustment, centered * centered, offdiag


def moment(distances: ArrayLike, returns: ArrayLike) -> FiniteMoment:
    """Equal-firm-year, equal-off-diagonal-reference through-origin moment."""
    _, _, product, square, offdiag = reference_components(distances, returns)
    return _aggregate(product, square, offdiag)


def _aggregate(
    product: FloatArray, square: FloatArray, offdiag: NDArray[np.bool_]
) -> FiniteMoment:
    denominator = float(np.mean(square[:, offdiag]))
    theta = float(np.mean(product[:, offdiag]))
    return {
        "beta": None if denominator == 0 else float(theta / denominator),
        "theta": theta,
        "second_moment": denominator,
    }


def _contributions(
    values: FloatArray, axis: int, labels: Iterable[object]
) -> list[Contribution]:
    total_cells = values.shape[0] * values.shape[1] * (values.shape[2] - 1)
    return [
        {
            "label": str(label),
            "contribution": float(np.sum(np.take(values, i, axis=axis)) / total_cells),
        }
        for i, label in enumerate(labels)
    ]


def _shares(square: FloatArray, axis: int, labels: Sequence[object]) -> ShareSummary:
    values = np.array(
        [np.sum(np.take(square, i, axis=axis)) for i in range(len(labels))]
    )
    total = float(values.sum())
    shares = np.zeros_like(values) if total == 0 else values / total
    return {
        "shares": [
            {"label": str(label), "share": float(share)}
            for label, share in zip(labels, shares, strict=True)
        ],
        "herfindahl": float(shares @ shares),
    }


def summarize(
    distances: ArrayLike,
    returns: ArrayLike,
    firm_ids: Sequence[str] | NDArray[np.str_],
    years: Sequence[int] = (2019, 2020, 2021, 2022),
) -> dict[str, object]:
    """Summarize the finite projection and descriptive role/year influence."""
    distance, financial, _ = _arrays(distances, returns)
    ids = np.asarray(firm_ids).astype(str)
    years = tuple(years)
    if len(ids) != distance.shape[1] or len(years) != financial.shape[0]:
        msg = "firm/year labels do not match the arrays"
        raise ValueError(msg)
    centered, adjustment, product, square, offdiag = reference_components(
        distance, financial
    )
    primary = moment(distance, financial)
    n = len(ids)
    annual_theta = _contributions(product, 0, years)
    annual_second = _contributions(square, 0, years)
    target_theta = _contributions(product, 1, ids)
    reference_theta = _contributions(product, 2, ids)
    target_denominator = _shares(square, 1, ids.tolist())
    reference_denominator = _shares(square, 2, ids.tolist())

    leave_firm = []
    for removed, ticker in enumerate(ids):
        keep = np.arange(n) != removed
        estimate = moment(distance[:, keep][:, :, keep], financial[:, keep][:, :, keep])
        leave_firm.append({"firm": ticker, **estimate})
    leave_year = []
    for removed, year in enumerate(years):
        keep = np.arange(len(years)) != removed
        estimate = _aggregate(product[keep], square[keep], offdiag)
        leave_year.append({"year": int(year), **estimate})

    return {
        **primary,
        "firms": n,
        "years": list(map(int, years)),
        "observations": int(len(years) * n * (n - 1)),
        "vector_count": int(len(years) * n),
        "directional_coordinate_count": int(len(years) * n * (n - 1)),
        "annual_theta_contributions": annual_theta,
        "annual_second_moment_contributions": annual_second,
        "target_theta_contributions": target_theta,
        "reference_theta_contributions": reference_theta,
        "target_denominator": target_denominator,
        "reference_denominator": reference_denominator,
        "leave_one_firm_out": leave_firm,
        "leave_one_year_out": leave_year,
        "scale": {
            "centered_return_rms": float(np.sqrt(np.mean(centered[:, offdiag] ** 2))),
            "reference_centered_adjustment_rms": float(
                np.sqrt(np.mean(adjustment[:, offdiag] ** 2))
            ),
            "reference_centered_adjustment_mean_abs": float(
                np.mean(np.abs(adjustment[:, offdiag]))
            ),
        },
    }
