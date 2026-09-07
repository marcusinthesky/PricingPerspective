"""Energy-test kernels: the mixture weights that maximise the test statistic.

The effective-sample-size normalizer these use lives beside them in
:mod:`jcor.discrepancy._mmd.energy.balancing.effective_size`.
"""

from __future__ import annotations

import jax.numpy as jnp

from jcor.discrepancy._mmd.components import _compute_energy_components
from jcor.discrepancy._mmd.energy.balancing._common import _MAX_DISTANCE_EXPONENT
from jcor.discrepancy._mmd.energy.balancing.effective_size import inverse_variance_m_eff
from jcor.ground.config import (  # noqa: TC001  # runtime annotations
    GroundDistanceSelection,
    resolve_ground_distance,
)
from jcor.ground.metrics import cdist
from jcor.optimize.simplex import pgd_simplex_affine


def _prepare_energy_test_inputs(
    x: jnp.ndarray,
    y_candidates: jnp.ndarray | list[jnp.ndarray],
    metric: GroundDistanceSelection,
    exponent: float,
) -> tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Resolve inputs and compute the shared energy-distance components."""
    resolved_metric = resolve_ground_distance(metric)
    candidates_list = (
        [y_candidates[i] for i in range(y_candidates.shape[0])]
        if isinstance(y_candidates, jnp.ndarray)
        else y_candidates
    )
    sample_sizes = jnp.array([len(candidate) for candidate in candidates_list])
    target_self = jnp.mean(cdist(x, x, metric=resolved_metric) ** exponent)
    candidate_gram, target_cross = _compute_energy_components(
        x, candidates_list, resolved_metric, exponent
    )
    return len(x), sample_sizes, target_self, candidate_gram, target_cross


def _energy_test_quadratic(
    n: int,
    sample_sizes: jnp.ndarray,
    target_self: jnp.ndarray,
    candidate_gram: jnp.ndarray,
    target_cross: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build the simplex QP surrogate used to select the mixture."""
    vertex_scales = (n * sample_sizes) / (n + sample_sizes)
    linear = vertex_scales * (2.0 * target_cross - target_self)
    scale_root = jnp.sqrt(vertex_scales)
    scaled_gram = scale_root[:, None] * scale_root[None, :] * candidate_gram
    return 2.0 * scaled_gram, -linear


def _solve_energy_test_weights(
    quadratic: jnp.ndarray,
    linear: jnp.ndarray,
    solver: str,
    tol: float,
    maxiter: int,
) -> jnp.ndarray:
    """Solve and normalize the simplex weights for the selected QP."""
    if solver not in ("boxosqp", "pgd", "eqcp"):
        message = f"Unknown solver: {solver}"
        raise ValueError(message)
    weights = pgd_simplex_affine(quadratic, linear, maxiter=maxiter, tol=tol)
    weights = jnp.clip(weights, 0.0, 1.0)
    return weights / jnp.sum(weights)


def _resolve_effective_size(
    weights: jnp.ndarray,
    sample_sizes: jnp.ndarray,
    mode: str,
    group_variances: jnp.ndarray | None,
) -> jnp.ndarray:
    """Resolve the requested legacy or inverse-variance effective size."""
    if mode == "linear":
        return jnp.dot(weights, sample_sizes)
    if mode == "inverse_variance":
        return inverse_variance_m_eff(weights, sample_sizes, group_variances)
    message = f"m_eff_mode must be 'inverse_variance' or 'linear', got {mode!r}"
    raise ValueError(message)


def energy_test_kernel(
    x: jnp.ndarray,
    y_candidates: jnp.ndarray | list[jnp.ndarray],
    exponent: float = 1.0,
    metric: GroundDistanceSelection = "euclidean",
    solver: str = "boxosqp",
    tol: float = 1e-6,
    maxiter: int = 1000,
    m_eff_mode: str = "inverse_variance",
    group_variances: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Find optimal mixture weights maximizing energy test statistic.

    Solves a quadratic program to find mixture weights that maximize
    the energy test statistic between sample x and a weighted mixture
    of candidate distributions. The test statistic accounts for the
    effective sample size:

        T(x, mixture) = (n·m_eff)/(n+m_eff) · E(x, mixture)

    The **outer harmonic factor** ``(n·m_eff)/(n+m_eff)`` is the correct
    two-sample energy normalizer (variance ``∝ 1/n + 1/m_eff``; Rizzo-Székely)
    and is *kept*.  Only the value of ``m_eff`` is corrected:

    - ``m_eff_mode="inverse_variance"`` (**default, correct**): the Kish /
      importance-sampling effective size ``1/m_eff = Σ_k w_k²/m_k`` (homoscedastic)
      or ``1/m_eff = (Σ_k w_k² s_k²/m_k)/(Σ_k w_k s_k²)`` (heteroscedastic, via
      ``group_variances``).  See :func:`inverse_variance_m_eff`.
    - ``m_eff_mode="linear"`` (**deprecated, buggy**): the legacy ``m_eff = wᵀm``.
      Retained only to reproduce the historical (over-optimistic) behavior; it is
      correct only at a simplex vertex.

    .. warning::
        ``m_eff`` is the **non-degenerate / studentization normalizer**, *not* an
        exact-null quantity.  Under ``H0`` the statistic is degenerate
        (``n·E_n ⇒ Σ_k λ_k Z_k²``, mixture-dependent eigenvalues); no scalar
        ``m_eff`` is exact there.  For calibrated inference prefer the studentized
        entry points (:func:`jcor.inference.studentized_mixture_statistic`) over
        this analytic scaling.

    This is useful for:
    - Testing x against the "most different" combination of candidates
    - Finding the optimal alternative hypothesis in permutation tests
    - Multi-sample two-sample testing with candidate selection

    The QP maximizes:
        T(x, w) = (n·m_eff)/(n+m_eff) · [2·E[d(x,y)] - E[d(x,x')] - E[d(y,y')]]

    subject to: sum(w) = 1, w >= 0

    Args:
        x: Reference sample, array of shape (n, d).
        y_candidates: Either:
            - List of K candidate distributions with shapes (m_k, d)
            - Array of shape (K, m, d) with uniform sizes
        exponent: Distance exponent α in (0, 2). Default 1.0.
        metric: Distance metric (see cdist for options).
        solver: Optimization solver ("boxosqp" or "eqcp").
        tol: Solver tolerance.
        maxiter: Maximum solver iterations.
        m_eff_mode: Effective-size mode, ``"inverse_variance"`` (default, correct)
            or ``"linear"`` (deprecated legacy ``wᵀm``).
        group_variances: Optional per-candidate variances ``s_k²`` (shape (K,)) for
            the heteroscedastic inverse-variance form.  Ignored for ``"linear"``.

    Returns:
        Tuple of (weights, test_statistic) where:
            weights: Array of shape (K,) with optimal mixture weights
            test_statistic: Maximum achievable energy test statistic

    Raises:
        ValueError: If exponent not in (0, 2).

    Examples:
        >>> # Find which candidate(s) are most different from x
        >>> x = jnp.array([[0.0, 0.0], [0.1, 0.1]])
        >>> cand1 = jnp.array([[0.0, 0.0], [0.1, 0.1]])  # Similar to x
        >>> cand2 = jnp.array([[10.0, 10.0], [11.0, 11.0]])  # Very different
        >>> candidates = [cand1, cand2]
        >>> weights, stat = energy_test_kernel(x, candidates)
        >>> weights  # Should put most weight on cand2
        Array([0.0, 1.0], dtype=float32)

    """
    if not 0.0 < exponent < _MAX_DISTANCE_EXPONENT:
        message = f"exponent must be in (0, 2), got {exponent}"
        raise ValueError(message)
    n, m_sizes, term_xx, candidate_gram, target_cross = _prepare_energy_test_inputs(
        x, y_candidates, metric, exponent
    )

    # Build the quadratic objective for the scaled energy test statistic. It
    # combines twice the weighted target-candidate distances, target self-distance,
    # and the weighted candidate Gram matrix.
    #
    # The exact scale (n·m_eff)/(n+m_eff) is nonlinear in w (m_eff = m_eff(w),
    # see inverse_variance_m_eff), so for the QP surrogate we weight each
    # candidate by its VERTEX effective size s_k = m_eff(e_k) = m_k — i.e. the
    # per-candidate harmonic factor (n·m_k)/(n+m_k). This is the correct value at
    # every simplex vertex under the inverse-variance m_eff (1/m_eff = Σ w²/m ⇒
    # m_eff(e_k) = m_k), and is a monotone, tractable surrogate off the vertices.
    #
    # NOTE (correction, ground: Martino 2017 Kish ESS): the legacy code linearized
    # the scale via m_eff ≈ wᵀm. That is REFUTED — wᵀm understates the effective
    # size off the vertices (K equal groups at equal weights give wᵀm = m instead
    # of the pooled Km). The FINAL statistic below uses the corrected
    # inverse-variance m_eff; the QP objective retains per-vertex s_k as a
    # documented surrogate for the argmax (the objective's role is candidate
    # SELECTION, not calibration — calibration is the studentized entry points).

    quadratic, linear = _energy_test_quadratic(
        n, m_sizes, term_xx, candidate_gram, target_cross
    )
    weights = _solve_energy_test_weights(quadratic, linear, solver, tol, maxiter)

    # Compute final test statistic with optimal weights.
    # CORRECTION (was m_eff = wᵀm, refuted): use the inverse-variance / Kish
    # effective size 1/m_eff = Σ_k w_k²/m_k (homoscedastic) or the
    # heteroscedastic form when group_variances is provided. The outer harmonic
    # factor (n·m_eff)/(n+m_eff) is CORRECT and kept. Ground: Martino 2017.
    m_eff = _resolve_effective_size(weights, m_sizes, m_eff_mode, group_variances)
    scale = (n * m_eff) / (n + m_eff)

    # Energy distance terms for mixture
    term_xy = jnp.dot(weights, target_cross)
    term_yy = jnp.dot(weights, jnp.dot(candidate_gram, weights))

    energy_dist = 2.0 * term_xy - term_xx - term_yy
    test_statistic = scale * energy_dist

    return weights, test_statistic


def energy_test_kernel_pairwise_max(
    distributions: jnp.ndarray | list[jnp.ndarray],
    exponent: float = 1.0,
    metric: GroundDistanceSelection = "euclidean",
    solver: str = "boxosqp",
    tol: float = 1e-6,
    maxiter: int = 1000,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Find optimal pairings maximizing energy test statistics.

    For each distribution, finds the optimal weighted mixture of all other
    distributions that maximizes the energy test statistic. Useful for:

    - Outlier detection: which distribution is most different from others?
    - Multi-sample homogeneity testing
    - Identifying subgroups in heterogeneous data

    Args:
        distributions: Either:
            - List of K distributions with shapes (n_k, d)
            - Array of shape (K, n, d) with uniform sizes
        exponent: Distance exponent α in (0, 2).
        metric: Distance metric (see cdist for options).
        solver: Optimization solver.
        tol: Solver tolerance.
        maxiter: Maximum solver iterations.

    Returns:
        Tuple of (weights, test_statistics) where:
            weights: Array of shape (K, K-1) with optimal mixture weights
            test_statistics: Array of shape (K,) with maximum test statistics

    Examples:
        >>> # Find outlier distribution
        >>> dist1 = jnp.array([[0.0, 0.0], [0.1, 0.1]])
        >>> dist2 = jnp.array([[0.0, 0.0], [0.1, 0.1]])  # Similar to dist1
        >>> dist3 = jnp.array([[10.0, 10.0], [11.0, 11.0]])  # Outlier
        >>> distributions = [dist1, dist2, dist3]
        >>> weights, stats = energy_test_kernel_pairwise_max(distributions)
        >>> outlier_idx = jnp.argmax(stats)
        >>> print(f"Outlier: distribution {outlier_idx}")
        Outlier: distribution 2

    """
    # Convert to list if needed
    if isinstance(distributions, jnp.ndarray):
        distributions_list = [distributions[i] for i in range(distributions.shape[0])]
    else:
        distributions_list = distributions

    distribution_count = len(distributions_list)
    resolved_metric = resolve_ground_distance(metric)

    weights_list = []
    stats_list = []

    for i in range(distribution_count):
        x = distributions_list[i]
        candidates = [
            distributions_list[j] for j in range(distribution_count) if j != i
        ]

        w, stat = energy_test_kernel(
            x,
            candidates,
            exponent=exponent,
            metric=resolved_metric,
            solver=solver,
            tol=tol,
            maxiter=maxiter,
        )

        weights_list.append(w)
        stats_list.append(stat)

    weights = jnp.stack(weights_list, axis=0)
    test_statistics = jnp.array(stats_list)

    return weights, test_statistics
