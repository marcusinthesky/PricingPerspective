"""Equivalence / relevance framing for the mixture two-sample test (Correction #2).

Replaces the inverted "high p ⇒ good hedge" logic with the relevance null
``H0: E ≥ δ`` vs ``H1: E < δ``; the PRIMARY metric is the point estimate
``Ê(X, Y_{w*})`` + bootstrap CI, with an equivalence curve over a δ-grid. Ground:
Liu & Gandy (2026), *Kernel Tests of Equivalence*; Wellek (2010); Schuirmann
(1987) TOST; Berger & Hsu (1996) IUT.

Also provides margin selection (:func:`poor_hedge_margin`). Family-wise control
across many verdicts moved to :mod:`jcor.decision.multiplicity` (t46.7).

Position
--------
rank 8 · consumes stage S7's ``inference`` results, produces a verdict

``decision → inference`` (S8 → S7) is the one cross-stage edge here and it runs
the legal direction; Ruff and Tach check the package direction.
"""

from __future__ import annotations

import jax.numpy as jnp

from jcor.ground.config import (
    GroundDistanceSelection,  # noqa: TC001  # runtime annotations
)
from jcor.inference.mixture import MixtureTestResult, corrected_mixture_test

__all__ = [
    "equivalence_pvalue_from_bootstrap",
    "equivalence_test",
    "poor_hedge_margin",
]


def equivalence_pvalue_from_bootstrap(
    bootstrap_energy: jnp.ndarray, delta: float
) -> float:
    """Relevance-null equivalence p-value from the bootstrap distribution of Ê.

    Tests H0: E(X, Y_w) ≥ δ vs H1: E(X, Y_w) < δ. Using the percentile bootstrap
    distribution of Ê, the (add-one smoothed) one-sided p-value is the bootstrap
    mass at or above the margin::

        p_equiv = (1 + #{Ê* ≥ δ}) / (1 + B).

    SMALL p_equiv ⇒ the bootstrap Ê sits confidently below δ ⇒ evidence FOR
    equivalence. This is consistent with the CI-based verdict ``ci_hi < δ``: when
    the upper (1-α/2) percentile of Ê* is below δ, fewer than α/2 of the bootstrap
    replicates lie above δ, so ``p_equiv < α/2`` and the two agree in direction.

    Args:
        bootstrap_energy: Bootstrap replicates of Ê, shape (B,).
        delta: Equivalence margin on the energy distance.

    Returns:
        Equivalence p-value in (0, 1]; small ⇒ evidence for equivalence.

    """
    b = bootstrap_energy.shape[0]
    if b == 0:
        return 1.0
    n_ge = float(jnp.sum(bootstrap_energy >= delta))
    return (1.0 + n_ge) / (1.0 + b)


def poor_hedge_margin(
    poor_hedge_energy_distances: jnp.ndarray, quantile: float = 0.1
) -> float:
    """Economically-anchored equivalence margin δ.

    Set δ at a low quantile of energy distances between assets known ex ante to be
    poor hedges (e.g. cross-sector random pairs). This ties "equivalent enough" to
    a domain-meaningful comparator rather than an arbitrary number. Ground: the
    economically-anchored route of the deep-research report (§1, route 1).

    Args:
        poor_hedge_energy_distances: Sample of ``Ê`` between poor-hedge pairs.
        quantile: Low quantile (default 0.1) defining the margin.

    Returns:
        δ as a float.

    """
    return float(jnp.quantile(jnp.asarray(poor_hedge_energy_distances), quantile))


def equivalence_test(
    x: jnp.ndarray,
    candidates: jnp.ndarray | list[jnp.ndarray],
    delta: float,
    num_permutations: int = 999,
    exponent: float = 1.0,
    metric: GroundDistanceSelection = "euclidean",
    solver: str = "boxosqp",
    tol: float = 1e-6,
    maxiter: int = 1000,
    *,
    reoptimize_in_permutation: bool = True,
    selection_split: float | None = 0.5,
    group_variances: jnp.ndarray | None = None,
    confidence_level: float = 0.95,
    num_bootstrap: int = 499,
    delta_grid: jnp.ndarray | None = None,
    seed: int | None = None,
    weight_method: str = "kernel_balance",
) -> MixtureTestResult:
    """Equivalence / relevance test: ``H0: E(X,Y_w) ≥ δ`` vs ``H1: E(X,Y_w) < δ``.

    Correction #2 — replaces the inverted "high p ⇒ good hedge" logic. Rejecting
    ``H0`` **positively** establishes equivalence within margin ``δ`` with a
    genuine Type-I guarantee on the claim of interest. The PRIMARY reported metric
    is ``Ê(X, Y_{w*})`` with a bootstrap CI; an equivalence curve over ``delta_grid``
    reports the verdict across margins so no single threshold is load-bearing.
    Composed on top of the corrected (studentized, re-optimized / split, correct
    m_eff) machinery of :func:`jcor.inference.mixture.corrected_mixture_test`.

    Decision rule (IUT / Wellek): declare equivalence iff the upper end of the
    ``100(1-2α)%`` CI on ``Ê`` is below ``δ`` (the ``equivalence`` flag). Note
    (Berger-Hsu 1996): a size-α equivalence test corresponds to a ``100(1-α)%``
    confidence SET, not ``1-2α`` — reported by
    :func:`jcor.decision.multiplicity.equivalence_multiplicity_control` for the
    simultaneous case.

    **Two p-values, opposite directions — do not confuse them:**

    - ``result.pvalue`` (the DIFFERENCE p): the studentized *permutation* p-value
      for H0: F_X = F_{Y_w}. SMALL ⇒ the distributions DIFFER ⇒ NOT equivalent.
      Kept for reference; it is NOT the equivalence claim and must NOT be fed to
      FWER control.
    - ``result.equivalence_pvalue`` (the EQUIVALENCE / relevance p): for the
      relevance null H0: E(X, Y_w) ≥ δ vs H1: E(X, Y_w) < δ, computed as the
      bootstrap mass of Ê at or above δ, ``(1 + #{Ê* ≥ δ}) / (1 + B)``. SMALL ⇒
      evidence FOR equivalence. This is the p-value that agrees in direction with
      the ``equivalence`` flag and that
      :func:`jcor.decision.multiplicity.equivalence_multiplicity_control`
      consumes.

    Ground: Liu & Gandy (2026); Wellek (2010); Schuirmann (1987) TOST;
    Berger & Hsu (1996) IUT.

    Args:
        x: Target observation cloud.
        candidates: Candidate observation clouds.
        delta: Equivalence margin on the energy distance (explicit, documented).
        num_permutations: Number of permutation replicates.
        exponent: Distance exponent in the open interval ``(0, 2)``.
        metric: Distance metric name or pairwise metric callable.
        solver: Mixture-weight solver.
        tol: Numerical solver tolerance.
        maxiter: Maximum solver iterations.
        reoptimize_in_permutation: Whether to refit weights in each permutation.
        selection_split: Fraction reserved for selecting mixture weights.
        group_variances: Optional candidate-specific variance estimates.
        confidence_level: Bootstrap interval confidence level.
        num_bootstrap: Number of bootstrap replicates.
        delta_grid: Optional δ-grid for the equivalence curve.
        seed: Random seed.
        weight_method: Mixture-weight estimation method.

    Returns:
        :class:`jcor.inference.mixture.MixtureTestResult` with
        ``equivalence_framed=True``, ``delta``, ``equivalence``, and
        ``equivalence_curve`` populated.

    """
    base = corrected_mixture_test(
        x,
        candidates,
        num_permutations=num_permutations,
        exponent=exponent,
        metric=metric,
        solver=solver,
        tol=tol,
        maxiter=maxiter,
        reoptimize_in_permutation=reoptimize_in_permutation,
        selection_split=selection_split,
        group_variances=group_variances,
        confidence_level=confidence_level,
        num_bootstrap=num_bootstrap,
        seed=seed,
        weight_method=weight_method,
    )
    ci_hi = base.energy_ci[1]
    equivalent = bool(ci_hi < delta)
    equiv_p = equivalence_pvalue_from_bootstrap(base.bootstrap_distribution, delta)

    curve = (
        [(margin, bool(ci_hi < margin)) for margin in map(float, delta_grid)]
        if delta_grid is not None
        else []
    )

    return MixtureTestResult(
        energy_point_estimate=base.energy_point_estimate,
        energy_ci=base.energy_ci,
        studentized_statistic=base.studentized_statistic,
        pvalue=base.pvalue,
        weights=base.weights,
        m_eff=base.m_eff,
        null_distribution=base.null_distribution,
        bootstrap_distribution=base.bootstrap_distribution,
        reoptimized_in_permutation=base.reoptimized_in_permutation,
        studentized=True,
        equivalence_framed=True,
        num_permutations=base.num_permutations,
        weight_method=base.weight_method,
        solver_used=base.solver_used,
        solver_converged=base.solver_converged,
        solver_frank_wolfe_gap=base.solver_frank_wolfe_gap,
        solver_cnd_tangent_max_eigenvalue=(base.solver_cnd_tangent_max_eigenvalue),
        solver_simplex_sum_error=base.solver_simplex_sum_error,
        solver_min_weight=base.solver_min_weight,
        equivalence_pvalue=equiv_p,
        equivalence=equivalent,
        delta=float(delta),
        equivalence_curve=curve,
    )
