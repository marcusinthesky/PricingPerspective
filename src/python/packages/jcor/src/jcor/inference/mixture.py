"""The corrected mixture test: sample splitting with a multi-split correction.

Drives :mod:`jcor.inference._permutation.split_runner` across repeated splits
and combines the per-split decisions into a single result.

:class:`~jcor.inference._results.MixtureTestResult` is re-exported here — the
container belongs with the procedure that builds it — but it is *defined* in
the leaf :mod:`jcor.inference._results`. Defining it in this module would make
``split_runner`` (which constructs it) import ``mixture`` while ``mixture``
imports ``split_runner``: a within-stage cycle that
``just python::analyze-cycles`` would reject.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jax import random

from jcor.discrepancy.energy import _pooled_from_candidates
from jcor.ground.config import (  # noqa: TC001  # runtime annotations
    GroundDistanceSelection,
    resolve_ground_distance,
)
from jcor.inference._permutation.common import _optimal_weights, _split_blocks
from jcor.inference._permutation.split_runner import _run_test_on_split
from jcor.inference._results import MixtureTestResult

if TYPE_CHECKING:
    import jax.numpy as jnp

__all__ = ["MixtureTestResult", "corrected_mixture_test"]


def _split_selection_and_test(
    x: jnp.ndarray,
    clist: list[jnp.ndarray],
    selection_split: float,
) -> tuple[jnp.ndarray, jnp.ndarray, list[jnp.ndarray], list[jnp.ndarray]]:
    """Cut the target and every candidate into selection-A and test-B blocks.

    Each sample is split independently by :func:`_split_blocks`, so block A is
    the contiguous leading fraction of that sample and block B the remainder.

    Args:
        x: Target sample, shape (n, d).
        clist: Candidate samples.
        selection_split: Fraction of each sample assigned to block A.

    Returns:
        Tuple ``(x_a, x_b, cand_a, cand_b)`` of the selection- and test-block
        target samples and candidate lists.

    """
    xa_idx, xb_idx = _split_blocks(len(x), selection_split)
    x_a, x_b = x[xa_idx], x[xb_idx]
    cand_a: list[jnp.ndarray] = []
    cand_b: list[jnp.ndarray] = []
    for c in clist:
        ai, bi = _split_blocks(len(c), selection_split)
        cand_a.append(c[ai])
        cand_b.append(c[bi])
    return x_a, x_b, cand_a, cand_b


def corrected_mixture_test(
    x: jnp.ndarray,
    candidates: jnp.ndarray | list[jnp.ndarray],
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
    seed: int | None = None,
    weight_method: str = "kernel_balance",
) -> MixtureTestResult:
    """Corrected permutation test of ``X`` vs the optimal candidate mixture.

    Composes corrections #1 (inverse-variance m_eff), #3 (re-optimize w* inside
    each permutation OR selection-split-A/test-split-B), and #4 (studentize). The
    reported PRIMARY metric is the energy point estimate ``Ê(X, Y_{w*})`` with a
    bootstrap CI; the studentized permutation p-value is secondary.

    **Architecture (Correction #3, default selection-split).** With
    ``selection_split`` set (default 0.5), ``w*`` is fit on a contiguous
    selection-block A and the studentized statistic + permutation calibration run
    on the independent test-block B — so ``w*`` is independent of the tested
    labels and no double-dipping occurs. With ``selection_split=None`` and
    ``reoptimize_in_permutation=True``, the full sample is used and the inner
    ``argmin_w`` is **re-optimized within every permutation replicate**, making
    "select-then-test" one fixed statistic (also valid). Plug-in-``w*``
    permutation (``reoptimize_in_permutation=False``, ``selection_split=None``) is
    **invalid** (double-dipping) and is disallowed.

    Disclosed-inapplicable: Fithian-Sun-Taylor (2014) polyhedral selective
    inference conditions on a finite partition of discrete models via linear
    inequalities; it does NOT characterize the selection event of a continuous
    ``argmin`` over the probability simplex against an energy objective, so we use
    re-optimization / splitting rather than polyhedral conditioning.

    Args:
        x: Target sample, shape (n, d).
        candidates: K candidate samples.
        num_permutations: Permutation replicates.
        exponent: Distance exponent α in (0, 2).
        metric: Distance metric.
        solver: Mixture-weight solver.
        tol: Numerical solver tolerance.
        maxiter: Maximum solver iterations.
        reoptimize_in_permutation: Re-solve w* per replicate (full-sample route).
        selection_split: Fraction for selection-block A (test on B). ``None`` ⇒
            full-sample re-optimization route.
        group_variances: Optional per-candidate variances (heteroscedastic m_eff).
        confidence_level: Bootstrap CI level for Ê.
        num_bootstrap: Bootstrap resamples for the Ê CI.
        seed: RNG seed.
        weight_method: ``"kernel_balance"`` for the legacy stationarity-residual
            surrogate or ``"energy_barycentre"`` for the exact finite-mixture
            energy objective.

    Returns:
        :class:`jcor.inference.mixture.MixtureTestResult`.

    """
    if selection_split is None and not reoptimize_in_permutation:
        message = (
            "Plug-in-w* permutation (selection_split=None and "
            "reoptimize_in_permutation=False) is invalid (double-dipping). "
            "Set selection_split (split-A/test-B) or reoptimize_in_permutation=True."
        )
        raise ValueError(message)

    clist, _ = _pooled_from_candidates(candidates)
    resolved_metric = resolve_ground_distance(metric)
    rng = random.PRNGKey(seed if seed is not None else 0)

    if selection_split is not None:
        # SELECTION-SPLIT-A / TEST-SPLIT-B (default). Fit w* on A, test on B.
        x_a, x_b, cand_a, cand_b = _split_selection_and_test(
            x,
            clist,
            selection_split,
        )
        w_star, selection_diagnostics, solver_used = _optimal_weights(
            x_a,
            cand_a,
            exponent,
            resolved_metric,
            solver,
            tol,
            maxiter,
            weight_method,
        )
        return _run_test_on_split(
            x_b,
            cand_b,
            w_star,
            num_permutations,
            exponent,
            resolved_metric,
            group_variances,
            confidence_level,
            num_bootstrap,
            rng,
            reoptimized=False,
            solver=solver,
            tol=tol,
            maxiter=maxiter,
            reoptimize=False,
            weight_method=weight_method,
            selection_diagnostics=selection_diagnostics,
            solver_used=solver_used,
        )

    # FULL-SAMPLE re-optimization-in-permutation route.
    w_star, selection_diagnostics, solver_used = _optimal_weights(
        x, clist, exponent, resolved_metric, solver, tol, maxiter, weight_method
    )
    return _run_test_on_split(
        x,
        clist,
        w_star,
        num_permutations,
        exponent,
        resolved_metric,
        group_variances,
        confidence_level,
        num_bootstrap,
        rng,
        reoptimized=True,
        solver=solver,
        tol=tol,
        maxiter=maxiter,
        reoptimize=True,
        weight_method=weight_method,
        selection_diagnostics=selection_diagnostics,
        solver_used=solver_used,
    )
