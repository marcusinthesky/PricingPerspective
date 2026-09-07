"""Certificate bounds for portfolio variance and hedging error.

All certificate functions operate in **float64** to maintain the tolerance
``1e-9`` required by the Lean theorem validation stages.  Callers should
configure JAX x64 (via :func:`simulation.backend.configure`) before passing
inputs to these functions.

Donors:
    ``src/python/apps/simulation/src/simulation/validation/variance_floor.py``:
        ``compute_floor_and_realized``,
        ``compute_lipschitz_constant``.
    ``src/python/apps/simulation/src/simulation/validation/pricefree.py``:
        ``certified_floor``,
        ``compute_lipschitz_constants``.
    ``src/python/apps/simulation/src/simulation/validation/hedging_error.py``:
        ``run_target`` (bound construction).
Float64 policy
--------------
The three certificate functions (:func:`variance_floor`,
:func:`certified_floor`, :func:`hedging_bound`) perform arithmetic on
well-conditioned double-precision inputs.  Callers must enable
``jax_enable_x64`` (via :func:`simulation.backend.configure`) before creating
the inputs; results are returned as Python floats for downstream YAML
serialisation.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

# Runtime import, not TYPE_CHECKING-guarded. `flake8-type-checking` in `strict`
# mode wants `Float` under the guard, but with `from __future__ import
# annotations` that makes it an unresolvable name at runtime and beartype then
# *silently skips* the annotation instead of failing it — so every shape string
# below would be decoration under `tests/conftest.py`'s `install_import_hook`.
# The probe is recorded in `jcor.core.typing`'s module docstring; not re-derived
# here. `ArrayLike` is the JAX/jaxtyping host-array contract and every input is
# coerced with `jnp.asarray` at the computation boundary.
from jaxtyping import (  # noqa: TC002  # third-party; measured, not guessed
    ArrayLike,
    Float,
)


class LipschitzResult(NamedTuple):
    """Return type of :func:`lipschitz_constants`.

    Attributes:
        L_max: Tight upper Lipschitz constant ``max‖βᵢ-βⱼ‖/dᵢⱼ``.
        ell_min: Minimum ratio ``min‖βᵢ-βⱼ‖/dᵢⱼ`` (conservative lower).
        n_skipped: Number of pairs with ``dᵢⱼ < eps`` (skipped).

    """

    # This spelling is part of the historical constructor and tuple schema.
    L_max: float
    ell_min: float
    n_skipped: int


def lipschitz_constants(
    betas: Float[ArrayLike, "n d"],
    d: Float[ArrayLike, "n n"],
    eps: float = 1e-12,
) -> LipschitzResult:
    """Compute pairwise Lipschitz ratio statistics over all asset pairs.

    For each pair ``(i, j)`` with ``i < j`` and ``dᵢⱼ >= eps``, computes
    the ratio ``‖βᵢ - βⱼ‖ / dᵢⱼ``.  Returns the maximum (``l_max``),
    minimum (``ell_min``), and skipped-pair count.

    ``l_max`` is the tight Lipschitz constant required by the Lean theorem
    ``portfolio_variance_lower_bound``.  ``ell_min`` is used by
    ``pf_dist_mvo_portfolio`` for a conservative ``Σ_dist`` construction.

    Args:
        betas: Exposure vectors, shape ``(n, d)``.
        d: Symmetric pairwise energy-distance matrix, shape ``(n, n)``.
        eps: Threshold below which a pair's ``dᵢⱼ`` is treated as zero
            and excluded from the ratio computation.

    Returns:
        :class:`LipschitzResult` with ``(L_max, ell_min, n_skipped)``.

    """
    betas = jnp.asarray(betas)
    d = jnp.asarray(d)
    n = betas.shape[0]

    # All pairwise diff norms: (n, n)
    diff = betas[:, None, :] - betas[None, :, :]  # (n, n, d)
    diff_norms = jnp.linalg.norm(diff, axis=-1)  # (n, n)

    # Upper triangle mask (i < j)
    mask = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)

    d_upper = jnp.where(mask, d, jnp.inf)
    dn_upper = jnp.where(mask, diff_norms, 0.0)

    valid = mask & (eps <= d)
    n_skipped = int(jnp.sum(mask & (eps > d)))

    # Ratios for valid pairs; fill invalid with 0/inf for min/max
    ratios = jnp.where(valid, dn_upper / jnp.where(valid, d_upper, 1.0), jnp.nan)

    if int(jnp.sum(valid)) == 0:
        return LipschitzResult(L_max=0.0, ell_min=0.0, n_skipped=n_skipped)

    l_max = float(jnp.nanmax(ratios))
    ell_min = float(jnp.nanmin(ratios))
    return LipschitzResult(L_max=l_max, ell_min=ell_min, n_skipped=n_skipped)


def variance_floor(
    w: Float[ArrayLike, " n"],
    betas: Float[ArrayLike, "n d"],
    distance_matrix: Float[ArrayLike, "n n"],
    lipschitz: float,
) -> tuple[float, float]:
    """Compute the Lean-theorem variance floor and realised portfolio variance.

    The floor from ``portfolio_variance_lower_bound``:

    .. code-block:: text

        floor = Σᵢⱼ wᵢ wⱼ · ½ · (‖βᵢ‖² + ‖βⱼ‖² − (L·Dᵢⱼ)²)

    Realised variance:

    .. code-block:: text

        realized = ‖Σᵢ wᵢ βᵢ‖²

    Theorem guarantee: ``floor <= realized``.

    Source: ``compute_floor_and_realized`` in
    ``validation/variance_floor.py``.

    Args:
        w: Portfolio weights, shape ``(n,)``.  Must be non-negative and sum to 1.
        betas: Exposure vectors, shape ``(n, d)``.
        distance_matrix: Pairwise energy-distance matrix, shape ``(n, n)``.
        lipschitz: Lipschitz constant from :func:`lipschitz_constants`.

    Returns:
        Tuple ``(floor, realized)`` as Python floats.

    """
    w = jnp.asarray(w)
    betas = jnp.asarray(betas)
    distance_matrix = jnp.asarray(distance_matrix)

    beta_norms_sq = jnp.sum(betas**2, axis=1)  # (n,)
    l2_d2 = (lipschitz * distance_matrix) ** 2  # (n, n)
    term = 0.5 * (beta_norms_sq[:, None] + beta_norms_sq[None, :] - l2_d2)  # (n, n)
    ww = jnp.outer(w, w)  # (n, n)
    floor = float(jnp.sum(ww * term))

    portfolio_beta = w @ betas  # (d,)
    realized = float(jnp.dot(portfolio_beta, portfolio_beta))
    return floor, realized


def certified_floor(
    w: Float[ArrayLike, " n"],
    betas: Float[ArrayLike, "n d"],
    d2: Float[ArrayLike, "n n"],
    l_max: float,
) -> float:
    """Compute the Lean-theorem certified lower bound on portfolio variance.

    .. code-block:: text

        floor = Σᵢ wᵢ‖βᵢ‖² − (l_max²/2) · wᵀd²w

    Theorem guarantee: ``floor <= ‖Σᵢ wᵢ βᵢ‖² <= wᵀ(ββᵀ)w <= wᵀΣ_true w``.

    Source: ``certified_floor`` in
    ``validation/pricefree.py``.

    Args:
        w: Portfolio weights, shape ``(n,)``.
        betas: Exposure vectors, shape ``(n, d)``.
        d2: Element-wise squared energy-distance matrix ``d²``, shape ``(n, n)``.
        l_max: Tight Lipschitz constant from :func:`lipschitz_constants`.

    Returns:
        Certified floor value as a Python float.

    """
    w = jnp.asarray(w)
    betas = jnp.asarray(betas)
    d2 = jnp.asarray(d2)

    beta_norms_sq = jnp.sum(betas**2, axis=1)  # (n,)
    weighted_norms_sq = float(w @ beta_norms_sq)
    q_w = float(w @ d2 @ w)
    return weighted_norms_sq - 0.5 * l_max**2 * q_w


def hedging_bound(
    w_opt: Float[ArrayLike, " K"],
    betas_cand: Float[ArrayLike, "K d"],
    beta_target: Float[ArrayLike, " d"],
    d_to_candidates: Float[ArrayLike, " K"],
    lipschitz: float,
) -> tuple[float, float, float]:
    """Compute the hedging-error bound for an optimal mixture portfolio.

    Bound: ``‖β_T − Σₖ w*ₖ βₖ‖ ≤ L · Σₖ w*ₖ · D_E(T, k)``.

    ``lipschitz`` is the tight ratio ``max_k ‖β_T − βₖ‖ / D_E(T, k)`` (over ``k``
    with ``D_E > 0``), so the bound holds by construction; violations
    indicate a computation bug.

    Source: bound construction in ``run_target`` in
    ``validation/hedging_error.py``.

    Args:
        w_opt: Optimal mixture weights, shape ``(K,)``.
        betas_cand: Candidate exposure vectors, shape ``(K, d)``.
        beta_target: Target asset exposure vector, shape ``(d,)``.
        d_to_candidates: Energy distances from target to each candidate,
            shape ``(K,)``.
        lipschitz: Lipschitz constant from :func:`lipschitz_constants`
            computed on target vs candidates).

    Returns:
        Tuple ``(tracking_error, bound_rhs, bound_slack)`` as Python floats.
        ``bound_slack = bound_rhs - tracking_error`` must be ``>= 0``.

    """
    w_opt = jnp.asarray(w_opt)
    betas_cand = jnp.asarray(betas_cand)
    beta_target = jnp.asarray(beta_target)
    d_to_candidates = jnp.asarray(d_to_candidates)

    beta_mix = betas_cand.T @ w_opt  # (d,)
    tracking_error = float(jnp.linalg.norm(beta_target - beta_mix))
    bound_rhs = lipschitz * float(jnp.dot(w_opt, d_to_candidates))
    bound_slack = bound_rhs - tracking_error
    return tracking_error, bound_rhs, bound_slack
