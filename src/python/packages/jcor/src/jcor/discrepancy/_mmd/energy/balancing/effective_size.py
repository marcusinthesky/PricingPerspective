"""Inverse-variance (Kish) effective sample size of a weighted mixture.

Split out of the energy-test kernels because it is the *studentization
normalizer* for anything built on a weighted empirical measure — the
permutation stage imports it directly, not through a kernel.
"""

from __future__ import annotations

import jax.numpy as jnp


def inverse_variance_m_eff(
    weights: jnp.ndarray,
    m_sizes: jnp.ndarray,
    group_variances: jnp.ndarray | None = None,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Inverse-variance (Kish) effective sample size of a weighted mixture.

    The ``Y``-side energy functionals are estimated from the weighted empirical
    measure ``Ŷ_w = Σ_k w_k Ŷ_k`` where group ``k`` supplies ``m_k`` iid draws.
    Because the groups are mutually independent, the sampling variance of the
    linear (non-degenerate / mean-embedding) part of the statistic is

        Var(Σ_k w_k μ̂_k) = Σ_k w_k² σ_k² / m_k.

    Matching this to ``σ̄² / m_eff`` (variance of a hypothetical unweighted
    sample of size ``m_eff`` at the pooled variance ``σ̄²``) defines the
    effective size.  In the **homoscedastic** case (``σ_k² ≡ σ²``) this collapses
    to the classical Kish / importance-sampling effective sample size

        1 / m_eff = Σ_k w_k² / m_k,      i.e.  m_eff = (Σ_k w_k² / m_k)^{-1}.

    In the **heteroscedastic** case (unequal ticker variances ``s_k²``) the
    inverse-variance-weighted form is

        1 / m_eff = (Σ_k w_k² s_k² / m_k) / (Σ_k w_k s_k²).

    This is the correct *non-degenerate / studentization normalizer*.  It is the
    replacement for the previous, over-optimistic ``m_eff = wᵀm`` — which is only
    correct at a simplex vertex ``w = e_k`` (where both give ``m_k``) and off the
    vertices *understates* the effective size (e.g. ``K`` equal groups of size
    ``m`` at equal weights give ``m_eff = Km`` here, but ``wᵀm = m``).

    Ground: Martino, Elvira & Louzada (2017), *Signal Processing* 131:386-401
    (Kish ESS); Rizzo-Székely energy-statistic variance ``∝ 1/n + 1/m_eff``.

    .. warning::
        This is the variance-normalizer for the **non-degenerate** part of the
        statistic and for studentization — **not** an exact-null quantity.  Under
        ``H0`` the energy statistic is degenerate, ``n·E_n ⇒ Σ_k λ_k Z_k²``
        (eigenvalues of the energy kernel's integral operator, mixture-dependent),
        and **no scalar ``m_eff`` is exact** in that regime.

    Args:
        weights: Mixture weights, shape (K,), on the simplex.
        m_sizes: Per-group sample sizes ``m_k``, shape (K,).
        group_variances: Optional per-group variances ``s_k²``, shape (K,).
            If ``None``, the homoscedastic form is used.
        eps: Numerical floor to avoid division by zero.

    Returns:
        Scalar effective sample size ``m_eff`` (>= 0).

    """
    w = jnp.asarray(weights, dtype=jnp.float32)
    m = jnp.asarray(m_sizes, dtype=jnp.float32)
    if group_variances is None:
        inv_m_eff = jnp.sum(w**2 / jnp.maximum(m, eps))
    else:
        s2 = jnp.asarray(group_variances, dtype=jnp.float32)
        numerator = jnp.sum(w**2 * s2 / jnp.maximum(m, eps))
        denominator = jnp.maximum(jnp.sum(w * s2), eps)
        inv_m_eff = numerator / denominator
    return 1.0 / jnp.maximum(inv_m_eff, eps)
