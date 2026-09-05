"""Portfolio construction utilities.

All weight-returning functions produce non-negative arrays that sum to 1
(simplex-feasible).  Optimisation calls (:func:`min_variance`,
:func:`max_spread`, :func:`dist_mvo`) delegate to
:mod:`simulation.optimize` and are therefore :func:`jax.jit`-friendly.

Donors:
    ``src/python/apps/simulation/src/simulation/validation/variance_floor.py``:
        ``make_weight_schemes``.
    ``src/python/apps/simulation/src/simulation/validation/pricefree.py``:
        ``ew_portfolio``,
        ``sample_mvo_portfolio``,
        ``pf_max_spread_portfolio``,
        ``pf_dist_mvo_portfolio``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# Runtime import, deliberately not TYPE_CHECKING-guarded. With `from __future__
# import annotations` a guarded `Float`/`Array` is an unresolvable name at
# runtime; jaxtyping's `_destring_annotation` swallows that and hands beartype
# `Any` for the whole annotation instead of failing, so every shape string below
# would be decoration under `tests/conftest.py`'s `install_import_hook`. The
# probe is recorded in `jcor.core.typing`'s module docstring; not re-derived here.
from jaxtyping import Array, Float  # noqa: TC002
from jcor.optimize import ridge_psd

from simulation.optimize import pgd_max_quadratic, pgd_min_variance


class WeightScheme(NamedTuple):
    """Named portfolio weight vector.

    Attributes:
        name: Human-readable scheme identifier.
        weights: Weight array, shape ``(n,)``, summing to 1.

    """

    name: str
    weights: Float[Array, " n"]


def equal_weight(n: int) -> Float[Array, " n"]:
    """Equal-weight (1/n) portfolio.

    Args:
        n: Number of assets.

    Returns:
        Weight array of shape ``(n,)`` with all entries ``1/n``.

    """
    return jnp.ones(n) / n


def _dirichlet_sample(key: jax.Array, n: int, alpha: float = 1.0) -> Float[Array, " n"]:
    """Sample from ``Dirichlet(alpha · 1_n)`` via Gamma variates.

    Args:
        key: JAX PRNG key.
        n: Number of assets (dimension of the Dirichlet).
        alpha: Concentration parameter.  Default ``1.0`` (uniform).

    Returns:
        Simplex-feasible weight vector of shape ``(n,)``.

    """
    gammas = jax.random.gamma(key, alpha, shape=(n,))
    return gammas / jnp.sum(gammas)


def weight_schemes(
    key: jax.Array,
    n: int,
    n_dirichlet: int = 5,
    dirichlet_alpha: float = 1.0,
) -> list[WeightScheme]:
    """Generate a standard battery of simplex-feasible weight schemes.

    Produces ``1 + n_dirichlet + n`` schemes:
      * ``"equal"``: uniform ``1/n``.
      * ``"dirichlet_{i}"`` for ``i in 0…n_dirichlet-1``: random long-only.
      * ``"corner_{i}"`` for ``i in 0…n-1``: full weight on single asset.

    Source: ``make_weight_schemes`` in
    ``validation/variance_floor.py``.

    Args:
        key: JAX PRNG key used for Dirichlet draws.
        n: Number of assets.
        n_dirichlet: Number of random Dirichlet draws.
        dirichlet_alpha: Concentration parameter for the Dirichlet.

    Returns:
        List of :class:`WeightScheme` named tuples.

    """
    schemes: list[WeightScheme] = []

    # Equal weight
    schemes.append(WeightScheme("equal", equal_weight(n)))

    # Random long-only Dirichlet
    for i in range(n_dirichlet):
        key, subkey = jax.random.split(key)
        w = _dirichlet_sample(subkey, n, dirichlet_alpha)
        schemes.append(WeightScheme(f"dirichlet_{i}", w))

    # Single-asset corners
    for i in range(n):
        w = jnp.zeros(n).at[i].set(1.0)
        schemes.append(WeightScheme(f"corner_{i}", w))

    return schemes


def min_variance(
    sigma: Float[Array, "n n"],
    *,
    n_steps: int = 5000,
    apply_ridge: bool = True,
    ridge_eps: float = 1e-6,
) -> tuple[Float[Array, " n"], float]:
    """Long-only minimum-variance portfolio via PGD on the simplex.

    Optionally applies ridge regularisation via :func:`jcor.ridge_psd`
    before optimisation (matches donor's ``sample_mvo_portfolio`` pattern).

    Args:
        sigma: Covariance matrix, shape ``(n, n)``.
        n_steps: PGD iterations.
        apply_ridge: If ``True``, regularise ``sigma`` before PGD.
        ridge_eps: Minimum ridge magnitude if ``apply_ridge=True``.

    Returns:
        Tuple ``(weights, predicted_variance)`` where ``predicted_variance``
        is ``wᵀΣ_ridgew`` (the variance the method itself believes).

    """
    sigma_opt = ridge_psd(sigma, eps=ridge_eps) if apply_ridge else sigma
    w = pgd_min_variance(sigma_opt, n_steps=n_steps)
    predicted_var = float(w @ sigma_opt @ w)
    return w, predicted_var


def max_spread(
    d2: Float[Array, "n n"],
    n_steps: int = 5000,
) -> Float[Array, " n"]:
    """Maximise the price-free spread ``wᵀD²w`` on the simplex.

    ``d2[i,j] = Dᵢⱼ²`` is the element-wise square of the energy-distance matrix,
    not the matrix square.

    Source: ``pf_max_spread_portfolio`` (line 264 of compare_pricefree).

    Args:
        d2: Element-wise squared energy-distance matrix, shape ``(n, n)``.
        n_steps: PGD ascent iterations.

    Returns:
        Approximate max-spread weights of shape ``(n,)``.

    """
    return pgd_max_quadratic(d2, n_steps=n_steps)


def dist_mvo(
    distance_matrix: Float[Array, "n n"],
    sigma_hat: float,
    ell_hat: float,
    n_steps: int = 5000,
    ridge_eps: float = 1e-6,
) -> tuple[Float[Array, " n"], float]:
    """Price-free distance-implied MVO: minimise variance on ``Σ_dist``.

    .. code-block:: text

        Σ_dist[i,j] = σ̂² − ½ · ℓ̂² · distance_matrixᵢⱼ²

    Ridge-regularises ``Σ_dist`` if needed before PGD.

    Source: ``pf_dist_mvo_portfolio`` (line 276 of compare_pricefree).

    Args:
        distance_matrix: Symmetric energy-distance matrix, shape ``(n, n)``.
        sigma_hat: Estimated asset volatility (oracle or data-driven).
        ell_hat: Lipschitz estimate for ``Σ_dist`` construction
            (use ``ell_min`` from :func:`~simulation.bounds.lipschitz_constants`
            for a conservative construction).
        n_steps: PGD iterations.
        ridge_eps: Minimum ridge for PSD regularisation.

    Returns:
        Tuple ``(weights, predicted_variance)`` where ``predicted_variance``
        is ``wᵀΣ_dist_ridgew``.

    """
    d2 = distance_matrix**2
    sigma_dist = sigma_hat**2 - 0.5 * ell_hat**2 * d2
    sigma_ridge = ridge_psd(sigma_dist, eps=ridge_eps)
    w = pgd_min_variance(sigma_ridge, n_steps=n_steps)
    predicted_var = float(w @ sigma_ridge @ w)
    return w, predicted_var
