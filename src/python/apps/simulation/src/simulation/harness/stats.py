"""Statistical summaries for size and power experiments.

Pure functions operating on JAX arrays of p-values.  Summary values are
converted to Python scalars at the host-artifact boundary for downstream
YAML/Parquet serialisation.

Source: ``src/python/apps/simulation/src/simulation/validation/size_power.py``,
``run_size_experiment`` and ``build_summary``.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax.typing import ArrayLike  # noqa: TC002  # runtime array contract

_THREE_STANDARD_ERRORS = 3.0


def rejection_rate(pvalues: ArrayLike, alpha: float) -> float:
    """Empirical rejection rate at nominal level ``alpha``.

    Args:
        pvalues: 1-D array of p-values from independent tests.
        alpha: Nominal significance level.

    Returns:
        Fraction of p-values below ``alpha`` as a Python float.

    """
    return float(jnp.mean(jnp.asarray(pvalues) < alpha))


def binomial_se(rate: float, n_sims: int) -> float:
    """Binomial standard error of an empirical rejection rate.

    .. code-block:: text

        SE = sqrt(rate · (1 − rate) / n_sims)

    Args:
        rate: Observed rejection rate.
        n_sims: Number of independent tests used to compute ``rate``.

    Returns:
        Standard error as a Python float.

    """
    return math.sqrt(rate * (1.0 - rate) / max(n_sims, 1))


def z_score(rate: float, alpha: float, n_sims: int) -> float:
    """Z-score of the observed rejection rate relative to nominal ``alpha``.

    .. code-block:: text

        z = (rate − alpha) / SE(alpha, n_sims)

    where ``SE(alpha, n_sims) = sqrt(alpha·(1−alpha)/n_sims)`` uses the
    nominal (null) standard error rather than the observed rate.

    Args:
        rate: Observed empirical rejection rate.
        alpha: Nominal significance level.
        n_sims: Number of independent tests.

    Returns:
        Z-score as a Python float; ``0.0`` if ``SE == 0``.

    """
    se = math.sqrt(alpha * (1.0 - alpha) / max(n_sims, 1))
    return (rate - alpha) / se if se > 0.0 else 0.0


def size_summary(
    pvalues: ArrayLike,
    alpha_levels: list[float],
    n_sims: int,
) -> list[dict[str, float | bool]]:
    """Build per-alpha size-check rows from a p-value array.

    Each returned dict contains: ``nominal_alpha``, ``empirical_rejection_rate``,
    ``binom_se``, ``z_score``, ``flag_outside_3se``.

    Args:
        pvalues: 1-D array of p-values, length ``n_sims``.
        alpha_levels: Nominal significance levels to check.
        n_sims: Number of simulations (``len(pvalues)``).

    Returns:
        List of dicts, one per alpha level.

    """
    rows: list[dict[str, float | bool]] = []
    for alpha in alpha_levels:
        rate = rejection_rate(pvalues, alpha)
        se = binomial_se(alpha, n_sims)  # null SE using nominal alpha
        z = z_score(rate, alpha, n_sims)
        rows.append(
            {
                "nominal_alpha": alpha,
                "empirical_rejection_rate": rate,
                "binom_se": se,
                "z_score": z,
                "flag_outside_3se": abs(z) > _THREE_STANDARD_ERRORS,
            }
        )
    return rows


def power_summary(pvalues: ArrayLike, alpha: float, n_sims: int) -> dict[str, float]:
    """Build a power-check row from a p-value array under an alternative.

    Returns: ``empirical_power``, ``binom_se``.

    Args:
        pvalues: 1-D array of p-values under the alternative hypothesis.
        alpha: Significance level at which power is evaluated.
        n_sims: Number of simulations.

    Returns:
        Dict with ``empirical_power`` and ``binom_se``.

    """
    power = rejection_rate(pvalues, alpha)
    se = binomial_se(power, n_sims)
    return {"empirical_power": power, "binom_se": se}
