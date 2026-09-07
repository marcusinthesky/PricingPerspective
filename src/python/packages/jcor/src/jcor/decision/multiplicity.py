"""S8 — multiplicity control across many equivalence verdicts.

Position
--------
rank 8 · consumes equivalence p-values, produces a controlled selection

Split from :mod:`jcor.decision.equivalence` (t46.7) on the natural seam: that
module answers "is THIS pair equivalent?", this one answers "which of these M
verdicts survive family-wise control?". The two constants below are used only
here, so they travel with the function.

Ground: Berger & Hsu (1996) IUT; TOST ``(k-1)`` scaling; interval-null
non-uniformity (Statistical Papers 2024).
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = ["equivalence_multiplicity_control"]

_MIN_MULTIPLICITY_GROUPS = 2
_ADAPTIVE_NULL_THRESHOLD = 0.5


def equivalence_multiplicity_control(
    equivalence_pvalues: jnp.ndarray,
    alpha: float = 0.05,
    method: str = "k_minus_1",
    n_groups: int | None = None,
) -> jnp.ndarray:
    """FWER control for MANY equivalence verdicts (guards against BH misuse).

    Correction (multiplicity): across many target/candidate equivalence verdicts,
    control FWER. Two IUT-consistent routes:

    - ``method="k_minus_1"``: scale the nominal level by ``1/(k-1)`` for all
      pairwise equivalence comparisons among ``k`` groups (``n_groups=k``). Less
      conservative than Bonferroni, controls FWER. Ground: TOST ``(k-1)`` scaling.
    - ``method="adaptive_bonferroni"``: plug in an estimate of the proportion of
      non-equivalent pairs to recover power (adaptive-Bonferroni for equivalence).

    **GUARD.** Raw TOST / equivalence p-values are **non-uniform under the interval
    null** (boundary- and ``n``-dependent), so feeding them to vanilla
    Benjamini-Hochberg (which assumes uniform-under-null p-values) is INVALID.
    ``method="bh"`` is therefore disallowed and raises. Ground: interval-null
    non-uniformity (Statistical Papers 2024); Berger-Hsu (1996) IUT.

    Args:
        equivalence_pvalues: Per-verdict equivalence p-values, shape (M,).
        alpha: Family-wise error level.
        method: ``"k_minus_1"`` or ``"adaptive_bonferroni"``.
        n_groups: ``k`` for the ``(k-1)`` scaling (required for that method).

    Returns:
        Boolean mask (shape (M,)); True ⇒ equivalence declared under FWER control.

    """
    p = jnp.asarray(equivalence_pvalues)
    if method == "bh":
        message = (
            "Vanilla Benjamini-Hochberg is INVALID for equivalence/TOST p-values: "
            "the interval null is non-uniform (boundary- and n-dependent). Use "
            "'k_minus_1' or 'adaptive_bonferroni' (IUT-consistent FWER)."
        )
        raise ValueError(message)
    if method == "k_minus_1":
        if n_groups is None or n_groups < _MIN_MULTIPLICITY_GROUPS:
            message = "method='k_minus_1' requires n_groups (k) >= 2."
            raise ValueError(message)
        threshold = alpha / (n_groups - 1)
        return p <= threshold
    if method == "adaptive_bonferroni":
        # Estimate proportion of non-equivalent (null) pairs via a simple
        # Schweder-Spjøtvoll-style plug-in on the equivalence p-values, then
        # apply Bonferroni over the estimated number of true nulls.
        m = p.shape[0]
        pi0_hat = jnp.minimum(1.0, 2.0 * jnp.mean(p > _ADAPTIVE_NULL_THRESHOLD))
        m0_hat = jnp.maximum(1.0, jnp.ceil(pi0_hat * m))
        return p <= (alpha / m0_hat)
    message = f"Unknown method {method!r}."
    raise ValueError(message)
