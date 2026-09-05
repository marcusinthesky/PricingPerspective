"""Bootstrap estimates with an explicit JAX/host result boundary.

:func:`bootstrap_estimate` is the primary numerical API: it accepts a PRNG key
and returns a registered :class:`jcor.core.results.EstimateResult` without
concretizing a leaf. :func:`bootstrap_confidence_interval` preserves the eager
``(float, float)`` API by converting a seed at entry and calling the named
``to_host_confidence_interval`` adapter only after numerical work completes.

The private grouped-energy bootstrap follows the same rule through a numerical
``*_result`` core and a compatibility wrapper for its eager mixture-test
consumer.
"""

from __future__ import annotations

from collections.abc import Callable  # runtime type aliases

import jax.numpy as jnp
from jax import random, vmap

from jcor.core.results import EstimateResult
from jcor.core.typing import (  # noqa: TC001  # runtime annotations; see core.typing
    Array,
    ArrayLike,
    Float,
    Num,
    PRNGKey,
    Static,
)
from jcor.inference._permutation.blocks import (
    grouped_block_reduction,
    map_replicates,
    replicate_counts,
)
from jcor.inference._permutation.common import _energy_from_group_blocks
from jcor.inference._results import to_host_confidence_interval

__all__ = ["bootstrap_confidence_interval", "bootstrap_estimate"]

type BootstrapStatistic = Callable[[Array], Float[Array, ""]]  # noqa: F722  # jaxtyping scalar shape
type EagerBootstrapStatistic = Callable[[Array], float | Array]


def _validate_bootstrap_config(confidence_level: float, num_bootstrap: int) -> None:
    """Validate shape-determining bootstrap configuration eagerly.

    Args:
        confidence_level: Requested percentile interval level.
        num_bootstrap: Requested number of bootstrap draws.

    Raises:
        ValueError: If either value is outside its valid domain.

    """
    if not 0.0 < confidence_level < 1.0:
        message = f"confidence_level must be in (0, 1), got {confidence_level}"
        raise ValueError(message)
    if num_bootstrap < 1:
        message = f"num_bootstrap must be positive, got {num_bootstrap}"
        raise ValueError(message)


def bootstrap_confidence_interval(
    data: ArrayLike,
    statistic_fn: EagerBootstrapStatistic,
    confidence_level: float = 0.95,
    num_bootstrap: int = 999,
    seed: int | None = None,
) -> tuple[float, float]:
    """Compute an eager percentile-bootstrap confidence interval.

    This compatibility/report adapter preserves the historical pair of Python
    floats. Transformable callers should use :func:`bootstrap_estimate` with a
    PRNG key and retain the returned ``EstimateResult``.

    Args:
        data: Array-like sample data whose leading axis indexes observations.
        statistic_fn: Function that computes statistic from a sample.
            Should return a scalar.
        confidence_level: Confidence level in (0, 1). Default 0.95.
        num_bootstrap: Number of bootstrap samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (lower_bound, upper_bound) for confidence interval.

    Raises:
        ValueError: If ``confidence_level`` is not in ``(0, 1)`` or
            ``num_bootstrap`` is not positive.

    Examples:
        >>> data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> mean_fn = lambda x: jnp.mean(x)
        >>> lower, upper = bootstrap_confidence_interval(
        ...     data, mean_fn, confidence_level=0.95, num_bootstrap=999
        ... )
        >>> 1.0 < lower < 3.0 < upper < 5.0
        True

    """
    _validate_bootstrap_config(confidence_level, num_bootstrap)
    values = jnp.asarray(data)

    def array_statistic(sample: Array) -> Float[Array, ""]:  # noqa: F722  # jaxtyping scalar shape
        """Narrow an eager scalar statistic to the numerical array contract.

        Args:
            sample: One bootstrap sample.

        Returns:
            The statistic as a scalar JAX array.

        """
        return jnp.asarray(statistic_fn(sample))

    result = bootstrap_estimate(
        values,
        array_statistic,
        random.PRNGKey(seed if seed is not None else 0),
        confidence_level=confidence_level,
        num_bootstrap=num_bootstrap,
    )
    return to_host_confidence_interval(result)


def bootstrap_estimate(
    data: Num[Array, "n ..."],  # noqa: F722  # jaxtyping variadic shape
    statistic_fn: Static[BootstrapStatistic],
    key: PRNGKey,
    confidence_level: Static[float] = 0.95,
    num_bootstrap: Static[int] = 999,
) -> EstimateResult[Float[Array, ""]]:  # noqa: F722  # jaxtyping scalar shape
    """Return a transformable percentile-bootstrap estimate pytree.

    ``statistic_fn``, ``confidence_level``, and ``num_bootstrap`` determine
    tracing or output shape and must be static under :func:`jax.jit`. ``data``
    and ``key`` remain dynamic and may be mapped over a leading axis.

    Args:
        data: On-device sample with observations on the leading axis.
        statistic_fn: Traceable scalar statistic.
        key: JAX PRNG key; numerical code never accepts or converts a seed.
        confidence_level: Static percentile interval level in ``(0, 1)``.
        num_bootstrap: Static positive number of bootstrap replicates.

    Returns:
        An ``EstimateResult`` containing the observed estimate, bootstrap
        standard error, percentile interval, a scalar JAX ``True`` convergence
        flag, and static observation count.

    Raises:
        ValueError: If bootstrap configuration is invalid.

    """
    _validate_bootstrap_config(confidence_level, num_bootstrap)
    n_obs = data.shape[0]

    def bootstrap_statistic(resample_key: PRNGKey) -> Float[Array, ""]:  # noqa: F722  # jaxtyping scalar shape
        """Evaluate the statistic on one with-replacement resample.

        Args:
            resample_key: Key for this bootstrap draw.

        Returns:
            The scalar statistic for the draw.

        """
        boot_idx = random.choice(resample_key, n_obs, shape=(n_obs,), replace=True)
        return jnp.asarray(statistic_fn(data[boot_idx]))

    keys = random.split(key, num_bootstrap)
    distribution: Float[Array, "resamples"] = vmap(  # noqa: F821, UP037
        bootstrap_statistic
    )(keys)
    alpha = 1.0 - confidence_level
    lower = jnp.percentile(distribution, 100 * alpha / 2, axis=0)
    upper = jnp.percentile(distribution, 100 * (1.0 - alpha / 2), axis=0)
    ddof = 1 if num_bootstrap > 1 else 0

    return EstimateResult(
        estimate=jnp.asarray(statistic_fn(data)),
        se=jnp.std(distribution, axis=0, ddof=ddof),
        ci=(lower, upper),
        converged=jnp.ones((), dtype=bool),
        n_obs=n_obs,
    )


def _bootstrap_energy_ci_result(
    distances: Float[Array, "n n"],  # noqa: F722  # jaxtyping shape
    membership: Float[Array, "n groups"],  # noqa: F722  # jaxtyping shape
    size_arr: Float[Array, "groups"],  # noqa: F821, UP037  # jaxtyping shape
    sizes: list[int],
    w_arr: Float[Array, "candidates"],  # noqa: F821, UP037  # jaxtyping shape
    confidence_level: Static[float],
    num_bootstrap: Static[int],
    rng: PRNGKey,
) -> tuple[
    tuple[Float[Array, ""], Float[Array, ""]],  # noqa: F722  # jaxtyping scalar shape
    Float[Array, "resamples"],  # noqa: F821, UP037  # jaxtyping shape
]:
    """Return the JAX interval/distribution pytree for Ê(X, Y_{w*}).

    **D-reuse (t05.2, author-approved 2026-07-18).** Each bootstrap replicate
    resamples rows *within* each pooled group (the target ``X`` and each
    candidate ``C_k``) with replacement, so its pairwise distances are exactly
    a reindexing of the single precomputed pooled matrix ``distances`` — no
    fresh ``cdist`` per replicate. The per-group draw uses
    the IDENTICAL PRNG derivation as the pre-D-reuse implementation
    (``kx, *kc = random.split(key, 1 + K)``; ``random.choice(kx, n,
    shape=(n,), replace=True)`` for ``X`` and ``random.choice(kc[i], m_i,
    shape=(m_i,), replace=True)`` per candidate), so the drawn indices are
    identical; only the distances they index into change from a fresh
    ``cdist`` to a reindexed view (~1e-8 XLA-matmul-tiling class). Group sizes
    are preserved under within-group resampling, so the fixed contiguous
    ``membership``/``size_arr`` evaluate ``Ê*`` via the same block-sum
    machinery (:func:`_energy_from_group_blocks`) as the observed estimate.

    That reindexing is applied to the **membership matrix, not the distance
    matrix**: :func:`jcor.inference._permutation.blocks.replicate_counts` turns
    ``idx`` into the multiplicity matrix ``C`` and
    :func:`~jcor.inference._permutation.blocks.grouped_block_reduction`
    contracts ``B = C D Cᵀ``, which is algebraically identical to the former
    ``Mᵀ (D[idx][:, idx]) M`` (bit-identical on exactly-representable entries)
    but never materialises the ``(N, N)`` gather. Replicates are driven with
    :func:`~jcor.inference._permutation.blocks.map_replicates` (chunked
    ``lax.map``, not a bare ``vmap``) to bound peak memory at the production
    ``num_bootstrap`` (2000).

    Args:
        distances: Pooled ``(N, N)`` pre-exponentiated distance matrix, rows
            ordered ``[X, C_1, ..., C_K]``.
        membership: One-hot ``(N, K+1)`` group membership for that row order.
        size_arr: ``(K+1,)`` float group sizes (``membership.sum(0)``).
        sizes: Python list ``[n, m_1, ..., m_K]`` of group sizes.
        w_arr: Mixture weights ``(K,)``.
        confidence_level: Percentile interval confidence level.
        num_bootstrap: Number of bootstrap replicates.
        rng: Bootstrap PRNG key.

    Returns:
        Tuple ((low, high) percentile CI, bootstrap Ê distribution). The
        distribution is reused for the equivalence p-value (P(Ê* ≥ δ)).

    """
    n = sizes[0]
    m_sizes = sizes[1:]
    # Contiguous global start offset of each group in the pooled row order.
    group_offsets = [0]
    for s in sizes:
        group_offsets.append(group_offsets[-1] + s)
    n_groups = len(sizes)
    n_pooled = distances.shape[0]
    # Slot -> group labels, recovered from the fixed contiguous one-hot. Group
    # sizes are preserved under within-group resampling, so this vector is the
    # same for every replicate and is hoisted out of the loop.
    slot_groups = jnp.argmax(membership, axis=1).astype(jnp.int32)

    def _boot(key: PRNGKey) -> Float[Array, ""]:  # noqa: F722  # jaxtyping scalar shape
        """Evaluate energy on one set of within-group resamples.

        Args:
            key: Key for this grouped bootstrap draw.

        Returns:
            The resampled energy estimate.

        """
        kx, *kc = random.split(key, 1 + len(m_sizes))
        parts = [group_offsets[0] + random.choice(kx, n, shape=(n,), replace=True)]
        for i, mi in enumerate(m_sizes):
            parts.append(
                group_offsets[i + 1]
                + random.choice(kc[i], mi, shape=(mi,), replace=True)
            )
        idx = jnp.concatenate(parts)
        # Resampling with replacement makes `idx` multi-valued, so the replicate
        # rides in a *multiplicity* matrix rather than a 0/1 indicator; the
        # scatter-add in `replicate_counts` accumulates the repeats, which is
        # exactly the C of `B = C D Cᵀ`. Identical to the former
        # `membership.T @ distances[idx][:, idx] @ membership`, without ever
        # materialising the (N, N) gather.
        counts = replicate_counts(idx, slot_groups, n_groups, n_pooled, distances.dtype)
        block_sums, _ = grouped_block_reduction(counts, distances)
        return _energy_from_group_blocks(block_sums, size_arr, w_arr)

    keys = random.split(rng, num_bootstrap)
    # Chunked map (scan over `batch_size`-wide vmapped slabs) keeps the peak
    # memory bound the per-replicate gather used to provide, while reproducing
    # the exact per-draw random.split derivation.
    dist: Float[Array, "resamples"] = map_replicates(  # noqa: F821, UP037
        _boot, keys
    )
    a = 1.0 - confidence_level
    lo = jnp.percentile(dist, 100 * a / 2)
    hi = jnp.percentile(dist, 100 * (1 - a / 2))
    return (lo, hi), dist


def _bootstrap_energy_ci(
    distances: Float[Array, "n n"],  # noqa: F722  # jaxtyping shape
    membership: Float[Array, "n groups"],  # noqa: F722  # jaxtyping shape
    size_arr: Float[Array, "groups"],  # noqa: F821, UP037  # jaxtyping shape
    sizes: list[int],
    w_arr: Float[Array, "candidates"],  # noqa: F821, UP037  # jaxtyping shape
    confidence_level: float,
    num_bootstrap: int,
    rng: PRNGKey,
) -> tuple[
    tuple[float, float],
    Float[Array, "resamples"],  # noqa: F821, UP037  # jaxtyping shape
]:
    """Preserve the mixture runner's eager confidence-interval surface.

    The numerical work lives entirely in :func:`_bootstrap_energy_ci_result`.
    This compatibility adapter concretizes only the two report bounds after
    that core returns; the bootstrap distribution remains a JAX array.

    Args:
        distances: Pooled pre-exponentiated distance matrix.
        membership: One-hot group membership matrix.
        size_arr: Floating group sizes.
        sizes: Static Python group sizes.
        w_arr: Candidate-mixture weights.
        confidence_level: Percentile interval level.
        num_bootstrap: Bootstrap replicate count.
        rng: Bootstrap PRNG key.

    Returns:
        Concrete confidence bounds and the on-device bootstrap distribution.

    """
    interval, distribution = _bootstrap_energy_ci_result(
        distances,
        membership,
        size_arr,
        sizes,
        w_arr,
        confidence_level,
        num_bootstrap,
        rng,
    )
    return to_host_confidence_interval(interval), distribution
