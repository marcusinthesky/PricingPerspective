"""Generic permutation testing with an explicit JAX/host result boundary.

``permutation_test_result`` is the primary numerical API. It accepts a PRNG key,
keeps every numerical value on device, and returns a registered
:class:`jcor.core.results.TestResult`. ``permutation_test_kernel`` is the smaller
statistic-plus-null assembly primitive used when callers already own resampling.

``permutation_test`` preserves the generic eager API. Energy statistics use the
separate ``energy_permutation_test`` door, whose single ``EnergyGeometry``
argument makes the statistic and hoisted matrix construction inseparable.
"""

from __future__ import annotations

from collections.abc import Callable  # runtime type aliases
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import random, vmap

from jcor.core.axioms import Law
from jcor.core.domains import ConstructionOrigin
from jcor.core.results import TestResult  # runtime annotations
from jcor.core.statistics import (  # typed result contract
    CalibratedTest,
    CalibrationProvenance,
    EnergyTwoSampleStatistic,
    Homogeneity,
    RealizedStatistic,
    ResampledCalibration,
    ResampledNull,
    StatisticProvenance,
    UngroupedDesign,
)
from jcor.core.typing import (  # noqa: TC001  # runtime annotations; see module
    Array,
    ArrayLike,
    Float,
    Int,
    Num,
    PRNGKey,
    Static,
)
from jcor.discrepancy._mmd.energy.geometry import (
    EnergyExponent,
    EnergyGeometry,  # noqa: TC001  # runtime jaxtyping contract
)
from jcor.discrepancy.energy import (
    energy_test_statistic_from_blocks,
    energy_test_statistic_from_distances,
)
from jcor.ground.metrics import cdist
from jcor.inference._permutation.blocks import (
    grouped_block_reduction,
    map_replicates,
    replicate_counts,
)
from jcor.inference._permutation.kernel import (
    _ALTERNATIVES,
    Alternative,  # noqa: TC001  # runtime annotations
    _alternative_message,
    permutation_test_kernel,
)
from jcor.inference._results import (  # noqa: TC001  # runtime annotations
    HypothesisTest,
    to_host_test_result,
)

__all__ = [
    "EnergyPermutationOrigin",
    "EnergyPermutationResult",
    "EnergyStatisticOrigin",
    "energy_permutation_test",
    "energy_permutation_test_result",
    "permutation_test",
    "permutation_test_kernel",
    "permutation_test_result",
]

type StatisticFn = Callable[[Array, Array], Float[Array, ""]]  # noqa: F722  # jaxtyping scalar shape

#: Group count of the two-sample energy design: X and Y, in that pooled order.
_TWO_SAMPLE_GROUPS = 2


class EnergyStatisticOrigin[
    GroundLawT: Law,
    ExponentT: EnergyExponent,
    EnergyLawT: Law,
](ConstructionOrigin):
    """Energy statistic whose geometry remains visible in its static type."""


class EnergyPermutationOrigin[
    GroundLawT: Law,
    ExponentT: EnergyExponent,
    EnergyLawT: Law,
](ConstructionOrigin):
    """Label permutation of one typed energy-statistic construction."""


#: The statistic and the calibration this module produces, each named once.
#: Both stay generic in the three geometry parameters, so the ground law,
#: exponent and energy law still survive into the test result — the bundling
#: shortens the spelling and changes no implication.
type EnergyStatisticProvenance[
    GroundLawA: Law,
    ExponentA: EnergyExponent,
    EnergyLawA: Law,
] = StatisticProvenance[
    EnergyTwoSampleStatistic,
    EnergyStatisticOrigin[GroundLawA, ExponentA, EnergyLawA],
]
type EnergyLabelPermutation[
    GroundLawA: Law,
    ExponentA: EnergyExponent,
    EnergyLawA: Law,
] = CalibrationProvenance[
    UngroupedDesign,
    Homogeneity,
    ResampledCalibration,
    EnergyPermutationOrigin[GroundLawA, ExponentA, EnergyLawA],
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EnergyPermutationResult[
    GroundLawT: Law,
    ExponentT: EnergyExponent,
    EnergyLawT: Law,
]:
    """Typed energy statistic, homogeneity test, null, and geometry."""

    test: CalibratedTest[
        Array,
        EnergyStatisticProvenance[GroundLawT, ExponentT, EnergyLawT],
        EnergyLabelPermutation[GroundLawT, ExponentT, EnergyLawT],
    ]
    null: ResampledNull[Array, ResampledCalibration]
    geometry: EnergyGeometry[GroundLawT, ExponentT, EnergyLawT] = field(
        metadata={"static": True}
    )

    @property
    def statistic(self) -> Array:
        """Observed energy statistic."""
        return self.test.statistic.value

    @property
    def pvalue(self) -> Array:
        """Monte-Carlo add-one p-value."""
        return self.test.pvalue

    @property
    def null_distribution(self) -> Array:
        """Resampled energy statistics."""
        return self.null.draws

    @property
    def n_resamples(self) -> int:
        """Static number of resamples."""
        return self.null.n_resamples


def permutation_test(
    x: ArrayLike,
    y: ArrayLike,
    statistic_fn: StatisticFn,
    num_permutations: int = 999,
    seed: int | None = None,
    alternative: Alternative = "two-sided",
) -> HypothesisTest:
    """Perform an eager two-sample permutation test.

    This compatibility/report adapter preserves the historical return type and
    validation. Transformable callers should use :func:`permutation_test_result`
    with a PRNG key and keep its ``TestResult`` on device.

    Args:
        x: First sample, shape ``(n, d)``.
        y: Second sample, shape ``(m, d)``.
        statistic_fn: Scalar JAX statistic; larger values represent greater
            separation for the ``"greater"`` alternative.
        num_permutations: Number of random permutations.
        seed: Eager reproducibility seed. ``None`` preserves the historical
            deterministic seed-zero behavior.
        alternative: ``"two-sided"``, ``"greater"``, or ``"less"``.

    Returns:
        A validated legacy :class:`HypothesisTest` with Python scalar report
        fields and the JAX null-distribution array.

    Raises:
        ValueError: If ``alternative`` is not recognized.

    Examples:
        >>> def mean_diff(x, y):
        ...     return jnp.abs(jnp.mean(x) - jnp.mean(y))
        >>> x = jnp.array([[1.0], [2.0], [3.0]])
        >>> y = jnp.array([[10.0], [11.0], [12.0]])
        >>> result = permutation_test(x, y, mean_diff, num_permutations=999)
        >>> result.pvalue < 0.05
        True

    """
    result = permutation_test_result(
        jnp.asarray(x),
        jnp.asarray(y),
        statistic_fn,
        random.PRNGKey(seed if seed is not None else 0),
        num_permutations=num_permutations,
        alternative=alternative,
    )
    return to_host_test_result(result)


def energy_permutation_test[
    GroundLawT: Law,
    ExponentT: EnergyExponent,
    EnergyLawT: Law,
](
    x: ArrayLike,
    y: ArrayLike,
    geometry: EnergyGeometry[GroundLawT, ExponentT, EnergyLawT],
    num_permutations: int = 999,
    seed: int | None = None,
    alternative: Alternative = "two-sided",
) -> HypothesisTest:
    """Perform an eager energy permutation test from one compatible geometry."""
    result = energy_permutation_test_result(
        jnp.asarray(x),
        jnp.asarray(y),
        geometry,
        random.PRNGKey(seed if seed is not None else 0),
        num_permutations=num_permutations,
        alternative=alternative,
    )
    return to_host_test_result(
        TestResult(
            statistic=result.statistic,
            pvalue=result.pvalue,
            null_distribution=result.null_distribution,
            n_resamples=result.n_resamples,
        )
    )


def permutation_test_result(
    x: Num[Array, "n ..."],  # noqa: F722  # jaxtyping variadic shape
    y: Num[Array, "m ..."],  # noqa: F722  # jaxtyping variadic shape
    statistic_fn: Static[StatisticFn],
    key: PRNGKey,
    num_permutations: Static[int] = 999,
    alternative: Static[Alternative] = "two-sided",
) -> TestResult[Float[Array, ""]]:  # noqa: F722  # jaxtyping scalar shape
    """Return the transformable JAX result of a two-sample permutation test.

    ``statistic_fn``, ``num_permutations``, and ``alternative`` determine
    tracing or shapes and must be marked static when this
    function is wrapped in :func:`jax.jit`. ``x``, ``y``, and ``key`` remain
    dynamic and may carry a leading axis through :func:`jax.vmap`.

    Args:
        x: First on-device sample, shape ``(n, d)``.
        y: Second on-device sample, shape ``(m, d)``.
        statistic_fn: Traceable scalar statistic.
        key: JAX PRNG key. Numerical code never accepts or converts a seed.
        num_permutations: Static resample count.
        alternative: Static tail convention.

    Returns:
        A registered ``TestResult`` containing only JAX array leaves plus its
        static resample count.

    Raises:
        ValueError: If ``alternative`` is not recognized.

    """
    if alternative not in _ALTERNATIVES:
        raise ValueError(_alternative_message(alternative))

    n = x.shape[0]
    m = y.shape[0]

    observed_stat = jnp.asarray(statistic_fn(x, y))
    combined = jnp.concatenate([x, y], axis=0)

    def permuted_statistic(resample_key: PRNGKey) -> Float[Array, ""]:  # noqa: F722  # jaxtyping scalar shape
        """Evaluate the statistic on one permuted label assignment.

        Args:
            resample_key: Key for this label permutation.

        Returns:
            The scalar statistic for the permuted split.

        """
        perm_idx = random.permutation(resample_key, n + m)
        perm_combined = combined[perm_idx]
        return jnp.asarray(statistic_fn(perm_combined[:n], perm_combined[n:]))

    keys = random.split(key, num_permutations)
    null_dist: Float[Array, "resamples"] = vmap(  # noqa: F821, UP037
        permuted_statistic
    )(keys)
    return permutation_test_kernel(
        observed_stat, null_dist, num_permutations, alternative
    )


def energy_permutation_test_result[
    GroundLawT: Law,
    ExponentT: EnergyExponent,
    EnergyLawT: Law,
](
    x: Num[Array, "n ..."],  # noqa: F722  # jaxtyping variadic shape
    y: Num[Array, "m ..."],  # noqa: F722  # jaxtyping variadic shape
    geometry: Static[EnergyGeometry[GroundLawT, ExponentT, EnergyLawT]],
    key: PRNGKey,
    num_permutations: Static[int] = 999,
    alternative: Static[Alternative] = "two-sided",
) -> EnergyPermutationResult[GroundLawT, ExponentT, EnergyLawT]:
    """Return a transformable distance-hoisted energy permutation result.

    Args:
        x: First on-device sample.
        y: Second on-device sample.
        geometry: Compatible ground, exponent regime, and result-law contract.
        key: Permutation PRNG key.
        num_permutations: Static number of null draws.
        alternative: Static tail convention.

    Returns:
        The observed statistic, p-value, and null draws as a ``TestResult``.

    """
    if alternative not in _ALTERNATIVES:
        raise ValueError(_alternative_message(alternative))

    n = x.shape[0]
    m = y.shape[0]
    combined = jnp.concatenate([x, y], axis=0)
    distances = cdist(combined, combined, metric=geometry.ground)
    distances = distances**geometry.exponent.value
    observed_stat = energy_test_statistic_from_distances(distances, n)

    keys = random.split(key, num_permutations)
    perms: Int[Array, "resamples pooled"] = vmap(  # noqa: F722  # jaxtyping shape
        lambda resample_key: random.permutation(resample_key, n + m)
    )(keys)

    # Slot -> group labels are fixed by construction: a permutation sends the
    # first n pooled slots to X and the remaining m to Y, so only *which*
    # pooled rows fill those slots varies per replicate. That is exactly the
    # membership-matrix parameterisation `jcor.inference._permutation.blocks`
    # exploits, letting the replicate ride in C rather than in a reindexed D.
    slot_groups = (jnp.arange(n + m) >= n).astype(jnp.int32)

    def null_statistic(perm: Array) -> Float[Array, ""]:  # noqa: F722  # jaxtyping scalar shape
        """Recover one energy statistic without reindexing the pooled distances.

        Args:
            perm: Permutation indices into the pooled distance matrix.

        Returns:
            The scalar energy statistic for that permutation.

        """
        counts = replicate_counts(
            perm, slot_groups, _TWO_SAMPLE_GROUPS, n + m, distances.dtype
        )
        block_sums, _ = grouped_block_reduction(counts, distances)
        return energy_test_statistic_from_blocks(block_sums, n, m)

    # Chunked, not a bare vmap: `map_replicates` keeps the scan memory bound the
    # per-replicate gather used to provide, while still handing XLA a batched
    # dot_general to fuse (see the blocks module docstring for the measurement).
    null_dist: Float[Array, "resamples"] = map_replicates(  # noqa: F821, UP037
        null_statistic, perms
    )
    raw = permutation_test_kernel(
        observed_stat, null_dist, num_permutations, alternative
    )
    statistic = RealizedStatistic(
        value=raw.statistic,
        provenance=StatisticProvenance(),
    )
    test = CalibratedTest(
        statistic=statistic,
        pvalue=raw.pvalue,
        provenance=CalibrationProvenance(),
    )
    return EnergyPermutationResult(
        test=test,
        null=ResampledNull(
            draws=raw.null_distribution,
            n_resamples=raw.n_resamples,
        ),
        geometry=geometry,
    )
