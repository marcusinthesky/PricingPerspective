"""Typed Monte-Carlo calibration of a pure DISCO decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic

import jax
import jax.numpy as jnp

from jcor.association.decomposition import (  # noqa: TC001  # typed carriers
    DiscoDecomposition,
    DiscoFRatioProvenance,
    DispersionDecomposableLaw,
    disco_components,
)
from jcor.association.dispersion import disco_decomposition
from jcor.core.axioms import Axioms, requires
from jcor.core.domains import ConstructionOrigin, DomainT, ObjectDomain, OriginT
from jcor.core.matrices import DMat  # noqa: TC001  # typed carrier
from jcor.core.random import cell_key, permutation_batch
from jcor.core.statistics import (  # noqa: TC001  # carriers and static axes
    CalibratedTest,
    CalibrationProvenance,
    GroupDesign,
    GroupedDesign,
    NoGroupEffect,
    ResampledCalibration,
    ResampledNull,
)
from jcor.core.typing import Array, ArrayLike  # noqa: TC001  # typed carriers
from jcor.inference.calibration import add_one_p_kernel

__all__ = [
    "DiscoPermutationOrigin",
    "DiscoPermutationResult",
    "disco_permutation_null",
    "disco_permutation_test",
]


class DiscoPermutationOrigin(ConstructionOrigin):
    """Label-permutation calibration with a Monte-Carlo add-one p-value."""


#: The calibration this module performs, named once. Four axes — grouped
#: design, no-group-effect null, Monte-Carlo resampling, label permutation —
#: which together identify the procedure and travel together everywhere.
type DiscoLabelPermutation = CalibrationProvenance[
    GroupedDesign,
    NoGroupEffect,
    ResampledCalibration,
    DiscoPermutationOrigin,
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DiscoPermutationResult(Generic[DomainT, OriginT]):  # noqa: UP046
    """Compose the pure decomposition, calibrated test, and resampled null."""

    decomposition: DiscoDecomposition[Array, DomainT, OriginT]
    test: CalibratedTest[Array, DiscoFRatioProvenance[OriginT], DiscoLabelPermutation]
    null: ResampledNull[Array, ResampledCalibration]


@jax.jit(static_argnames=("n_groups", "n_permutations"))
def _disco_permutation_null_kernel(
    dist: jax.Array,
    group_codes: jax.Array,
    key: jax.Array,
    n_groups: int,
    n_permutations: int,
) -> jax.Array:
    """Reuse one pooled matrix while permuting labels only."""
    permuted_codes = permutation_batch(key, group_codes, n_permutations)
    _, _, _, _, f_ratio = jax.vmap(disco_decomposition, in_axes=(None, 0, None))(
        dist, permuted_codes, n_groups
    )
    return f_ratio


def disco_permutation_null(
    dist: ArrayLike,
    group_codes: ArrayLike,
    n_groups: int,
    n_permutations: int,
    random_state: int,
) -> jax.Array:
    """Draw the label-permutation null while holding the matrix fixed."""
    if isinstance(n_permutations, bool) or n_permutations <= 0:
        message = "n_permutations must be a positive integer"
        raise ValueError(message)
    key = cell_key(random_state, "association", "disco_permutation_null")
    return _disco_permutation_null_kernel(
        jnp.asarray(dist),
        jnp.asarray(group_codes),
        key,
        n_groups,
        n_permutations,
    )


@requires(Axioms.NONNEGATIVE | Axioms.ZERO_DIAGONAL | Axioms.SYMMETRY)
def disco_permutation_test[DomainA: ObjectDomain, OriginA: ConstructionOrigin](
    matrix: DMat[DomainA, DispersionDecomposableLaw, OriginA],
    design: GroupDesign,
    *,
    n_permutations: int,
    random_state: int,
) -> DiscoPermutationResult[DomainA, OriginA]:
    """Calibrate a typed DISCO F-ratio by label permutation."""
    decomposition = disco_components(matrix, design)
    # Both paths stay in JAX end to end now that the door returns `jax.Array`;
    # before t65 this had to reach past it to avoid a JAX -> NumPy -> JAX trip.
    draws = disco_permutation_null(
        matrix.values,
        design.codes,
        design.n_groups,
        n_permutations,
        random_state,
    )
    pvalue = add_one_p_kernel(decomposition.f_ratio.value, draws)
    test = CalibratedTest(
        statistic=decomposition.f_ratio,
        pvalue=pvalue,
        provenance=CalibrationProvenance(),
    )
    return DiscoPermutationResult(
        decomposition=decomposition,
        test=test,
        null=ResampledNull(draws=draws, n_resamples=n_permutations),
    )
