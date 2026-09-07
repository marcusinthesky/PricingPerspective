"""Branded MMD matrix producers — squared estimates and rooted distances.

Position
--------
rank 3 (WAIST 1) · private implementation behind :mod:`jcor.discrepancy.mmd`

Why this is a separate module from :mod:`jcor.discrepancy._mmd.matrix`
----------------------------------------------------------------------
The t62 producers there own the numerics: same packed
traversal, same ``where``-based masked reductions, same true-count divisors, and
no equal-cloud-size signature anywhere. This module adds only the T55.3 brand,
by delegation, so the ragged contract cannot drift between the two spellings.
It also keeps the three-axis :class:`jcor.core.matrices.DMat` as the sole
distance carrier returned by the public matrix surface.

The two containers here are siblings for a mathematical reason
--------------------------------------------------------------
``MMD²`` is not a dissimilarity. The diagonal-debiased U-statistic estimate is
routinely negative and its diagonal need not vanish, so
:func:`mmd_squared_estimate_matrix` returns a
:class:`~jcor.core.matrices.PairwiseEstimateMatrix` for **both** schemes — a
V-statistic grid is nonnegative but is still a squared estimate, not a distance.
Only :func:`rooted_mmd_matrix`, which clamps and roots the V-statistic form and
zeroes the diagonal, produces a ``DMat``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from jcor.core.domains import EmpiricalDistribution  # runtime contract
from jcor.core.matrices import (  # noqa: TC001  # runtime contract
    DMat,
    HilbertianMetricLaw,
    HilbertianPseudometricLaw,
    PairwiseEstimateMatrix,
    VStatisticScheme,
    _dmat,
    _pairwise_estimate,
)
from jcor.core.statistics import (  # runtime contract
    UStatisticEstimator,
    VStatisticEstimator,
)
from jcor.core.typing import Array, Float  # noqa: TC001  # runtime contract
from jcor.discrepancy._mmd.matrix import mmd_matrix_values, mmd_squared_matrix
from jcor.discrepancy.provenance import (  # noqa: TC001  # runtime contract
    RootedMmd,
    SquaredMmdConstruction,
    SquaredMmdFunctional,
)
from jcor.ground.similarities import LINEAR_KERNEL

if TYPE_CHECKING:
    from jcor.ground.similarities import (
        CharacteristicKernelLaw,
        PositiveSemidefiniteKernelLaw,
        SimilarityStrategy,
    )

__all__ = ["mmd_squared_estimate_matrix", "rooted_mmd_matrix"]

#: The squared-MMD estimate matrix at the diagonal-debiased scheme. Its values
#: may be negative; no distance law is attached, and none may be.
type _UStatisticEstimates = PairwiseEstimateMatrix[
    EmpiricalDistribution,
    SquaredMmdFunctional,
    UStatisticEstimator,
    SquaredMmdConstruction,
]

#: The squared-MMD estimate matrix at the biased plug-in scheme. Nonnegative
#: under a positive-definite kernel, and still an estimate rather than a
#: distance — it is ``MMD²``, not ``MMD``.
type _VStatisticEstimates = PairwiseEstimateMatrix[
    EmpiricalDistribution,
    SquaredMmdFunctional,
    VStatisticEstimator,
    SquaredMmdConstruction,
]


@overload
def mmd_squared_estimate_matrix(
    sample_sets: list[Array],
    *,
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw] = ...,
    reference: Float[Array, " d"] | None = ...,
    unbiased: Literal[True],
) -> _UStatisticEstimates: ...


@overload
def mmd_squared_estimate_matrix(
    sample_sets: list[Array],
    *,
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw] = ...,
    reference: Float[Array, " d"] | None = ...,
    unbiased: Literal[False] = ...,
) -> _VStatisticEstimates: ...


def mmd_squared_estimate_matrix(
    sample_sets: list[Array],
    *,
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw] = LINEAR_KERNEL,
    reference: Float[Array, " d"] | None = None,
    unbiased: bool = False,
) -> _UStatisticEstimates | _VStatisticEstimates:
    """Brand :func:`jcor.discrepancy.mmd.mmd_squared_matrix` as a realized estimate.

    The scheme is resolved *statically* from the ``unbiased`` literal, so a
    consumer written against the V-statistic plug-in cannot be handed the signed
    U-statistic grid and vice versa. Neither is a
    :class:`~jcor.core.matrices.DMat`: the U-statistic estimate can be negative,
    and even the nonnegative V-statistic grid is ``MMD²`` rather than a
    distance. Root it with :func:`rooted_mmd_matrix` to obtain one.

    Delegates wholly to the t62 producer, so ragged clouds are admitted on
    exactly the same terms and every divisor remains a product of true counts.

    Args:
        sample_sets: Nonempty list of rank-two ``(m_k, d)`` clouds; row counts
            may differ, the feature width may not, and no cloud may be empty.
        kernel: Declared PSD strategy with parameters and law closed.
        reference: Reference shared by every distance-induced Gram block.
        unbiased: Must remain false; a signed U-statistic cannot be rooted.
        unbiased: Drop the within-sample diagonals. Pass a literal; the return
            brand is chosen by it.

    Returns:
        The branded squared-MMD estimates at the scheme ``unbiased`` selected.

    Raises:
        ValueError: Whatever the delegate raises for an inadmissible cloud list.

    """
    values = mmd_squared_matrix(
        sample_sets,
        kernel=kernel,
        reference=reference,
        unbiased=unbiased,
    )
    if unbiased:
        debiased: _UStatisticEstimates = _pairwise_estimate(values)
        return debiased
    plugin: _VStatisticEstimates = _pairwise_estimate(values)
    return plugin


@overload
def rooted_mmd_matrix(
    sample_sets: list[Array],
    *,
    kernel: SimilarityStrategy[CharacteristicKernelLaw],
    reference: Float[Array, " d"] | None = ...,
    unbiased: Literal[False] = ...,
) -> DMat[
    EmpiricalDistribution,
    HilbertianMetricLaw,
    RootedMmd[VStatisticScheme],
]: ...


@overload
def rooted_mmd_matrix(
    sample_sets: list[Array],
    *,
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw] = ...,
    reference: Float[Array, " d"] | None = ...,
    unbiased: Literal[False] = ...,
) -> DMat[
    EmpiricalDistribution,
    HilbertianPseudometricLaw,
    RootedMmd[VStatisticScheme],
]: ...


def rooted_mmd_matrix(
    sample_sets: list[Array],
    *,
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw] = LINEAR_KERNEL,
    reference: Float[Array, " d"] | None = None,
    unbiased: Literal[False] = False,
) -> DMat[
    EmpiricalDistribution,
    HilbertianPseudometricLaw,
    RootedMmd[VStatisticScheme],
]:
    """Brand the rooted V-statistic MMD matrix with domain, law, and origin.

    The typed successor to :func:`jcor.discrepancy.mmd.mmd_matrix`, which it
    calls — the clamp, the root, and the zeroed diagonal all stay in one place.
    There is no ``unbiased`` parameter, and that absence is the point: rooting a
    signed U-statistic grid is meaningless, and ``RootedMmd[UStatisticScheme]``
    is unspellable because :data:`jcor.core.matrices.SchemeT` is bounded at the
    nonnegative schemes.

    Every declared PSD kernel yields a Hilbertian pseudometric. A strategy whose
    law additionally declares ``characteristic`` selects the metric overload;
    separation is therefore carried by the kernel contract rather than inferred
    from one observed Gram matrix or from a built-in name.

    Args:
        sample_sets: Nonempty list of rank-two ``(m_k, d)`` clouds, admitted on
            exactly the terms :func:`mmd_squared_estimate_matrix` admits.
        kernel: Declared PSD strategy with parameters and law closed.
        reference: Reference shared by every distance-induced Gram block.
        unbiased: Must remain false; a signed U-statistic cannot be rooted.

    Returns:
        The branded symmetric MMD distance matrix with a zero diagonal.

    Raises:
        ValueError: Whatever the delegate raises for an inadmissible cloud list.

    """
    values = mmd_matrix_values(
        sample_sets,
        kernel=kernel,
        reference=reference,
        unbiased=unbiased,
    )
    branded: DMat[
        EmpiricalDistribution,
        HilbertianPseudometricLaw,
        RootedMmd[VStatisticScheme],
    ] = _dmat(values)
    return branded
