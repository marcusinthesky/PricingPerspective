"""Estimating-equation contracts — ``psi``, its Jacobian, and its variability.

Position
--------
rank 0 · importable by every stage

What this module is for
-----------------------
:mod:`jcor.core.matrices` carries what a *dissimilarity* is and
:mod:`jcor.core.statistics` carries what a *number* is. Neither says what an
**estimator** is, and an estimator is identified by the equation it solves
rather than by the model family it belongs to. OLS, a Poisson GLM, GEE, 2SLS,
quantile regression and M-estimation differ only in ``psi``; everything
downstream of ``psi`` is shared::

    psi     the estimating function, evaluated per observation   (T, m)
    J       its Jacobian in the parameters — the *bread*         (T, m, k)
    Omega   the variability matrix of psi — the *meat*           (m, m)

Writing that triple once is what lets :func:`sandwich_covariance` — and later
the Wald/score/LR trio — be written once instead of per model family. A large
part of why comparable packages re-derive covariance and testing for every
family is that they have no name for this triple.

Why the contract is at rank 0 and its producers are not
-------------------------------------------------------
For the same reason :class:`jcor.core.matrices.DMat` is here while ``cdist`` is
in ``ground``, and the same reason
:func:`jcor.core.matrices.metrize_energy` moved here rather than staying in
``discrepancy``: only rank 0 can name every party to the composition. A
long-run covariance is produced at rank 5 (:mod:`jcor.operators.longrun`), a
moment condition at rank 7 (:mod:`jcor.inference.gmm`), and a calibrated test
above that — so no stage can state the rule relating them without importing a
later stage. **The stage rank fences the producers; this module holds the
contract.** Filing the contract inside a producing stage would place it above
the models that need to name it.

Omega's scheme is a soundness fact, not a label
------------------------------------------------
:mod:`jcor.inference.gmm` already depends on this distinction and carries it as
a ``centered: Static[bool]`` flag with the reasoning in prose: a Wald statistic
wants the plain Newey-West long-run covariance, because there the moment
process is mean-zero by construction, while a Hansen ``J`` wants Hall's (1987)
mean-centered form, because under misspecification ``E[g_t] != 0`` and the
plain weighting matrix diverges at ``O(T / b_T)``. A boolean cannot express
that one consumer requires one form and not the other.

:class:`PlainLongRun` and :class:`MeanCenteredLongRun` are therefore nominal
**siblings** under :class:`VariabilityScheme`, neither usable where the other is
required — exactly as :class:`jcor.core.statistics.VStatisticEstimator` and
:class:`jcor.core.statistics.UStatisticEstimator` are siblings, and for the same
reason: a consumer that type-checks against one having been handed the other is
silently wrong with no static symptom.

Nothing semantic is a runtime leaf
----------------------------------
The axis markers are plain empty classes and the descriptors are zero-field
frozen dataclasses, exactly as in :mod:`jcor.core.domains`. Every carrier below
has one array leaf and a static producer token, so ``tree_leaves`` sees only the
numbers and ``jit``/``vmap`` are unaffected.

Known gaps, recorded rather than papered over
---------------------------------------------
1. **The live producer is unmigrated.** ``inference/gmm.py`` still selects its
   long-run form with ``centered: Static[bool]`` and returns an unbranded
   ``(m, m)`` array; ``hansen_j`` should demand
   ``VariabilityMatrix[..., MeanCenteredLongRun]``. The brands here are that
   migration's target, not a description of it.
2. **Only :class:`MomentCondition` has a producer.** Likelihood scores,
   M-estimator equations and quasi-scores are named by the stage that first
   produces one, never in advance — an empty marker is an affordance a reader
   will act on.
3. **Provenance is unenforced beyond construction**, as in
   :mod:`jcor.core.statistics`. The producer token stops arbitrary code writing
   ``EstimatingFunction(values=...)``; it does not check that the values are the
   equation the brand names.
4. **No checked door.** :func:`jcor.core.matrices.checked_dmat` exists because
   persisted distance matrices are read back. No jcor path reads back a
   persisted ``psi`` or ``Omega`` yet, so adding one now would ship an untested
   boundary; the widest-brand rule it would follow is recorded on
   :class:`VariabilityScheme`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar, final

import jax
import jax.numpy as jnp

from jcor.core.domains import OriginT  # runtime: `EquationContract` Generic base
from jcor.core.typing import (  # noqa: TC001  # runtime; see jcor.core.typing docstring
    Array,
    Float,
)

__all__ = [
    "EquationContract",
    "EquationDescriptor",
    "EquationJacobian",
    "EquationKindT",
    "EquationT",
    "EstimatingEquationKind",
    "EstimatingFunction",
    "MeanCenteredLongRun",
    "MomentCondition",
    "PlainLongRun",
    "SandwichCovariance",
    "VariabilityMatrix",
    "VariabilityScheme",
    "VariabilityT",
    "sandwich_covariance",
    "unsafe_assume_equation_jacobian",
    "unsafe_assume_estimating_function",
    "unsafe_assume_variability_matrix",
]


# --- The equation axis. Which equation `psi` is, never how it is solved. ---


class EstimatingEquationKind:
    """Root of the estimating-equation axis; names no equation.

    Deliberately *not* an estimator axis. ``2SLS``, ``LIML`` and ``IV-GMM``
    solve the same moment condition by different criteria, so the estimator is a
    fact about the algorithm, while what travels through the sandwich is the
    equation. :class:`jcor.core.statistics.EstimatorScheme` carries the former.
    """


class MomentCondition(EstimatingEquationKind):
    """``E[g(Z, theta)] = 0`` — a moment restriction, possibly overidentified.

    The only inhabitant with a producer in jcor today (``inference/gmm``,
    ``gel``, ``gms``, ``wolak``). Overidentification is a property of the
    ``(m, k)`` shape rather than a separate marker: ``m > k`` is visible on the
    Jacobian and is what gives the Hansen ``J`` its ``m - k`` degrees of
    freedom.
    """


# --- The variability scheme axis. How Omega was formed; a soundness fact. ---


class VariabilityScheme:
    """Root of the variability-scheme axis; the uninformative scheme.

    The widest brand, and therefore what any future checked door must return: a
    persisted ``(m, m)`` array shows squareness, finiteness and symmetry, and
    none of those recover whether the sample mean was subtracted before the
    autocovariances were formed. A result at this brand satisfies neither
    :class:`PlainLongRun` nor :class:`MeanCenteredLongRun`.
    """


@final
class PlainLongRun(VariabilityScheme):
    """Long-run covariance of ``psi`` formed **without** mean centering.

    Correct where the estimating function is mean-zero by construction — a Wald
    statistic on a correctly specified moment condition. Produced by
    :func:`jcor.operators.longrun.newey_west_variability`.
    """


@final
class MeanCenteredLongRun(VariabilityScheme):
    """Hall (1987) long-run covariance, formed after subtracting ``gbar``.

    An unrelated **sibling** of :class:`PlainLongRun`, not a refinement of it.
    Consistent for the long-run variance of ``g_t - E[g_t]`` even when
    ``E[g_t] != 0``, which is what keeps the Hansen ``J`` weighting matrix from
    diverging at ``O(T / b_T)`` under misspecification. Substitution is unsound
    in **both** directions: centering a genuinely mean-zero process discards a
    degree of freedom, and failing to centre a misspecified one is the
    divergence above. Produced by
    :func:`jcor.operators.longrun.hall_centered_variability`.
    """


# --- Axis parameters. Variance names the flow it blocks; phantom otherwise. ---

#: **Covariant** equation kind. Explicit ``TypeVar`` for the reason given in
#: :mod:`jcor.core.domains`: these are phantom parameters, so PEP 695 has
#: nothing to infer variance from and settles on invariant.
EquationKindT = TypeVar(  # noqa: PLC0105
    "EquationKindT",
    bound=EstimatingEquationKind,
    covariant=True,
)

#: **Covariant** variability scheme. Blocks: a plain long-run covariance
#: supplied where a mean-centered one is required, and conversely. Named
#: ``VariabilityT`` rather than ``SchemeT`` because
#: :mod:`jcor.core.matrices` already binds that name to the *energy* estimation
#: scheme, and the two axes are unrelated.
VariabilityT = TypeVar(  # noqa: PLC0105
    "VariabilityT",
    bound="VariabilityScheme",
    covariant=True,
)


# --- The descriptor. Many axes here, so the carriers can take one token. ---


@dataclass(frozen=True, slots=True)
class EquationDescriptor:
    """Non-generic root for a complete static estimating-equation descriptor."""


class EquationContract(
    EquationDescriptor,
    Generic[EquationKindT, OriginT],  # noqa: UP046  # PEP 695 loses variance
):
    """Zero-field descriptor tying an equation kind to its construction origin.

    Descriptors carry axes; **carriers carry descriptors**. A call site names
    the combination once with a ``type`` alias and then spells one token, the
    same discipline :data:`jcor.core.transforms.L2NormalizationContract` and
    :data:`jcor.core.domains.L2UnitSphereSpace` already use. That is what keeps
    the carriers below at one and two parameters while the number of semantic
    axes grows.
    """


#: **Covariant** equation descriptor, the single parameter the carriers take.
EquationT = TypeVar(  # noqa: PLC0105
    "EquationT",
    bound=EquationDescriptor,
    covariant=True,
)


# --- The carriers. One array leaf each; every semantic axis is phantom. ---


@final
@dataclass(frozen=True, slots=True)
class _ProducerToken:
    """Private constructor token carried as static pytree metadata.

    Deliberately a second token rather than an import of
    :mod:`jcor.core.matrices`'s: a distance-matrix producer must not be able to
    mint an estimating function, and keeping this module a leaf of the intra-
    ``core`` import order is what lets the fence rank rank-0 modules at all.
    """


_PRODUCER_TOKEN = _ProducerToken()


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EstimatingFunction(Generic[EquationT]):  # noqa: UP046  # PEP 695 loses variance
    """``psi`` evaluated at every observation — the estimating function itself.

    Attributes:
        values: ``psi(Z_t, theta)`` per observation, shape
            ``(observations, moments)`` — or ``(*batch, observations, moments)``
            once a transform has batched it. Rows are observations and columns
            are equation components, matching the ``g`` argument
            :mod:`jcor.inference.gmm` already takes.

    Examples:
        >>> import jax.numpy as jnp
        >>> psi = unsafe_assume_estimating_function(jnp.zeros((8, 3)))
        >>> psi.values.shape
        (8, 3)

    """

    values: Float[Array, "*batch observations moments"]  # noqa: F722  # jaxtyping shape
    _token: _ProducerToken = field(repr=False, metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EquationJacobian(Generic[EquationT]):  # noqa: UP046  # PEP 695 loses variance
    """``J`` — the bread: the derivative of ``psi`` in the parameters.

    Held as the per-observation panel rather than as its average, because the
    average is what the sandwich needs but the panel is what a clustered or
    two-way meat needs; discarding it at construction would make those
    unreachable. :func:`sandwich_covariance` takes the mean itself.

    Attributes:
        values: ``d psi / d theta'`` per observation, shape
            ``(observations, moments, parameters)`` — or with a leading
            ``*batch``. This is exactly the ``derivative_panel`` argument
            :func:`jcor.inference.gmm.linear_hansen_j_kernel` already takes.

    """

    values: Float[
        Array,
        "*batch observations moments parameters",  # noqa: F722  # jaxtyping shape
    ]
    _token: _ProducerToken = field(repr=False, metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class VariabilityMatrix(
    Generic[EquationT, VariabilityT],  # noqa: UP046  # PEP 695 loses variance
):
    """``Omega`` — the meat: the variability matrix of ``psi``.

    The scheme is the load-bearing parameter; see the module docstring. HAC,
    cluster-robust, two-way, Driscoll-Kraay and spatial-robust forms are all
    *meat variants* and belong on this carrier at their own scheme brand, not on
    separate containers.

    Attributes:
        values: ``Omega``, shape ``(moments, moments)`` — or with a leading
            ``*batch``.

    Examples:
        >>> import jax.numpy as jnp
        >>> meat = unsafe_assume_variability_matrix(jnp.eye(3))
        >>> meat.values.shape
        (3, 3)

    """

    values: Float[Array, "*batch moments moments"]  # noqa: F722  # jaxtyping shape
    _token: _ProducerToken = field(repr=False, metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SandwichCovariance(
    Generic[EquationT, VariabilityT],  # noqa: UP046  # PEP 695 loses variance
):
    """Asymptotic parameter covariance, retaining the meat scheme it was built from.

    A **sibling** of :class:`VariabilityMatrix`, never a subtype: this is
    ``(parameters, parameters)`` and that is ``(moments, moments)``, and only the
    just-identified case makes them the same shape. Keeping the scheme means a
    consumer can demand that a reported standard error came from the centered
    meat.

    Attributes:
        values: ``Avar(theta_hat)``, shape ``(parameters, parameters)`` — or
            with a leading ``*batch``.

    """

    values: Float[Array, "*batch parameters parameters"]  # noqa: F722  # jaxtyping
    _token: _ProducerToken = field(repr=False, metadata={"static": True})


# --- Doors. One private producer for in-package code; loud external doors. ---


def _variability_matrix[
    EquationA: EquationDescriptor,
    VariabilityA: VariabilityScheme,
](
    values: Float[Array, "*batch moments moments"],  # noqa: F722  # jaxtyping shape
) -> VariabilityMatrix[EquationA, VariabilityA]:
    """Construct ``Omega`` from a producer that knows which scheme it formed.

    The private door in-package meat producers use, so a producer writes its
    scheme once, in its return type. :mod:`jcor.operators.longrun` is the only
    caller today.

    Args:
        values: The variability matrix the producer computed.

    Returns:
        The matrix branded with the equation and scheme the call site requires.

    """
    return VariabilityMatrix(values=values, _token=_PRODUCER_TOKEN)


def unsafe_assume_estimating_function[EquationA: EquationDescriptor](
    values: Float[Array, "*batch observations moments"],  # noqa: F722  # jaxtyping
) -> EstimatingFunction[EquationA]:
    """Assert an equation descriptor that nothing in this call verifies.

    The name is the review trigger, as for
    :func:`jcor.core.matrices.unsafe_assume_dmat`. Every use is a claim the
    *caller* owes evidence for.

    Args:
        values: The ``(observations, moments)`` array to brand.

    Returns:
        The array wrapped and branded exactly as the call site declares.

    """
    return EstimatingFunction(values=values, _token=_PRODUCER_TOKEN)


def unsafe_assume_equation_jacobian[EquationA: EquationDescriptor](
    values: Float[
        Array,
        "*batch observations moments parameters",  # noqa: F722  # jaxtyping
    ],
) -> EquationJacobian[EquationA]:
    """Assert an equation descriptor for a derivative panel, unverified.

    Args:
        values: The ``(observations, moments, parameters)`` array to brand.

    Returns:
        The array wrapped and branded exactly as the call site declares.

    """
    return EquationJacobian(values=values, _token=_PRODUCER_TOKEN)


def unsafe_assume_variability_matrix[
    EquationA: EquationDescriptor,
    VariabilityA: VariabilityScheme,
](
    values: Float[Array, "*batch moments moments"],  # noqa: F722  # jaxtyping shape
) -> VariabilityMatrix[EquationA, VariabilityA]:
    """Assert an equation and a **scheme** that nothing in this call verifies.

    The scheme is the dangerous half: no inspection of a finished ``(m, m)``
    array recovers whether the sample mean was subtracted before the
    autocovariances were formed, so this door is the only place that claim can
    be written without a producer standing behind it.

    Args:
        values: The ``(moments, moments)`` array to brand.

    Returns:
        The array wrapped and branded exactly as the call site declares.

    """
    return VariabilityMatrix(values=values, _token=_PRODUCER_TOKEN)


# --- The composition rule. Written once; this is the whole point of the waist. ---


def sandwich_covariance[EquationA: EquationDescriptor, VariabilityA: VariabilityScheme](
    bread: EquationJacobian[EquationA],
    meat: VariabilityMatrix[EquationA, VariabilityA],
    weight: Float[Array, "*batch moments moments"] | None = None,  # noqa: F722
) -> SandwichCovariance[EquationA, VariabilityA]:
    """Combine a bread and a meat into an asymptotic parameter covariance.

    With ``J`` the mean Jacobian and ``W`` the weighting matrix,

    ``Avar = (J' W J)^-1 J' W Omega W J (J' W J)^-1``

    which collapses to the efficient ``(J' Omega^-1 J)^-1`` at ``W = Omega^-1``
    and to the classical ``J^-1 Omega J^-T`` when the equation is just
    identified. Passing ``weight=None`` selects the efficient form directly
    rather than computing the general expression at ``W = pinv(Omega)``.

    This is the function the second waist exists for. Every estimator that can
    state a ``psi`` reaches its standard errors through here, so HAC,
    cluster-robust, two-way, Driscoll-Kraay and spatial-robust covariance are
    *meat variants* rather than per-family code. Note what the signature keeps
    and what it refuses:

    * ``EquationA`` is shared, so a bread and a meat from unrelated equations
      have no common descriptor below :class:`EquationDescriptor` to unify on;
    * ``VariabilityA`` **propagates into the result**, so a consumer written
      against ``SandwichCovariance[..., MeanCenteredLongRun]`` rejects a
      standard error built from the plain long-run meat. That rejection is the
      static replacement for ``inference/gmm.py``'s ``centered`` boolean.

    Traceable: no eager validation, no host comparison. Shape agreement is
    carried by jaxtyping and by the matrix products themselves.

    Args:
        bread: The per-observation derivative panel ``J``, averaged here.
        meat: The variability matrix ``Omega`` at its declared scheme.
        weight: The GMM weighting matrix ``W``. ``None`` selects the efficient
            weighting ``W = Omega^-1``.

    Returns:
        The parameter covariance, branded with the meat's scheme.

    Examples:
        >>> import jax.numpy as jnp
        >>> panel = jnp.tile(jnp.eye(2)[None], (5, 1, 1))
        >>> bread = unsafe_assume_equation_jacobian(panel)
        >>> meat = unsafe_assume_variability_matrix(jnp.eye(2) * 4.0)
        >>> sandwich_covariance(bread, meat).values
        Array([[4., 0.],
               [0., 4.]], dtype=float64)

    """
    jacobian = jnp.mean(bread.values, axis=-3)
    omega = meat.values
    transposed = jnp.swapaxes(jacobian, -1, -2)
    if weight is None:
        normal = transposed @ jnp.linalg.pinv(omega) @ jacobian
        covariance = jnp.linalg.pinv(normal)
    else:
        normal = jnp.linalg.pinv(transposed @ weight @ jacobian)
        middle = transposed @ weight @ omega @ weight @ jacobian
        covariance = normal @ middle @ normal
    symmetric = 0.5 * (covariance + jnp.swapaxes(covariance, -1, -2))
    return SandwichCovariance(values=symmetric, _token=_PRODUCER_TOKEN)
