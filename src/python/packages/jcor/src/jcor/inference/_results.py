"""Host result adapters and procedure-specific containers for stage S7.

Position
--------
rank 7 · depends only on rank-0 result contracts, so every sibling may depend on it

Numerical paths return the registered pytrees from :mod:`jcor.core.results`.
This module is the explicit host boundary: :func:`to_host_test_result` and
:func:`to_host_confidence_interval` perform scalar concretization only after a
computation has left ``jit``/``vmap``. Calling either adapter while tracing
fails with :class:`HostAdapterError` instead of leaking an incidental JAX
``ConcretizationTypeError``.

It also holds three legacy/procedure-specific containers:

``MixtureTestResult``
    Kept as-is, by design. Its solver/equivalence fields are procedure-specific
    rather than a contract (t46.1 deviation 4), and ``pipeline``'s Paper 2
    ``_hedge`` stage reads ``equivalence_pvalue`` / ``equivalence`` / ``delta`` /
    ``energy_point_estimate`` off it, so the field set is frozen until Paper 2
    outputs are re-derived. It is defined *here* rather than in
    :mod:`jcor.inference.mixture` so that
    :mod:`jcor.inference._permutation.split_runner`, which constructs it, does
    not close a within-stage import cycle; ``mixture`` re-exports it.

``HypothesisTest`` / ``PermutationTestResult``
    Compatibility/report containers. ``permutation_test`` preserves this eager
    surface by calling :func:`to_host_test_result`; transformable code calls
    ``permutation_test_result`` or ``permutation_test_kernel`` and receives a
    :class:`jcor.core.results.TestResult` directly.

The dangling ``ArrayScalar = TypeVar("ArrayScalar")`` this module used to carry
was declared and never used; it is deleted, not moved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from jcor.core.results import (  # noqa: TC001  # runtime annotations
    EstimateResult,
    TestResult,
)
from jcor.core.typing import Array, Float  # noqa: TC001  # runtime annotations

__all__ = [
    "HostAdapterError",
    "HypothesisTest",
    "MixtureTestResult",
    "PermutationTestResult",
    "to_host_confidence_interval",
    "to_host_test_result",
]


class HostAdapterError(RuntimeError):
    """Raised when a host-only result adapter is called during JAX tracing."""


def _reject_tracers(tree: object, *, adapter: str) -> None:
    """Fail explicitly when a host adapter is reached from compiled code.

    Args:
        tree: Result pytree whose dynamic leaves are inspected.
        adapter: Public adapter name included in the diagnostic.

    Raises:
        HostAdapterError: If any result leaf is a JAX tracer.

    """
    if any(
        isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(tree)
    ):
        message = (
            f"{adapter} is host-only; return the JAX result pytree from compiled "
            "code and call the adapter after jit/vmap completes"
        )
        raise HostAdapterError(message)


@dataclass(frozen=True)
class HypothesisTest:
    """Results of a hypothesis test.

    This dataclass contains the results from a statistical hypothesis test,
    including the observed test statistic, p-value, number of permutations used,
    and the null distribution of test statistics generated under the null
    hypothesis of no effect.

    Attributes:
        statistic: The observed test statistic computed from the original data.
        pvalue: The finite-sample add-one permutation p-value
            ``(1 + b) / (1 + B)``, where ``b`` permuted statistics are as
            extreme as or more extreme than the observed statistic. Must be in
            [0, 1].
        num_permutations: Number of random permutations performed to generate
            the null distribution.
        null_distribution: Array of test statistics computed from permuted data
            under the null hypothesis. Length should equal num_permutations.

    Raises:
        ValueError: If pvalue is not in [0, 1] or if length of null_distribution
            does not match num_permutations.

    Examples:
        >>> result = HypothesisTest(
        ...     statistic=2.5,
        ...     pvalue=0.03,
        ...     num_permutations=999,
        ...     null_distribution=jnp.array([...]),  # 999 values
        ... )
        >>> print(f"Reject null hypothesis: {result.pvalue < 0.05}")
        Reject null hypothesis: True

    """

    statistic: float
    pvalue: float
    num_permutations: int
    null_distribution: Array

    def __post_init__(self) -> None:
        """Validate hypothesis test result data.

        Raises:
            ValueError: If pvalue is not in [0, 1] or if length of
                null_distribution does not match num_permutations.

        """
        if not 0.0 <= self.pvalue <= 1.0:
            message = f"pvalue must be in [0, 1], got {self.pvalue}"
            raise ValueError(message)

        if len(self.null_distribution) != self.num_permutations:
            message = (
                f"null_distribution length ({len(self.null_distribution)}) must equal "
                f"num_permutations ({self.num_permutations})"
            )
            raise ValueError(message)


@dataclass(frozen=True)
class PermutationTestResult(HypothesisTest):
    """Extended hypothesis test result with permutation-specific metadata.

    This subclass adds no additional fields but provides semantic clarity
    for permutation test results.

    Examples:
        >>> result = PermutationTestResult(
        ...     statistic=3.2,
        ...     pvalue=0.001,
        ...     num_permutations=999,
        ...     null_distribution=jnp.array([...]),
        ... )
        >>> print(
        ...     f"Minimum achievable p-value: {1 / (result.num_permutations + 1):.4f}"
        ... )
        Minimum achievable p-value: 0.0010

    """


def to_host_test_result(
    result: TestResult[Float[Array, ""]],  # noqa: F722  # jaxtyping scalar shape
) -> HypothesisTest:
    """Convert a numerical test-result pytree to the legacy eager container.

    Scalar conversion and ``HypothesisTest`` validation happen together at this
    named boundary. The null distribution deliberately remains a JAX array to
    preserve the public eager surface; only its scalar report fields are
    concretized.

    Args:
        result: Scalar numerical test result returned by a transformable core.

    Returns:
        The validated legacy hypothesis-test container.

    Raises:
        HostAdapterError: If called while ``result`` contains JAX tracers.
        ValueError: If the concrete p-value or null length is invalid.

    """
    _reject_tracers(result, adapter="to_host_test_result")
    return HypothesisTest(
        statistic=float(jax.device_get(result.statistic)),
        pvalue=float(jax.device_get(result.pvalue)),
        num_permutations=result.n_resamples,
        null_distribution=result.null_distribution,
    )


def to_host_confidence_interval(
    result: EstimateResult[Float[Array, ""]]  # noqa: F722  # jaxtyping scalar shape
    | tuple[Float[Array, ""], Float[Array, ""]],  # noqa: F722  # jaxtyping scalar
) -> tuple[float, float]:
    """Return JAX confidence bounds as the legacy pair of floats.

    Args:
        result: Scalar estimate result or interval pytree returned by a
            transformable bootstrap.

    Returns:
        Concrete ``(lower, upper)`` confidence bounds.

    Raises:
        HostAdapterError: If called while ``result`` contains JAX tracers.

    """
    _reject_tracers(result, adapter="to_host_confidence_interval")
    lower, upper = result.ci if isinstance(result, EstimateResult) else result
    return float(jax.device_get(lower)), float(jax.device_get(upper))


@dataclass(frozen=True)
class MixtureTestResult:
    """Result of a studentized / re-optimized mixture two-sample test.

    Attributes:
        energy_point_estimate: Ê(X, Y_{w*}) — the PRIMARY hedge-quality metric
            (magnitude; monotone in hedge quality). Reported *instead of* an
            inverted p-value.
        energy_ci: (low, high) bootstrap CI on Ê.
        studentized_statistic: The studentized (Chung-Romano) test statistic.
        pvalue: **Difference** permutation p-value of the studentized statistic.
            Tests H0: F_X = F_{Y_w} (one-sided, greater). SMALL p ⇒ the
            distributions DIFFER ⇒ NOT equivalent. This is NOT an equivalence
            p-value; do not feed it to
            :func:`jcor.decision.multiplicity.equivalence_multiplicity_control`.
        equivalence_pvalue: **Equivalence / relevance** p-value for the relevance
            null H0: E(X, Y_w) ≥ δ vs H1: E(X, Y_w) < δ. Derived from the
            bootstrap distribution of Ê as ``(1 + #{Ê* ≥ δ}) / (1 + B)``. SMALL p
            ⇒ evidence FOR equivalence (Ê is confidently below δ). Consistent with
            the CI-based ``equivalence`` flag (``ci_hi < δ``). ``None`` outside the
            equivalence framing (no δ). THIS is what FWER control consumes.
        equivalence: True iff the equivalence test rejects H0: E ≥ δ
            (``ci_hi < δ``).
        delta: The equivalence margin used (None if not an equivalence test).
        weights: The selected mixture weights w*.
        m_eff: Inverse-variance effective size at w*.
        null_distribution: Permutation null of the studentized statistic.
        bootstrap_distribution: Bootstrap distribution of Ê (source of the CI and
            the equivalence p-value).
        reoptimized_in_permutation: True iff w* was re-optimized per replicate.
        studentized: Always True for this constructor (documents composition).
        equivalence_framed: True iff built via equivalence_test.
        num_permutations: Number of permutation replicates.
        solver_used: Numerical solver actually used for weight selection.
        solver_converged: Exact-solver convergence flag when available.
        solver_frank_wolfe_gap: Exact-solver KKT gap when available.
        solver_cnd_tangent_max_eigenvalue: Largest tangent-space Gram
            eigenvalue used by the CND check when available.
        solver_simplex_sum_error: Absolute simplex-sum error when available.
        solver_min_weight: Minimum selected weight when available.
        equivalence_curve: Optional list of (delta, rejected) over a δ-grid.

    """

    energy_point_estimate: float
    energy_ci: tuple[float, float]
    studentized_statistic: float
    pvalue: float
    weights: Array
    m_eff: float
    null_distribution: Array
    reoptimized_in_permutation: bool
    studentized: bool
    equivalence_framed: bool
    num_permutations: int
    weight_method: str = "kernel_balance"
    solver_used: str = ""
    solver_converged: bool | None = None
    solver_frank_wolfe_gap: float | None = None
    solver_cnd_tangent_max_eigenvalue: float | None = None
    solver_simplex_sum_error: float | None = None
    solver_min_weight: float | None = None
    bootstrap_distribution: Array = field(default_factory=lambda: jnp.array([]))
    equivalence_pvalue: float | None = None
    equivalence: bool | None = None
    delta: float | None = None
    equivalence_curve: list[tuple[float, bool]] = field(default_factory=list)
