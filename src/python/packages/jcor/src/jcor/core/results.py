"""Generic result containers — what stage S7 (``inference``) and S6 (``model``) return.

Position
--------
rank 0 · importable by every stage

Naming grammar (t46): ``*_test`` returns :class:`TestResult`; ``*_fit`` /
``*_estimate`` returns :class:`EstimateResult`.

Both are pytree-registered, so a ``vmap`` over a Monte-Carlo grid returns
``TestResult[Float[Array, "cells"]]`` rather than failing to flatten. That is the
whole reason they are generic: ``S`` is the statistic's own type, scalar in the
single-run case and batched after a transform, with no second container class.

No ``__post_init__`` validation — deliberate
--------------------------------------------
Host validation of values such as ``0 <= pvalue <= 1`` does not survive
tracing: comparing a tracer raises ``TracerBoolConversionError``. These
containers therefore enforce structure only. Static configuration and shape
checks live in constructing kernels; concrete value validation lives in named
``to_host_*`` adapters after a computation leaves the transform boundary.

Field mapping for the three ``hypothesis.py`` dataclasses
---------------------------------------------------------
=========================================  ==================================
``hypothesis.py``                          here
=========================================  ==================================
``HypothesisTest.statistic``               ``TestResult.statistic``
``HypothesisTest.pvalue``                  ``TestResult.pvalue``
``HypothesisTest.null_distribution``       ``TestResult.null_distribution``
``HypothesisTest.num_permutations``        ``TestResult.n_resamples``
``PermutationTestResult`` (no new fields)  ``TestResult`` — the subclass carried
                                           no information and does not survive
``MixtureTestResult``                      stays in ``inference/mixture.py``:
                                           its solver/equivalence fields are
                                           procedure-specific, not a contract
=========================================  ==================================

The dangling ``ArrayScalar = TypeVar("ArrayScalar")`` at ``hypothesis.py:9`` was
declared and never used; it is not carried over. ``S`` below replaces it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax

from jcor.core.typing import Array, Bool, Float  # noqa: TC001  # runtime fields

__all__ = ["EstimateResult", "TestResult"]

# `S` is the statistic's own type: `Float[Array, ""]` for a single run,
# `Float[Array, "cells"]` after a `vmap` over a parameter grid. It deliberately
# has no `S: Array` bound: jaxtyping's parameterized hints are runtime-generated
# classes, and beartype rejects `Float[Array, ""]` as violating that nominal
# bound while importing a specialized `TestResult`. Result-producing APIs pin
# `S` to a JAX-array hint instead; the concrete non-generic fields below do too.


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TestResult[S]:
    """The result of a hypothesis test — what every ``*_test`` returns.

    Attributes:
        statistic: Observed test statistic.
        pvalue: p-value in ``[0, 1]``. Not validated here; see the module
            docstring.
        null_distribution: Statistics under the null, shape ``(resamples,)`` —
            or ``(*batch, resamples)`` once a transform has batched it; see
            :mod:`jcor.core.typing` for why every pytree field carries
            ``*batch``.
        n_resamples: Number of resamples (permutations / bootstrap replicates).
            A **static** pytree field: it is a shape, so it must not become a
            traced leaf, and it stays constant across a ``vmap``-ed grid.

    Examples:
        >>> r = TestResult(
        ...     statistic=jnp.asarray(2.5),
        ...     pvalue=jnp.asarray(0.03),
        ...     null_distribution=jnp.zeros(999),
        ...     n_resamples=999,
        ... )
        >>> [leaf.shape for leaf in jax.tree_util.tree_leaves(r)]
        [(), (), (999,)]

    """

    # pytest's `python_classes = "Test*"` matches this name wherever a test
    # module imports it, and warns that it cannot collect a class with
    # `__init__`. Declared once here so no downstream suite has to.
    __test__ = False

    statistic: S
    pvalue: S
    null_distribution: Float[Array, "*batch resamples"]  # noqa: F722  # jaxtyping
    n_resamples: int = field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EstimateResult[S]:
    """A point estimate with uncertainty — the ``*_fit`` / ``*_estimate`` return type.

    Attributes:
        estimate: The point estimate.
        se: Standard error of ``estimate``, on the same scale.
        ci: ``(low, high)`` confidence interval. A tuple of arrays flattens as
            two pytree leaves, so it survives ``vmap`` unchanged.
        converged: Solver convergence flag as a JAX **boolean** array — a Python
            ``bool`` cannot be a traced leaf. Pass ``jnp.asarray(True)`` for a
            closed-form estimator that cannot fail to converge. The dtype is
            checked, so ``tree_map(lambda x: x * 2, ...)`` over the whole
            container is a type error rather than a silent cast to int.
        n_obs: Number of observations. A **static** pytree field, for the same
            reason as ``TestResult.n_resamples``.

    """

    estimate: S
    se: S
    ci: tuple[S, S]
    converged: Bool[Array, "*batch"]  # noqa: F722  # jaxtyping shape
    n_obs: int = field(metadata={"static": True})
