"""Unified pairwise-distance dispatch for sample-cloud panels.

Sliced metrics reuse one random projection set for every pair. This preserves
comparability across the resulting symmetric distance matrix.

Why jaxtyping is imported at runtime here
-----------------------------------------
``flake8-type-checking`` runs in ``strict`` mode, so it wants
``from jaxtyping import ...`` under ``if TYPE_CHECKING:``. Paired with
``from __future__ import annotations`` that makes the names *unresolvable* at
runtime, and beartype then **silently skips** the annotation rather than
failing — the probe is recorded in :mod:`jcor.core.typing`'s module docstring.
The import hook installed by ``apps/simulation/tests/conftest.py`` reaches this
module's shape strings only because the import stays at runtime; the
``noqa: TC002`` is what that costs.

Why the ``jcor.ground`` names stay guarded
------------------------------------------
They are documentation only, and deliberately so: neither reaches a position
beartype checks, so un-guarding them would buy zero enforcement.

* ``_DEFAULT_GROUND_DISTANCE`` is a module-level annotated assignment.
  jaxtyping's transformer rewrites ``ClassDef``/``FunctionDef`` alone
  (``jaxtyping/_import_hook.py:170,190``) and PEP 563 leaves the annotation an
  unevaluated string, so nothing ever reads it.
* ``_MetricKwargs.distance_metric`` is reached only through
  ``**kwargs: Unpack[_MetricKwargs]``, and beartype 0.22.9 reduces a PEP 692
  ``Unpack[TypedDict]`` to ``HINT_SANE_IGNORABLE`` — an explicit noop — at
  ``beartype/_check/convert/_reduce/_pep/redpep692.py:106``.

The ground distance forwarded from here *is* checked one call downstream, at
:func:`simulation.metrics.energy.pairwise_energy_distance`: there the same
alias annotates a real parameter, and there the import is therefore runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, Unpack

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int  # noqa: TC002  # runtime; see docstring
from jcor.discrepancy.transport import w1_1d, w2_1d

from .energy import pairwise_energy_distance

if TYPE_CHECKING:
    from jcor.ground import GroundDistanceName, GroundDistanceSelection

_SLICED_METRICS = frozenset({"sliced_w1", "sliced_w2"})
_ONE_DIMENSIONAL_METRICS = frozenset({"w1_1d", "w2_1d"})
_VALID_METRICS = ("energy", "sliced_w2", "sliced_w1", "w2_1d", "w1_1d")
_DEFAULT_GROUND_DISTANCE: GroundDistanceName = "euclidean"


class _MetricKwargs(TypedDict, total=False):
    """Keyword arguments accepted by metric implementations."""

    exponent: float
    distance_metric: GroundDistanceSelection
    row_chunk: int
    n_projections: int


def pairwise_distance_matrix(
    samples: Float[Array, "n m d"],
    metric: str = "energy",
    *,
    key: jax.Array | None = None,
    **kwargs: Unpack[_MetricKwargs],
) -> Float[Array, "n n"]:
    """Compute a symmetric pairwise-distance matrix for a sample-cloud panel.

    Args:
        samples: Sample clouds with shape ``(n, m, d)``.
        metric: Energy, sliced Wasserstein, or one-dimensional Wasserstein mode.
        key: Required PRNG key for sliced metrics.
        **kwargs: Metric-specific exponent, base metric, chunk, or projection count.

    Returns:
        Symmetric ``(n, n)`` array with an exact zero diagonal.

    Raises:
        ValueError: If the metric or its required dimensional inputs are invalid.

    """
    if metric == "energy":
        return pairwise_energy_distance(
            samples,
            exponent=kwargs.get("exponent", 1.0),
            metric=kwargs.get("distance_metric", _DEFAULT_GROUND_DISTANCE),
            row_chunk=kwargs.get("row_chunk", 8),
        )
    if metric in _SLICED_METRICS:
        return _sliced_distance_matrix(
            samples,
            metric=metric,
            key=key,
            n_projections=kwargs.get("n_projections", 128),
        )
    if metric in _ONE_DIMENSIONAL_METRICS:
        return _one_dimensional_distance_matrix(samples, metric=metric)
    valid = ", ".join(repr(name) for name in _VALID_METRICS)
    message = f"Unknown metric {metric!r}. Choose from: {valid}."
    raise ValueError(message)


def _sliced_distance_matrix(
    samples: Float[Array, "n m d"],
    *,
    metric: str,
    key: jax.Array | None,
    n_projections: int,
) -> Float[Array, "n n"]:
    if key is None:
        message = f"metric={metric!r} requires a JAX PRNGKey via key="
        raise ValueError(message)
    dimension = samples.shape[2]
    power = 2 if metric == "sliced_w2" else 1
    raw = jax.random.normal(
        key,
        shape=(n_projections, dimension),
        dtype=samples.dtype,
    )
    directions = raw / jnp.linalg.norm(raw, axis=1, keepdims=True)

    def pair(left: Float[Array, "m d"], right: Float[Array, "m d"]) -> Array:
        left_projection = left @ directions.T
        right_projection = right @ directions.T

        # `index` is a *traced* scalar, never a Python `int`: the `jax.vmap`
        # below maps this over `jnp.arange(n_projections)`, so what arrives is
        # `i64[](jax)` under the suite's `jax_enable_x64`
        # (`apps/simulation/tests/conftest.py`) and `i32[]` without it.
        # Measured, not reasoned: with `index: int` — which resolves at runtime,
        # so beartype does check it — the import hook fails
        # `test_unified_api_sliced_shape_sym_zero_diag` with
        # `Actual value: i64[](jax)  Expected type: <class 'int'>`.
        # `Int[Array, ""]` is rank-0 and width-agnostic, so it states the truth
        # under either x64 setting without widening to a bare `Array`.
        def projected_distance(index: Int[Array, ""]) -> Array:
            difference = jnp.abs(
                jnp.sort(left_projection[:, index])
                - jnp.sort(right_projection[:, index])
            )
            return jnp.mean(difference**power)

        values: Array = jax.vmap(projected_distance)(jnp.arange(n_projections))
        return jnp.mean(values) ** (1.0 / power)

    matrix: Array = jax.vmap(
        lambda left: jax.vmap(lambda right: pair(left, right))(samples)
    )(samples)
    return matrix - jnp.diag(jnp.diag(matrix))


def _one_dimensional_distance_matrix(
    samples: Float[Array, "n m d"], *, metric: str
) -> Float[Array, "n n"]:
    dimension = samples.shape[2]
    if dimension != 1:
        message = (
            f"metric={metric!r} requires d==1, got d={dimension}. "
            "Squeeze the last dimension before calling."
        )
        raise ValueError(message)
    flattened_samples = samples[:, :, 0]
    distance = w2_1d if metric == "w2_1d" else w1_1d
    matrix: Array = jax.vmap(
        lambda left: jax.vmap(lambda right: distance(left, right))(flattened_samples)
    )(flattened_samples)
    return matrix - jnp.diag(jnp.diag(matrix))
