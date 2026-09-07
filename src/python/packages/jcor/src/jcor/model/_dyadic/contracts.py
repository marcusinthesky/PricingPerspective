"""Entity-neutral records and errors for undirected dyadic estimation.

The records split by direction. **Result** records — ``DyadicBootstrapResult``
and ``DyadicFitResult`` — hold ``jax.Array``: jcor hands JAX back to
``apps/pipeline``/``apps/simulation``, which own the conversion to
NumPy/pandas/pyarrow at their own reporting edges. **Request** records —
``DyadicSample``, ``DyadicModel``, ``DyadicBootstrapDesign``,
``NodeCountSchedule`` — accept :class:`jcor.operators.design.ArrayView`, the
structural array Protocol that admits a NumPy panel read off parquet as
readily as a JAX array, and that keeps the shape and indexing guarantees
``jax.typing.ArrayLike`` erases by also admitting bare scalars.

Numeric arrays request float64 to preserve the contract inherited from the
retired NumPy carriers (``t65`` binding decision 4), but jcor opens no scope of
its own: the caller must own ``jax_enable_x64=True``. Without it the rank
selector in :mod:`jcor.operators.design` loses its threshold signal on these
designs and every node-bootstrap draw is rejected — measured 0/1999 on
``p1_dyadic_confound``.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping  # noqa: TC003  # runtime hint
from dataclasses import dataclass, field
from typing import cast

import jax  # noqa: TC002  # runtime: beartype resolves `jax.Array` annotations
import jax.numpy as jnp

from jcor.operators.design import (
    ArrayView,  # noqa: TC001  # runtime dataclass hint
    NodeIndex,  # noqa: TC001  # runtime dataclass hint
)

DEFAULT_BOOTSTRAP_ITERS = 1_999
DEFAULT_BOOTSTRAP_SEED = 42
MIN_ACTIVE_DYADS = 3


class DyadicInputError(ValueError):
    """Report invalid or incompatible dyadic estimation inputs."""


class DyadicComputationError(RuntimeError):
    """Report a failed dyadic fit or bootstrap computation."""


@dataclass(frozen=True)
class NodeCountSchedule[NodeIdT: Hashable = Hashable]:
    """Multinomial counts tied to the exact entity-to-column mapping they use."""

    node_index: NodeIndex[NodeIdT]
    #: Accepts any array-like; ``__post_init__`` normalizes it to a
    #: ``jax.Array``, which is what every reader after construction sees.
    counts: ArrayView

    def __post_init__(self) -> None:
        """Own an immutable count matrix disjoint from the caller's buffer.

        The retired NumPy carrier bought this with a ``copy=True`` plus
        ``setflags(write=False)`` pair. A JAX array is immutable by
        construction and ``jnp.asarray`` does not alias a host buffer, so
        both guarantees now follow from the array type itself.

        The caller's dtype is deliberately preserved rather than cast to
        ``int64``: ``validate_node_schedule`` rejects fractional and
        non-finite counts, and a silent integer cast here would truncate
        exactly the values it exists to reject.
        """
        object.__setattr__(self, "counts", jnp.asarray(self.counts))


@dataclass(frozen=True)
class DyadicSample[NodeIdT: Hashable = Hashable]:
    """Outcome and canonical endpoint IDs for one undirected dyadic panel.

    ``endpoint_i``/``endpoint_j`` stay host label arrays: their elements are
    arbitrary hashables, not tensor values.
    """

    outcome: ArrayView
    endpoint_i: ArrayView
    endpoint_j: ArrayView


@dataclass(frozen=True)
class DyadicModel[NodeIdT: Hashable = Hashable]:
    """Regressor selection and aligned arrays for one dyadic model."""

    regressor_names: list[str]
    columns: Mapping[str, ArrayView]
    sample: DyadicSample[NodeIdT]


@dataclass(frozen=True)
class DyadicInference[NodeIdT: Hashable = Hashable]:
    """Symmetric node-effect and multinomial node-bootstrap controls."""

    node_effects: bool = False
    effect_name_prefix: str = "node_effect"
    bootstrap_iters: int = DEFAULT_BOOTSTRAP_ITERS
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED
    node_schedule: NodeCountSchedule[NodeIdT] | None = None
    required_bootstrap_columns: tuple[str, ...] = ()


def _default_inference[NodeIdT: Hashable]() -> DyadicInference[NodeIdT]:
    """Specialize the schedule-free default to the fit carrier's node ID."""
    return cast("DyadicInference[NodeIdT]", DyadicInference())


@dataclass(frozen=True)
class DyadicFit[NodeIdT: Hashable = Hashable]:
    """Complete point-estimation and inference request."""

    model: DyadicModel[NodeIdT]
    inference: DyadicInference[NodeIdT] = field(default_factory=_default_inference)


@dataclass(frozen=True)
class DyadicBootstrapDesign[NodeIdT: Hashable = Hashable]:
    """Rank-cleaned design and shared node-bootstrap schedule."""

    design: ArrayView
    column_names: list[str]
    sample: DyadicSample[NodeIdT]
    node_schedule: NodeCountSchedule[NodeIdT]
    node_effects: bool = False
    effect_name_prefix: str = "node_effect"
    required_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class DyadicBootstrapResult:
    """Array-valued coefficient draws and per-draw validity diagnostics."""

    draw_id: jax.Array
    coefficients: dict[str, jax.Array]
    sse: jax.Array
    overall_r2: jax.Array
    within_r2: jax.Array
    active_nodes: jax.Array
    active_dyads: jax.Array
    effective_columns: jax.Array
    effective_rank: jax.Array
    solve_ok: jax.Array

    def coefficient(self, name: str) -> jax.Array:
        """Return one aligned coefficient series, or all-``NaN`` if absent."""
        values = self.coefficients.get(name)
        if values is not None:
            return values
        return jnp.full(self.draw_id.shape, jnp.nan, dtype=jnp.float64)


@dataclass(frozen=True)
class DyadicFitResult:
    """Point fit, node-bootstrap draws, and design diagnostics."""

    n_dyads: int
    regressor_names: tuple[str, ...]
    coefficients: dict[str, float]
    effect_one_sd: dict[str, float]
    bootstrap_inference: dict[str, tuple[float, float, float, float, float]]
    bootstrap_draws: DyadicBootstrapResult
    bootstrap_valid_draws: int
    bootstrap_requested_draws: int
    bootstrap_seed: int
    overall_r2: float
    within_r2: float | None
    sse_node_effects_only: float | None
    sse_full: float
    design_rank: int
    design_columns: int
    dropped_collinear_columns: tuple[str, ...]
    fwl_coefficients: dict[str, float]
    point_design: jax.Array
    point_design_names: tuple[str, ...]
    point_fitted: jax.Array
    point_residual: jax.Array
