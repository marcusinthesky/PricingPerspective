"""Fixed-shape JAX solve behind dyadic host-side rank selection."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jcor.core.typing import ArrayLike  # noqa: TC001
from jcor.model._dyadic.contracts import DyadicComputationError
from jcor.optimize.backends import least_squares


def least_squares_host(
    design: ArrayLike,
    outcome: ArrayLike,
    *,
    context: str,
) -> jax.Array:
    """Solve one rank-selected design through the shared Lineax QR kernel.

    Column/rank selection is deliberately host-only and may change width from
    draw to draw. Once that decision is made, this named eager door delegates
    the sole least-squares implementation to the fixed-shape JAX backend.
    """
    if isinstance(design, jax.core.Tracer) or isinstance(outcome, jax.core.Tracer):
        message = (
            "dyadic rank selection is a host-only boundary; call least_squares "
            "directly only after selecting a fixed-width design"
        )
        raise RuntimeError(message)  # noqa: TRY004  # tracing is an execution mode
    try:
        return least_squares(
            jnp.asarray(design, dtype=jnp.float64),
            jnp.asarray(outcome, dtype=jnp.float64),
        )
    except (RuntimeError, ValueError) as error:
        message = f"dyadic {context} failed for a validated finite design"
        raise DyadicComputationError(message) from error
