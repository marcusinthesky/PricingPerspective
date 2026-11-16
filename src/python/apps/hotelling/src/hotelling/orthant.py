"""Checked JAX extragradient for strongly monotone orthant fields.

The caller supplies an analytical positive modulus. Host deadlines are checked
before evaluation and between synchronized blocks of 32 iterations; a compiled
block cannot be preempted. Bounds are numerical diagnostics, not interval proofs.
"""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, NamedTuple, Protocol

import jax
import jax.numpy as jnp
from statadic.core.typing import Array, ArrayLike, Bool, Float, Int  # noqa: TC002

from hotelling.precision import require_float64

if TYPE_CHECKING:
    from collections.abc import Callable


SOLVE_TARGET = 1e-10
LINE_SEARCH_STEPS = 80
BLOCK_ITERATIONS = 32
_RUNNING = 0
_SUCCESS = 1
_LIMIT = 2
_LINE_SEARCH = 3
_NONFINITE = 4


class OrthantField(Protocol):
    """A pure field; registered numerical state remains dynamic under JAX."""

    def operator(self, q: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
        """Return the negative own-profit gradient on a joint profile."""
        ...


class _Progress(NamedTuple):
    q: Array
    step: Array
    iteration: Array
    bound: Array
    status: Array
    evaluations: Array


def _apply(game: OrthantField, q: Array) -> Array:
    return game.operator(q)


def _field(game: OrthantField) -> Callable[[Array], Array]:
    leaves = jax.tree_util.tree_leaves(game)
    if len(leaves) == 1 and leaves[0] is game:
        # Simple third-party pure callables may be static; production Game is
        # a registered pytree and passes its arrays explicitly instead.
        return jax.tree_util.Partial(game.operator)
    return jax.tree_util.Partial(_apply, game)


@jax.jit
def _block(
    field: Callable[[Array], Array],
    state: _Progress,
    modulus: Array,
    target: Array,
    *,
    end: int,
    limit: int,
) -> _Progress:
    def condition(value: _Progress) -> Array:
        return (value.status == _RUNNING) & (value.iteration < end)

    def body(value: _Progress) -> _Progress:
        q, step, iteration, _, _, evaluations = value
        f = field(q) / modulus
        residual = jnp.where(q == 0, jnp.minimum(f, 0), f)
        bound = jnp.linalg.norm(residual)
        finite = jnp.isfinite(bound) & jnp.all(jnp.isfinite(f))
        accepted = finite & (bound <= target * jnp.maximum(1.0, jnp.linalg.norm(q)))
        status = jnp.where(finite, _RUNNING, _NONFINITE)
        status = jnp.where(accepted, _SUCCESS, status)
        status = jnp.where((status == _RUNNING) & (iteration >= limit), _LIMIT, status)
        evaluations = evaluations + 1

        def move() -> _Progress:
            def searching(carry: tuple[Array, Array, Array, Array, Array]) -> Array:
                return (carry[3] < LINE_SEARCH_STEPS) & ~carry[4]

            def trial(
                carry: tuple[Array, Array, Array, Array, Array],
            ) -> tuple[Array, Array, Array, Array, Array]:
                eta, _, _, count, _ = carry
                y = jnp.maximum(0, q - eta * f)
                fy = field(y) / modulus
                good = eta * jnp.linalg.norm(fy - f) <= 0.5 * jnp.linalg.norm(y - q)
                return jnp.where(good, eta, eta * 0.5), y, fy, count + 1, good

            eta, _, fy, count, good = jax.lax.while_loop(
                searching,
                trial,
                (step, q, f, jnp.asarray(0), jnp.asarray(0, dtype=jnp.bool_)),
            )
            following = jnp.where(good, jnp.maximum(0, q - eta * fy), q)
            return _Progress(
                following,
                jnp.minimum(1.0, 1.1 * eta),
                iteration + good.astype(iteration.dtype),
                bound,
                jnp.where(good, _RUNNING, _LINE_SEARCH),
                evaluations + count,
            )

        return jax.lax.cond(
            status == _RUNNING,
            move,
            lambda: _Progress(q, step, iteration, bound, status, evaluations),
        )

    return jax.lax.while_loop(condition, body, state)


def _initial(start: Array) -> _Progress:
    return _Progress(
        jnp.maximum(0, jnp.asarray(start)),
        jnp.asarray(1.0),
        jnp.asarray(0),
        jnp.asarray(jnp.inf),
        jnp.asarray(_RUNNING),
        jnp.asarray(0),
    )


def certificate_arrays(
    game: OrthantField,
    q: Float[Array, "*shape"],
    modulus: Float[Array, ""] | float,
    target: float = SOLVE_TARGET,
) -> tuple[Float[Array, ""], Float[Array, ""], Bool[Array, ""]]:
    """Return distance bound, profile norm and acceptance without host conversion."""
    q = jnp.asarray(q)
    f = game.operator(q) / modulus
    residual = jnp.where(q == 0, jnp.minimum(f, 0), f)
    bound, norm = jnp.linalg.norm(residual), jnp.linalg.norm(q)
    valid = (
        jnp.isfinite(bound)
        & jnp.isfinite(norm)
        & jnp.isfinite(modulus)
        & (modulus > 0)
        & jnp.all(q >= 0)
    )
    return bound, norm, valid & (bound <= target * jnp.maximum(1.0, norm))


def solve_arrays(
    game: OrthantField,
    start: ArrayLike,
    modulus: Float[Array, ""] | float,
    target: float = SOLVE_TARGET,
    limit: int = 400000,
) -> tuple[Float[Array, "*shape"], Float[Array, ""], Int[Array, ""], Int[Array, ""]]:
    """Compile a fixed-budget solve; status zero means verified convergence.

    This door has no wall-clock deadline or converged-solution derivative claim.
    Inputs must already be finite and the analytical modulus positive.
    """
    outcome = _block(
        _field(game),
        _initial(jnp.asarray(start)),
        jnp.asarray(modulus),
        jnp.asarray(target),
        end=limit + 1,
        limit=limit,
    )
    return (
        outcome.q,
        outcome.bound,
        outcome.iteration,
        jnp.where(outcome.status == _SUCCESS, 0, outcome.status),
    )


def solve_certified(
    game: OrthantField,
    start: ArrayLike,
    modulus: Float[Array, ""] | float,
    target: float = SOLVE_TARGET,
    limit: int = 400000,
    *,
    deadline: float | None = None,
) -> tuple[Float[Array, "*shape"], float, int]:
    """Solve with field/modulus normalization and a checked relative bound."""
    if deadline is not None and perf_counter() >= deadline:
        message = "time limit reached before certification"
        raise TimeoutError(message)
    require_float64()
    if not jnp.isfinite(modulus) or modulus <= 0:
        message = "positive modulus required"
        raise ValueError(message)
    field = _field(game)
    state = _initial(jnp.asarray(start, dtype=jnp.float64))
    while int(state.status) == _RUNNING:
        if deadline is not None and perf_counter() >= deadline:
            message = "time limit reached before certification"
            raise TimeoutError(message)
        end = min(int(state.iteration) + BLOCK_ITERATIONS, limit + 1)
        if end <= int(state.iteration):
            message = "iteration limit reached before certification"
            raise ArithmeticError(message)
        state = _block(
            field,
            state,
            jnp.asarray(modulus),
            jnp.asarray(target),
            end=end,
            limit=limit,
        )
        jax.block_until_ready(state)
        if deadline is not None and perf_counter() >= deadline:
            message = "time limit reached before certification"
            raise TimeoutError(message)
    status = int(state.status)
    if status == _SUCCESS:
        return state.q, float(state.bound), int(state.iteration)
    if status == _LINE_SEARCH:
        message = "line search failed"
        raise ArithmeticError(message)
    if status == _NONFINITE:
        message = "nonfinite field before certification"
        raise ArithmeticError(message)
    message = "iteration limit reached before certification"
    raise ArithmeticError(message)
