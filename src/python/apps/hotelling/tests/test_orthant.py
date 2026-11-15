"""Analytical boundary and evaluation-budget checks for the orthant solver."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from hotelling.orthant import (
    SOLVE_TARGET,
    _block,
    _field,
    _initial,
    certificate_arrays,
    solve_certified,
)

if TYPE_CHECKING:
    from statadic.core.typing import Array

BOUNDARY_SOLVE_EVALUATIONS = 4
INVARIANCE_ITERATION_LIMIT = 200


class CountingAffineField:
    """A unit-modulus field whose constrained optimum is exactly zero."""

    def __init__(self) -> None:
        """Start with no evaluations of the analytical field."""
        self.evaluated: list[Array] = []

    def operator(self, q: Array) -> Array:
        """Record evaluations independently of the solver's iteration counter."""
        self.evaluated.append(q.copy())
        return q + 1


def test_boundary_solution_reuses_current_field_evaluation() -> None:
    """A known one-step boundary solution needs no duplicate initial evaluation."""
    game = CountingAffineField()
    start = np.array([0.5], dtype=np.float64)
    result, bound, iterations = solve_certified(game, start, modulus=1)
    np.testing.assert_array_equal(result, np.zeros(1, dtype=np.float64))
    assert bound == 0
    assert iterations == 1
    assert result.dtype == np.float64
    # A compiled field has no Python evaluation side effects. Its block state
    # counts actual numerical calls independently of the iteration counter.
    outcome = _block(
        _field(game),
        _initial(jnp.asarray(start)),
        jnp.asarray(1.0),
        jnp.asarray(SOLVE_TARGET),
        end=32,
        limit=400000,
    )
    assert int(outcome.evaluations) == BOUNDARY_SOLVE_EVALUATIONS
    np.testing.assert_array_equal(start, [0.5])


def test_expired_deadline_precedes_field_evaluation() -> None:
    """An expired solve cannot spend time evaluating the economic operator."""
    game = CountingAffineField()
    with pytest.raises(TimeoutError, match="time limit"):
        solve_certified(game, np.array([0.5]), 1, deadline=perf_counter() - 1)
    assert not game.evaluated


@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
def test_positive_payoff_units_preserve_solution_and_iterations(scale: float) -> None:
    """Changing payoff units preserves the distance certificate and convergence."""

    class ScaledField:
        def __init__(self, units: float) -> None:
            self.units = units

        def operator(self, q: Array) -> Array:
            return self.units * (jnp.array([2.0, 3.0]) * q - jnp.array([1.0, -1.0]))

    result, bound, iterations = solve_certified(
        ScaledField(scale),
        np.array([0.1, 0.5]),
        modulus=2 * scale,
        limit=INVARIANCE_ITERATION_LIMIT,
    )
    np.testing.assert_allclose(result, [0.5, 0.0], atol=1e-10, rtol=0)
    assert bound <= SOLVE_TARGET
    assert iterations < INVARIANCE_ITERATION_LIMIT
    reference, reference_bound, reference_iterations = solve_certified(
        ScaledField(1.0),
        np.array([0.1, 0.5]),
        modulus=2,
        limit=INVARIANCE_ITERATION_LIMIT,
    )
    assert iterations == reference_iterations
    np.testing.assert_allclose(result, reference, rtol=0, atol=1e-14)
    assert bound == pytest.approx(reference_bound, abs=1e-14)


@pytest.mark.parametrize("modulus", [0.0, -1.0, np.inf, np.nan])
def test_invalid_modulus_rejected_before_evaluation(modulus: float) -> None:
    """A nonfinite normalization cannot silently manufacture a zero certificate."""
    game = CountingAffineField()
    with pytest.raises(ValueError, match="positive modulus"):
        solve_certified(game, np.array([0.5]), modulus)
    assert not game.evaluated


@pytest.mark.parametrize("modulus", [0.0, -1.0, np.inf, np.nan])
def test_array_certificate_rejects_invalid_normalization(modulus: float) -> None:
    """An infinite modulus cannot turn a nonzero field into an accepted bound."""
    _, _, accepted = certificate_arrays(
        CountingAffineField(), jnp.array([0.5]), modulus
    )
    assert not bool(accepted)
