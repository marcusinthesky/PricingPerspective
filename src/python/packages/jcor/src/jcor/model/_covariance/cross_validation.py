"""Cross-validated sample/target covariance blend selection."""

# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.
from __future__ import annotations

import jax
import jax.numpy as jnp

from jcor.core.random import cell_key
from jcor.core.typing import Array, ArrayLike, Float, Int  # noqa: TC001
from jcor.model._covariance.estimators import sample_covariance
from jcor.model._covariance.validation import (
    MIN_COVARIANCE_OBSERVATIONS,
    MIN_CV_FOLD_OBSERVATIONS,
    as_index,
    validate_covariance_target,
    validate_observation_panel,
)
from jcor.optimize.psd import ridge_psd

__all__ = ["cv_lambda_grid"]


def _split_folds(indices: Int[Array, " t"], n_folds: int) -> list[Array]:
    """Split a shuffled observation index into NumPy-compatible fold slices.

    ``np.array_split`` had no ``jnp`` counterpart, but it never needed one: the
    fold count is a validated Python int, so the split points are static and
    plain slicing reproduces NumPy's rule exactly — the first ``n % k`` folds
    get one extra element. Sizes still differ by one when the panel does not
    divide evenly, so ``sample_covariance_kernel`` compiles twice, as before.
    """
    quotient, remainder = divmod(int(indices.shape[0]), n_folds)
    bounds = [0]
    for fold in range(n_folds):
        bounds.append(bounds[-1] + quotient + (1 if fold < remainder else 0))
    return [indices[bounds[i] : bounds[i + 1]] for i in range(n_folds)]


def _blend_fold_score(
    panel: Float[Array, "t n"],
    target_matrix: Float[Array, "n n"],
    folds: list[Array],
    sample_weight: Float[Array, ""],
) -> Float[Array, ""]:
    """Return the mean out-of-fold realized variance for one sample weight."""
    n_folds = len(folds)
    n_variables = panel.shape[1]
    out_of_sample = []
    for fold_index in range(n_folds):
        test = folds[fold_index]
        train = jnp.concatenate(
            [folds[index] for index in range(n_folds) if index != fold_index]
        )
        sample = sample_covariance(panel[train], ddof=0)
        sigma = sample_weight * sample + (1.0 - sample_weight) * target_matrix
        # `ridge_psd` is traced and now consumes the device array directly:
        # the former host round trip through `np.array(..., np.float64)` on
        # every (grid x fold) iteration bought nothing but a materialization.
        sigma = ridge_psd(sigma, eps=1e-8)
        inverse_ones = jnp.linalg.solve(sigma, jnp.ones(n_variables))
        weights = inverse_ones / inverse_ones.sum()
        test_returns = panel[test]
        out_of_sample.append(jnp.var(test_returns @ weights))
    return jnp.mean(jnp.stack(out_of_sample))


def cv_lambda_grid(
    returns: ArrayLike,
    target: ArrayLike,
    grid: ArrayLike | list[float] | None = None,
    n_folds: int = 5,
    seed: int = 0,
    *,
    assume_target_fixed: bool = False,
) -> tuple[float, Float[Array, " g"], Float[Array, " g"]]:
    """Select the sample weight by out-of-fold realized variance."""
    panel = validate_observation_panel(
        returns,
        ddof=0,
        min_observations=MIN_COVARIANCE_OBSERVATIONS,
    )
    n_observations, n_variables = panel.shape
    if not isinstance(assume_target_fixed, bool):
        message = "assume_target_fixed must be a boolean."
        raise TypeError(message)
    if not assume_target_fixed:
        message = (
            "cv_lambda_grid requires a fixed/independent target; a target "
            "estimated from the full panel leaks held-out folds."
        )
        raise ValueError(message)
    target_matrix = validate_covariance_target(target, n_variables)
    if (
        as_index(n_folds) is None
        or n_folds < MIN_CV_FOLD_OBSERVATIONS
        or n_folds > n_observations // MIN_CV_FOLD_OBSERVATIONS
    ):
        message = (
            "n_folds must be an integer in [2, floor(n_observations / 2)] "
            "so every validation fold has at least two observations."
        )
        raise ValueError(message)
    candidate_grid = jnp.linspace(0.0, 1.0, 11) if grid is None else jnp.asarray(grid)
    if (
        candidate_grid.ndim != 1
        or candidate_grid.size == 0
        or not bool(jnp.all(jnp.isfinite(candidate_grid)))
        or bool(jnp.any((candidate_grid < 0.0) | (candidate_grid > 1.0)))
    ):
        message = "grid must be a nonempty finite vector with values in [0, 1]."
        raise ValueError(message)

    key = cell_key(seed, "model", "cv_lambda_grid")
    indices = jax.random.permutation(key, n_observations)
    folds = _split_folds(indices, n_folds)

    scores = [
        _blend_fold_score(panel, target_matrix, folds, sample_weight)
        for sample_weight in candidate_grid
    ]
    score_vector = jnp.stack(scores)
    best = int(jnp.argmin(score_vector))
    return float(candidate_grid[best]), candidate_grid, score_vector
