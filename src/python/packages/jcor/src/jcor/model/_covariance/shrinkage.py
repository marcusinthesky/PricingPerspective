"""Linear shrinkage of sample covariance toward a supplied target.

The sample-weight convention is
``Sigma(lambda) = lambda * S + (1 - lambda) * target``. Fixed-target optimality
requires an explicit acknowledgement; same-window targets use the separately
named descriptive plug-in rule.
"""

# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.
from __future__ import annotations

import math
from numbers import Real
from typing import NamedTuple

import jax
import jax.numpy as jnp

from jcor.core.typing import Array, ArrayLike  # noqa: TC001
from jcor.core.typing import Float as JaxFloat  # noqa: TC001
from jcor.model._covariance.estimators import sample_covariance_kernel
from jcor.model._covariance.validation import (
    MIN_COVARIANCE_OBSERVATIONS,
    validate_covariance_target,
    validate_observation_panel,
)

type ShrinkageWeight = int | float

__all__ = [
    "PlugInLambdaKernelResult",
    "ShrinkageKernelResult",
    "ShrinkageResult",
    "optimal_lambda_dist",
    "plug_in_lambda_dist",
    "plug_in_lambda_dist_kernel",
    "sample_cov_entry_variance",
    "sample_cov_entry_variance_kernel",
    "shrink_to_dist",
    "shrink_to_dist_kernel",
    "target_weight_dist",
]


class ShrinkageResult(NamedTuple):
    """Blended covariance and diagnostics for its sample weight."""

    sigma: JaxFloat[Array, "n n"]
    lam: float
    var_s_sum: float
    bias_sq_sum: float

    def __getattr__(self, name: str) -> float:
        """Preserve the historical ``var_S_sum`` attribute for consumers."""
        if name == "var_S_sum":
            return self.var_s_sum
        raise AttributeError(name)


class PlugInLambdaKernelResult(NamedTuple):
    """Fixed-structure traced plug-in weight and its diagnostic masses."""

    sample_weight: JaxFloat[Array, ""]
    variance_sum: JaxFloat[Array, ""]
    adjusted_gap: JaxFloat[Array, ""]


class ShrinkageKernelResult(NamedTuple):
    """Fixed-structure traced covariance blend and diagnostics."""

    sigma: JaxFloat[Array, "n n"]
    sample_weight: JaxFloat[Array, ""]
    variance_sum: JaxFloat[Array, ""]
    adjusted_gap: JaxFloat[Array, ""]


@jax.jit
def sample_cov_entry_variance_kernel(
    returns: JaxFloat[Array, "t n"],
) -> JaxFloat[Array, "n n"]:
    """Estimate entrywise covariance variance by fourth moments on graph."""
    n_observations = returns.shape[0]
    centered = returns - jnp.mean(returns, axis=0, keepdims=True)
    sample = sample_covariance_kernel(returns, ddof=0)
    products = centered[:, :, None] * centered[:, None, :]
    fourth = jnp.mean((products - sample[None, :, :]) ** 2, axis=0)
    return fourth / n_observations


@jax.jit
def plug_in_lambda_dist_kernel(
    returns: JaxFloat[Array, "t n"],
    target: JaxFloat[Array, "n n"],
) -> PlugInLambdaKernelResult:
    """Evaluate the same-window plug-in sample weight entirely on graph."""
    sample = sample_covariance_kernel(returns, ddof=0)
    variance_sum = jnp.sum(sample_cov_entry_variance_kernel(returns))
    raw_gap = jnp.sum((target - sample) ** 2)
    adjusted_gap = jnp.maximum(raw_gap - variance_sum, 0.0)
    denominator = variance_sum + adjusted_gap
    sample_weight = jnp.where(denominator > 0.0, adjusted_gap / denominator, 0.0)
    return PlugInLambdaKernelResult(
        sample_weight=jnp.clip(sample_weight, 0.0, 1.0),
        variance_sum=variance_sum,
        adjusted_gap=adjusted_gap,
    )


@jax.jit
def shrink_to_dist_kernel(
    returns: JaxFloat[Array, "t n"],
    target: JaxFloat[Array, "n n"],
    sample_weight: JaxFloat[Array, ""],
) -> ShrinkageKernelResult:
    """Blend sample and target covariance with diagnostics on the JAX graph."""
    sample = sample_covariance_kernel(returns, ddof=0)
    variance_sum = jnp.sum(sample_cov_entry_variance_kernel(returns))
    adjusted_gap = jnp.maximum(
        jnp.sum((target - sample) ** 2) - variance_sum,
        0.0,
    )
    sigma = sample_weight * sample + (1.0 - sample_weight) * target
    return ShrinkageKernelResult(
        sigma=sigma,
        sample_weight=sample_weight,
        variance_sum=variance_sum,
        adjusted_gap=adjusted_gap,
    )


def sample_cov_entry_variance(returns: ArrayLike) -> JaxFloat[Array, "n n"]:
    """Estimate entrywise sample-covariance variance by fourth moments."""
    panel = validate_observation_panel(
        returns,
        ddof=0,
        min_observations=MIN_COVARIANCE_OBSERVATIONS,
    )
    return sample_cov_entry_variance_kernel(panel)


def optimal_lambda_dist(
    returns: ArrayLike,
    target: ArrayLike,
    *,
    assume_target_fixed: bool = False,
) -> tuple[float, float, float]:
    """Estimate the MSE-optimal sample weight for a fixed target.

    The target must be fixed or estimated independently of ``returns``. A
    same-window target has omitted variance and covariance terms and must use
    :func:`plug_in_lambda_dist` under its explicitly weaker contract.
    """
    if not isinstance(assume_target_fixed, bool):
        message = "assume_target_fixed must be a boolean."
        raise TypeError(message)
    if not assume_target_fixed:
        message = (
            "optimal_lambda_dist is valid only for a fixed/independent target; "
            "pass assume_target_fixed=True only when that condition holds, or "
            "use plug_in_lambda_dist for the explicitly heuristic same-window rule."
        )
        raise ValueError(message)
    return plug_in_lambda_dist(returns, target)


def plug_in_lambda_dist(
    returns: ArrayLike,
    target: ArrayLike,
) -> tuple[float, float, float]:
    """Return the descriptive same-window plug-in sample weight.

    Computes ``max(||target-S||^2-V, 0)`` over its sum with the sampling-noise
    mass ``V``. The adjusted gap has a target-bias interpretation only when the
    target is fixed or independent.
    """
    panel = validate_observation_panel(
        returns,
        ddof=0,
        min_observations=MIN_COVARIANCE_OBSERVATIONS,
    )
    target_matrix = validate_covariance_target(target, panel.shape[1])
    result = plug_in_lambda_dist_kernel(
        jnp.asarray(panel),
        jnp.asarray(target_matrix),
    )
    return (
        float(result.sample_weight),
        float(result.variance_sum),
        float(result.adjusted_gap),
    )


def target_weight_dist(
    returns: ArrayLike,
    target: ArrayLike,
    *,
    assume_target_fixed: bool = False,
) -> float:
    """Return the complementary fixed-target weight ``1 - lambda``."""
    sample_weight, _, _ = optimal_lambda_dist(
        returns,
        target,
        assume_target_fixed=assume_target_fixed,
    )
    return 1.0 - sample_weight


def shrink_to_dist(
    returns: ArrayLike,
    target: ArrayLike,
    lam: ShrinkageWeight | None = None,
    *,
    assume_target_fixed: bool = False,
) -> ShrinkageResult:
    """Blend sample covariance toward a checked positive-semidefinite target.

    Convex combinations of the sample covariance and the checked target remain
    positive semidefinite for every admitted ``lam`` in ``[0, 1]``.
    """
    panel = validate_observation_panel(
        returns,
        ddof=0,
        min_observations=MIN_COVARIANCE_OBSERVATIONS,
    )
    target_matrix = validate_covariance_target(target, panel.shape[1])
    if lam is None:
        sample_weight, variance_sum, adjusted_gap = optimal_lambda_dist(
            returns,
            target,
            assume_target_fixed=assume_target_fixed,
        )
    else:
        if not isinstance(lam, Real) or isinstance(lam, bool):
            message = "lam must be a real number, not a boolean."
            raise TypeError(message)
        sample_weight = float(lam)
        if not math.isfinite(sample_weight) or not 0.0 <= sample_weight <= 1.0:
            message = "lam must be finite and lie in [0, 1]."
            raise ValueError(message)
        diagnostics = plug_in_lambda_dist_kernel(
            jnp.asarray(panel),
            jnp.asarray(target_matrix),
        )
        variance_sum = float(diagnostics.variance_sum)
        adjusted_gap = float(diagnostics.adjusted_gap)
    sample_weight = min(max(sample_weight, 0.0), 1.0)
    result = shrink_to_dist_kernel(
        jnp.asarray(panel),
        jnp.asarray(target_matrix),
        jnp.asarray(sample_weight),
    )
    return ShrinkageResult(
        sigma=result.sigma,
        lam=sample_weight,
        var_s_sum=variance_sum,
        bias_sq_sum=adjusted_gap,
    )
