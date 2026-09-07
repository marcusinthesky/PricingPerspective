"""Boundary calibration of the Paper-2 energy-equivalence procedure.

This stage implements ROADMAP P0-2.  Its null is the relevance/equivalence
boundary

    H0: E(F, G) >= delta    versus    H1: E(F, G) < delta,

not the ordinary two-sample null ``F = G``.  The boundary data-generating
process is therefore constructed from Gaussian laws whose *population* energy
distance is exactly ``delta``.  A studentized multiplier bootstrap estimates
the one-sided boundary p-value.  This is deliberately separate from the
permutation distribution used for the point null ``E = 0``.

Families share a target sample and correlated candidate innovations.  This
creates correlated target-level claims while retaining a closed-form marginal
energy distance for every target.  All-null families measure the probability
of at least one boundary rejection; mixed
families measure false equivalence claims in the presence of true discoveries.
Benjamini--Hochberg is reported only as a diagnostic comparator, never as an
pre-specified diagnostic rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import pandas as pd
from jax import random
from jcor.core.random import cell_key
from jcor.core.typing import ArrayLike  # noqa: TC002  # runtime array contract

from simulation.harness.replications import execute_replication_cell, map_replications

if TYPE_CHECKING:
    from collections.abc import Callable

_ENERGY_BOUNDARY_TOLERANCE = 1e-12
_MINIMUM_SAMPLE_SIZE = 3
_MINIMUM_FAMILY_SIZE = 2
_MINIMUM_MONTE_CARLO_REPLICATIONS = 2
_PRODUCTION_REPLICATIONS = 2_000


@dataclass(frozen=True)
class VarianceCell:
    """Marginal Gaussian scales for one calibration cell."""

    name: str
    target_sd: float
    candidate_sd: float


def _normal_absolute_moment(mean: float, sd: float) -> float:
    """Return ``E|Z|`` for ``Z ~ Normal(mean, sd^2)``."""
    if sd <= 0.0:
        message = "sd must be positive"
        raise ValueError(message)
    z = abs(mean) / sd
    return float(
        sd * math.sqrt(2.0 / math.pi) * math.exp(-0.5 * z * z)
        + abs(mean) * math.erf(z / math.sqrt(2.0))
    )


def gaussian_energy_distance(
    mean_shift: float,
    target_sd: float,
    candidate_sd: float,
) -> float:
    """Compute population energy distance between univariate Gaussian laws.

    The target law is ``N(0, target_sd^2)`` and the candidate law is
    ``N(mean_shift, candidate_sd^2)``.  For the Euclidean ground metric,

    ``E = 2 E|X-Y| - E|X-X'| - E|Y-Y'|``.
    """
    if target_sd <= 0.0 or candidate_sd <= 0.0:
        message = "Gaussian standard deviations must be positive"
        raise ValueError(message)
    cross_sd = math.sqrt(target_sd**2 + candidate_sd**2)
    cross = _normal_absolute_moment(mean_shift, cross_sd)
    within_target = 2.0 * target_sd / math.sqrt(math.pi)
    within_candidate = 2.0 * candidate_sd / math.sqrt(math.pi)
    return 2.0 * cross - within_target - within_candidate


#: Safety cap on bisection halvings. Never binding in practice: the loop's real
#: exit is the double-precision floor below, reached in ~50-60 iterations for any
#: bracket this module builds.
_BISECTION_MAX_ITERATIONS = 200


def _bisect_increasing(
    objective: Callable[[float], float],
    lower: float,
    upper: float,
) -> float:
    """Return the root of a continuous increasing ``objective`` on a sign bracket.

    Replaces ``scipy.optimize.brentq`` so SciPy leaves the shipped wheel and
    survives only as a dev-group reference oracle — see
    ``docs/numpy-scipy-boundary-ledger.md``. Plain bisection rather than Brent's
    method is deliberate: the one caller's objective is a smooth, strictly
    increasing scalar in the mean shift, so inverse-quadratic interpolation buys
    nothing that ~50 halvings do not, and the convergence path is unconditional
    rather than dependent on interpolation behaving on the bracket.

    **It bisects to the double-precision floor, not to a tolerance argument.**
    That is not gold-plating, it is what parity costs: bisection stops *at* the
    width you ask for, whereas Brent's superlinear last step incidentally lands
    far inside its own ``xtol + rtol*|x|`` rule. Reproducing brentq's accuracy
    therefore means halving until the midpoint stops moving, not until the
    bracket is 1e-13 wide — a first cut with brentq's literal tolerance
    reproduced the root only to 1.6e-13 relative and left the stage's
    ``absolute_boundary_error`` within 7x of its 1e-11 test bound.

    Host scalar Python, not a JAX kernel, and cannot become one: the caller
    establishes the bracket with a data-dependent doubling loop that is not
    traceable at any tolerance. See the ledger's "SciPy: retired" section.
    """
    if objective(lower) > 0.0:
        message = f"objective must be nonpositive at the lower end {lower}"
        raise ValueError(message)
    if objective(upper) < 0.0:
        message = f"objective must be nonnegative at the upper end {upper}"
        raise ValueError(message)
    for _ in range(_BISECTION_MAX_ITERATIONS):
        midpoint = 0.5 * (lower + upper)
        # Exhausted the representable doubles between the ends: no further
        # halving can change either bound.
        if midpoint <= lower or midpoint >= upper:
            break
        if objective(midpoint) < 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return lower if abs(objective(lower)) <= abs(objective(upper)) else upper


def mean_shift_at_energy(
    energy: float,
    target_sd: float,
    candidate_sd: float,
) -> float:
    """Solve for the nonnegative Gaussian mean shift at a requested energy."""
    if energy <= 0.0:
        message = "energy must be positive"
        raise ValueError(message)
    minimum = gaussian_energy_distance(0.0, target_sd, candidate_sd)
    if energy < minimum - 1e-12:
        message = (
            f"requested energy {energy} is below the variance-cell minimum {minimum}"
        )
        raise ValueError(message)
    if abs(energy - minimum) <= _ENERGY_BOUNDARY_TOLERANCE:
        return 0.0

    def objective(shift: float) -> float:
        return gaussian_energy_distance(shift, target_sd, candidate_sd) - energy

    upper = max(1.0, math.sqrt(energy))
    while objective(upper) < 0.0:
        upper *= 2.0
    return _bisect_increasing(objective, 0.0, upper)


def _energy_x_terms(x: ArrayLike) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Target-only energy terms, invariant across a candidate family.

    Returns ``(x_flat, within_x, within_x_i)``. Hoisted out of the per-candidate
    loop in :func:`_studentized_family_pvalues` (the ``dxx`` block depends only
    on ``x``), reproducing the exact values the inlined form computed.
    """
    x = jnp.asarray(x).reshape(-1)
    if len(x) < _MINIMUM_SAMPLE_SIZE:
        message = "each sample must contain at least three observations"
        raise ValueError(message)
    dxx = jnp.abs(x[:, None] - x[None, :])
    within_x = dxx.sum() / (len(x) * (len(x) - 1))
    within_x_i = dxx.sum(axis=1) / (len(x) - 1)
    return x, within_x, within_x_i


def _energy_influence_given_x(
    x: jax.Array,
    within_x: jax.Array,
    within_x_i: jax.Array,
    y: ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Energy estimate + influence for candidate ``y`` given precomputed x-terms.

    ``x`` is the flattened target and ``within_x``/``within_x_i`` its
    :func:`_energy_x_terms` outputs. Byte-for-byte identical to the inlined
    computation in the original single-call function.
    """
    y = jnp.asarray(y).reshape(-1)
    if len(y) < _MINIMUM_SAMPLE_SIZE:
        message = "each sample must contain at least three observations"
        raise ValueError(message)

    dxy = jnp.abs(x[:, None] - y[None, :])
    dyy = jnp.abs(y[:, None] - y[None, :])

    cross = dxy.mean()
    within_y = dyy.sum() / (len(y) * (len(y) - 1))
    estimate = 2.0 * cross - within_x - within_y

    cross_x = dxy.mean(axis=1)
    cross_y = dxy.mean(axis=0)
    within_y_i = dyy.sum(axis=1) / (len(y) - 1)
    psi_x = 2.0 * (cross_x - cross) - 2.0 * (within_x_i - within_x)
    psi_y = 2.0 * (cross_y - cross) - 2.0 * (within_y_i - within_y)
    return estimate, psi_x, psi_y


def _energy_estimate_and_influence(
    x: ArrayLike,
    y: ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Unbiased energy estimate and first-order influence values.

    The nonzero equivalence boundary is a non-degenerate functional, so its
    first-order influence representation supports the multiplier bootstrap.
    Within-sample terms use U-statistic (off-diagonal) averages.
    """
    x_flat, within_x, within_x_i = _energy_x_terms(x)
    return _energy_influence_given_x(x_flat, within_x, within_x_i, y)


def _studentized_family_pvalues(
    key: jax.Array,
    x: jax.Array,
    candidates: jax.Array,
    delta: float,
    n_resamples: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return array-only primary and diagnostic p-values for one family."""
    _m_targets, n_candidate = candidates.shape
    n_target = len(x)
    dxx = jnp.abs(x[:, None] - x[None, :])
    within_x = dxx.sum() / (n_target * (n_target - 1))
    within_x_i = dxx.sum(axis=1) / (n_target - 1)

    def candidate_terms(y: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        dxy = jnp.abs(x[:, None] - y[None, :])
        dyy = jnp.abs(y[:, None] - y[None, :])
        cross = dxy.mean()
        within_y = dyy.sum() / (n_candidate * (n_candidate - 1))
        estimate = 2.0 * cross - within_x - within_y
        candidate_psi_x = 2.0 * (dxy.mean(axis=1) - cross) - 2.0 * (
            within_x_i - within_x
        )
        within_y_i = dyy.sum(axis=1) / (n_candidate - 1)
        candidate_psi_y = 2.0 * (dxy.mean(axis=0) - cross) - 2.0 * (
            within_y_i - within_y
        )
        return estimate, candidate_psi_x, candidate_psi_y

    estimates, psi_x_rows, psi_y_rows = jax.vmap(candidate_terms)(candidates)
    psi_x = psi_x_rows.T
    psi_y = psi_y_rows.T
    var_x = psi_x.var(axis=0, ddof=1)
    var_y = psi_y.var(axis=0, ddof=1)
    standard_errors = jnp.sqrt(var_x / n_target + var_y / n_candidate)
    standard_errors = jnp.maximum(standard_errors, jnp.finfo(x.dtype).eps)
    observed = (estimates - delta) / standard_errors

    x_key, y_key = random.split(key)
    multipliers_x = random.normal(x_key, (n_resamples, n_target), dtype=x.dtype)
    multipliers_y = random.normal(y_key, (n_resamples, n_candidate), dtype=x.dtype)
    perturbations = (
        multipliers_x @ psi_x / n_target + multipliers_y @ psi_y / n_candidate
    )
    bootstrap_t = perturbations / standard_errors[None, :]
    primary = (1.0 + jnp.sum(bootstrap_t <= observed[None, :], axis=0)) / (
        n_resamples + 1.0
    )

    target_scale_se = jnp.sqrt(var_x * (1.0 / n_target + 1.0 / n_candidate))
    target_scale_se = jnp.maximum(target_scale_se, jnp.finfo(x.dtype).eps)
    diagnostic_observed = (estimates - delta) / target_scale_se
    diagnostic_bootstrap_t = perturbations / target_scale_se[None, :]
    diagnostic = (
        1.0 + jnp.sum(diagnostic_bootstrap_t <= diagnostic_observed[None, :], axis=0)
    ) / (n_resamples + 1.0)
    return primary, diagnostic, estimates


def _bh_rejections(pvalues: jax.Array, alpha: float) -> jax.Array:
    """Benjamini--Hochberg mask, used only as a documented diagnostic."""
    order = jnp.argsort(pvalues)
    ordered = pvalues[order]
    indices = jnp.arange(len(ordered))
    passed = ordered <= alpha * (indices + 1) / len(ordered)
    largest = jnp.max(jnp.where(passed, indices, -1))
    ordered_rejected = indices <= largest
    inverse_order = jnp.argsort(order)
    return ordered_rejected[inverse_order]


def _k_minus_1_rejections(pvalues: jax.Array, alpha: float) -> jax.Array:
    """Apply the repository's ``alpha / (k - 1)`` equivalence threshold."""
    return pvalues <= alpha / (len(pvalues) - 1)


def _mc_summary(values: jax.Array) -> dict[str, float]:
    """Mean, Monte-Carlo SE, and clipped normal 95% interval."""
    values = jnp.asarray(values)
    rate = values.mean()
    se = (
        values.std(ddof=1) / jnp.sqrt(jnp.asarray(len(values), dtype=values.dtype))
        if len(values) > 1
        else jnp.asarray(0.0, dtype=values.dtype)
    )
    return {
        "estimate": float(rate),
        "mc_se": float(se),
        "ci_low": max(0.0, float(rate - 1.96 * se)),
        "ci_high": min(1.0, float(rate + 1.96 * se)),
    }


def _parse_variance_cells(cfg: dict[str, Any]) -> list[VarianceCell]:
    cells = [VarianceCell(**cell) for cell in cfg["variance_cells"]]
    if not cells:
        message = "variance_cells must not be empty"
        raise ValueError(message)
    if len({cell.name for cell in cells}) != len(cells):
        message = "variance-cell names must be unique"
        raise ValueError(message)
    return cells


def build_boundary_design(cfg: dict[str, Any]) -> pd.DataFrame:
    """Return the exact population parameters used by every DGP cell."""
    delta = float(cfg["delta"])
    power_ratio = float(cfg["power_energy_ratio"])
    if not 0.0 < power_ratio < 1.0:
        message = "power_energy_ratio must lie in (0, 1)"
        raise ValueError(message)

    rows: list[dict[str, float | str]] = []
    for cell in _parse_variance_cells(cfg):
        for hypothesis, energy in (
            ("boundary_null", delta),
            ("equivalent_alternative", delta * power_ratio),
        ):
            shift = mean_shift_at_energy(energy, cell.target_sd, cell.candidate_sd)
            achieved = gaussian_energy_distance(
                shift, cell.target_sd, cell.candidate_sd
            )
            rows.append(
                {
                    "variance_cell": cell.name,
                    "hypothesis": hypothesis,
                    "target_sd": cell.target_sd,
                    "candidate_sd": cell.candidate_sd,
                    "mean_shift": shift,
                    "requested_energy": energy,
                    "achieved_energy": achieved,
                    "absolute_boundary_error": abs(achieved - energy),
                }
            )
    return pd.DataFrame(rows)


def _draw_family(
    key: jax.Array,
    n_target: int,
    n_candidate: int,
    m_targets: int,
    target_sd: float,
    candidate_sd: float,
    correlation: float,
    mean_shifts: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Draw one family with a shared target and correlated candidates."""
    target_key, common_key, idiosyncratic_key = random.split(key, 3)
    x = target_sd * random.normal(target_key, (n_target,))
    common = random.normal(common_key, (n_candidate,))
    idiosyncratic = random.normal(idiosyncratic_key, (m_targets, n_candidate))
    innovations = (
        jnp.sqrt(correlation) * common[None, :]
        + jnp.sqrt(1.0 - correlation) * idiosyncratic
    )
    y = mean_shifts[:, None] + candidate_sd * innovations
    return x, y


_METHODS = (
    "studentized_marginal",
    "studentized_k_minus_1",
    "studentized_bh_diagnostic",
    "target_scale_k_minus_1_diagnostic",
)


class EquivalenceReplication(NamedTuple):
    """Array-only summary for one equivalence-calibration replication."""

    family_false: jax.Array
    target_false: jax.Array
    target_power: jax.Array
    mean_estimate_null: jax.Array
    mean_estimate_alt: jax.Array


def _equivalence_replication(
    key: jax.Array,
    mean_shifts: jax.Array,
    *,
    n_null: int,
    n_alternative: int,
    n_target: int,
    n_candidate: int,
    m_targets: int,
    target_sd: float,
    candidate_sd: float,
    correlation: float,
    delta: float,
    n_resamples: int,
    alpha: float,
) -> EquivalenceReplication:
    """Draw and summarize one correlated equivalence family."""
    data_key, bootstrap_key = random.split(key)
    x, candidates = _draw_family(
        data_key,
        n_target,
        n_candidate,
        m_targets,
        target_sd,
        candidate_sd,
        correlation,
        mean_shifts,
    )
    pvalues, diagnostic_pvalues, estimates = _studentized_family_pvalues(
        bootstrap_key, x, candidates, delta, n_resamples
    )

    rejection_masks = jnp.stack(
        (
            pvalues <= alpha,
            _k_minus_1_rejections(pvalues, alpha),
            _bh_rejections(pvalues, alpha),
            _k_minus_1_rejections(diagnostic_pvalues, alpha),
        ),
        axis=0,
    )
    false = rejection_masks[:, :n_null]
    family_false = jnp.any(false, axis=1).astype(estimates.dtype)
    target_false = false.mean(axis=1)
    target_power = (
        rejection_masks[:, n_null:].mean(axis=1)
        if n_alternative
        else jnp.full(len(_METHODS), jnp.nan)
    )
    mean_estimate_alt = (
        estimates[n_null:].mean() if n_alternative else jnp.asarray(jnp.nan)
    )
    return EquivalenceReplication(
        family_false,
        target_false,
        target_power,
        estimates[:n_null].mean(),
        mean_estimate_alt,
    )


@partial(
    jax.jit,
    static_argnames=(
        "n_null",
        "n_alternative",
        "n_target",
        "n_candidate",
        "m_targets",
        "target_sd",
        "candidate_sd",
        "correlation",
        "delta",
        "n_resamples",
        "alpha",
        "n_sims",
        "batch_size",
    ),
)
def _equivalence_cell(
    key: jax.Array,
    mean_shifts: jax.Array,
    *,
    n_null: int,
    n_alternative: int,
    n_target: int,
    n_candidate: int,
    m_targets: int,
    target_sd: float,
    candidate_sd: float,
    correlation: float,
    delta: float,
    n_resamples: int,
    alpha: float,
    n_sims: int,
    batch_size: int,
) -> EquivalenceReplication:
    kernel = partial(
        _equivalence_replication,
        mean_shifts=mean_shifts,
        n_null=n_null,
        n_alternative=n_alternative,
        n_target=n_target,
        n_candidate=n_candidate,
        m_targets=m_targets,
        target_sd=target_sd,
        candidate_sd=candidate_sd,
        correlation=correlation,
        delta=delta,
        n_resamples=n_resamples,
        alpha=alpha,
    )
    return map_replications(kernel, key, n_sims, batch_size)


def _validate_equivalence_config(cfg: dict[str, Any]) -> None:
    """Validate dimensions and probabilities required by the calibration."""
    n_sims = int(cfg["n_sims"])
    n_target = int(cfg["n_target"])
    n_candidate = int(cfg["n_candidate"])
    alpha = float(cfg["alpha"])
    correlation = float(cfg["target_correlation"])
    m_values = [int(value) for value in cfg["m_values"]]
    null_fractions = [float(value) for value in cfg["null_fractions"]]

    if n_sims < _MINIMUM_MONTE_CARLO_REPLICATIONS:
        message = "n_sims must be at least two to estimate Monte-Carlo SE"
        raise ValueError(message)
    if n_target < _MINIMUM_SAMPLE_SIZE or n_candidate < _MINIMUM_SAMPLE_SIZE:
        message = "n_target and n_candidate must be at least three"
        raise ValueError(message)
    if not 0.0 < alpha < 1.0:
        message = "alpha must lie in (0, 1)"
        raise ValueError(message)
    if not 0.0 <= correlation < 1.0:
        message = "target_correlation must lie in [0, 1)"
        raise ValueError(message)
    if not m_values or any(value < _MINIMUM_FAMILY_SIZE for value in m_values):
        message = "all m_values must be at least two"
        raise ValueError(message)
    if not null_fractions or any(not 0.0 < value <= 1.0 for value in null_fractions):
        message = "null_fractions must lie in (0, 1]"
        raise ValueError(message)


def _aggregate_equivalence_group(
    cfg: dict[str, Any],
    group: tuple[VarianceCell, int, float, int, int],
    group_results: EquivalenceReplication,
) -> list[dict[str, Any]]:
    """Aggregate one variance/family-design cell across replications."""
    cell, m_targets, null_fraction, n_null, n_alternative = group
    family_false = group_results.family_false
    target_false = group_results.target_false
    target_power = group_results.target_power
    mean_estimate_null = group_results.mean_estimate_null
    mean_estimate_alt = group_results.mean_estimate_alt

    rows = []
    for method_index, method in enumerate(_METHODS):
        fwer = _mc_summary(family_false[:, method_index])
        size = _mc_summary(target_false[:, method_index])
        power = _mc_summary(target_power[:, method_index]) if n_alternative else None
        rows.append(
            {
                "variance_cell": cell.name,
                "m_targets": m_targets,
                "null_fraction": null_fraction,
                "n_boundary_null": n_null,
                "n_equivalent_alternative": n_alternative,
                "method": method,
                "nominal_alpha": float(cfg["alpha"]),
                "fwer": fwer["estimate"],
                "fwer_mc_se": fwer["mc_se"],
                "fwer_ci_low": fwer["ci_low"],
                "fwer_ci_high": fwer["ci_high"],
                "marginal_boundary_size": size["estimate"],
                "size_mc_se": size["mc_se"],
                "size_ci_low": size["ci_low"],
                "size_ci_high": size["ci_high"],
                "marginal_power": power["estimate"] if power else math.nan,
                "power_mc_se": power["mc_se"] if power else math.nan,
                "power_ci_low": power["ci_low"] if power else math.nan,
                "power_ci_high": power["ci_high"] if power else math.nan,
                "mean_boundary_energy_estimate": float(mean_estimate_null.mean()),
                "mean_alternative_energy_estimate": (
                    float(mean_estimate_alt.mean()) if n_alternative else math.nan
                ),
                "n_sims": int(cfg["n_sims"]),
                "n_resamples": int(cfg["n_resamples"]),
            }
        )
    return rows


def _build_equivalence_groups(
    cfg: dict[str, Any], boundary_design: pd.DataFrame
) -> list[tuple[tuple[VarianceCell, int, float, int, int], jax.Array]]:
    """Build deterministic group metadata and mean-shift arrays."""
    shifts = {
        (str(row["variance_cell"]), str(row["hypothesis"])): float(row["mean_shift"])
        for row in boundary_design.to_dict(orient="records")
    }
    groups = []
    for cell in _parse_variance_cells(cfg):
        for m_targets in (int(value) for value in cfg["m_values"]):
            for null_fraction in (float(value) for value in cfg["null_fractions"]):
                n_null = min(m_targets, max(1, round(m_targets * null_fraction)))
                n_alternative = m_targets - n_null
                group = (cell, m_targets, null_fraction, n_null, n_alternative)
                mean_shifts = jnp.full(
                    m_targets, shifts[(cell.name, "equivalent_alternative")]
                )
                mean_shifts = mean_shifts.at[:n_null].set(  # noqa: PD008
                    shifts[(cell.name, "boundary_null")]
                )
                groups.append((group, mean_shifts))
    return groups


def run_equivalence_calibration(
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run P0-2 boundary rejection and power diagnostics.

    Returns:
        ``(results, boundary_design, summary)``.  ``results`` contains one row
        per cell and multiplicity method, with family-level Monte-Carlo SEs and
        confidence intervals.  No result is labelled pass/fail in advance.

    """
    n_sims = int(cfg["n_sims"])
    n_resamples = int(cfg["n_resamples"])
    replication_batch_size = int(cfg.get("replication_batch_size", 64))
    _validate_equivalence_config(cfg)
    boundary_design = build_boundary_design(cfg)
    groups = _build_equivalence_groups(cfg, boundary_design)
    result_rows: list[dict[str, Any]] = []

    for group, mean_shifts in groups:
        cell, m_targets, null_fraction, n_null, n_alternative = group
        key = cell_key(
            int(cfg["seed"]),
            "mc_equivalence_calibration",
            "variance_cell",
            cell.name,
            "m_targets",
            m_targets,
            "null_fraction",
            null_fraction,
        )
        group_results = execute_replication_cell(
            _equivalence_cell,
            key,
            mean_shifts,
            name="mc_equivalence_calibration.cell",
            replication_count=n_sims,
            replication_batch_size=replication_batch_size,
            static={
                "variance_cell": cell.name,
                "m_targets": m_targets,
                "null_fraction": null_fraction,
            },
            n_null=n_null,
            n_alternative=n_alternative,
            n_target=int(cfg["n_target"]),
            n_candidate=int(cfg["n_candidate"]),
            m_targets=m_targets,
            target_sd=cell.target_sd,
            candidate_sd=cell.candidate_sd,
            correlation=float(cfg["target_correlation"]),
            delta=float(cfg["delta"]),
            n_resamples=n_resamples,
            alpha=float(cfg["alpha"]),
            n_sims=n_sims,
            batch_size=replication_batch_size,
        )
        result_rows.extend(_aggregate_equivalence_group(cfg, group, group_results))

    results = pd.DataFrame(result_rows)
    production_scale = (
        n_sims >= _PRODUCTION_REPLICATIONS and n_resamples >= _PRODUCTION_REPLICATIONS
    )
    summary_cells = []
    for row in results.to_dict(orient="records"):
        marginal_power = float(row["marginal_power"])
        summary_cells.append(
            {
                "variance_cell": row["variance_cell"],
                "m_targets": int(row["m_targets"]),
                "null_fraction": float(row["null_fraction"]),
                "method": row["method"],
                "fwer": round(float(row["fwer"]), 6),
                "fwer_mc_se": round(float(row["fwer_mc_se"]), 6),
                "marginal_boundary_size": round(
                    float(row["marginal_boundary_size"]), 6
                ),
                "marginal_power": (
                    None if math.isnan(marginal_power) else round(marginal_power, 6)
                ),
            }
        )
    summary = {
        "design": {
            "null": "H0: population energy distance >= delta",
            "alternative": "H1: population energy distance < delta",
            "boundary_construction": (
                "closed-form univariate Gaussian energy distance; mean shift "
                "root-solved so population energy equals delta"
            ),
            "correlation": (
                "common target sample and correlated candidate innovations"
            ),
            "primary_method": "studentized multiplier bootstrap + k-minus-1 rule",
            "bh_status": "comparison only; not the pre-specified diagnostic rule",
        },
        "production_scale": production_scale,
        "params": cfg,
        "max_absolute_boundary_error": float(
            boundary_design["absolute_boundary_error"].max()
        ),
        "cells": summary_cells,
        "interpretation": (
            "Empirical marginal rejection, family rejection, power, Monte-Carlo "
            "SEs, and intervals are "
            "reported without an ex-ante pass/fail label. Draw conclusions only "
            "from a production-scale run."
        ),
    }
    return results, boundary_design, summary
