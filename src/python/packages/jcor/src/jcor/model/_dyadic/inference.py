"""Entity-neutral multinomial node-bootstrap inference for dyadic models."""

from __future__ import annotations

from collections.abc import Hashable
from contextlib import suppress
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike  # noqa: TC002  # runtime jaxtyping contract

from jcor.core.precision import require_x64
from jcor.core.random import cell_key
from jcor.model._dyadic.contracts import (
    MIN_ACTIVE_DYADS,
    ArrayView,  # noqa: TC001  # runtime jaxtyping contract
    DyadicBootstrapDesign,
    DyadicBootstrapResult,
    DyadicComputationError,
    DyadicInputError,
    NodeCountSchedule,
)
from jcor.model._dyadic.solve import least_squares_host
from jcor.model._dyadic.validation import (
    validate_bootstrap_design,
    validate_inference,
    validate_node_schedule,
    validate_sample,
)
from jcor.operators.design import NodeIndex, drop_collinear_columns

if TYPE_CHECKING:
    from jcor.model._dyadic.contracts import DyadicInference, DyadicSample

_MATRIX_DIMENSIONS = 2
_INTEGER_TYPES = (int, jnp.integer)
_BOOLEAN_TYPES = (bool, jnp.bool_)


def _effective_rank(weighted_full: jax.Array, kept: jax.Array) -> int:
    """Return the rank of the retained columns without reslicing the design.

    ``matrix_rank(weighted_full[:, kept])`` recompiles once per retained width,
    and the width is exactly what varies from draw to draw -- the live log shows
    widths 4 through 41 on one schedule, against the "handful of distinct
    values" :func:`_fit_bootstrap_draw` assumes. Masking the dropped columns to
    zero holds the shape fixed instead: it leaves ``x @ x.T`` unchanged, so the
    nonzero singular values, and therefore the rank, are those of the slice.

    The tolerance is NumPy's default reproduced *for the sliced design*
    (``sigma_max * max(rows, kept) * eps``), not for the padded one. Only the
    ``max(rows, n)`` term could differ between the two, and only when the design
    is wider than it is tall; passing it explicitly keeps the verdict identical
    there too, which matters because the caller gates the solve on it.
    """
    width = weighted_full.shape[1]
    retained = set(kept.tolist())
    mask = jnp.asarray(
        [1.0 if position in retained else 0.0 for position in range(width)],
        dtype=jnp.float64,
    )
    singular = jnp.linalg.svd(weighted_full * mask[None, :], compute_uv=False)
    tolerance = (
        float(singular[0])
        * max(weighted_full.shape[0], len(retained))
        * float(jnp.finfo(jnp.float64).eps)
    )
    return int(jnp.sum(singular > tolerance))


def draw_node_counts[NodeIdT: Hashable](
    node_index: NodeIndex[NodeIdT],
    n_boot: int,
    seed: int,
) -> NodeCountSchedule[NodeIdT]:
    """Draw multinomial counts tied to an explicit canonical node index."""
    if not isinstance(node_index, NodeIndex):
        message = "node bootstrap requires a NodeIndex"
        raise DyadicInputError(message)
    if (
        not isinstance(n_boot, _INTEGER_TYPES)
        or isinstance(n_boot, _BOOLEAN_TYPES)
        or n_boot < 1
    ):
        message = "node bootstrap requires a positive integer draw count"
        raise DyadicInputError(message)
    if not isinstance(seed, _INTEGER_TYPES) or isinstance(seed, _BOOLEAN_TYPES):
        message = "node bootstrap seed must be an integer"
        raise DyadicInputError(message)
    n_boot = int(n_boot)
    seed = int(seed)
    n_nodes = len(node_index)
    key = cell_key(seed, "model", "draw_node_counts")
    probabilities = jnp.full(n_nodes, 1.0 / n_nodes, dtype=jnp.float64)
    counts = jax.random.multinomial(
        key,
        n_nodes,
        probabilities,
        shape=(n_boot, n_nodes),
        dtype=jnp.float64,
    )
    return NodeCountSchedule(
        node_index=node_index,
        counts=jnp.asarray(counts, dtype=jnp.int64),
    )


def _base_bootstrap_record(
    draw_id: int,
    counts: jax.Array,
    active: jax.Array,
) -> dict[str, object]:
    return {
        "draw_id": int(draw_id),
        "coefficients": {},
        "sse": float("nan"),
        "overall_r2": float("nan"),
        "within_r2": float("nan"),
        "active_nodes": int(jnp.sum(counts > 0)),
        "active_dyads": int(jnp.sum(active)),
        "effective_columns": 0,
        "effective_rank": 0,
        "solve_ok": False,
    }


def _bootstrap_within_r2(
    design: jax.Array,
    weighted_outcome: jax.Array,
    root_weights: jax.Array,
    column_names: list[str],
    effect_name_prefix: str,
    sse: float,
) -> float:
    """Return the within-R^2 of the node-effect-only fit for one draw.

    Takes the full-height design and the draw's ``root_weights``, matching
    :func:`_fit_bootstrap_draw`: inactive rows carry zero weight rather than
    being sliced out, so this second solve is shape-stable across draws too.
    """
    effect_prefix = f"{effect_name_prefix}["
    effect_positions = [
        position
        for position, name in enumerate(column_names)
        if name == "intercept" or name.startswith(effect_prefix)
    ]
    effects = design[:, jnp.asarray(effect_positions)]
    effects_full = effects * root_weights[:, None]
    kept = drop_collinear_columns(effects_full)
    weighted_effects = effects_full[:, kept]
    residual = weighted_outcome - weighted_effects @ least_squares_host(
        weighted_effects,
        weighted_outcome,
        context="bootstrap node-effect fit",
    )
    effect_sse = float(residual @ residual)
    return 1.0 - sse / effect_sse if effect_sse > 0.0 else float("nan")


def _records_to_bootstrap_result(
    records: list[dict[str, object]],
    column_names: list[str],
) -> DyadicBootstrapResult:
    coefficients = {
        name: jnp.asarray(
            [
                cast("dict[str, float]", record["coefficients"]).get(name, float("nan"))
                for record in records
            ],
            dtype=jnp.float64,
        )
        for name in column_names
    }

    def values(name: str, dtype: DTypeLike) -> jax.Array:
        return jnp.asarray([record[name] for record in records], dtype=dtype)

    return DyadicBootstrapResult(
        draw_id=values("draw_id", jnp.int64),
        coefficients=coefficients,
        sse=values("sse", jnp.float64),
        overall_r2=values("overall_r2", jnp.float64),
        within_r2=values("within_r2", jnp.float64),
        active_nodes=values("active_nodes", jnp.int64),
        active_dyads=values("active_dyads", jnp.int64),
        effective_columns=values("effective_columns", jnp.int64),
        effective_rank=values("effective_rank", jnp.int64),
        solve_ok=values("solve_ok", jnp.bool_),
    )


def _fit_bootstrap_draw[NodeIdT: Hashable](
    inputs: DyadicBootstrapDesign[NodeIdT],
    design: jax.Array,
    outcome: jax.Array,
    weights: jax.Array,
    record: dict[str, object],
) -> dict[str, object]:
    """Populate one active draw record, leaving invalid designs unsolved.

    Rows are weighted, never sliced. ``active`` is exactly ``weights > 0``, so
    ``sqrt(weights)`` already annihilates every inactive row: a zero row adds
    nothing to ``X'X``, ``X'y``, the residual, or the spectrum, making the
    full-height fit algebraically identical to the sliced one. Keeping the row
    count fixed is what makes it *fast* -- slicing made the design height vary
    per draw, so the jitted least-squares kernel missed its shape-keyed cache
    and recompiled every draw (measured 197.1 ms/draw and 47 compilations per
    15 draws, against 1.5 ms/draw and 0 at fixed height).

    Column width still varies with the host-only rank selection below. An
    earlier revision of this docstring claimed that took "only a handful of
    distinct values across the whole schedule"; a ``JAX_LOG_COMPILES=1`` run
    showed widths 4 through 41 on one schedule, so it is nearer one per draw.
    :func:`_effective_rank` no longer pays that cost -- it masks instead of
    slicing -- but the slice below still does, once per distinct width, for the
    ``gather``, ``matmul`` and ``linear_solve`` it feeds. That is deliberate
    rather than overlooked: ``least_squares_host`` documents fixed-width
    rank-selected input as its contract, and the guard above admits only
    full-rank designs, so handing it a zero-padded one would break the
    precondition it is written around.
    """
    root_weights = jnp.sqrt(weights.astype(jnp.float64))
    weighted_full = design * root_weights[:, None]
    weighted_outcome = outcome * root_weights
    kept = drop_collinear_columns(weighted_full)
    active_names = [inputs.column_names[position] for position in kept.tolist()]
    if any(name not in active_names for name in inputs.required_columns):
        return record
    rank = _effective_rank(weighted_full, kept)
    record["effective_columns"] = int(kept.size)
    record["effective_rank"] = rank
    if rank != int(kept.size):
        return record
    # Reached only for a full-rank retained set, which is the precondition
    # ``least_squares_host`` is written around; the slice stays here rather than
    # moving up so a rejected draw never pays for it.
    weighted_design = weighted_full[:, kept]
    beta = least_squares_host(
        weighted_design,
        weighted_outcome,
        context="bootstrap draw fit",
    )
    residual = weighted_outcome - weighted_design @ beta
    sse = float(residual @ residual)
    # Both moments are weighted sums, so the zero-weight rows drop out of the
    # numerator and the denominator alike -- identical to summing the active
    # subset, at a shape that does not change from draw to draw.
    draw_weights = weights.astype(jnp.float64)
    outcome_mean = float(jnp.sum(draw_weights * outcome) / jnp.sum(draw_weights))
    total_ss = float(jnp.sum(draw_weights * (outcome - outcome_mean) ** 2))
    within_r2 = float("nan")
    if inputs.node_effects:
        within_r2 = _bootstrap_within_r2(
            design,
            weighted_outcome,
            root_weights,
            inputs.column_names,
            inputs.effect_name_prefix,
            sse,
        )
    record.update(
        coefficients=dict(
            zip(active_names, (float(value) for value in beta), strict=True)
        ),
        sse=sse,
        overall_r2=(1.0 - sse / total_ss if total_ss > 0.0 else float("nan")),
        within_r2=within_r2,
        solve_ok=bool(jnp.all(jnp.isfinite(beta))),
    )
    return record


def bootstrap_dyadic[NodeIdT: Hashable](
    inputs: DyadicBootstrapDesign[NodeIdT],
) -> DyadicBootstrapResult:
    """Fit one design over a shared matrix of multinomial node-count draws."""
    require_x64("model.dyadic.bootstrap_dyadic")
    sample = inputs.sample
    if not isinstance(inputs.node_schedule, NodeCountSchedule):
        message = "node bootstrap requires a NodeCountSchedule"
        raise DyadicInputError(message)
    node_index = inputs.node_schedule.node_index
    validate_sample(sample, node_index=node_index)
    validate_bootstrap_design(inputs, node_index)
    i_position = node_index.positions_for(sample.endpoint_i)
    j_position = node_index.positions_for(sample.endpoint_j)
    records: list[dict[str, object]] = []
    design = jnp.asarray(inputs.design, dtype=jnp.float64)
    outcome = jnp.asarray(sample.outcome, dtype=jnp.float64)
    for draw_id, counts in enumerate(inputs.node_schedule.counts):
        weights = counts[i_position] * counts[j_position]
        active = weights > 0
        record = _base_bootstrap_record(draw_id, counts, active)
        if int(jnp.sum(active)) < MIN_ACTIVE_DYADS:
            records.append(record)
            continue
        # A draw-level numerical failure is one invalid bootstrap draw, not
        # a new exception law for an otherwise valid caller-owned schedule.
        # The retired NumPy path suppressed ``LinAlgError``/
        # ``FloatingPointError``; the JAX solve reports the same class of
        # failure as ``DyadicComputationError`` from ``least_squares_host``.
        with suppress(DyadicComputationError):
            record = _fit_bootstrap_draw(inputs, design, outcome, weights, record)
        records.append(record)
    return _records_to_bootstrap_result(records, inputs.column_names)


def bootstrap_coefficient_inference(
    draws: ArrayView,
    coefficient: float,
) -> tuple[float, float, float, float, float]:
    """Return bootstrap SE, t-statistic, two-sided tail p-value, and interval."""
    values = jnp.asarray(draws, dtype=jnp.float64)
    standard_error = float(jnp.std(values, ddof=1)) if values.size > 1 else float("nan")
    t_statistic = coefficient / standard_error if standard_error > 0 else float("nan")
    if not values.size:
        return standard_error, t_statistic, float("nan"), float("nan"), float("nan")
    lower_tail = (1.0 + float(jnp.sum(values <= 0.0))) / (values.size + 1.0)
    upper_tail = (1.0 + float(jnp.sum(values >= 0.0))) / (values.size + 1.0)
    p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))
    return (
        standard_error,
        t_statistic,
        p_value,
        float(jnp.percentile(values, 2.5)),
        float(jnp.percentile(values, 97.5)),
    )


def node_bootstrap_coefficient[NodeIdT: Hashable](
    design: ArrayView,
    sample: DyadicSample[NodeIdT],
    focal_column: int,
    inference: DyadicInference[NodeIdT],
) -> jax.Array:
    """Return valid node-bootstrap draws for one selected design column."""
    design_shape = getattr(design, "shape", None)
    if (
        not isinstance(design_shape, tuple)
        or len(design_shape) != _MATRIX_DIMENSIONS
        or not all(isinstance(size, int) for size in design_shape)
    ):
        message = "dyadic bootstrap design must be a two-dimensional array"
        raise DyadicInputError(message)
    if (
        not isinstance(focal_column, _INTEGER_TYPES)
        or isinstance(focal_column, _BOOLEAN_TYPES)
        or not 0 <= focal_column < design_shape[1]
    ):
        message = "focal column must be a valid nonnegative design-column index"
        raise DyadicInputError(message)
    validate_inference(inference)
    node_index = validate_sample(sample)
    names = [f"x{position}" for position in range(design_shape[1])]
    focal_name = names[focal_column]
    schedule = inference.node_schedule
    if schedule is None:
        schedule = draw_node_counts(
            node_index,
            inference.bootstrap_iters,
            inference.bootstrap_seed,
        )
    validate_node_schedule(
        schedule,
        node_index,
        expected_draws=inference.bootstrap_iters,
    )
    result = bootstrap_dyadic(
        DyadicBootstrapDesign(
            design=design,
            column_names=names,
            sample=sample,
            node_schedule=schedule,
            node_effects=False,
            effect_name_prefix=inference.effect_name_prefix,
            required_columns=(focal_name,),
        )
    )
    draws = result.coefficient(focal_name)
    return draws[result.solve_ok & jnp.isfinite(draws)]
