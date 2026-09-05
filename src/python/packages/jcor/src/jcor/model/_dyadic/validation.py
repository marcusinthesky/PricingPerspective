"""Validation shared by dyadic fit and node-bootstrap inference."""

from __future__ import annotations

from collections.abc import Hashable, Mapping  # noqa: TC003  # runtime hint
from typing import TYPE_CHECKING, cast

import jax.numpy as jnp

from jcor.model._dyadic.contracts import (
    ArrayView,  # noqa: TC001  # runtime jaxtyping contract
    DyadicInputError,
    NodeCountSchedule,
)
from jcor.operators.design import NodeIndex

if TYPE_CHECKING:
    from jcor.model._dyadic.contracts import (
        DyadicBootstrapDesign,
        DyadicInference,
        DyadicModel,
        DyadicSample,
    )

MATRIX_DIMENSIONS = 2
MIN_DYADIC_OBSERVATIONS = 2
_INTEGER_TYPES = (int, jnp.integer)
_BOOLEAN_TYPES = (bool, jnp.bool_)


def _array_shape(value: object) -> tuple[int, ...] | None:
    """Return an array-like's shape, or ``None`` if it is not array-shaped.

    Deliberately dtype-neutral: it must report the shape of a complex or
    string-labelled array so the dimension guards fire before the
    finite-real guard, preserving the retired NumPy ordering of errors.
    """
    shape = getattr(value, "shape", None)
    if isinstance(shape, tuple) and all(isinstance(size, int) for size in shape):
        return shape
    return None


def _is_finite_real_array(value: object) -> bool:
    """Report whether ``value`` is an all-finite real numeric array."""
    if _array_shape(value) is None:
        return False
    try:
        array = jnp.asarray(value)
    except (TypeError, ValueError):
        return False
    dtype = array.dtype
    return (
        jnp.issubdtype(dtype, jnp.number)
        and not jnp.issubdtype(dtype, jnp.bool_)
        and not jnp.issubdtype(dtype, jnp.complexfloating)
        and bool(jnp.all(jnp.isfinite(array)))
    )


def _validate_endpoint_index[NodeIdT: Hashable](
    endpoint_i: ArrayView,
    endpoint_j: ArrayView,
    node_index: NodeIndex[NodeIdT] | None = None,
) -> NodeIndex[NodeIdT]:
    try:
        if node_index is None:
            # NumPy erases arbitrary Python object element types, so this is the
            # one runtime-validated bridge into the otherwise propagated axis.
            return cast(
                "NodeIndex[NodeIdT]",
                NodeIndex.from_endpoints(endpoint_i, endpoint_j),
            )
        i_position = node_index.positions_for(endpoint_i)
        j_position = node_index.positions_for(endpoint_j)
    except ValueError as error:
        raise DyadicInputError(str(error)) from error
    if _array_shape(endpoint_i) != _array_shape(endpoint_j):
        message = "dyadic endpoint arrays must be aligned one-dimensional vectors"
        raise DyadicInputError(message)
    observed = set(i_position.tolist()) | set(j_position.tolist())
    if observed != set(range(len(node_index))):
        message = "node index must exactly match the observed dyadic endpoint IDs"
        raise DyadicInputError(message)
    if bool(jnp.any(i_position == j_position)):
        message = "undirected dyadic samples cannot contain self-dyads"
        raise DyadicInputError(message)
    return node_index


def validate_sample[NodeIdT: Hashable](
    sample: DyadicSample[NodeIdT],
    *,
    node_index: NodeIndex[NodeIdT] | None = None,
) -> NodeIndex[NodeIdT]:
    """Validate a nondegenerate real outcome and its distinct entity pairs."""
    shapes = [
        _array_shape(sample.outcome),
        _array_shape(sample.endpoint_i),
        _array_shape(sample.endpoint_j),
    ]
    if any(shape is None or len(shape) != 1 for shape in shapes):
        message = "outcome and endpoint IDs must be aligned one-dimensional arrays"
        raise DyadicInputError(message)
    outcome_shape, i_shape, j_shape = cast("list[tuple[int, ...]]", shapes)
    n_dyads = outcome_shape[0]
    if i_shape[0] != n_dyads or j_shape[0] != n_dyads:
        message = "outcome and endpoint IDs must be aligned one-dimensional arrays"
        raise DyadicInputError(message)
    if n_dyads < MIN_DYADIC_OBSERVATIONS:
        message = "dyadic estimation requires at least two observed dyads"
        raise DyadicInputError(message)
    if not _is_finite_real_array(sample.outcome):
        message = "dyadic outcomes must be finite real numeric values"
        raise DyadicInputError(message)
    outcome = jnp.asarray(sample.outcome)
    if bool(jnp.all(outcome == outcome[0])):
        message = "dyadic outcomes must be nonconstant so R-squared is defined"
        raise DyadicInputError(message)
    return _validate_endpoint_index(sample.endpoint_i, sample.endpoint_j, node_index)


def _validate_regressors(
    regressor_names: list[str],
    columns: Mapping[str, ArrayView],
    n_dyads: int,
) -> None:
    if not isinstance(regressor_names, list) or any(
        not isinstance(name, str) or not name for name in regressor_names
    ):
        message = "dyadic regressor names must be non-empty strings"
        raise DyadicInputError(message)
    if len(regressor_names) != len(set(regressor_names)):
        message = "dyadic regressor names must be unique"
        raise DyadicInputError(message)
    if "intercept" in regressor_names:
        message = "dyadic regressor name 'intercept' is reserved"
        raise DyadicInputError(message)
    if not isinstance(columns, dict):
        message = "dyadic regressor columns must be supplied by name"
        raise DyadicInputError(message)
    missing = [name for name in regressor_names if name not in columns]
    if missing:
        message = f"missing dyadic regressor columns: {missing}"
        raise DyadicInputError(message)
    misaligned = [
        name for name in regressor_names if _array_shape(columns[name]) != (n_dyads,)
    ]
    if misaligned:
        message = f"misaligned dyadic regressor columns: {misaligned}"
        raise DyadicInputError(message)
    invalid = [
        name for name in regressor_names if not _is_finite_real_array(columns[name])
    ]
    if invalid:
        message = f"dyadic regressors must be finite real numeric values: {invalid}"
        raise DyadicInputError(message)


def validate_design_inputs[NodeIdT: Hashable](
    regressor_names: list[str],
    columns: Mapping[str, ArrayView],
    endpoint_i: ArrayView,
    endpoint_j: ArrayView,
    *,
    node_index: NodeIndex[NodeIdT] | None = None,
) -> NodeIndex[NodeIdT]:
    """Validate the component-wise public design-builder boundary."""
    index = _validate_endpoint_index(endpoint_i, endpoint_j, node_index)
    endpoint_shape = _array_shape(endpoint_i)
    if endpoint_shape is None or len(endpoint_shape) != 1:
        message = "dyadic endpoint arrays must be aligned one-dimensional vectors"
        raise DyadicInputError(message)
    _validate_regressors(regressor_names, columns, endpoint_shape[0])
    return index


def validate_model[NodeIdT: Hashable](
    model: DyadicModel[NodeIdT],
) -> NodeIndex[NodeIdT]:
    """Validate one aligned finite regressor specification."""
    index = validate_sample(model.sample)
    outcome_shape = cast("tuple[int, ...]", _array_shape(model.sample.outcome))
    _validate_regressors(model.regressor_names, model.columns, outcome_shape[0])
    return index


def validate_inference[NodeIdT: Hashable](
    inference: DyadicInference[NodeIdT],
) -> None:
    """Validate eager bootstrap controls before JAX or linear algebra sees them."""
    if not isinstance(inference.node_effects, _BOOLEAN_TYPES):
        message = "node_effects must be a boolean"
        raise DyadicInputError(message)
    if (
        not isinstance(inference.effect_name_prefix, str)
        or not inference.effect_name_prefix
    ):
        message = "dyadic effect name prefix must be a non-empty string"
        raise DyadicInputError(message)
    if (
        not isinstance(inference.bootstrap_iters, _INTEGER_TYPES)
        or isinstance(inference.bootstrap_iters, _BOOLEAN_TYPES)
        or inference.bootstrap_iters < 1
    ):
        message = "node bootstrap requires a positive integer draw count"
        raise DyadicInputError(message)
    if not isinstance(inference.bootstrap_seed, _INTEGER_TYPES) or isinstance(
        inference.bootstrap_seed, _BOOLEAN_TYPES
    ):
        message = "node bootstrap seed must be an integer"
        raise DyadicInputError(message)
    required = inference.required_bootstrap_columns
    if not isinstance(required, tuple) or any(
        not isinstance(name, str) or not name for name in required
    ):
        message = "required bootstrap columns must be non-empty string names"
        raise DyadicInputError(message)
    if len(required) != len(set(required)):
        message = "required bootstrap columns must be unique"
        raise DyadicInputError(message)


def validate_node_schedule[NodeIdT: Hashable](
    schedule: NodeCountSchedule[NodeIdT],
    node_index: NodeIndex[NodeIdT],
    *,
    expected_draws: int | None = None,
) -> None:
    """Validate a multinomial schedule and its exact entity-column mapping."""
    if not isinstance(schedule, NodeCountSchedule):
        message = "node bootstrap requires a NodeCountSchedule"
        raise DyadicInputError(message)
    if schedule.node_index != node_index:
        message = "node-count schedule IDs do not match the dyadic sample"
        raise DyadicInputError(message)
    node_counts = schedule.counts
    n_nodes = len(node_index)
    counts_shape = _array_shape(node_counts)
    if counts_shape is None:
        message = "node counts must be supplied as a two-dimensional count matrix"
        raise DyadicInputError(message)
    if len(counts_shape) != MATRIX_DIMENSIONS or counts_shape[1] != n_nodes:
        width = counts_shape[1] if len(counts_shape) == MATRIX_DIMENSIONS else None
        message = f"node-count matrix has {width} columns for {n_nodes} nodes"
        raise DyadicInputError(message)
    if counts_shape[0] < 1:
        message = "node-count matrix must contain at least one draw"
        raise DyadicInputError(message)
    if expected_draws is not None and counts_shape[0] != expected_draws:
        message = (
            f"node-count matrix has {counts_shape[0]} draws, expected {expected_draws}"
        )
        raise DyadicInputError(message)
    if not _is_finite_real_array(node_counts):
        message = "node counts must be finite nonnegative integers"
        raise DyadicInputError(message)
    counts = jnp.asarray(node_counts)
    if bool(jnp.any(counts < 0)) or bool(jnp.any(counts != jnp.floor(counts))):
        message = "node counts must be finite nonnegative integers"
        raise DyadicInputError(message)
    if bool(jnp.any(jnp.sum(counts, axis=1) != n_nodes)):
        message = f"each node-count draw must sum to the {n_nodes} sampled nodes"
        raise DyadicInputError(message)


def _validate_bootstrap_matrix_and_names[NodeIdT: Hashable](
    inputs: DyadicBootstrapDesign[NodeIdT],
) -> list[str]:
    """Validate the numeric matrix and its injective column namespace."""
    design = inputs.design
    column_names = inputs.column_names
    if not isinstance(column_names, list) or not column_names:
        message = "dyadic design column names must be a non-empty list"
        raise DyadicInputError(message)
    outcome_shape = _array_shape(inputs.sample.outcome)
    if outcome_shape is None or _array_shape(design) != (
        outcome_shape[0],
        len(column_names),
    ):
        message = "design shape does not match the dyadic sample and column names"
        raise DyadicInputError(message)
    if not _is_finite_real_array(design):
        message = "dyadic bootstrap design must contain finite real numeric values"
        raise DyadicInputError(message)
    if any(not isinstance(name, str) or not name for name in column_names):
        message = "dyadic design column names must be non-empty strings"
        raise DyadicInputError(message)
    if len(column_names) != len(set(column_names)):
        message = "dyadic design column names must be unique"
        raise DyadicInputError(message)
    return column_names


def _validate_bootstrap_configuration[NodeIdT: Hashable](
    inputs: DyadicBootstrapDesign[NodeIdT],
    column_names: list[str],
) -> None:
    """Validate bootstrap flags and coefficient-retention requirements."""
    if not isinstance(inputs.node_effects, _BOOLEAN_TYPES):
        message = "node_effects must be a boolean"
        raise DyadicInputError(message)
    if not isinstance(inputs.effect_name_prefix, str) or not inputs.effect_name_prefix:
        message = "dyadic effect name prefix must be a non-empty string"
        raise DyadicInputError(message)
    required = inputs.required_columns
    if not isinstance(required, tuple) or any(
        not isinstance(name, str) or not name for name in required
    ):
        message = "required bootstrap columns must be non-empty string names"
        raise DyadicInputError(message)
    if len(required) != len(set(required)):
        message = "required bootstrap columns must be unique"
        raise DyadicInputError(message)
    missing_required = [name for name in required if name not in column_names]
    if missing_required:
        message = f"required bootstrap columns are absent: {missing_required}"
        raise DyadicInputError(message)
    if inputs.node_effects:
        effect_prefix = f"{inputs.effect_name_prefix}["
        if "intercept" not in column_names or not any(
            name.startswith(effect_prefix) for name in column_names
        ):
            message = (
                "node-effect bootstrap designs require intercept and effect columns"
            )
            raise DyadicInputError(message)


def validate_bootstrap_design[NodeIdT: Hashable](
    inputs: DyadicBootstrapDesign[NodeIdT],
    node_index: NodeIndex[NodeIdT],
) -> None:
    """Validate a complete public bootstrap-design request."""
    column_names = _validate_bootstrap_matrix_and_names(inputs)
    _validate_bootstrap_configuration(inputs, column_names)
    validate_node_schedule(inputs.node_schedule, node_index)
