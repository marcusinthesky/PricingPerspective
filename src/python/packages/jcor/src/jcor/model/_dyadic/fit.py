"""Point estimation for entity-neutral undirected dyadic models."""

from __future__ import annotations

from collections.abc import Hashable, Mapping  # noqa: TC003  # runtime hint
from typing import TYPE_CHECKING

import jax  # noqa: TC002  # runtime: beartype resolves `jax.Array` annotations
import jax.numpy as jnp

from jcor.core.precision import require_x64
from jcor.model._dyadic.contracts import (
    ArrayView,  # noqa: TC001  # runtime jaxtyping contract
    DyadicBootstrapDesign,
    DyadicComputationError,
    DyadicFitResult,
    DyadicInputError,
)
from jcor.model._dyadic.inference import (
    bootstrap_coefficient_inference,
    bootstrap_dyadic,
    draw_node_counts,
)
from jcor.model._dyadic.solve import least_squares_host
from jcor.model._dyadic.validation import (
    validate_design_inputs,
    validate_inference,
    validate_model,
    validate_node_schedule,
)
from jcor.operators.design import (
    drop_collinear_columns,
    fwl_coefficient,
    symmetric_dyadic_effects,
)

if TYPE_CHECKING:
    from jcor.model._dyadic.contracts import DyadicBootstrapResult, DyadicFit
    from jcor.operators.design import NodeIndex


def build_dyadic_design[NodeIdT: Hashable](
    regressor_names: list[str],
    columns: Mapping[str, ArrayView],
    endpoint_i: ArrayView,
    endpoint_j: ArrayView,
    *,
    node_effects: bool,
    effect_name_prefix: str = "node_effect",
    node_index: NodeIndex[NodeIdT] | None = None,
) -> tuple[jax.Array, list[str], list[str]]:
    """Construct and rank-clean an intercept-first dyadic design matrix."""
    if not isinstance(node_effects, (bool, jnp.bool_)):
        message = "node_effects must be a boolean"
        raise DyadicInputError(message)
    if not isinstance(effect_name_prefix, str) or not effect_name_prefix:
        message = "dyadic effect name prefix must be a non-empty string"
        raise DyadicInputError(message)
    index = validate_design_inputs(
        regressor_names,
        columns,
        endpoint_i,
        endpoint_j,
        node_index=node_index,
    )
    names = ["intercept", *regressor_names]
    x = jnp.column_stack(
        [
            jnp.ones(len(endpoint_i), dtype=jnp.float64),
            *[
                jnp.asarray(columns[name], dtype=jnp.float64)
                for name in regressor_names
            ],
        ]
    )
    if node_effects:
        try:
            effects, effect_names = symmetric_dyadic_effects(
                endpoint_i,
                endpoint_j,
                name_prefix=effect_name_prefix,
                node_index=index,
            )
        except ValueError as error:
            raise DyadicInputError(str(error)) from error
        x = jnp.column_stack([x, effects])
        names.extend(effect_names)
    if len(names) != len(set(names)):
        message = "regressor names collide with generated dyadic design columns"
        raise DyadicInputError(message)
    # The retired NumPy path caught ``np.linalg.LinAlgError`` here. The JAX
    # rank core cannot raise it, so the failure is now detected from the
    # result: a validated finite design must retain at least one column.
    kept = drop_collinear_columns(x)
    if kept.size == 0:
        message = "dyadic rank cleaning failed for a validated finite design"
        raise DyadicComputationError(message)
    kept_set = set(kept.tolist())
    dropped = [
        names[position] for position in range(x.shape[1]) if position not in kept_set
    ]
    return x[:, kept], [names[position] for position in kept], dropped


def fit_dyadic_model[NodeIdT: Hashable](
    request: DyadicFit[NodeIdT],
) -> DyadicFitResult:
    """Fit one dyadic model and its shared multinomial node-bootstrap schedule."""
    require_x64("model.dyadic.fit_dyadic_model")
    model = request.model
    inference = request.inference
    validate_inference(inference)
    node_index = validate_model(model)
    sample = model.sample
    design, kept_names, dropped_columns = build_dyadic_design(
        model.regressor_names,
        model.columns,
        sample.endpoint_i,
        sample.endpoint_j,
        node_effects=inference.node_effects,
        effect_name_prefix=inference.effect_name_prefix,
        node_index=node_index,
    )
    # Explicit float64 despite the guard above: an array born under x64 keeps
    # its float64 label outside the scope but silently computes in float32
    # thereafter (measured 2.5e-8 relative error on this SSE path), which t65
    # decision 4 rates as an escalation rather than benign drift.
    outcome = jnp.asarray(sample.outcome, dtype=jnp.float64)
    beta = least_squares_host(design, outcome, context="least-squares fit")
    fitted = design @ beta
    residual = outcome - fitted
    return _assemble_fit_result(
        request=request,
        node_index=node_index,
        design=design,
        kept_names=kept_names,
        dropped_columns=dropped_columns,
        outcome=outcome,
        beta=beta,
        fitted=fitted,
        residual=residual,
    )


def _run_node_bootstrap[NodeIdT: Hashable](
    *,
    request: DyadicFit[NodeIdT],
    node_index: NodeIndex[NodeIdT],
    design: jax.Array,
    kept_names: list[str],
) -> DyadicBootstrapResult:
    """Resolve the shared node schedule and bootstrap one rank-cleaned design."""
    inference = request.inference
    node_schedule = inference.node_schedule
    if node_schedule is None:
        node_schedule = draw_node_counts(
            node_index,
            inference.bootstrap_iters,
            inference.bootstrap_seed,
        )
    validate_node_schedule(
        node_schedule,
        node_index,
        expected_draws=inference.bootstrap_iters,
    )
    unavailable_required = [
        name for name in inference.required_bootstrap_columns if name not in kept_names
    ]
    if unavailable_required:
        message = (
            "required bootstrap columns are absent from the rank-cleaned "
            f"point design: {unavailable_required}"
        )
        raise DyadicInputError(message)
    return bootstrap_dyadic(
        DyadicBootstrapDesign(
            design=design,
            column_names=kept_names,
            sample=request.model.sample,
            node_schedule=node_schedule,
            node_effects=inference.node_effects,
            effect_name_prefix=inference.effect_name_prefix,
            required_columns=inference.required_bootstrap_columns,
        )
    )


def _regressor_reports[NodeIdT: Hashable](
    *,
    request: DyadicFit[NodeIdT],
    bootstrap: DyadicBootstrapResult,
    coefficients: Mapping[str, float],
    design: jax.Array,
    outcome: jax.Array,
    kept_names: list[str],
) -> tuple[
    dict[str, tuple[float, float, float, float, float]],
    dict[str, float],
    dict[str, float],
]:
    """Report per-regressor bootstrap inference, one-SD effects, and FWL slopes."""
    model = request.model
    valid = bootstrap.solve_ok
    uncertainty: dict[str, tuple[float, float, float, float, float]] = {}
    effect_one_sd: dict[str, float] = {}
    fwl: dict[str, float] = {}
    for name in model.regressor_names:
        coefficient = float(coefficients[name])
        draws = bootstrap.coefficient(name)
        valid_draws = draws[valid & jnp.isfinite(draws)]
        uncertainty[name] = bootstrap_coefficient_inference(valid_draws, coefficient)
        column = jnp.asarray(model.columns[name], dtype=jnp.float64)
        effect_one_sd[name] = coefficient * float(jnp.std(column, ddof=1))
        if name in kept_names:
            # The retired NumPy path caught ``np.linalg.LinAlgError`` from a
            # non-convergent LAPACK SVD here. ``fwl_coefficient`` is now a JAX
            # least-squares that returns NaN instead of raising, and NaN is
            # already its documented "no variation after projection" outcome,
            # so the exception law is retired rather than relocated.
            fwl[name] = fwl_coefficient(design, outcome, kept_names.index(name))
    return uncertainty, effect_one_sd, fwl


def _restricted_node_effect_sse[NodeIdT: Hashable](
    *,
    request: DyadicFit[NodeIdT],
    node_index: NodeIndex[NodeIdT],
    outcome: jax.Array,
) -> float:
    """Return the SSE of the intercept-plus-node-effect restricted fit."""
    sample = request.model.sample
    effects, _ = symmetric_dyadic_effects(
        sample.endpoint_i,
        sample.endpoint_j,
        name_prefix=request.inference.effect_name_prefix,
        node_index=node_index,
    )
    restricted_design = jnp.column_stack(
        [jnp.ones(outcome.shape[0], dtype=jnp.float64), effects]
    )
    restricted_residual = outcome - restricted_design @ least_squares_host(
        restricted_design,
        outcome,
        context="restricted node-effect fit",
    )
    return float(restricted_residual @ restricted_residual)


def _assemble_fit_result[NodeIdT: Hashable](  # one eager door's state
    *,
    request: DyadicFit[NodeIdT],
    node_index: NodeIndex[NodeIdT],
    design: jax.Array,
    kept_names: list[str],
    dropped_columns: list[str],
    outcome: jax.Array,
    beta: jax.Array,
    fitted: jax.Array,
    residual: jax.Array,
) -> DyadicFitResult:
    """Run the bootstrap and assemble the host report for one fitted design."""
    model = request.model
    inference = request.inference
    coefficients = dict.fromkeys([*kept_names, *dropped_columns], 0.0)
    coefficients.update(
        dict(zip(kept_names, (float(value) for value in beta), strict=True))
    )
    bootstrap = _run_node_bootstrap(
        request=request,
        node_index=node_index,
        design=design,
        kept_names=kept_names,
    )
    valid = bootstrap.solve_ok
    uncertainty, effect_one_sd, fwl = _regressor_reports(
        request=request,
        bootstrap=bootstrap,
        coefficients=coefficients,
        design=design,
        outcome=outcome,
        kept_names=kept_names,
    )

    total_ss = float(jnp.sum((outcome - jnp.mean(outcome)) ** 2))
    full_sse = float(residual @ residual)
    overall_r2 = 1.0 - full_sse / total_ss
    within_r2: float | None = None
    restricted_sse: float | None = None
    if inference.node_effects:
        restricted_sse = _restricted_node_effect_sse(
            request=request,
            node_index=node_index,
            outcome=outcome,
        )
        within_r2 = 1.0 - full_sse / restricted_sse if restricted_sse > 0.0 else None
    return DyadicFitResult(
        n_dyads=int(outcome.shape[0]),
        regressor_names=tuple(model.regressor_names),
        coefficients={name: float(value) for name, value in coefficients.items()},
        effect_one_sd=effect_one_sd,
        bootstrap_inference=uncertainty,
        bootstrap_draws=bootstrap,
        bootstrap_valid_draws=int(jnp.sum(valid)),
        bootstrap_requested_draws=int(inference.bootstrap_iters),
        bootstrap_seed=int(inference.bootstrap_seed),
        overall_r2=overall_r2,
        within_r2=within_r2,
        sse_node_effects_only=restricted_sse,
        sse_full=full_sse,
        # ``build_dyadic_design`` already retained a maximal independent set.
        design_rank=int(design.shape[1]),
        design_columns=int(design.shape[1]),
        dropped_collinear_columns=tuple(dropped_columns),
        fwl_coefficients=fwl,
        point_design=design,
        point_design_names=tuple(kept_names),
        point_fitted=fitted,
        point_residual=residual,
    )
