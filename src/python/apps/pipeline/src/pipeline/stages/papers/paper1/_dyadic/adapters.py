"""Paper-1 NumPy adapters for ``statadic.structure.dyadic`` carriers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from statadic.inference.dyadic import NodeCountSchedule, draw_node_counts
from statadic.numerics.linalg.dyadic import (
    NodeIndex,
    fwl_residualize,
    symmetric_dyadic_effects,
)
from statadic.observation.data import Observations
from statadic.observation.design import Intercept, Linear
from statadic.regression import Regression
from statadic.structure.dyadic import Dyads, EndpointEffects

if TYPE_CHECKING:
    from statadic.law.dyadic import DyadicModel, DyadicSample
    from statadic.observation.design import Term
    from statadic.structure.dyadic import DyadicResult, PreparedDyadic

_OUTCOME = "outcome"
_ENDPOINTS = ("endpoint_i", "endpoint_j")


def dyadic_regression(
    model: DyadicModel[str],
) -> tuple[Regression[DyadicResult, PreparedDyadic], Observations]:
    """Express a native model as the Statadic specification and aligned table."""
    sample = model.sample
    columns = {_OUTCOME: np.asarray(sample.outcome, dtype=np.float64)}
    columns.update(
        {
            name: np.asarray(model.columns[name], dtype=np.float64)
            for name in model.regressor_names
        }
    )
    data = Observations(
        columns,
        labels={
            _ENDPOINTS[0]: np.asarray(sample.endpoint_i).tolist(),
            _ENDPOINTS[1]: np.asarray(sample.endpoint_j).tolist(),
        },
    )
    terms: Term = Intercept()
    for name in model.regressor_names:
        terms += Linear(name)
    spec = Regression(
        _OUTCOME,
        terms,
        structure=Dyads(*_ENDPOINTS),
        effects=(
            EndpointEffects(model.effect_name_prefix) if model.node_effects else None
        ),
    )
    return spec, data


def node_effect_projection(
    values: dict[str, np.ndarray],
    firm_i: np.ndarray,
    firm_j: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return ``(fitted, residual)`` of each column on intercept plus firm effects.

    Pure partialling, so it uses the ``fwl_residualize`` kernel with the column
    as its focal series against the intercept-plus-effects nuisance design.
    """
    effects, _ = symmetric_dyadic_effects(firm_i.tolist(), firm_j.tolist())
    nuisance = np.column_stack([np.ones(len(firm_i)), np.asarray(effects)])
    projections: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, column in values.items():
        series = np.asarray(column, dtype=np.float64)
        residual = np.asarray(
            fwl_residualize(
                np.column_stack([nuisance, series]), series, nuisance.shape[1]
            ).focal_residual
        )
        projections[name] = (series - residual, residual)
    return projections


def draw_legacy_node_counts(n_nodes: int, n_boot: int, seed: int) -> np.ndarray:
    """Return the bare count matrix of the native node-count schedule."""
    return np.asarray(
        draw_node_counts(NodeIndex(tuple(range(n_nodes))), n_boot, seed).counts
    )


def legacy_node_count_schedule(
    sample: DyadicSample[str], node_counts: np.ndarray
) -> NodeCountSchedule[str]:
    """Tie a retained count matrix to the sample's canonical node index."""
    return NodeCountSchedule(
        node_index=NodeIndex.from_endpoints(sample.endpoint_i, sample.endpoint_j),
        counts=node_counts,
    )
