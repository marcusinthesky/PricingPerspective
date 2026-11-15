"""Independent formal-mode and geometry-invariance benchmarks."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from hotelling.dynamics import engine
from hotelling.dynamics.engine import (
    build_canonical_economy,
    canonical_capacity,
    operating_cash_flow,
)
from hotelling.dynamics.geometry import interaction, squared_distance
from hotelling.dynamics.valuation import derive_model, permanent_tilt_readout

MODE_TOLERANCE = 1e-8
GEOMETRY_TOLERANCE = 1e-14


@pytest.mark.parametrize("kappa", [0.0, 1.0, 4.0])
def test_permanent_tilt_payout_matches_lean_horizon_zero_and_stage_scale(
    kappa: float,
) -> None:
    """Price actual payouts on the canonical permanent path, including h=0."""
    firms, horizon = 6, 7
    economy = build_canonical_economy(firms, kappa=kappa)
    baseline = np.array(canonical_capacity(firms), copy=True)
    model = derive_model(economy, baseline)
    readout = permanent_tilt_readout(model, horizon)
    np.testing.assert_array_equal(readout.capacity_response[0], 0)
    signed_share = 2 * baseline[:, 0] - 1
    denominator = 1 + firms / 2
    np.testing.assert_allclose(
        readout.payout_loading[0], signed_share / denominator, atol=MODE_TOLERANCE
    )
    # The Lean stageScale uses nodeResponse at Q=1, not the largest observed
    # firm's share. Reconstruct that endpoint through the affine node response.
    shares = baseline[:, 0]
    for h, direction in enumerate(readout.capacity_response):
        slope = (direction[-1, 0] - direction[0, 0]) / (shares[-1] - shares[0])
        endpoint_response = direction[0, 0] + (1 - shares[0]) * slope
        stage_scale = (
            1 / denominator
            + (endpoint_response - direction[:, 0].sum()) / denominator**2
        )
        np.testing.assert_allclose(
            readout.payout_loading[h],
            signed_share * stage_scale,
            atol=MODE_TOLERANCE,
            rtol=MODE_TOLERANCE,
        )
        # Independent central differences of the nonlinear operating payout
        # check that both direct opportunity news and all rival effects enter.
        step = 1e-6
        news = np.array([step, 0.0])
        derivative = (
            operating_cash_flow(economy, baseline + step * direction, news)
            - operating_cash_flow(economy, baseline - step * direction, -news)
        ) / (2 * step)
        np.testing.assert_allclose(
            readout.payout_loading[h],
            derivative,
            atol=MODE_TOLERANCE,
            rtol=MODE_TOLERANCE,
        )


def test_permanent_tilt_valuation_is_discounted_actual_horizon_payouts() -> None:
    """A long sum of actual payout tangents verifies the permanent resolvent."""
    firms = 6
    economy = build_canonical_economy(firms)
    model = derive_model(economy, canonical_capacity(firms))
    readout = permanent_tilt_readout(model, 800)
    discount = economy.config.discount ** np.arange(len(readout.payout_loading))
    summed = discount @ readout.payout_loading
    np.testing.assert_allclose(
        summed, readout.total_payoff_loading, atol=MODE_TOLERANCE, rtol=MODE_TOLERANCE
    )
    np.testing.assert_allclose(
        readout.return_loading,
        summed / model.baseline_prices,
        atol=MODE_TOLERANCE,
        rtol=MODE_TOLERANCE,
    )


@pytest.mark.parametrize("firms", [4, 100])
@pytest.mark.parametrize("kappa", [0.0, 1.0, 4.0])
def test_canonical_permanent_response_matches_both_lean_modes(
    firms: int, kappa: float
) -> None:
    """The derivative recurrence reproduces nodeResponse, including both decays."""
    economy = build_canonical_economy(firms, kappa=kappa)
    baseline = np.array(canonical_capacity(firms), copy=True)
    model = derive_model(economy, baseline)
    denominator = 1 + firms / 2
    node_d = economy.config.own_curvature + 1 / denominator**2
    aggregate_u = firms * (denominator - 1) / denominator**3
    node_h = node_d + aggregate_u
    aggregate_force = firms * (denominator - 0.5) / denominator**2
    node_u = (denominator - 2 * baseline) / denominator**3
    node_force = (denominator - baseline) / denominator**2
    perpendicular = node_force - aggregate_force / aggregate_u * node_u
    own_decay = kappa / (node_d + kappa)
    aggregate_decay = kappa / (node_h + kappa)
    deviation = np.zeros_like(baseline)
    for horizon in range(1, 9):
        deviation, _ = model.step(deviation, np.array([1.0, 0.0]))
        expected = perpendicular / node_d * (
            1 - own_decay**horizon
        ) + aggregate_force / aggregate_u * node_u / node_h * (
            1 - aggregate_decay**horizon
        )
        expected[:, 1] *= -1
        np.testing.assert_allclose(
            deviation, expected, rtol=MODE_TOLERANCE, atol=MODE_TOLERANCE
        )


def test_dimension_normalized_isometric_embedding() -> None:
    """Embedding preserves economic geometry when normalized distances agree."""
    points = np.random.default_rng(17).normal(size=(8, 3))
    embedded = np.pad(points * 2, ((0, 0), (0, 9)))
    np.testing.assert_allclose(
        squared_distance(points, points),
        squared_distance(embedded, embedded),
        atol=GEOMETRY_TOLERANCE,
    )
    np.testing.assert_allclose(
        interaction(points, points, 2),
        interaction(embedded, embedded, 2),
        atol=GEOMETRY_TOLERANCE,
    )


def test_numerical_core_keeps_orchestration_boundary() -> None:
    """Framework and filesystem orchestration cannot become core dependencies."""
    root = Path(engine.__file__).parent
    forbidden = {"mesa", "pipeline", "simulation", "pathlib", "tempfile", "os"}
    paths = [
        root / f"{name}.py"
        for name in (
            "config",
            "geometry",
            "engine",
            "valuation",
            "observation",
            "transport",
        )
    ]
    paths.append(root.parent / "orthant.py")
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
                assert node.module not in {
                    "hotelling.dynamics.mesa",
                    "hotelling.dynamics.artifacts",
                    "hotelling.dynamics.resources",
                }
        assert not imports & forbidden
