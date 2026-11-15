"""Focused tests for Paper 5's sequential horizon-and-universe bridge."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from pipeline.figures.paper5.universe_horizon_bridge import (
    render_paper5_universe_horizon_bridge_table,
)
from pipeline.stages.papers.paper5 import universe_horizon_bridge as bridge

if TYPE_CHECKING:
    from pathlib import Path


def _identity() -> dict[str, str]:
    """Return the operator identity shared by synthetic governed results."""
    return {
        "provider_id": "qwen3-embedding-8b",
        "representation_id": "qwen3-embedding-8b-unit",
        "barycentre_arm_id": "wasserstein_w2_loo",
        "geometry_id": "wasserstein_w2",
    }


def _write_governed_result(
    path: Path,
    *,
    rho: float,
    lower: float,
    upper: float,
    start: str,
    end: str,
    trading_days: int,
) -> None:
    """Write one minimal governed post-corpus result fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "identity": _identity(),
                "primary_period": "pooled_2023_2026",
                "results": {
                    "pooled_2023_2026": {
                        "w_flat": {
                            "rho_hat": rho,
                            "rho_ci_lower": lower,
                            "rho_ci_upper": upper,
                            "start": start,
                            "end": end,
                            "trading_days": trading_days,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_bridge_estimates_sequential_cells_and_descriptive_contrasts(  # noqa: PLR0915
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bridge changes one axis per step and bootstraps only the new cell."""
    paths = bridge.Paper5UniverseHorizonBridgePaths(
        returns_dir=tmp_path / "returns",
        current_oos_returns_dir=tmp_path / "current-oos",
        current_barycentre_dir=tmp_path / "current-w",
        current_results_json=tmp_path / "current-results.json",
        reference_oos_returns_dir=tmp_path / "reference-oos",
        reference_barycentre_dir=tmp_path / "reference-w",
        reference_results_json=tmp_path / "reference-results.json",
        output_dir=tmp_path / "bridge",
    )
    config = bridge.Paper5UniverseHorizonBridgeConfig(
        expected_reference_universe=2,
        expected_current_universe=4,
        n_bootstrap=4,
    )
    reference_tickers = ["A", "B"]
    current_tickers = ["A", "B", "C", "D"]
    reference_weights = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    current_weights = np.full((4, 4), 1.0 / 3.0)
    np.fill_diagonal(current_weights, 0.0)
    common_dates = np.asarray(
        ["2023-01-03", "2023-01-04", "2023-01-05"],
        dtype="datetime64[D]",
    )
    common_returns = np.asarray(
        [[0.01, 0.02], [0.03, -0.01], [-0.02, 0.01]],
        dtype=np.float64,
    )
    long_dates = np.append(common_dates, np.datetime64("2023-01-09"))
    long_returns = np.vstack([common_returns, [0.02, 0.04]])
    current_dates = common_dates[[0, 2]]
    current_returns = np.asarray(
        [[0.01, 0.02, -0.01, 0.03], [-0.02, 0.01, 0.04, -0.03]],
        dtype=np.float64,
    )

    def fake_operator(
        artifact_dir: Path,
        _config: object,
    ) -> tuple[list[str], np.ndarray, dict[str, object]]:
        if artifact_dir == paths.current_barycentre_dir:
            return current_tickers, current_weights, {}
        return reference_tickers, reference_weights, {}

    def fake_panel(
        _returns_dir: Path,
        tickers: list[str],
        _start: str | None = None,
        end: str | None = None,
        oos_returns_dir: Path | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if tickers == current_tickers:
            return current_dates, current_returns
        if end == config.long_end:
            return long_dates, long_returns
        assert oos_returns_dir in {
            paths.current_oos_returns_dir,
            paths.reference_oos_returns_dir,
        }
        return common_dates, common_returns.copy()

    def fake_qmle(
        returns: np.ndarray,
        _weights: np.ndarray,
        *,
        compute_se: bool = True,
    ) -> dict[str, float | bool]:
        del compute_se
        if returns.shape == long_returns.shape:
            rho = 0.3
        elif returns.shape == current_returns.shape:
            rho = 0.4
        else:
            rho = 0.2
        return {"rho_hat": rho, "loglik": 10.0 + rho, "converged": True}

    rho_draws = np.asarray([0.10, 0.20, 0.30, 0.25], dtype=np.float64)

    def fake_bootstrap_rho(
        _returns: np.ndarray,
        _weights: np.ndarray,
        _indices: np.ndarray,
    ) -> np.ndarray:
        return rho_draws

    monkeypatch.setattr(bridge, "load_operator", fake_operator)
    monkeypatch.setattr(bridge, "load_aligned_return_panel", fake_panel)
    monkeypatch.setattr(bridge.est, "sar_qmle", fake_qmle)
    monkeypatch.setattr(bridge, "_bootstrap_rho", fake_bootstrap_rho)
    _write_governed_result(
        paths.reference_results_json,
        rho=0.3,
        lower=0.25,
        upper=0.35,
        start="2023-01-03",
        end="2023-01-09",
        trading_days=4,
    )
    _write_governed_result(
        paths.current_results_json,
        rho=0.4,
        lower=0.35,
        upper=0.45,
        start="2023-01-03",
        end="2023-01-05",
        trading_days=2,
    )

    payload = bridge.run_paper5_universe_horizon_bridge(paths, config)

    cells = payload["cells"]
    assert isinstance(cells, dict)
    assert list(cells) == [
        "reference_52_long",
        "reference_52_short",
        "current_100_short",
    ]
    assert all(cell["status"] == "observed" for cell in cells.values())
    assert cells["reference_52_short"]["rho_hat"] == pytest.approx(0.2)
    assert cells["reference_52_short"]["rho_ci_lower"] == pytest.approx(
        np.quantile(rho_draws, 0.025)
    )
    assert cells["reference_52_long"][
        "governed_rho_absolute_difference"
    ] == pytest.approx(0.0)
    assert cells["current_100_short"]["trading_days"] == len(current_returns)
    contrasts = payload["contrasts"]
    assert isinstance(contrasts, dict)
    horizon = contrasts["horizon_long_minus_short_at_52"]
    universe = contrasts["universe_100_minus_52_at_short"]
    assert horizon["rho_difference"] == pytest.approx(0.1)
    assert universe["rho_difference"] == pytest.approx(0.2)
    assert horizon["lambda_difference"] == pytest.approx(0.17857142857142855)
    assert universe["lambda_difference"] == pytest.approx(0.41666666666666674)
    assert payload["scope"] == {
        "interpretation": "descriptive_noncausal",
        "reference_long_role": "historical_comparison",
    }
    design = payload["design"]
    assert isinstance(design, dict)
    assert design["reference_revision"] == config.reference_revision
    common_check = payload["common_return_check"]
    assert isinstance(common_check, dict)
    assert common_check["max_abs_difference"] == pytest.approx(0.0)
    draws = pd.read_parquet(paths.output_dir / "bootstrap_draws.parquet")
    assert draws["cell_id"].unique().tolist() == ["reference_52_short"]
    np.testing.assert_allclose(draws["rho_draw"], rho_draws)
    assert (paths.output_dir / "results.json").read_text(encoding="utf-8") == (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def test_common_return_check_rejects_a_material_difference() -> None:
    """A shared-return discrepancy beyond the pinned tolerance fails loudly."""
    dates = np.asarray(["2023-01-03", "2023-01-04"], dtype="datetime64[D]")
    reference = np.asarray([[0.01, 0.02], [0.03, 0.04]], dtype=np.float64)
    current = reference.copy()
    current[1, 1] += 1e-8

    with pytest.raises(ValueError, match="above tolerance"):
        bridge._verify_common_return_panel(  # noqa: SLF001 - invariant unit test
            dates,
            current,
            dates.copy(),
            reference,
            tolerance=1e-12,
        )


def test_renderer_emits_one_three_specification_booktabs_table(tmp_path: Path) -> None:
    """The table is a compact five-column sequence naming its return specification."""

    def cell(
        cell_id: str,
        universe: int,
        horizon: str,
        rho: float,
        lower: float,
        upper: float,
        trading_days: int,
        end: str,
    ) -> dict[str, object]:
        return {
            "cell_id": cell_id,
            "universe_size": universe,
            "horizon": horizon,
            "status": "observed",
            "window_start": "2023-01-03",
            "window_end": end,
            "trading_days": trading_days,
            "rho_hat": rho,
            "rho_ci_lower": lower,
            "rho_ci_upper": upper,
            "lambda_hat": rho / (1.0 - rho),
            "lambda_ci_lower": lower / (1.0 - lower),
            "lambda_ci_upper": upper / (1.0 - upper),
        }

    summary = {
        "schema_version": 1,
        "design": {
            "specification_order": [
                "reference_52_long",
                "reference_52_short",
                "current_100_short",
            ]
        },
        "scope": {
            "interpretation": "descriptive_noncausal",
            "reference_long_role": "historical_comparison",
        },
        "cells": {
            "reference_52_long": cell(
                "reference_52_long", 52, "long", 0.70, 0.65, 0.75, 884, "2026-07-15"
            ),
            "reference_52_short": cell(
                "reference_52_short",
                52,
                "short",
                0.68,
                0.63,
                0.73,
                816,
                "2026-04-06",
            ),
            "current_100_short": cell(
                "current_100_short",
                100,
                "short",
                0.84,
                0.81,
                0.87,
                815,
                "2026-04-06",
            ),
        },
        "contrasts": {
            "horizon_long_minus_short_at_52": {
                "contrast_id": "horizon_long_minus_short_at_52",
                "status": "descriptive_only",
                "rho_difference": 0.02,
                "lambda_difference": 0.21,
                "inference": "none",
            },
            "universe_100_minus_52_at_short": {
                "contrast_id": "universe_100_minus_52_at_short",
                "status": "descriptive_only",
                "rho_difference": 0.16,
                "lambda_difference": 3.13,
                "inference": "none",
            },
        },
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    output_path = tmp_path / "generated" / "universe_horizon_bridge.tex"

    render_paper5_universe_horizon_bridge_table(summary_path, output_path)

    rendered = output_path.read_text(encoding="utf-8")
    assert rendered.count(r"\begin{table}") == 1
    assert r"\toprule" in rendered
    assert "Historical reference" in rendered
    assert "Horizon-matched" in rendered
    assert "Current universe" in rendered
    assert "Panel B" not in rendered
    assert "tabularx" not in rendered
    assert r"\hat\lambda" not in rendered
    assert "15 Jul 2026" in rendered
    assert "6 Apr 2026" in rendered
    assert "3 January 2023" in rendered
    assert "raw-return, common-intercept specification" in rendered
    assert "changes both the roster and the operator" in rendered
    assert "Complete-case trading-day counts can differ across universes" in rendered
    assert r"\label{tab:p5-universe-horizon-bridge}" in rendered
    assert "unavailable" not in rendered.lower()
    assert "imputation" not in rendered.lower()
