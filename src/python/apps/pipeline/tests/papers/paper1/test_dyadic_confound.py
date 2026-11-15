"""Inference-contract tests for Paper 1's undirected dyadic ladder."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from pipeline.stages.papers.paper1._dyadic.inputs import liquidity_size_proxy
from pipeline.stages.papers.paper1.dyadic_confound import (
    _SQUARED_DISTANCE_ARBITRATION,
    DyadicFitRequest,
    DyadicInference,
    DyadicModel,
    DyadicSample,
    _BootstrapDesign,
    _design_diagnostic_row,
    _draw_node_counts,
    _embedding_dispersions,
    _fit_spec,
    _fwl_display,
    _influence_diagnostics,
    _partial_r2_block,
    _return_chord_translation,
    _shared_endpoint_scale_statistic,
    _symmetric_firm_effects,
)
from statadic.estimation.least_squares.dyadic import DyadicEstimator
from statadic.inference.dyadic import (
    DyadicBootstrapDesign,
    NodeCountSchedule,
    bootstrap_coefficient_inference,
    bootstrap_dyadic,
    draw_node_counts,
    node_bootstrap_coefficient,
)
from statadic.numerics.linalg.dyadic import NodeIndex, symmetric_dyadic_effects

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

NODE_BOOTSTRAP_DRAW_COUNT = 80
EXPECTED_VALID_DRAWS = 120
NONBASE_FIRM_COUNT = 11
PANEL_NODE_COUNT = 12
ACTIVE_NODE_COUNT = 7


@pytest.fixture(autouse=True)
def _float64() -> Iterator[None]:
    """Own float64 per test, as the stage's CLI command does.

    Without this the module passed only when an earlier module had flipped the
    global flag, so its result depended on collection order.
    """
    with jax.enable_x64(new_val=True):
        yield


def test_legacy_node_schedule_adapter_preserves_draws() -> None:
    """The historical matrix adapter preserves typed multinomial entity counts."""
    assert _draw_node_counts is not draw_node_counts
    legacy = _draw_node_counts(PANEL_NODE_COUNT, 4, 7)
    typed = draw_node_counts(NodeIndex(tuple(range(PANEL_NODE_COUNT))), 4, 7)
    np.testing.assert_array_equal(legacy, typed.counts)
    assert _symmetric_firm_effects is symmetric_dyadic_effects


def test_embedding_dispersions_omit_undefined_direction_panels(
    tmp_path: Path,
) -> None:
    """Dyadic dispersion controls never repair invalid embedding rows."""
    invalid = np.array([[0.0, 0.0], [1.0, 0.0]])
    pd.DataFrame({"embedding": list(invalid)}).to_parquet(tmp_path / "AAA.parquet")
    assert _embedding_dispersions(tmp_path, ["AAA"]) is None


def _panel() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4)
    firms = np.array([f"f{i:02d}" for i in range(12)])
    i_idx, j_idx = np.triu_indices(len(firms), k=1)
    firm_i, firm_j = firms[i_idx], firms[j_idx]
    energy = rng.uniform(0.1, 0.8, size=len(i_idx))
    effects = rng.normal(scale=0.2, size=len(firms))
    y = 0.7 * energy + effects[i_idx] + effects[j_idx]
    columns = {"energy_distance": energy}
    return columns, y, firm_i, firm_j


def _fit_request(
    regressor_names: list[str],
    panel: tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray],
    inference: DyadicInference[str],
    *,
    node_effects: bool = False,
    effect_name_prefix: str = "node_effect",
) -> DyadicFitRequest[str]:
    """Build a typed fit request from the compact synthetic-panel fixture."""
    columns, outcome, firm_i, firm_j = panel
    return DyadicFitRequest[str](
        DyadicModel[str](
            regressor_names,
            columns,
            DyadicSample[str](outcome, firm_i.tolist(), firm_j.tolist()),
            node_effects=node_effects,
            effect_name_prefix=effect_name_prefix,
        ),
        inference,
    )


def test_symmetric_fe_point_and_bootstrap_are_endpoint_order_invariant() -> None:
    """Keep point estimates and bootstrap errors invariant to endpoint order."""
    columns, y, firm_i, firm_j = _panel()
    fit = _fit_spec(
        _fit_request(
            ["energy_distance"],
            (columns, y, firm_i, firm_j),
            DyadicInference[str](bootstrap_iters=120, bootstrap_seed=11),
            node_effects=True,
        )
    )
    swapped = _fit_spec(
        _fit_request(
            ["energy_distance"],
            (columns, y, firm_j, firm_i),
            DyadicInference[str](bootstrap_iters=120, bootstrap_seed=11),
            node_effects=True,
        )
    )
    assert fit["coef_energy_distance"] == pytest.approx(
        swapped["coef_energy_distance"], abs=1e-12
    )
    assert fit["se_energy_distance"] == pytest.approx(
        swapped["se_energy_distance"], abs=1e-12
    )


def test_pipeline_adapter_preserves_the_statadic_fit_and_draws() -> None:
    """The dataframe/JSON adapter must not alter the promoted estimator."""
    request = _fit_request(
        ["energy_distance"],
        _panel(),
        DyadicInference[str](bootstrap_iters=120, bootstrap_seed=13),
        node_effects=True,
        effect_name_prefix="firm_fe",
    )

    point = DyadicEstimator().estimate(request.model)
    schedule = draw_node_counts(
        point.node_index,
        request.inference.bootstrap_iters,
        request.inference.bootstrap_seed,
    )
    draws = bootstrap_dyadic(
        DyadicBootstrapDesign(
            design=point.design,
            column_names=list(point.column_names),
            sample=request.model.sample,
            node_schedule=schedule,
            node_effects=request.model.node_effects,
            effect_name_prefix=request.model.effect_name_prefix,
            required_columns=request.inference.required_bootstrap_columns,
        )
    )
    series = draws.coefficient("energy_distance")
    valid = draws.solve_ok & jnp.isfinite(series)
    standard_error = bootstrap_coefficient_inference(
        series[valid], point.coefficients["energy_distance"]
    ).standard_error
    adapted = _fit_spec(request, include_draws=True)
    frame = cast("pd.DataFrame", adapted["_bootstrap_draws"])

    assert adapted["coef_energy_distance"] == pytest.approx(
        point.coefficients["energy_distance"], abs=1e-12
    )
    assert adapted["se_energy_distance"] == pytest.approx(standard_error, abs=1e-12)
    np.testing.assert_allclose(
        frame["beta_energy"].to_numpy(dtype=float),
        series,
        equal_nan=True,
    )
    point_names = cast("list[str]", adapted["_point_design_names"])
    assert any(name.startswith("firm_fe[") for name in point_names)
    assert not any(name.startswith("node_effect[") for name in point_names)


def test_symmetric_fe_is_ticker_rename_invariant_and_full_rank() -> None:
    """Keep symmetric effects full-rank and invariant to ticker renaming."""
    columns, y, firm_i, firm_j = _panel()
    renamed_i = np.array([f"renamed_{firm}" for firm in firm_i])
    renamed_j = np.array([f"renamed_{firm}" for firm in firm_j])
    fit = _fit_spec(
        _fit_request(
            ["energy_distance"],
            (columns, y, firm_i, firm_j),
            DyadicInference[str](bootstrap_iters=120, bootstrap_seed=3),
            node_effects=True,
        )
    )
    renamed = _fit_spec(
        _fit_request(
            ["energy_distance"],
            (columns, y, renamed_i, renamed_j),
            DyadicInference[str](bootstrap_iters=120, bootstrap_seed=3),
            node_effects=True,
        )
    )
    assert fit["coef_energy_distance"] == pytest.approx(
        renamed["coef_energy_distance"], abs=1e-12
    )
    assert fit["se_energy_distance"] == pytest.approx(
        renamed["se_energy_distance"], abs=1e-12
    )
    assert fit["design_rank"] == fit["design_columns"]
    assert fit["coef_energy_distance"] == pytest.approx(
        fit["fwl_energy_distance"], abs=1e-10
    )
    assert fit["effect_one_sd_energy_distance"] == pytest.approx(
        cast("float", fit["coef_energy_distance"])
        * np.std(columns["energy_distance"], ddof=1)
    )


def test_node_bootstrap_is_deterministic_and_uses_node_weights() -> None:
    """Use deterministic node-weight draws in the dyadic bootstrap."""
    columns, y, firm_i, firm_j = _panel()
    x = np.column_stack([np.ones(len(y)), columns["energy_distance"]])
    sample = DyadicSample[str](y, firm_i.tolist(), firm_j.tolist())
    first = node_bootstrap_coefficient(
        x, sample, 1, DyadicInference[str](bootstrap_iters=80, bootstrap_seed=19)
    )
    second = node_bootstrap_coefficient(
        x, sample, 1, DyadicInference[str](bootstrap_iters=80, bootstrap_seed=19)
    )
    other_seed = node_bootstrap_coefficient(
        x, sample, 1, DyadicInference[str](bootstrap_iters=80, bootstrap_seed=20)
    )
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other_seed)
    assert first.size == NODE_BOOTSTRAP_DRAW_COUNT


def test_published_rung_uses_only_node_bootstrap_inference() -> None:
    """Use node-bootstrap inference exclusively for the published rung."""
    columns, y, firm_i, firm_j = _panel()
    fit = _fit_spec(
        _fit_request(
            ["energy_distance"],
            (columns, y, firm_i, firm_j),
            DyadicInference[str](bootstrap_iters=120, bootstrap_seed=7),
        )
    )
    assert fit["se_type"] == "multinomial node bootstrap (dyad weight w_i*w_j)"
    assert fit["bootstrap_valid_draws"] == EXPECTED_VALID_DRAWS
    assert "analytic" not in str(fit["se_type"]).lower()


def test_symmetric_fe_has_one_column_per_nonbase_firm() -> None:
    """Allocate one symmetric-effect column per non-base firm."""
    _columns, _y, firm_i, firm_j = _panel()
    matrix, labels = _symmetric_firm_effects(firm_i.tolist(), firm_j.tolist())
    assert matrix.shape == (len(firm_i), NONBASE_FIRM_COUNT)
    assert len(labels) == NONBASE_FIRM_COUNT


def test_common_node_count_array_is_deterministic_and_sums_to_node_count() -> None:
    """Reproduce node counts whose rows sum to the panel node count."""
    first = _draw_node_counts(12, 40, 17)
    second = _draw_node_counts(12, 40, 17)
    assert np.array_equal(first, second)
    assert np.all(first.sum(axis=1) == PANEL_NODE_COUNT)
    assert np.any(first == 0)


def test_bootstrap_rank_is_against_active_fe_design() -> None:
    """Measure bootstrap rank against the active fixed-effect design."""
    columns, y, firm_i, firm_j = _panel()
    fe, fe_names = _symmetric_firm_effects(firm_i.tolist(), firm_j.tolist())
    x = np.column_stack([np.ones(len(y)), columns["energy_distance"], fe])
    counts = np.zeros((1, 12), dtype=int)
    counts[0, :7] = [2, 2, 2, 2, 2, 1, 1]
    sample = DyadicSample[str](y, firm_i.tolist(), firm_j.tolist())
    result = bootstrap_dyadic(
        _BootstrapDesign[str](
            design=x,
            column_names=["intercept", "energy_distance", *fe_names],
            sample=sample,
            node_schedule=NodeCountSchedule[str](
                NodeIndex[str](
                    tuple(sorted(set(firm_i.tolist()) | set(firm_j.tolist())))
                ),
                counts,
            ),
            node_effects=True,
        ),
    )
    assert bool(result.solve_ok[0])
    assert result.active_nodes[0] == ACTIVE_NODE_COUNT
    assert result.effective_rank[0] == result.effective_columns[0]
    assert result.effective_rank[0] < x.shape[1]


def test_design_audit_records_complete_graph_overlap() -> None:
    """Record complete-graph degrees and endpoint overlap in the audit."""
    columns, _y, firm_i, firm_j = _panel()
    row = _design_diagnostic_row(
        6,
        DyadicModel[str](
            ["energy_distance"],
            columns,
            DyadicSample[str](_y, firm_i.tolist(), firm_j.tolist()),
        ),
        firm_effects=True,
    )
    assert row["degree_min"] == NONBASE_FIRM_COUNT
    assert row["degree_max"] == NONBASE_FIRM_COUNT
    assert row["design_rank"] == row["design_columns"]
    overlap_share = float(cast("float | int | str", row["endpoint_overlap_share"]))
    assert 0.0 < overlap_share < 1.0


def test_squared_residual_statistic_detects_shared_endpoint_scale() -> None:
    """Detect shared-endpoint scale in squared residuals."""
    _columns, _y, firm_i, firm_j = _panel()
    rng = np.random.default_rng(9)
    iid = rng.normal(size=len(firm_i))
    node_scale = np.where((firm_i == "f00") | (firm_j == "f00"), 8.0, 1.0)
    clustered = iid * node_scale
    assert _shared_endpoint_scale_statistic(
        clustered**2, firm_i, firm_j
    ) > _shared_endpoint_scale_statistic(iid**2, firm_i, firm_j)


def test_influence_diagnostics_identify_dyad_and_descriptive_threshold() -> None:
    """Identify the most influential dyad and descriptive threshold status."""
    primary_fit = pd.DataFrame(
        {
            "ticker_i": ["AAL", "CMCSA"],
            "ticker_j": ["UAL", "NFLX"],
            "leverage": [0.0867, 0.0620],
            "studentized_residual": [-5.46, 4.57],
        }
    )
    lofo = pd.DataFrame(
        {
            "delta_beta": [-0.1, 0.04],
            "delta_in_bootstrap_se": [-0.52, 0.2],
            "sign_flip": [False, False],
        }
    )

    result = _influence_diagnostics(primary_fit, lofo, n_design_columns=1)

    assert result["max_leverage_dyad"] == "AAL--UAL"
    assert result["most_extreme_studentized_residual"] == pytest.approx(-5.46)
    assert result["leverage_exceeds_heuristic"] is False
    assert result["lofo_sign_flip_count"] == 0


@pytest.mark.parametrize(
    ("focal_name", "prefix"), [("energy_distance", "energy"), ("w2_distance", "w2")]
)
def test_fwl_display_slope_reproduces_the_reported_primary_coefficient(
    focal_name: str, prefix: str
) -> None:
    """Pin the FWL identity: the displayed slope is the reported coefficient."""
    columns, y, firm_i, firm_j = _panel()
    columns = {focal_name: columns["energy_distance"]}
    fit = _fit_spec(
        _fit_request(
            [focal_name],
            (columns, y, firm_i, firm_j),
            DyadicInference[str](bootstrap_iters=120, bootstrap_seed=11),
            node_effects=True,
        ),
        include_draws=True,
    )
    design = cast("np.ndarray", fit["_point_design"])
    names = cast("list[str]", fit["_point_design_names"])

    frame, summary = _fwl_display(design, names, y, "f00", focal_name)

    reported = float(cast("float", fit[f"coef_{focal_name}"]))
    # beta_E = (W_tilde' D_tilde)/(W_tilde' W_tilde) must equal the full-OLS
    # coefficient exactly, not approximately: that is what makes the display a
    # true FWL plot rather than a partial residual scatter.
    assert summary["slope"] == pytest.approx(reported, abs=1e-10)
    assert frame.columns.tolist() == [
        f"{prefix}_fwl_residual",
        "outcome_fwl_residual",
        "fwl_fitted",
    ]
    assert summary["focal_regressor"] == focal_name
    assert summary["n_dyads"] == len(y)
    energy_residual = frame[f"{prefix}_fwl_residual"].to_numpy()
    outcome_residual = frame["outcome_fwl_residual"].to_numpy()
    slope = float(
        (energy_residual @ outcome_residual) / (energy_residual @ energy_residual)
    )
    assert slope == pytest.approx(reported, abs=1e-10)
    assert frame["fwl_fitted"].to_numpy() == pytest.approx(slope * energy_residual)
    # Both sides are residualized on the same nuisance projector, so each
    # residual series is mean-zero and orthogonal to the firm-effect design.
    assert float(np.mean(energy_residual)) == pytest.approx(0.0, abs=1e-10)
    assert float(np.mean(outcome_residual)) == pytest.approx(0.0, abs=1e-10)
    assert summary["nuisance_columns"] == design.shape[1] - 1
    assert summary["omitted_firm_indicator"] == "f00"


def test_fwl_display_uses_the_rank_cleaned_point_design_projector() -> None:
    """Residualize on the estimated design, not a rebuilt firm-effect matrix."""
    columns, y, firm_i, firm_j = _panel()
    # A dyad-level constant is dropped from the point design for collinearity
    # with the intercept; a rebuilt nuisance design would not know that.
    columns = {**columns, "constant_control": np.ones(len(y))}
    fit = _fit_spec(
        _fit_request(
            ["energy_distance", "constant_control"],
            (columns, y, firm_i, firm_j),
            DyadicInference[str](bootstrap_iters=120, bootstrap_seed=11),
            node_effects=True,
        ),
        include_draws=True,
    )
    assert "constant_control" in cast("list[str]", fit["dropped_collinear_columns"])
    design = cast("np.ndarray", fit["_point_design"])
    names = cast("list[str]", fit["_point_design_names"])

    _frame, summary = _fwl_display(design, names, y, "f00")

    assert summary["slope"] == pytest.approx(
        float(cast("float", fit["coef_energy_distance"])), abs=1e-10
    )


def test_partial_r2_matches_its_stated_sse_formula() -> None:
    """Report partial R-squared as 1 - SSE_(FE+E)/SSE_FE, components included."""
    columns, y, firm_i, firm_j = _panel()
    fit = _fit_spec(
        _fit_request(
            ["energy_distance"],
            (columns, y, firm_i, firm_j),
            DyadicInference[str](bootstrap_iters=120, bootstrap_seed=11),
            node_effects=True,
        )
    )
    restricted = float(cast("float", fit["sse_firm_effects_only"]))
    full = float(cast("float", fit["sse_firm_effects_plus_regressors"]))

    block = _partial_r2_block(restricted, full, cast("float", fit["within_r2"]))

    assert block["definition"] == "partial_r2 = 1 - SSE_(FE+E)/SSE_FE"
    assert block["partial_r2"] == pytest.approx(1.0 - full / restricted)
    assert 0.0 < full < restricted


def test_pooled_rungs_report_no_partial_r2() -> None:
    """Leave conditional fit undefined where no restricted FE model exists."""
    columns, y, firm_i, firm_j = _panel()
    fit = _fit_spec(
        _fit_request(
            ["energy_distance"],
            (columns, y, firm_i, firm_j),
            DyadicInference[str](bootstrap_iters=120, bootstrap_seed=11),
        )
    )
    assert fit["within_r2"] is None
    assert fit["sse_firm_effects_only"] is None
    assert fit["sse_firm_effects_plus_regressors"] is None


def test_return_chord_quantiles_and_correlation_translation_are_exact() -> None:
    """Emit governed chord quantiles and the signed rho = 1 - D^2/2 mapping."""
    y = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    effect = 0.05

    translation = _return_chord_translation(y, effect)

    quantiles = cast("dict[str, float]", translation["return_chord_distance_quantiles"])
    assert set(quantiles) == {"q25", "q50", "q75"}
    assert quantiles["q25"] == pytest.approx(float(np.percentile(y, 25.0)))
    assert quantiles["q50"] == pytest.approx(0.6)
    assert quantiles["q75"] == pytest.approx(float(np.percentile(y, 75.0)))
    anchors = cast("dict[str, dict[str, float]]", translation["anchors"])
    for key, chord in quantiles.items():
        anchor = anchors[key]
        assert anchor["correlation"] == pytest.approx(1.0 - chord**2 / 2.0)
        assert anchor["chord_distance_after_one_sd"] == pytest.approx(chord + effect)
        assert anchor["correlation_after_one_sd"] == pytest.approx(
            1.0 - (chord + effect) ** 2 / 2.0
        )
        # d rho = -D dD - (dD)^2 / 2, exactly, and negative for a positive
        # chord-distance effect: a wider news gap means a lower correlation.
        assert anchor["delta_correlation"] == pytest.approx(
            -chord * effect - effect**2 / 2.0
        )
        assert anchor["delta_correlation"] < 0.0


def test_squared_distance_sensitivity_rejection_is_recorded() -> None:
    """Keep the t34.2 requirement-5 arbitration auditable in the payload."""
    assert _SQUARED_DISTANCE_ARBITRATION["decision"] == "rejected"
    reason = str(_SQUARED_DISTANCE_ARBITRATION["reason"])
    assert "additive firm margin" in reason
    assert "exposure scales" in reason


def test_liquidity_proxy_preserves_missing_volume_exclusions(tmp_path: Path) -> None:
    """Missing historical volume stays missing and unrelated firms stay excluded."""
    path = tmp_path / "proxy.json"
    path.write_text(json.dumps({"A": 1.25, "C": 5.0}))
    assert liquidity_size_proxy(path, ["B", "A"]) == {"A": 1.25}


@pytest.mark.parametrize("value", [None, "1.25", True, float("nan"), float("inf")])
def test_liquidity_proxy_rejects_invalid_numeric_values(
    tmp_path: Path, value: object
) -> None:
    """A changed JSON type or nonfinite control cannot silently enter estimation."""
    path = tmp_path / "proxy.json"
    path.write_text(json.dumps({"A": value}))
    with pytest.raises(ValueError, match="finite numeric"):
        liquidity_size_proxy(path, ["A"])
