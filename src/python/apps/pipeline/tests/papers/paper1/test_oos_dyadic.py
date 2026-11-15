"""t37 contract for the out-of-sample PRIMARY dyadic association.

Written before the stage was ever executed, so the convention is pinned in
advance rather than after an estimate was seen. It pins four things:

1. the DVC graph wires the FIXED in-sample energy matrix and the FIXED in-sample
   embeddings against the OOS covariance matrix, and never touches the in-sample
   covariance matrix (the look-ahead channel) or an OOS text matrix (which does
   not exist);
2. the inference convention is the in-sample one -- 1,999 multinomial node
   bootstrap draws, seed 42, 2.5/97.5 percentile interval, symmetric additive
   firm effects -- and is not swept over encoders;
3. the W2 arm runs on the realized panel with the governed node-count array;
4. a null or negative result round-trips into the generated
   numbers unaltered.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import yaml
from pipeline.figures.paper1.numbers import Paper1NumberInputs, write_paper1_numbers
from pipeline.io.contracts import write_manifest
from pipeline.precision import own_float64
from pipeline.stages.papers.paper1.dyadic_oos import (
    _DEFAULT_BOOTSTRAP_ITERS,
    _DEFAULT_BOOTSTRAP_SEED,
    DyadicOosConfig,
    run_dyadic_oos,
)

LEAD_W2_DISTANCE = (
    "${shared}/typed_distances/${scientific_selectors.baseline_model}/"
    "${scientific_selectors.baseline_model}-unit/wasserstein_w2"
)
OOS_COVARIANCE = "${shared}/oos/covariance_matrix.parquet"
IN_SAMPLE_COVARIANCE = "${shared}/covariance_matrix.parquet"

_PANEL_FIRMS = 9
_TEST_ITERS = 200
_IN_SAMPLE_ITERS = 1999
_IN_SAMPLE_SEED = 42
_PUBLICATION_FIRMS = 100
_PUBLICATION_DYADS = _PUBLICATION_FIRMS * (_PUBLICATION_FIRMS - 1) // 2
_PUBLICATION_DESIGN_COLUMNS = _PUBLICATION_FIRMS + 1
_PUBLICATION_RESIDUAL_DOF = _PUBLICATION_DYADS - _PUBLICATION_DESIGN_COLUMNS


def _repository_root() -> Path:
    return next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "dvc.yaml").exists()
    )


def _stages() -> dict[str, dict[str, object]]:
    root = _repository_root()
    document = yaml.safe_load((root / "dvc.yaml").read_text(encoding="utf-8"))
    return cast("dict[str, dict[str, object]]", document["stages"])


def _out_paths(stage: dict[str, object]) -> set[str]:
    outputs = cast("list[object]", stage.get("outs", [])) + cast(
        "list[object]", stage.get("metrics", [])
    )
    return {
        str(next(iter(output))) if isinstance(output, dict) else str(output)
        for output in outputs
    }


def _synthetic_panel(
    tmp_path: Path, *, drop_ticker: str | None = None, seed: int = 7
) -> tuple[Path, Path]:
    """Write a small energy/covariance/embedding triple to ``tmp_path``.

    ``drop_ticker`` removes one firm from the covariance side only, so the test
    can check that attrition is derived from the realized intersection instead of
    inherited from the frozen text universe.
    """
    rng = np.random.default_rng(seed)
    firms = [f"F{index:02d}" for index in range(_PANEL_FIRMS)]

    energy_values = np.zeros((len(firms), len(firms)), dtype=np.float64)
    covariance_rows: list[dict[str, object]] = []
    effects = rng.normal(scale=0.2, size=len(firms))
    for left in range(len(firms)):
        for right in range(left + 1, len(firms)):
            functional = float(rng.uniform(0.05, 0.6))
            energy_values[left, right] = functional
            energy_values[right, left] = functional
            if drop_ticker in {firms[left], firms[right]}:
                continue
            chord = (
                0.9
                + 0.4 * np.sqrt(functional)
                + effects[left]
                + effects[right]
                + 0.02 * rng.normal()
            )
            chord = float(np.clip(chord, 0.05, 1.99))
            covariance_rows.append(
                {
                    "ticker_i": firms[left],
                    "ticker_j": firms[right],
                    "covariance": 0.0,
                    "correlation": 1.0 - chord * chord / 2.0,
                }
            )

    distance_artifact_dir = tmp_path / "typed-distance"
    distance_artifact_dir.mkdir()
    distance_path = distance_artifact_dir / "distances.parquet"
    pd.DataFrame(
        [
            {"item_i": left, "item_j": right, "value": energy_values[i, j]}
            for i, left in enumerate(firms)
            for j, right in enumerate(firms)
        ]
    ).to_parquet(distance_path, index=False)
    digest = hashlib.sha256(distance_path.read_bytes()).hexdigest()
    output_identity = {
        "path": str(distance_path),
        "kind": "parquet",
        "sha256": digest,
        "bytes": distance_path.stat().st_size,
    }
    metadata = {
        "provider_id": "qwen3-embedding-8b",
        "representation_id": "qwen3-embedding-8b-unit",
        "distance_id": "wasserstein_w2",
        "shape": [len(firms), len(firms)],
        "output_dtype": "float64",
    }
    summary = {
        "provider_id": metadata["provider_id"],
        "representation_id": metadata["representation_id"],
        "distance_id": metadata["distance_id"],
        "item_ids": firms,
        "metadata": metadata,
        "output_inventory": [output_identity],
    }
    (distance_artifact_dir / "summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    write_manifest(
        distance_artifact_dir / "provenance.manifest.json",
        {
            "schema_version": "1.0",
            "stage_key": "test_typed_distance",
            "lane": "pipeline",
            "phase": "compute",
            "protocol_id": "test.typed-distance.v1",
            "code_identity": {"kind": "sha256", "value": "test"},
            "parameter_identity": {"kind": "sha256", "value": "test"},
            "seed_policy": "fixed-test-seed",
            "upstream_artifacts": [],
            "environment": {},
            "outputs": [output_identity],
            "risk_class": "low",
            "metadata": metadata,
        },
    )
    covariance_path = tmp_path / "oos_covariance.parquet"
    pd.DataFrame(covariance_rows).to_parquet(covariance_path, index=False)
    return distance_artifact_dir, covariance_path


def _run(tmp_path: Path, drop_ticker: str | None = None) -> dict[str, object]:
    own_float64()
    distance_artifact_dir, covariance_path = _synthetic_panel(
        tmp_path, drop_ticker=drop_ticker
    )
    return run_dyadic_oos(
        DyadicOosConfig(
            distance_artifact_dir=distance_artifact_dir,
            covariance_matrix=covariance_path,
            output_file=tmp_path / "out" / "summary.yaml",
            bootstrap_iters=_TEST_ITERS,
            bootstrap_seed=_DEFAULT_BOOTSTRAP_SEED,
        )
    )


# ---------------------------------------------------------------------------
# 1. DVC wiring: frozen text side, OOS return side, no look-ahead channel
# ---------------------------------------------------------------------------


def test_oos_dyadic_stage_holds_the_text_side_fixed() -> None:
    """Only returns are out of sample; the text corpus is frozen."""
    stage = _stages()["p1_dyadic_oos"]
    cmd = str(stage["cmd"])
    deps = set(cast("list[str]", stage["deps"]))
    assert f"--distance-artifact-dir {LEAD_W2_DISTANCE}" in cmd
    assert f"--covariance-matrix {OOS_COVARIANCE}" in cmd
    assert "--embeddings-dir" not in cmd
    assert LEAD_W2_DISTANCE in deps
    assert OOS_COVARIANCE in deps
    # The in-sample covariance matrix is the look-ahead channel: never a dep.
    assert IN_SAMPLE_COVARIANCE not in deps
    assert IN_SAMPLE_COVARIANCE not in cmd
    # No OOS text side exists; it must not be invented.
    assert not any(dep.startswith("${shared}/oos/energy") for dep in deps)
    assert not any(dep.startswith("${shared}/oos/embeddings") for dep in deps)


def test_oos_dyadic_stage_is_not_swept_over_encoders() -> None:
    """A four-encoder sweep would open a model-selection channel."""
    stage = _stages()["p1_dyadic_oos"]
    assert "matrix" not in stage
    assert "foreach" not in stage


def test_oos_dyadic_outputs_are_governed_and_cached() -> None:
    """Pin the governed artifact set for the stage."""
    assert _out_paths(_stages()["p1_dyadic_oos"]) == {
        "${p1}/dyadic_oos/results.json",
        "${p1}/dyadic_oos/bootstrap_draws.parquet",
        "${p1}/dyadic_oos/node_bootstrap_counts.parquet",
    }


# ---------------------------------------------------------------------------
# 2. The convention is fixed in advance by the in-sample design
# ---------------------------------------------------------------------------


def test_inference_convention_matches_the_in_sample_design() -> None:
    """1,999 node-bootstrap draws at seed 42, as in sample."""
    assert _DEFAULT_BOOTSTRAP_ITERS == _IN_SAMPLE_ITERS
    assert _DEFAULT_BOOTSTRAP_SEED == _IN_SAMPLE_SEED


def test_w2_arm_reports_the_percentile_interval_convention(tmp_path: Path) -> None:
    """The primary W2 arm declares the governed bootstrap and interval."""
    result = _run(tmp_path)
    arm = cast("dict[str, object]", result["w2_arm"])
    assert arm["regressor"] == "w2_distance"
    assert arm["se_type"] == "multinomial node bootstrap (dyad weight w_i*w_j)"
    assert arm["interval_type"] == "percentile [2.5, 97.5]"
    assert arm["bootstrap_requested_draws"] == _TEST_ITERS
    assert cast("float", arm["ci_low"]) <= cast("float", arm["ci_high"])
    assert result["bootstrap_common_node_counts"] is True


def test_realized_panel_and_degrees_of_freedom_are_derived(tmp_path: Path) -> None:
    """Counts come from the surviving panel, never from an assumed size."""
    dropped = "F03"
    result = _run(tmp_path, drop_ticker=dropped)
    panel = cast("dict[str, object]", result["panel"])
    surviving = _PANEL_FIRMS - 1
    assert panel["n_firms"] == surviving
    assert panel["n_dyads"] == surviving * (surviving - 1) // 2
    assert panel["graph_is_complete"] is True
    assert panel["firm_effect_columns"] == surviving - 1
    # intercept + focal regressor + (N-1) firm effects
    assert panel["design_columns"] == surviving + 1
    assert panel["residual_degrees_of_freedom"] == cast("int", panel["n_dyads"]) - cast(
        "int", panel["design_columns"]
    )
    assert isinstance(panel["degrees_of_freedom_statement"], str)
    attrition = cast("dict[str, object]", panel["attrition"])
    assert attrition["frozen_text_universe_firms"] == _PANEL_FIRMS
    assert attrition["dropped_from_text_universe"] == [dropped]


def test_governed_artifacts_are_written(tmp_path: Path) -> None:
    """Summary, per-draw records, and node multiplicities are all persisted."""
    result = _run(tmp_path)
    directory = tmp_path / "out"
    assert (directory / "summary.yaml").exists()
    draws = pd.read_parquet(directory / "bootstrap_draws.parquet")
    counts = pd.read_parquet(directory / "node_bootstrap_counts.parquet")
    assert set(draws["arm"]) == {"wasserstein_w2"}
    panel = cast("dict[str, object]", result["panel"])
    assert len(counts) == _TEST_ITERS * cast("int", panel["n_firms"])
    assert draws["draw_id"].nunique() == _TEST_ITERS


# ---------------------------------------------------------------------------
# 4. A null or reversing result must survive the render unaltered
# ---------------------------------------------------------------------------


def _oos_dyadic_summary(coef: float, ci: tuple[float, float]) -> dict[str, object]:
    return {
        "panel": {
            "n_firms": _PUBLICATION_FIRMS,
            "n_dyads": _PUBLICATION_DYADS,
            "design_columns": _PUBLICATION_DESIGN_COLUMNS,
            "residual_degrees_of_freedom": _PUBLICATION_RESIDUAL_DOF,
            "graph_is_complete": True,
            "attrition": {
                "frozen_text_universe_firms": _PUBLICATION_FIRMS,
                "surviving_firms": _PUBLICATION_FIRMS,
                "dropped_from_text_universe": [],
            },
        },
        "w2_arm": {
            "coef": coef,
            "se": 0.3,
            "pvalue": 0.42,
            "ci_low": ci[0],
            "ci_high": ci[1],
            "effect_one_sd": coef * 0.05,
            "within_r2": 0.1,
        },
    }


def test_oos_dyadic_bindings_are_emitted(tmp_path: Path) -> None:
    """Route the primary W2 arm through the generated numbers."""
    numbers = tmp_path / "numbers.tex"
    write_paper1_numbers(
        numbers,
        Paper1NumberInputs(
            oos_dyadic=_oos_dyadic_summary(1.5, (1.0, 2.0)),
        ),
    )
    emitted = numbers.read_text(encoding="utf-8")
    assert r"\ppDeclareValue{oos-dyadic-w2-coef}{1.500000}" in emitted
    assert r"\ppDeclareValue{oos-dyadic-w2-ci-low}{1.000000}" in emitted
    assert r"\ppDeclareValue{oos-dyadic-w2-ci-high}{2.000000}" in emitted
    assert rf"\ppDeclareValue{{oos-dyadic-n-firms}}{{{_PUBLICATION_FIRMS}}}" in emitted
    assert rf"\ppDeclareValue{{oos-dyadic-n-dyads}}{{{_PUBLICATION_DYADS}}}" in emitted
    assert (
        rf"\ppDeclareValue{{oos-dyadic-residual-dof}}{{{_PUBLICATION_RESIDUAL_DOF}}}"
        in emitted
    )


def test_null_oos_dyadic_result_round_trips_intact(tmp_path: Path) -> None:
    """A negative coefficient whose interval crosses zero survives unaltered.

    This is the guard that matters. The task ships its finding either way, so a
    sign-clipping or significance-gating formatter is a correctness failure.
    """
    numbers = tmp_path / "numbers.tex"
    write_paper1_numbers(
        numbers,
        Paper1NumberInputs(
            oos_dyadic=_oos_dyadic_summary(-0.234567, (-1.234567, 0.765432)),
        ),
    )
    emitted = numbers.read_text(encoding="utf-8")
    assert r"\ppDeclareValue{oos-dyadic-w2-coef}{-0.234567}" in emitted
    assert r"\ppDeclareValue{oos-dyadic-w2-ci-low}{-1.234567}" in emitted
    assert r"\ppDeclareValue{oos-dyadic-w2-ci-high}{0.765432}" in emitted


def test_missing_oos_dyadic_summary_leaves_numbers_unchanged(tmp_path: Path) -> None:
    """Absent OOS dyadic artifacts must not break the default numbers render."""
    with_oos = tmp_path / "with.tex"
    without_oos = tmp_path / "without.tex"
    write_paper1_numbers(
        without_oos,
        Paper1NumberInputs(),
    )
    write_paper1_numbers(
        with_oos,
        Paper1NumberInputs(oos_dyadic={}),
    )
    baseline = without_oos.read_text(encoding="utf-8")
    assert "oos-dyadic" not in baseline
    assert with_oos.read_text(encoding="utf-8") == baseline
