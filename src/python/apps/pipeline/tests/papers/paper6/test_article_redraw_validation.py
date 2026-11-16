"""Behavioral checks for joint article resampling and fixed-history calibration."""

from __future__ import annotations

from collections import Counter
from itertools import product
from math import factorial
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from hotelling.dynamics.artifacts import read_arrays, write_arrays
from pipeline._kernels.paper6.article_redraw_validation import (
    HISTORIES,
    SCENARIOS,
    ArticleRedrawValidationConfig,
    archive_panel,
    bootstrap_archive,
    measure_outer,
    measure_regimes,
    measure_regimes_batch,
    outer_archive,
    sensitivity_quantiles,
    summarize_outer,
)
from pipeline._kernels.paper6.corporate_analysis import (
    exact_transport,
    point_masses,
    return_readout,
)
from pipeline._kernels.paper6.directional_validation import panel_arrays
from pipeline._kernels.paper6.reference_alignment import pool_dated_laws
from pipeline.stages.papers.paper6.article_redraw_validation import (
    _checkpoint,
    _failed_outer,
    _load_checkpoint,
    _manifest,
    _outer_task,
)
from pipeline.stages.papers.paper6.artifacts import write

if TYPE_CHECKING:
    from pathlib import Path

    from pipeline._kernels.paper6.corporate_analysis import Cloud


def _fixture() -> tuple[dict[str, np.ndarray], np.ndarray]:
    counts = np.ones((2, 5, 12, 4), dtype=np.int64)
    counts[:, :, ::3] = 0
    counts[1, :, :, 0] = 0
    base = np.linspace(0.2, 0.5, 4)[None, None, :]
    time = np.arange(60).reshape(5, 12, 1)
    activity = np.stack(
        (base + time * 0.001, np.broadcast_to(base, (5, 12, 4)), base + time * 0.0005)
    )
    laws = np.stack(
        (activity / 2, activity / 2, (1 - activity) / 2, (1 - activity) / 2), axis=-1
    )
    news = np.zeros((60, 4))
    news[[18, 30, 42, 54]] = np.linspace(-0.1, 0.2, 4)
    pooled = np.stack([pool_dated_laws(law, counts.sum(axis=0)) for law in laws])
    targets = measure_regimes(pooled, counts.sum(axis=0), news)
    return {
        "laws": laws,
        "news": news,
        **{f"target_{key}": value for key, value in targets.items()},
        "target_arithmetic_beta": targets["beta"].copy(),
        "target_arithmetic_q": targets["q"].copy(),
    }, counts


def test_outer_strata_fix_sizes_and_keep_regime_dates_paired() -> None:
    """Core/additional date laws remain separate and absent strata remain empty."""
    design, counts = _fixture()
    archive = outer_archive(design["laws"], counts, 3, 4)
    np.testing.assert_array_equal(archive, outer_archive(design["laws"], counts, 3, 4))
    assert not np.array_equal(
        archive, outer_archive(design["laws"], counts, 3, 4, pilot=True)
    )
    dates = archive.sum(axis=-1)
    for regime in range(len(SCENARIOS)):
        np.testing.assert_array_equal(dates[regime], dates[0])
        np.testing.assert_array_equal(dates[regime].sum(axis=2), counts.sum(axis=2))
    assert not archive[:, 1, :, :, 0].any()


def test_inner_resampling_keeps_joint_support_and_shared_dates() -> None:
    """A resampled date cannot acquire atoms absent from that archive's date."""
    design, counts = _fixture()
    archive = outer_archive(design["laws"], counts, 2, 1)
    sample = bootstrap_archive(archive, 2, 1, 5)
    np.testing.assert_array_equal(sample, bootstrap_archive(archive, 2, 1, 5))
    assert not sample[archive == 0].any()
    np.testing.assert_array_equal(sample.sum(axis=(3, 5)), archive.sum(axis=(3, 5)))
    dates = sample.sum(axis=-1)
    for regime in range(len(SCENARIOS)):
        np.testing.assert_array_equal(dates[regime], dates[0])
    assert not sample[:, 1, :, :, 0].any()


def _explicit_clouds(
    joint: np.ndarray, news: np.ndarray
) -> tuple[dict[tuple[int, int], Cloud], dict[int, tuple[np.ndarray, np.ndarray]]]:
    points = np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    clouds: dict[tuple[int, int], Cloud] = {}
    panels = {}
    for year in range(5):
        months = np.arange(
            np.datetime64(f"{2018 + year}-01", "M"),
            np.datetime64(f"{2019 + year}-01", "M"),
        )
        ends = (months + 1).astype("datetime64[D]") - np.timedelta64(1, "D")
        if year:
            panels[2018 + year] = (
                months.astype("datetime64[D]"),
                news[year * 12 : (year + 1) * 12],
            )
        for firm in range(4):
            coordinates: list[np.ndarray] = []
            article_dates: list[np.datetime64] = []
            flags: list[bool] = []
            for stratum in range(2):
                for month in range(12):
                    for atom in range(4):
                        copies = int(joint[0, stratum, year, month, firm, atom])
                        coordinates.extend([points[atom]] * copies)
                        article_dates.extend([ends[month]] * copies)
                        flags.extend([stratum == 0] * copies)
            clouds[firm, 2018 + year] = {
                "points": np.asarray(coordinates),
                "dates": np.asarray(article_dates),
                "core": np.asarray(flags),
            }
    return clouds, panels


def test_compressed_archive_matches_explicit_dated_weighted_corporate_clouds() -> None:
    """The retained corporate geometry and strict-tail adapter give the same arrays."""
    design, counts = _fixture()
    joint = outer_archive(design["laws"], counts, 1, 2)[0:1]
    pooled, dates = archive_panel(joint)
    financial, approach = panel_arrays(pooled[0], dates, design["news"])
    clouds, panels = _explicit_clouds(joint, design["news"])
    masses = point_masses(clouds, "expanded")
    explicit_financial = return_readout(clouds, panels, masses, 4)
    np.testing.assert_allclose(financial, explicit_financial, atol=1e-14)
    distance = np.full((5, 4, 4), np.nan)
    for year in range(5):
        for target in range(4):
            for reference in range(4):
                if target == reference:
                    continue
                left, right = (
                    clouds[target, 2018 + year]["points"],
                    clouds[reference, 2018]["points"],
                )
                cost = ((left[:, None] - right[None]) ** 2).sum(axis=-1)
                distance[year, target, reference] = exact_transport(
                    cost, masses[target, 2018 + year], masses[reference, 2018]
                )
    offdiag = ~np.eye(4, dtype=bool)
    np.testing.assert_allclose(
        approach[:, offdiag], (distance[:-1] - distance[1:])[:, offdiag], atol=1e-12
    )


def test_zero_population_law_can_have_finite_article_movement() -> None:
    """The stationary control retains a financial denominator and finite noise."""
    design, counts = _fixture()
    assert design["target_beta"][1] == pytest.approx(0, abs=1e-12)
    assert design["target_d"][1] > 0
    inner, rows = measure_outer(design, counts, 0, 0, inner_redraws=7)
    assert not np.allclose(inner["beta"][1], 0, atol=1e-12)
    assert all(row["inner_attempts"] == len(inner["beta"][0]) for row in rows)
    assert all(
        row["oracle_beta"] == design["target_beta"][i] for i, row in enumerate(rows)
    )


def test_batched_measurement_matches_scalar_centered_estimator() -> None:
    """Execution batching leaves each history/panel/regime ratio unchanged."""
    design, counts = _fixture()
    archive = outer_archive(design["laws"], counts, 4, 1)
    panels = [
        archive_panel(bootstrap_archive(archive, 4, 1, inner)) for inner in range(3)
    ]
    laws = np.stack([panel[0] for panel in panels])
    dates = np.stack([panel[1] for panel in panels])
    batched = measure_regimes_batch(laws, dates, design["news"])
    for index, (law, months) in enumerate(panels):
        scalar = measure_regimes(law, months, design["news"])
        for key in scalar:
            np.testing.assert_allclose(
                batched[key][index], scalar[key], rtol=1e-12, atol=1e-12
            )


def test_one_failed_inner_retains_other_results_and_execution_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch retry isolates one failing inner without losing valid replicates."""
    design, counts = _fixture()
    original = bootstrap_archive

    def failing(
        archive: np.ndarray,
        history: int,
        outer: int,
        inner: int,
        *,
        pilot: bool = False,
    ) -> np.ndarray:
        if inner == 1:
            msg = "injected numerical failure"
            raise ArithmeticError(msg)
        return original(archive, history, outer, inner, pilot=pilot)

    monkeypatch.setattr(
        "pipeline._kernels.paper6.article_redraw_validation.bootstrap_archive", failing
    )
    arrays, rows = measure_outer(design, counts, 0, 0, inner_redraws=3)
    np.testing.assert_array_equal(
        arrays["execution_succeeded"], [[1, 0, 1]] * len(SCENARIOS)
    )
    assert all(
        row["inner_execution_failed"] == 1 and row["inner_undefined"] == 1
        for row in rows
    )
    assert all(row["quantile_status"] == "conditional_on_defined" for row in rows)


def test_conditional_quantiles_preserve_undefined_counts_and_empty_reason() -> None:
    """Undefined estimates cannot become zero uncertainty or disappear from counts."""
    result = sensitivity_quantiles(
        np.asarray([-2.0, 1.0, 9.0]), np.asarray([1.0, 1.0, 0.0])
    )
    assert result["article_sensitivity_q05"] == pytest.approx(
        np.quantile([-2, 1], 0.05, method="linear")
    )
    assert result["quantile_status"] == "conditional_on_defined"
    assert result["inner_undefined"] == 1
    empty = sensitivity_quantiles(np.zeros(3), np.zeros(3))
    assert empty["article_sensitivity_q05"] is None
    assert empty["quantile_reason"] == "no_defined_inner_estimates"


def _rows() -> list[dict[str, Any]]:
    return [
        {
            "scenario_id": scenario,
            "history": history,
            "outer_replicate": outer,
            "execution_succeeded": True,
            "premises_held": True,
            "oracle_defined": True,
            "oracle_beta": 0.0,
            "article_sensitivity_q025": -1.0,
            "article_sensitivity_q975": 2.0,
            "article_sensitivity_q05": -0.1 if outer % 2 else 0.1,
            "inner_undefined": 0,
        }
        for scenario in SCENARIOS
        for history in range(HISTORIES)
        for outer in range(4)
    ]


def test_mcse_uses_outer_replications_within_fixed_history_strata() -> None:
    """Inner samples and dyads contribute no independent-replication count."""
    result = summarize_outer(_rows(), 4)
    rate = result["scenarios"][0]["positive_q05"]
    assert rate["equal_history_rate"] == pytest.approx(0.5)
    assert rate["mcse"] == pytest.approx(np.sqrt((1 / 3) / (HISTORIES * 4)))
    assert rate["attempts"] == HISTORIES * 4


def test_missing_oracle_withholds_aggregate_but_one_failed_outer_does_not() -> None:
    """Unknown fixed targets differ from unavailable bounds for a known history."""
    rows = _rows()
    rows[0].update(
        execution_succeeded=False,
        oracle_defined=False,
        oracle_beta=None,
        article_sensitivity_q05=None,
    )
    result = summarize_outer(rows, 4)["scenarios"][0]
    assert result["central_containment"]["equal_history_rate"] == pytest.approx(63 / 64)
    for row in rows:
        if row["scenario_id"] == "A" and row["history"] == 0:
            row.update(oracle_defined=False, oracle_beta=None)
    result = summarize_outer(rows, 4)["scenarios"][0]
    assert result["central_containment"]["equal_history_rate"] is None
    assert result["positive_q05"]["equal_history_rate"] is not None
    assert (
        result["central_containment"]["unavailable_reason"]
        == "missing_declared_history_oracle"
    )


def test_outer_checkpoint_integrity_and_replay_are_deterministic(
    tmp_path: Path,
) -> None:
    """Resume readers verify complete scalar arrays and their outer membership."""
    design, counts = _fixture()
    history_path = tmp_path / "histories/history-00.npz"
    write_arrays(history_path, design)
    write(history_path.with_suffix(".json"), {"history": 0, "premises_held": True})
    first = _outer_task(tmp_path, 0, 1, counts, 3, pilot=False)
    second = _outer_task(tmp_path, 0, 1, counts, 3, pilot=False)
    for key in first[0]:
        np.testing.assert_array_equal(first[0][key], second[0][key])
    manifest: dict[str, Any] = {"completed": {}}
    _checkpoint(tmp_path, manifest, 0, 1, first, pilot=False)
    key = next(iter(manifest["completed"]))
    records = _load_checkpoint(tmp_path, key, manifest["completed"][key], 3)
    assert len(records) == len(SCENARIOS)
    path = tmp_path / key
    stored = read_arrays(path)
    stored["beta"][0, 0] += 1
    write_arrays(path, stored)
    with pytest.raises(ValueError, match="integrity mismatch"):
        _load_checkpoint(tmp_path, key, manifest["completed"][key], 3)


def test_failed_outer_keeps_known_economic_target() -> None:
    """An execution failure does not erase a population target already computed."""
    design, _ = _fixture()
    arrays, rows = _failed_outer(0, 1, 3, "failed", design=design)
    assert not arrays["defined"].any()
    assert all(row["oracle_defined"] for row in rows)
    assert not any(row["execution_succeeded"] for row in rows)


def test_compressed_inner_law_equals_enumerated_article_index_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All ordered three-core/two-added article redraws give the same joint law."""
    # Three distinct core articles occupy (month,atom) cells (0,0),(0,1),(1,0).
    # Two additional articles occupy (0,1),(1,1). Enumerate their independent
    # uniform article-index draws directly: 3**3 * 2**2 equiprobable outcomes.
    outcomes: Counter[tuple[int, ...]] = Counter()
    for core in product(range(3), repeat=3):
        for added in product(range(2), repeat=2):
            outcomes[
                tuple(np.bincount(core, minlength=3))
                + tuple(np.bincount(added, minlength=2))
            ] += 1
    archive = np.zeros((1, 2, 5, 12, 1, 2), dtype=np.int64)
    archive[0, 0, 0, 0, 0] = [1, 1]
    archive[0, 0, 0, 1, 0, 0] = 1
    archive[0, 0, 1:, 0, 0, 0] = 1
    archive[0, 1, 0, :2, 0, 1] = 1
    probabilities = []
    date_role = 2
    for outcome, frequency in outcomes.items():
        desired = np.zeros_like(archive)
        desired[0, 0, 0, 0, 0] = outcome[:2]
        desired[0, 0, 0, 1, 0, 0] = outcome[2]
        desired[0, 0, 1:, 0, 0, 0] = 1
        desired[0, 1, 0, :2, 0, 1] = outcome[3:]
        probability = [1.0]

        class EnumeratedDraw:
            """Evaluate the PMF actually passed to a scripted multinomial draw."""

            def __init__(self, sample: np.ndarray, accumulator: list[float]) -> None:
                self.sample = sample
                self.accumulator = accumulator

            def multinomial(self, n: np.ndarray, p: np.ndarray) -> np.ndarray:
                np.testing.assert_array_equal(self.sample.sum(axis=-1), n)
                for total, weights, counts in zip(
                    np.ravel(n),
                    p.reshape(-1, p.shape[-1]),
                    self.sample.reshape(-1, p.shape[-1]),
                    strict=True,
                ):
                    coefficient = factorial(int(total))
                    for count in counts:
                        coefficient /= factorial(int(count))
                    self.accumulator[0] *= coefficient * float(np.prod(weights**counts))
                return self.sample.copy()

        def enumerated_rng(
            _history: int,
            _outer: int,
            role: int,
            stratum: int,
            *,
            _desired: np.ndarray = desired,
            _probability: list[float] = probability,
            **_: object,
        ) -> EnumeratedDraw:
            joint = _desired[0, stratum]
            sample = (
                np.moveaxis(joint.sum(axis=-1), 1, -1) if role == date_role else joint
            )
            return EnumeratedDraw(sample, _probability)

        monkeypatch.setattr(
            "pipeline._kernels.paper6.article_redraw_validation._rng", enumerated_rng
        )
        np.testing.assert_array_equal(bootstrap_archive(archive, 0, 0, 0), desired)
        assert probability[0] == pytest.approx(
            frequency / sum(outcomes.values()), abs=1e-15
        )
        probabilities.append(probability[0])
    assert sum(probabilities) == pytest.approx(1, abs=1e-14)


def test_manifest_rejects_changed_code_design_inputs_and_verifies_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resume accepts identical provenance and refuses changed or damaged evidence."""
    source = tmp_path / "source-design.json"
    write(source, {"design": "frozen"})
    original_design = source.read_bytes()
    output = tmp_path / "analysis"
    output.mkdir()
    code = {"identity": "original-code"}
    monkeypatch.setattr(
        "pipeline.stages.papers.paper6.article_redraw_validation.entrypoint_code_identity",
        lambda _command: code["identity"],
    )
    config = ArticleRedrawValidationConfig()
    inputs: dict[str, object] = {"hashes": {"support.npz": "original-input"}}
    saved = _manifest(output, source, config, inputs)
    assert _manifest(output, source, config, inputs) == saved
    code["identity"] = "changed-code"
    with pytest.raises(ValueError, match="incompatible design, code or inputs"):
        _manifest(output, source, config, inputs)
    code["identity"] = "original-code"
    with pytest.raises(ValueError, match="incompatible design, code or inputs"):
        _manifest(output, source, config, {"hashes": {"support.npz": "changed-input"}})
    write(source, {"design": "changed"})
    with pytest.raises(ValueError, match="incompatible design, code or inputs"):
        _manifest(output, source, config, inputs)
    source.write_bytes(original_design)
    write(output / "design.json", {"design": "corrupted-snapshot"})
    with pytest.raises(ValueError, match="saved output hash mismatch"):
        _manifest(output, source, config, inputs)
    (output / "design.json").write_bytes(original_design)
    assert _manifest(output, source, config, inputs) == saved
