"""Tests for the point-in-time representation sensitivity exhibit."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from pipeline.figures.paper5.point_in_time import (
    Paper5PointInTimeTableError,
    render_paper5_point_in_time_table,
)
from pipeline.stages.papers.paper5 import point_in_time as stage
from pipeline.stages.papers.paper5.point_in_time import _encoder_training_cutoff

if TYPE_CHECKING:
    from pathlib import Path


def _write_fixture(directory: Path, *, provider_id: str = "chronobert-2017") -> Path:
    """Write a minimal point-in-time result payload."""
    path = directory / "results.json"
    path.write_text(
        json.dumps(
            {
                "identity": {
                    "provider_id": provider_id,
                    "representation_id": "chronobert-2017-unit",
                    "factor_model": "ff5_mom",
                },
                "design": {
                    "encoder_training_cutoff": "2017-12-31",
                    "article_window_start": "2018-01-01",
                    "calibration_start": "2018-01-01",
                    "representation_is_pre_period": True,
                },
                "model_rows": [
                    {
                        "model": "factor_only",
                        "label": "Factor-only",
                        "rho_hat": 0.0,
                        "held_out_score": 0.1,
                        "held_out_score_gain_vs_factor_only": 0.0,
                        "held_out_gain_ci_lower": 0.0,
                        "held_out_gain_ci_upper": 0.0,
                    },
                    {
                        "model": "w2",
                        "label": "ChronoBERT 2017 W2 barycentric",
                        "rho_hat": 0.2,
                        "held_out_score": 0.2,
                        "held_out_score_gain_vs_factor_only": 0.1,
                        "held_out_gain_ci_lower": 0.05,
                        "held_out_gain_ci_upper": 0.15,
                    },
                    {
                        "model": "rbf_median",
                        "label": "ChronoBERT 2017 median-bandwidth W2 RBF",
                        "rho_hat": 0.1,
                        "held_out_score": 0.15,
                        "held_out_score_gain_vs_factor_only": 0.05,
                        "held_out_gain_ci_lower": 0.01,
                        "held_out_gain_ci_upper": 0.09,
                    },
                ],
                "paired_contrasts": {
                    "w2_minus_rbf": {
                        "estimate": 0.05,
                        "ci_lower": 0.01,
                        "ci_upper": 0.09,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_renders_point_in_time_table(tmp_path: Path) -> None:
    """The table identifies the pre-period encoder and paired contrast."""
    output_path = tmp_path / "point_in_time_encoder.tex"

    render_paper5_point_in_time_table(_write_fixture(tmp_path), output_path)

    rendered = output_path.read_text(encoding="utf-8")
    assert "ChronoBERT 2017" in rendered
    assert "W2 minus median RBF" in rendered
    assert "pre-period" in rendered
    assert "\\label{tab:p5-point-in-time-encoder}" in rendered


def test_rejects_post_period_encoder(tmp_path: Path) -> None:
    """The exhibit cannot silently relabel a post-period representation."""
    with pytest.raises(Paper5PointInTimeTableError, match="pre-period"):
        render_paper5_point_in_time_table(
            _write_fixture(tmp_path, provider_id="qwen3-embedding-8b"),
            tmp_path / "point_in_time_encoder.tex",
        )


def _write_params(
    directory: Path, *, cutoff: str = "2017-12-31", article_start: str = "2018-01-01"
) -> Path:
    """Write real YAML-compatible configuration for cutoff admission."""
    model_id = "manelalab/chrono-bert-v1-20171231"
    path = directory / "params.yaml"
    path.write_text(
        json.dumps(
            {
                "start_date": article_start,
                "chronobert": {
                    "models": {
                        "chronobert-2017": {
                            "id": model_id,
                            "revision": "02bde7f9d3d8c55c1e44d7e3ee071f374a1b9bb4",
                            "training_cutoff": cutoff,
                        }
                    }
                },
                "typed_analysis": {
                    "providers": {"chronobert-2017": {"model_id": model_id}}
                },
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("cutoff", "article_start", "calibration_start"),
    [
        ("2018-12-31", "2018-01-01", "2018-01-01"),
        ("2024-12-31", "2018-01-01", "2018-01-01"),
        ("2017-12-31", "2017-12-31", "2018-01-01"),
        ("2017-12-31", "2018-01-01", "2017-12-31"),
    ],
)
def test_cutoff_must_precede_both_windows_before_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cutoff: str,
    article_start: str,
    calibration_start: str,
) -> None:
    """Later/equal cutoffs cannot load operators, fit, or publish results."""
    params_file = _write_params(tmp_path, cutoff=cutoff, article_start=article_start)
    output_dir = tmp_path / "chronobert_2017"
    paths = stage.Paper5PointInTimePaths(
        returns_dir=tmp_path / "returns",
        oos_returns_dir=tmp_path / "oos_returns",
        w2_barycentre_dir=tmp_path / "barycentres",
        distance_artifact_dir=tmp_path / "distances",
        params_file=params_file,
        output_dir=output_dir,
    )

    def forbidden(*_args: object, **_kwargs: object) -> None:
        message = "invalid cutoff reached operator loading or fitting"
        raise AssertionError(message)

    monkeypatch.setattr(stage, "_load_w2_operator", forbidden)
    monkeypatch.setattr(stage, "_fit", forbidden)
    monkeypatch.setattr(stage, "_factor_only_fit", forbidden)
    with pytest.raises(stage.Paper5PointInTimeError, match="must precede"):
        stage.run_paper5_point_in_time(
            paths, stage.Paper5PointInTimeConfig(calibration_start=calibration_start)
        )
    assert not output_dir.exists()


def test_pinned_encoder_cutoff_agrees_with_typed_provider(tmp_path: Path) -> None:
    """Admission resolves genuine identity and rejects a mismatched typed model."""
    params_file = _write_params(tmp_path)
    cutoff, article_start, calibration_start = _encoder_training_cutoff(
        params_file, stage.Paper5PointInTimeConfig()
    )
    assert cutoff.isoformat() == "2017-12-31"
    assert cutoff < article_start
    assert cutoff < calibration_start
    params = json.loads(params_file.read_text(encoding="utf-8"))
    params["typed_analysis"]["providers"]["chronobert-2017"]["model_id"] = (
        "manelalab/chrono-bert-v1-20181231"
    )
    params_file.write_text(json.dumps(params), encoding="utf-8")
    with pytest.raises(stage.Paper5PointInTimeError, match="agree"):
        _encoder_training_cutoff(params_file, stage.Paper5PointInTimeConfig())


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("encoder_training_cutoff", "2018-12-31"),
        ("encoder_training_cutoff", "not-a-date"),
        ("article_window_start", "2017-12-31"),
        ("calibration_start", "2017-12-31"),
        ("representation_is_pre_period", False),
    ],
)
def test_renderer_rejects_untruthful_pre_period_metadata(
    tmp_path: Path, key: str, value: object
) -> None:
    """A valid label cannot launder a later cutoff or unsupported pre-period flag."""
    results_path = _write_fixture(tmp_path)
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    payload["design"][key] = value
    results_path.write_text(json.dumps(payload), encoding="utf-8")
    output_path = tmp_path / "point_in_time_encoder.tex"
    with pytest.raises(Paper5PointInTimeTableError, match="pre-period"):
        render_paper5_point_in_time_table(results_path, output_path)
    assert not output_path.exists()


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("training_cutoff", "not-a-date", "parseable"),
        ("training_cutoff", None, "parseable"),
        ("revision", "main", "immutable"),
    ],
)
def test_rejects_unpinned_or_unparseable_encoder_config(
    tmp_path: Path, key: str, value: str | None, message: str
) -> None:
    """A claimed pre-period encoder needs an actual pinned, dated config."""
    params_file = _write_params(tmp_path)
    params = json.loads(params_file.read_text(encoding="utf-8"))
    params["chronobert"]["models"]["chronobert-2017"][key] = value
    params_file.write_text(json.dumps(params), encoding="utf-8")
    with pytest.raises(stage.Paper5PointInTimeError, match=message):
        _encoder_training_cutoff(params_file, stage.Paper5PointInTimeConfig())
