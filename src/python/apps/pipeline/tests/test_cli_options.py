"""Validation and forwarding of the real market command's typed options."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pipeline.cli import app
from pipeline.stages.substrate import market
from typer.testing import CliRunner

if TYPE_CHECKING:
    from pathlib import Path

_USAGE_ERROR = 2
# Keep error text plain even when GitHub Actions forces Rich terminal styling.
runner = CliRunner(env={"TERM": "dumb"})


def _covariance_args(returns_dir: Path, output_file: Path) -> list[str]:
    return [
        "compute-covariance",
        "--returns-dir",
        str(returns_dir),
        "--output-file",
        str(output_file),
        "--min-observations",
        "3",
        "--expected-tickers",
        "2",
        "--expected-observations",
        "4",
        "--expected-start-date",
        "2020-01-01",
        "--expected-end-date",
        "2020-01-06",
    ]


def test_covariance_options_reach_the_stage_as_the_declared_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Forward validated user values without adapter defaults or model bridges."""
    observed: list[tuple[Path, Path, market.ReturnPanelContract]] = []

    def capture(
        returns_dir: Path, output_file: Path, contract: market.ReturnPanelContract
    ) -> None:
        observed.append((returns_dir, output_file, contract))

    monkeypatch.setattr(market, "compute_covariance", capture)
    output_file = tmp_path / "covariance.parquet"
    result = runner.invoke(app, _covariance_args(tmp_path, output_file))

    assert result.exit_code == 0, result.output
    assert observed == [
        (
            tmp_path,
            output_file,
            market.ReturnPanelContract(
                min_observations=3,
                expected_tickers=2,
                expected_observations=4,
                expected_start_date="2020-01-01",
                expected_end_date="2020-01-06",
            ),
        )
    ]


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--min-observations", "0"),
        ("--expected-tickers", "-1"),
        ("--expected-observations", "invalid"),
    ],
)
def test_covariance_rejects_invalid_counts_before_stage_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, option: str, value: str
) -> None:
    """Let native Typer validation enforce count bounds and integer types."""
    called = False

    def unexpected(*_args: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(market, "compute_covariance", unexpected)
    arguments = _covariance_args(tmp_path, tmp_path / "covariance.parquet")
    arguments[arguments.index(option) + 1] = value
    result = runner.invoke(app, arguments)

    assert result.exit_code == _USAGE_ERROR
    assert f"Invalid value for '{option}'" in result.output
    assert not called
