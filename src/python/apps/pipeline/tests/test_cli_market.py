"""Semantic Typer validation for substrate market commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pipeline.cli import app
from typer.testing import CliRunner

if TYPE_CHECKING:
    from pathlib import Path


runner = CliRunner()
CLI_USAGE_ERROR = 2
CONTRACT_OPTIONS = (
    "--min-observations",
    "100",
    "--expected-tickers",
    "100",
    "--expected-observations",
    "1207",
    "--expected-start-date",
    "2018-03-19",
    "--expected-end-date",
    "2022-12-30",
)


def test_compute_covariance_rejects_invalid_path_role_and_count(tmp_path: Path) -> None:
    """Reject a file where a directory is required and a nonpositive count."""
    returns_file = tmp_path / "returns.parquet"
    returns_file.touch()
    output_file = tmp_path / "covariance.parquet"

    wrong_role = runner.invoke(
        app,
        [
            "compute-covariance",
            "--returns-dir",
            str(returns_file),
            "--output-file",
            str(output_file),
            *CONTRACT_OPTIONS,
        ],
    )
    assert wrong_role.exit_code == CLI_USAGE_ERROR
    assert "directory" in wrong_role.output.lower()

    returns_dir = tmp_path / "returns"
    returns_dir.mkdir()
    invalid_count = runner.invoke(
        app,
        [
            "compute-covariance",
            "--returns-dir",
            str(returns_dir),
            "--output-file",
            str(output_file),
            *CONTRACT_OPTIONS[2:],
            "--min-observations",
            "0",
        ],
    )
    assert invalid_count.exit_code == CLI_USAGE_ERROR
    assert invalid_count.exception is not None
