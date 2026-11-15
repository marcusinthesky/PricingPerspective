"""The retained runner cannot mutate the archived scientific campaign."""

from pathlib import Path

from pipeline.cli.commands.p6.recovery import commands
from pipeline.io.paths import PAPERS
from typer.testing import CliRunner


def test_legacy_runner_requires_explicit_destination() -> None:
    """An omitted destination cannot select historical data implicitly."""
    result = CliRunner().invoke(commands, ["paper6-information-recovery"])
    assert result.exit_code != 0
    assert "--output" in result.output


def test_legacy_runner_rejects_archive_and_descendants() -> None:
    """Neither the archive nor a nested destination admits execution."""
    archive = PAPERS / "paper6/information_recovery"
    for destination in (archive, archive / "campaign", archive / "new"):
        result = CliRunner().invoke(
            commands, ["paper6-information-recovery", "--output", str(destination)]
        )
        assert result.exit_code != 0
        assert "immutable" in result.output


def test_archive_guard_resolves_relative_paths() -> None:
    """Normalized aliases cannot bypass the immutable namespace guard."""
    destination = Path("data/papers/paper6/information_recovery/campaign")
    # The test recipe runs in the app, so use an absolute normalized alias.
    archive = PAPERS / "paper6/information_recovery"
    destination = archive / ".." / "information_recovery" / destination.name
    result = CliRunner().invoke(
        commands, ["paper6-information-recovery", "--output", str(destination)]
    )
    assert result.exit_code != 0
    assert "immutable" in result.output
