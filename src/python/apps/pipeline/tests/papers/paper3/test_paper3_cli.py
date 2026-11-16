"""Paper 3 command adapters: names, required options, and stage wiring."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pipeline.cli.commands.p3 import (
    certificate,
    certificate_validation,
    illustrations,
    numbers,
    pit_w2,
    portfolio_anatomy,
    representation_ablation,
)
from pipeline.stages.papers.paper3 import representation_ablation as stage
from typer.testing import CliRunner

if TYPE_CHECKING:
    from pathlib import Path

    import typer

_MODULES = (
    pit_w2,
    certificate,
    portfolio_anatomy,
    representation_ablation,
    certificate_validation,
    numbers,
    illustrations,
)
_EXPECTED = {
    "paper3-pit-w2",
    "paper3-certificate",
    "render-paper3-certificate",
    "paper3-portfolio-anatomy",
    "render-paper3-portfolio-anatomy",
    "paper3-representation-ablation-cells",
    "paper3-representation-ablation",
    "render-paper3-representation-ablation",
    "paper3-certificate-validation",
    "render-paper3-numbers",
    "render-paper3-illustrations",
}
_RUNNER = CliRunner()


def _names(app: typer.Typer) -> set[str]:
    return {str(command.name) for command in app.registered_commands}


def test_command_roster_is_one_module_per_experiment() -> None:
    """Every Paper 3 command is registered once under its convention name."""
    names = [name for module in _MODULES for name in _names(module.commands)]
    assert sorted(names) == sorted(_EXPECTED)


@pytest.mark.parametrize("module", _MODULES, ids=lambda module: module.__name__)
def test_no_option_has_a_data_default(module: object) -> None:
    """Invoking without options fails: paths and values are never defaulted."""
    app = module.commands  # type: ignore[attr-defined]
    for name in _names(app):
        args = [name] if len(_names(app)) > 1 else []
        result = _RUNNER.invoke(app, args)
        assert result.exit_code == 2, (name, result.output)  # noqa: PLR2004
        assert "Missing option" in result.output


def test_render_numbers_adapter_writes_numbers(tmp_path: Path) -> None:
    """The numbers adapter wires its folders through to the renderer."""
    folders = []
    for name in ("certificate", "validation", "anatomy"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "results.json").write_text("{}\n", encoding="utf-8")
        folders.append(folder)
    out = tmp_path / "numbers.tex"
    result = _RUNNER.invoke(
        numbers.commands,
        [
            "--numbers-path",
            str(out),
            "--certificate-dir",
            str(folders[0]),
            "--validation-dir",
            str(folders[1]),
            "--anatomy-dir",
            str(folders[2]),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.is_file()


def test_merge_adapter_passes_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The merge adapter maps every option onto the stage path record."""
    seen: list[object] = []
    monkeypatch.setattr(stage, "merge_paper3_representation_ablation", seen.append)
    params = tmp_path / "params.yaml"
    params.write_text("{}\n", encoding="utf-8")
    certificate_results = tmp_path / "results.json"
    certificate_results.write_text("{}\n", encoding="utf-8")
    result = _RUNNER.invoke(
        representation_ablation.commands,
        [
            "paper3-representation-ablation",
            "--params-file",
            str(params),
            "--cells-dir",
            str(tmp_path),
            "--canonical-certificate-summary",
            str(certificate_results),
            "--output-summary",
            str(tmp_path / "merged.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    (paths,) = seen
    assert paths == stage.Paper3RepresentationMergePaths(
        params_file=params,
        cells_dir=tmp_path,
        canonical_certificate_summary=certificate_results,
        output_summary=tmp_path / "merged.json",
        contrast_draws_dir=None,
    )
