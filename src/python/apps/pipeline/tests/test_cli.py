"""Smoke tests for the pipeline package."""

from __future__ import annotations

import shlex
from importlib.metadata import version
from pathlib import Path

import pipeline
import typer
import yaml
from pipeline.cli import app
from typer._click.exceptions import UsageError
from typer.core import TyperGroup
from typer.main import get_command

EXPECTED_FLAT_COMMANDS = {
    "compute-covariance",
    "compute-co-mentions",
    "validate-public-inputs",
    "paper1-pcoa",
    "compute-sample-descriptives",
    "paper1-diagnostics",
    "paper1-w2-permanova",
    "paper1-dyadic-confound",
    "paper1-dyadic-oos",
    "paper1-etnic-alignment",
    "paper1-neighbour-outcome-split",
    "paper1-representation-sensitivity",
    "paper1-pretraining-distances",
    "paper1-tfidf-controls",
    "paper1-text-controls",
    "paper1-wasserstein-comparator",
    "paper3-certificate",
    "paper3-certificate-validation",
    "paper3-pit-w2",
    "paper3-portfolio-anatomy",
    "paper3-representation-ablation",
    "paper3-representation-ablation-cells",
    "paper5-economic-value",
    "paper5-gmv-validation",
    "paper5-network-certificate",
    "paper5-peer-definitions",
    "paper5-point-in-time-encoder",
    "paper5-post-corpus-qmle",
    "paper5-representation-ablation",
    "paper5-spatial-covariance",
    "paper5-spatial-value-comparison",
    "paper5-transport-dimension-ladder",
    "paper5-transport-order-sensitivity",
    "paper5-two-field-qmle",
    "paper5-universe-horizon-bridge",
    "paper6-information-recovery",
    "paper6-information-recovery-audit",
    "paper6-directional-validation",
    "paper6-corporate-analysis",
    "paper6-article-redraw-validation",
    "paper6-inference-audit",
    "render-paper1-diagnostics",
    "render-paper1-w2-permanova",
    "render-paper1-dyadic-confound",
    "render-paper1-illustrations",
    "render-paper1-pcoa",
    "render-paper1-neighbour-outcome-split",
    "render-paper1-numbers",
    "render-paper1-representation-sensitivity",
    "render-paper1-text-controls",
    "render-paper1-sample-descriptives",
    "render-paper3-certificate",
    "render-paper3-illustrations",
    "render-paper3-numbers",
    "render-paper3-portfolio-anatomy",
    "render-paper3-representation-ablation",
    "render-paper5-barycentric-mixture",
    "render-paper5-economic-value",
    "render-paper5-gmv-validation",
    "render-paper5-illustrations",
    "render-paper5-numbers",
    "render-paper5-peer-definitions",
    "render-paper5-point-in-time-encoder",
    "render-paper5-representation-ablation",
    "render-paper5-sample-descriptives",
    "render-paper5-transport-dimension-ladder",
    "render-paper5-two-field-qmle",
    "render-paper5-universe-horizon-bridge",
    "render-paper6-information-recovery",
    "render-paper6-corporate-analysis",
    "render-paper6-information-audit",
    "render-paper6-article-redraw-validation",
    "render-paper6-inference-audit",
    "render-sample-universe-table",
    "typed-barycentre",
    "typed-distance",
    "typed-geometry-component",
    "validate-coverage",
}


def test_cli_imports() -> None:
    """Verify the complete experiment-oriented command surface is registered."""
    command = get_command(app)
    assert isinstance(command, TyperGroup)
    assert set(command.commands) == EXPECTED_FLAT_COMMANDS | {"semflow"}

    grouped_names = [group.name for group in app.registered_groups]
    assert "semflow" in grouped_names


def test_version() -> None:
    """Verify package version is set."""
    assert pipeline.__version__ == version("pipeline")


def test_dvc_pipeline_commands_match_the_registered_cli() -> None:
    """Parse command templates without stage execution or a shared DVC lock."""
    root = next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "dvc.yaml").is_file()
    )
    document = yaml.safe_load((root / "dvc.yaml").read_text(encoding="utf-8"))
    command = get_command(app)
    assert isinstance(command, TyperGroup)
    violations: set[str] = set()
    for stage_name, stage in document["stages"].items():
        body = stage.get("do", stage)
        tokens = shlex.split(body["cmd"])
        if "${py_pipeline}" not in tokens:
            continue
        invocation = tokens[tokens.index("${py_pipeline}") + 1 :]
        for separator in ("&&", "||", ";"):
            if separator in invocation:
                invocation = invocation[: invocation.index(separator)]
        command_name, *arguments = invocation
        if command_name not in command.commands:
            violations.add(f"{stage_name}: unregistered command {command_name}")
            continue
        child = command.commands[command_name]
        context = typer.Context(child, info_name=command_name)
        try:
            values, remaining, _order = child.make_parser(context).parse_args(arguments)
        except UsageError as exc:
            violations.add(f"{stage_name}: {exc}")
            continue
        if remaining:
            violations.add(f"{stage_name}: unexpected arguments {remaining}")
        missing = [
            parameter.name
            for parameter in child.params
            if parameter.required and parameter.name not in values
        ]
        if missing:
            violations.add(f"{stage_name}: missing required options {missing}")
    assert not violations, "\n".join(sorted(violations))
