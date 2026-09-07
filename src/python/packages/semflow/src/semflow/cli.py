"""Command-line interface for semantic fingerprint checks and patches."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from semflow.api import check as check_project
from semflow.api import patch as patch_project
from semflow.api import refresh as refresh_project
from semflow.config import ConfigError
from semflow.dvc import DvcPatchError

if TYPE_CHECKING:
    from semflow.model import PatchPlan

app = typer.Typer(
    help="Semantic Python fingerprints for DVC dependencies.",
    no_args_is_help=True,
)

ConfigOption = Annotated[
    Path | None,
    typer.Option(
        "--config",
        help="semflow.toml or pyproject.toml containing [tool.semflow].",
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
]


@app.command("check")
def check_command(config: ConfigOption = None) -> None:
    """Fail when DVC instrumentation or semantic hash JSON is stale."""
    try:
        plan = check_project(config)
    except (ConfigError, DvcPatchError) as error:
        typer.echo(f"semflow: {error}", err=True)
        raise typer.Exit(code=2) from error
    if plan.clean:
        typer.echo("semflow: fingerprints are current.")
        return
    typer.echo(_summary(plan), err=True)
    typer.echo("Run `semflow patch` to review the required changes.", err=True)
    raise typer.Exit(code=1)


@app.command("patch")
def patch_command(
    config: ConfigOption = None,
    *,
    apply: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="Write the reviewed DVC YAML and fingerprint changes.",
        ),
    ] = False,
) -> None:
    """Preview a unified diff, or atomically apply it with --apply."""
    try:
        plan = patch_project(config, apply=apply)
    except (ConfigError, DvcPatchError) as error:
        typer.echo(f"semflow: {error}", err=True)
        raise typer.Exit(code=2) from error
    if plan.clean:
        typer.echo("semflow: no patch required.")
        return
    if apply:
        typer.echo(f"semflow: applied {len(plan.changes)} file change(s).")
        return
    typer.echo(plan.diff(), nl=False)


@app.command("refresh")
def refresh_command(
    config: ConfigOption = None,
    *,
    apply: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="Write only the reviewed fingerprint JSON changes.",
        ),
    ] = False,
) -> None:
    """Preview or apply fingerprint JSON changes without rewriting DVC YAML."""
    try:
        plan = refresh_project(config, apply=apply)
    except (ConfigError, DvcPatchError) as error:
        typer.echo(f"semflow: {error}", err=True)
        raise typer.Exit(code=2) from error
    if plan.clean:
        typer.echo("semflow: fingerprints are current.")
        return
    if apply:
        typer.echo(f"semflow: refreshed {len(plan.changes)} fingerprint file(s).")
        return
    typer.echo(plan.diff(), nl=False)


def _summary(plan: PatchPlan) -> str:
    changes = plan.changes
    counts = {
        operation: sum(change.operation == operation for change in changes)
        for operation in ("create", "update", "delete")
    }
    rendered = ", ".join(
        f"{count} {operation}" for operation, count in counts.items() if count
    )
    return f"semflow: patch required ({rendered})."


if __name__ == "__main__":
    app()
