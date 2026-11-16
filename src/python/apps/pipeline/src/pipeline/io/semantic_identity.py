"""Read or compute numerical identities using the shared Semflow policy."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class _EntrypointIdentity(BaseModel):
    """The identity fields of a generated Semflow entrypoint document."""

    algorithm: Literal["sha256"]
    entrypoint: str
    digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


def entrypoint_code_identity(command: str) -> str:
    """Return a generated pipeline command's digest without its algorithm prefix.

    Run from the repository root, after ``semflow refresh --apply``. Missing or
    invalid fingerprints fail rather than substituting a different hash policy.
    """
    path = Path(".semflow/hashes/entrypoints/pipeline") / f"{command}.json"
    identity = _EntrypointIdentity.model_validate_json(path.read_text(encoding="utf-8"))
    if identity.entrypoint != f"pipeline {command}":
        message = f"Semflow entrypoint mismatch in {path}: {identity.entrypoint}"
        raise ValueError(message)
    return identity.digest.removeprefix(f"{identity.algorithm}:")


def standalone_entrypoint_code_identity(command: str) -> str:
    """Fingerprint a retained command outside DVC with the same semantic engine.

    Semflow refresh retains generated fingerprints only for DVC commands. A
    standalone runner computes its reachable closure directly without creating
    another generated fingerprint or requiring a producer stage.
    """
    # DVC readers do not need to load the analysis engine or CLI discovery.
    from semflow.analysis import ModuleIndex, fingerprint_entrypoint  # noqa: PLC0415
    from semflow.config import load_config  # noqa: PLC0415
    from semflow.discovery import discover_command  # noqa: PLC0415
    from semflow.tracking import tracking_spec  # noqa: PLC0415

    config = load_config()
    target = discover_command("pipeline", (command,), overrides=config.applications)
    fingerprint = fingerprint_entrypoint(
        target,
        index=ModuleIndex(
            project_root=config.project_root, source_roots=config.source_roots
        ),
        policy=config.policy,
        track_spec=tracking_spec(target.callback),
    )
    return fingerprint.digest.removeprefix(f"{fingerprint.algorithm}:")
