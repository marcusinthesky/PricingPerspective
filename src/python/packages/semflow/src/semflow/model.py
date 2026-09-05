"""Immutable public models for semantic fingerprints and patch plans."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import unified_diff
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@dataclass(frozen=True)
class HashPolicy:
    """Normalization choices that define a semantic-hash contract."""

    ignore_docstrings: bool = True
    ignore_annotations: bool = False
    ignore_type_checking_blocks: bool = True

    def to_dict(self) -> dict[str, bool]:
        """Return a JSON-ready representation with stable field names."""
        return asdict(self)


@dataclass(frozen=True)
class MemberFingerprint:
    """Semantic digest for one Python file within a source dependency."""

    path: str
    digest: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-ready representation."""
        return asdict(self)


@dataclass(frozen=True)
class Fingerprint:
    """Canonical semantic fingerprint for a file or Python package directory."""

    schema_version: int
    source: str
    kind: Literal["file", "directory"]
    algorithm: Literal["sha256"]
    policy: HashPolicy
    digest: str
    members: tuple[MemberFingerprint, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON document shape."""
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "kind": self.kind,
            "algorithm": self.algorithm,
            "policy": self.policy.to_dict(),
            "digest": self.digest,
            "members": [member.to_dict() for member in self.members],
        }


@dataclass(frozen=True)
class TrackedMember:
    """One reachable symbol or conservative module fallback."""

    path: str
    symbol: str
    kind: Literal["function", "class", "global", "binding", "module"]
    digest: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-ready representation."""
        return asdict(self)


@dataclass(frozen=True)
class AnalysisFallback:
    """Explain why one module was hashed as a whole."""

    module: str
    path: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-ready representation."""
        return asdict(self)


@dataclass(frozen=True)
class EntrypointFingerprint:
    """Semantic fingerprint of one command callback's reachable code."""

    schema_version: int
    entrypoint: str
    callback: str
    algorithm: Literal["sha256"]
    policy: HashPolicy
    digest: str
    members: tuple[TrackedMember, ...]
    fallbacks: tuple[AnalysisFallback, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON document shape."""
        return {
            "schema_version": self.schema_version,
            "entrypoint": self.entrypoint,
            "callback": self.callback,
            "algorithm": self.algorithm,
            "policy": self.policy.to_dict(),
            "digest": self.digest,
            "members": [member.to_dict() for member in self.members],
            "fallbacks": [fallback.to_dict() for fallback in self.fallbacks],
        }


@dataclass(frozen=True)
class FileChange:
    """One proposed file creation, update, or deletion."""

    path: Path
    before: str | None
    after: str | None

    @property
    def operation(self) -> Literal["create", "update", "delete"]:
        """Classify the change for diagnostics."""
        if self.before is None:
            return "create"
        if self.after is None:
            return "delete"
        return "update"

    def unified_diff(self, *, root: Path) -> str:
        """Render this change as a standard unified diff."""
        relative = self.path.relative_to(root).as_posix()
        before_name = "/dev/null" if self.before is None else f"a/{relative}"
        after_name = "/dev/null" if self.after is None else f"b/{relative}"
        return "".join(
            unified_diff(
                () if self.before is None else self.before.splitlines(keepends=True),
                () if self.after is None else self.after.splitlines(keepends=True),
                fromfile=before_name,
                tofile=after_name,
            )
        )


@dataclass(frozen=True)
class PatchPlan:
    """A deterministic, reviewable set of repository file changes."""

    root: Path
    pipeline_file: Path
    changes: tuple[FileChange, ...]

    @property
    def clean(self) -> bool:
        """Return whether the repository already matches the computed state."""
        return not self.changes

    def diff(self) -> str:
        """Render every proposed change in path order."""
        return "".join(change.unified_diff(root=self.root) for change in self.changes)

    def __iter__(self) -> Iterator[FileChange]:
        """Iterate over proposed changes."""
        return iter(self.changes)
