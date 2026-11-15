"""Filesystem guards shared by numerical artifacts and read-only archive audits."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def reject_symlinks(path: Path) -> None:
    """Reject pre-existing symlink components, including dangling leaf links.

    The caller chooses the root explicitly; even that root must not be a link.
    This preflight protects ordinary local execution, not concurrent hostile
    directory replacement (which requires directory-relative no-follow I/O).
    """
    for component in (path, *path.parents):
        if component.is_symlink():
            msg = f"symlinked artifact destination: {component}"
            raise ValueError(msg)
