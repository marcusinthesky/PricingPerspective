"""Atomic numerical artifacts preserve completed results on failure."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from hotelling.dynamics.artifacts import read_arrays, write_arrays

if TYPE_CHECKING:
    from pathlib import Path


def test_nonfinite_array_write_preserves_previous_artifact(tmp_path: Path) -> None:
    """Invalid numerical results cannot replace an already completed artifact."""
    target = tmp_path / "result.npz"
    write_arrays(target, {"value": np.array([1.0])})
    before = target.read_bytes()
    with pytest.raises(ArithmeticError, match="must be finite"):
        write_arrays(target, {"value": np.array([float("nan")])})
    assert target.read_bytes() == before
    assert not list(tmp_path.glob(".partial-*"))


@pytest.mark.parametrize("linked_parent", [False, True])
def test_array_write_rejects_symlink_destination(
    tmp_path: Path, *, linked_parent: bool
) -> None:
    """Reject both leaf and parent links before changing an artifact."""
    actual = tmp_path / "actual"
    actual.mkdir()
    target = actual / "result.npz"
    write_arrays(target, {"value": np.array([1.0])})
    before = target.read_bytes()
    link = tmp_path / "link"
    link.symlink_to(actual if linked_parent else target)
    with pytest.raises(ValueError, match="symlinked artifact destination"):
        write_arrays(
            link / "result.npz" if linked_parent else link,
            {"value": np.array([2.0])},
        )
    assert target.read_bytes() == before


def test_failed_array_write_preserves_previous_complete_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A serialization failure cannot replace an existing complete artifact."""
    destination = tmp_path / "complete.npz"
    expected = np.arange(4, dtype=np.float64)
    write_arrays(destination, {"capacity": expected})

    def fail_write(*_args: object, **_kwargs: object) -> None:
        msg = "simulated interrupted serialization"
        raise OSError(msg)

    monkeypatch.setattr(np, "savez_compressed", fail_write)
    with pytest.raises(OSError, match="interrupted"):
        write_arrays(destination, {"capacity": expected + 1})
    np.testing.assert_array_equal(read_arrays(destination)["capacity"], expected)
    assert not list(tmp_path.glob(".partial-*"))
