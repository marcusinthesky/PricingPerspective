"""Admission and failed-publication safeguards for paired sector-test artifacts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pipeline.io.w2_permanova_artifacts import (
    read_w2_permanova_results,
    write_w2_permanova_results,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_archive_roundtrip_and_read_only_output_replacement(tmp_path: Path) -> None:
    """Keep the actual draws, return the digest, and safely replace DVC-like files."""
    summary = {"schema_version": 1, "n_firms": 3}
    draws = np.array([[0, 2, 1], [2, 1, 0]])
    result = write_w2_permanova_results(tmp_path, summary, {"schedule": draws})
    assert "audit_sha256" not in summary
    assert read_w2_permanova_results(tmp_path / "results.json") == result
    with np.load(tmp_path / "permutation_audit.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["schedule"], draws)
    (tmp_path / "results.json").chmod(0o444)
    (tmp_path / "permutation_audit.npz").chmod(0o444)
    updated = write_w2_permanova_results(
        tmp_path, summary, {"schedule": draws[:, ::-1]}
    )
    assert updated["audit_sha256"] != result["audit_sha256"]
    assert read_w2_permanova_results(tmp_path / "results.json") == updated


@pytest.mark.parametrize("version", [2, "1", True])
def test_reader_rejects_unsupported_result_schema(
    tmp_path: Path, version: object
) -> None:
    """An unsupported schema cannot inherit the current interpretation silently."""
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"schema_version": version}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        read_w2_permanova_results(path)


def test_missing_digest_rejects_old_unbound_summary(tmp_path: Path) -> None:
    """A previous summary without its archive identity is inadmissible."""
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="audit digest"):
        read_w2_permanova_results(path)


def test_failed_archive_write_invalidates_previous_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed replacement must leave no accepted result from the prior run."""
    summary = {"schema_version": 1}
    write_w2_permanova_results(tmp_path, summary, {"schedule": np.arange(3)})

    def fail_save(*_args: object, **_kwargs: object) -> None:
        message = "interrupted archive write"
        raise OSError(message)

    monkeypatch.setattr(np, "savez_compressed", fail_save)
    with pytest.raises(OSError, match="interrupted"):
        write_w2_permanova_results(tmp_path, summary, {"schedule": np.arange(3)})
    assert not (tmp_path / "results.json").exists()
    assert not list(tmp_path.glob("*.tmp"))
