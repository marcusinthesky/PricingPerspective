"""Declaration-extraction and provenance-classification tests."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import toolchain_provenance

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_settings_extracts_command_head_and_skips_non_bash(tmp_path: Path) -> None:
    """Extract only safe command heads from Bash permission declarations."""
    settings = tmp_path / ".agents" / "claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps(
            {
                "permissions": {
                    "allow": [
                        "Bash(gh pr create:*)",
                        "Bash(just:*)",
                        "Bash(rumdl check --no-config:*)",
                        "Read(//home/**)",
                        "Bash($(echo evil):*)",
                    ]
                }
            }
        )
    )
    assert [tool.name for tool in toolchain_provenance.settings_tools(tmp_path)] == [
        "gh",
        "just",
        "rumdl",
    ]


def test_system_hook_entries_reads_folded_block_scalars() -> None:
    """Read both inline and block-scalar system hook commands."""
    # The root config declares a hook this way; a naive scanner reads `>-`
    # itself as the binary name and reports a bogus missing tool.
    document = """
repos:
  - repo: local
    hooks:
      - id: plain
        entry: tombi format
        language: system
      - id: folded
        entry: >-
          bash -c 'exit 0' --
        language: system
      - id: not-system
        entry: never-extracted
        language: python
"""
    assert list(toolchain_provenance.system_hook_entries(document)) == [
        "tombi format",
        "bash -c 'exit 0' --",
    ]


def test_classification_separates_store_from_ambient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classify Nix-store, ambient, and unresolved command paths."""
    resolved = {
        "shipped": "/nix/store/abc-rumdl-0.2.31/bin/shipped",
        "leaked": "/etc/profiles/per-user/someone/bin/leaked",
    }
    monkeypatch.setattr(toolchain_provenance.shutil, "which", resolved.get)
    assert toolchain_provenance.classify("shipped")[0] == "store"
    assert toolchain_provenance.classify("leaked")[0] == "ambient"
    assert toolchain_provenance.classify("absent") == ("missing", "-")


def test_only_non_store_and_non_exempt_tools_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject ambient and missing tools while retaining declared sources."""
    resolved = {
        "shipped": "/nix/store/abc-x/bin/shipped",
        "leaked": "/usr/bin/leaked",
        "nix": "/run/current-system/sw/bin/nix",
    }
    monkeypatch.setattr(toolchain_provenance.shutil, "which", resolved.get)
    results = toolchain_provenance.findings(
        [
            toolchain_provenance.Tool("shipped", "a.yaml"),
            toolchain_provenance.Tool("leaked", "a.yaml"),
            toolchain_provenance.Tool("leaked", "b.yaml"),
            toolchain_provenance.Tool("nix", "a.yaml"),
            toolchain_provenance.Tool("absent", "b.yaml"),
        ]
    )
    failed = toolchain_provenance.violations(results)
    # `nix` is exempt (it bootstraps the shell); `shipped` is fine.
    assert [finding.name for finding in failed] == ["absent", "leaked"]
    # Declarations collapse by tool name but retain every declaring source.
    leaked = next(finding for finding in failed if finding.name == "leaked")
    assert leaked.sources == ("a.yaml", "b.yaml")
