"""Static architecture gates for the pipeline CLI package."""

from __future__ import annotations

import ast
from pathlib import Path

CLI_ROOT = Path(__file__).parents[1] / "src/pipeline/cli"
PIPELINE_ROOT = CLI_ROOT.parent
COMMAND_ROOT = CLI_ROOT / "commands"
IMPLEMENTATION_PREFIXES = (
    "pipeline._kernels",
    "pipeline.figures",
    "pipeline.io",
    "pipeline.stages",
)
FORBIDDEN_DYNAMIC_CALLS = {
    "_load_command",
    "_load_result_command",
    "getattr",
    "globals",
    "import_module",
}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports(tree: ast.AST) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def test_command_names_are_literal_and_unique() -> None:
    """Keep the flat DVC command surface statically discoverable."""
    names: list[str] = []
    for path in COMMAND_ROOT.rglob("*.py"):
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "command":
                continue
            if not node.args:
                continue
            name = node.args[0]
            assert isinstance(name, ast.Constant)
            assert isinstance(name.value, str)
            names.append(name.value)
    assert len(names) == len(set(names))


def test_cli_uses_no_reflective_dispatch() -> None:
    """Reject the dynamic loader and string/globals registration pattern."""
    violations: list[str] = []
    for path in CLI_ROOT.rglob("*.py"):
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.Call):
                continue
            function_name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else None
            )
            if function_name in FORBIDDEN_DYNAMIC_CALLS:
                violations.append(f"{path.relative_to(CLI_ROOT)}:{node.lineno}")
    assert violations == []


def test_composition_imports_only_cli_modules() -> None:
    """Keep stage implementations below the command-adapter boundary."""
    violations: list[str] = []
    composition_files = [CLI_ROOT / "app.py", *(CLI_ROOT / "views").rglob("*.py")]
    for path in composition_files:
        violations.extend(
            f"{path.relative_to(CLI_ROOT)} -> {imported}"
            for imported in _imports(_tree(path))
            if imported.startswith(IMPLEMENTATION_PREFIXES)
        )
    assert violations == []


def test_commands_do_not_import_composition() -> None:
    """Keep command adapters independent of the root app and grouped views."""
    violations: list[str] = []
    for path in COMMAND_ROOT.rglob("*.py"):
        violations.extend(
            f"{path.relative_to(CLI_ROOT)} -> {imported}"
            for imported in _imports(_tree(path))
            if imported.startswith(("pipeline.cli.app", "pipeline.cli.views"))
        )
    assert violations == []


def test_implementation_does_not_import_cli() -> None:
    """Prevent stages, figures, IO, and kernels from depending on adapters."""
    violations: list[str] = []
    for path in PIPELINE_ROOT.rglob("*.py"):
        if path.is_relative_to(CLI_ROOT):
            continue
        violations.extend(
            f"{path.relative_to(PIPELINE_ROOT)} -> {imported}"
            for imported in _imports(_tree(path))
            if imported == "pipeline.cli" or imported.startswith("pipeline.cli.")
        )
    assert violations == []
