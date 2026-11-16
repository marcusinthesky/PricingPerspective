"""Static boundary checks for Paper 1's stable dyadic facade."""

from __future__ import annotations

import ast
from pathlib import Path

_PIPELINE_ROOT = Path(__file__).parents[3]
_FACADE = "pipeline.stages.papers.paper1.dyadic_confound"
_PRIVATE_PREFIX = "pipeline.stages.papers.paper1._dyadic"
_DOWNSTREAM_MODULES = (
    _PIPELINE_ROOT / "src/pipeline/stages/papers/paper1/dyadic_oos.py",
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_downstream_dyadic_consumers_import_the_stable_facade() -> None:
    """Owner topology stays private behind the documented compatibility seam."""
    for path in _DOWNSTREAM_MODULES:
        imports = _imports(path)
        private_imports = sorted(
            imported for imported in imports if imported.startswith(_PRIVATE_PREFIX)
        )
        assert not private_imports, (
            f"{path.name} bypasses the facade: {private_imports}"
        )
        assert _FACADE in imports, f"{path.name} must import {_FACADE}"
