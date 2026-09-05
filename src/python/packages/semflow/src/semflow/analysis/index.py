"""Index importable modules, top-level definitions, globals, and bindings."""

from __future__ import annotations

import ast
import importlib.util
import tokenize
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class ModuleIndexError(ValueError):
    """Raised when configured source roots do not form one module index."""


@dataclass(frozen=True)
class ImportBinding:
    """One local name bound by an import statement."""

    local_name: str
    module: str
    symbol: str | None
    node: ast.Import | ast.ImportFrom


@dataclass(frozen=True)
class ModuleInfo:
    """Parsed source information for one first-party module."""

    name: str
    path: Path
    is_package: bool
    tree: ast.Module
    definitions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]
    globals: dict[str, ast.stmt]
    bindings: dict[str, ImportBinding]
    future_imports: tuple[ast.ImportFrom, ...]


class ModuleIndex:
    """Resolve source modules and imported symbols below configured roots."""

    def __init__(self, *, project_root: Path, source_roots: tuple[Path, ...]) -> None:
        """Build an index over every Python file below the import roots."""
        self.project_root = project_root.resolve()
        self._modules = self._build_modules(source_roots)
        self._populate_bindings()

    @property
    def modules(self) -> frozenset[str]:
        """Return every indexed first-party module name."""
        return frozenset(self._modules)

    def module(self, name: str) -> ModuleInfo | None:
        """Return parsed module information when the module is first-party."""
        return self._modules.get(name)

    def relative_path(self, module: ModuleInfo) -> str:
        """Return the module path relative to the project root."""
        return module.path.relative_to(self.project_root).as_posix()

    def bindings_in(
        self,
        module: ModuleInfo,
        node: ast.AST,
    ) -> dict[str, ImportBinding]:
        """Return bindings introduced by imports anywhere below an AST node."""
        return {
            binding.local_name: binding
            for binding in self.all_bindings_in(module, node)
        }

    def all_bindings_in(
        self,
        module: ModuleInfo,
        node: ast.AST,
    ) -> tuple[ImportBinding, ...]:
        """Return every import binding below a node without alias deduplication."""
        bindings: list[ImportBinding] = []
        for child in ast.walk(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                bindings.extend(self._bindings_for_node(module, child).values())
        return tuple(bindings)

    def resolve_binding(
        self,
        binding: ImportBinding,
        attributes: tuple[str, ...] = (),
    ) -> tuple[str, str | None] | None:
        """Resolve a binding plus attribute chain to a module and optional symbol."""
        if binding.symbol is not None:
            symbol = ".".join((binding.symbol, *attributes))
            return (binding.module, symbol)

        parts = (*binding.module.split("."), *attributes)
        for length in range(len(parts), 0, -1):
            module_name = ".".join(parts[:length])
            if module_name in self._modules:
                remainder = parts[length:]
                return (module_name, ".".join(remainder) or None)
        return None

    def _build_modules(self, source_roots: tuple[Path, ...]) -> dict[str, ModuleInfo]:
        modules: dict[str, ModuleInfo] = {}
        for source_root in source_roots:
            root = source_root.resolve()
            for path in sorted(root.rglob("*.py")):
                resolved = path.resolve()
                if not resolved.is_relative_to(self.project_root):
                    msg = f"Indexed Python path escapes project root: {path}"
                    raise ModuleIndexError(msg)
                relative = path.relative_to(root)
                is_package = path.name == "__init__.py"
                module_parts = (
                    relative.parent.parts
                    if is_package
                    else relative.with_suffix("").parts
                )
                if not module_parts:
                    continue
                name = ".".join(module_parts)
                if name in modules:
                    msg = f"Duplicate module {name!r} below configured source-roots."
                    raise ModuleIndexError(msg)
                tree = _parse_python(path)
                modules[name] = ModuleInfo(
                    name=name,
                    path=resolved,
                    is_package=is_package,
                    tree=tree,
                    definitions=_definitions(tree.body),
                    globals=_globals(tree.body),
                    bindings={},
                    future_imports=tuple(
                        statement
                        for statement in tree.body
                        if isinstance(statement, ast.ImportFrom)
                        and statement.module == "__future__"
                    ),
                )
        return modules

    def _populate_bindings(self) -> None:
        for name, module in tuple(self._modules.items()):
            bindings: dict[str, ImportBinding] = {}
            for statement in module.tree.body:
                if isinstance(statement, (ast.Import, ast.ImportFrom)):
                    bindings.update(self._bindings_for_node(module, statement))
            self._modules[name] = ModuleInfo(
                name=module.name,
                path=module.path,
                is_package=module.is_package,
                tree=module.tree,
                definitions=module.definitions,
                globals=module.globals,
                bindings=bindings,
                future_imports=module.future_imports,
            )

    def _bindings_for_node(
        self,
        current: ModuleInfo,
        node: ast.Import | ast.ImportFrom,
    ) -> dict[str, ImportBinding]:
        if isinstance(node, ast.Import):
            return {
                alias.asname or alias.name.split(".")[0]: ImportBinding(
                    local_name=alias.asname or alias.name.split(".")[0],
                    module=alias.name if alias.asname else alias.name.split(".")[0],
                    symbol=None,
                    node=node,
                )
                for alias in node.names
            }

        base = _import_from_base(current, node)
        if base is None:
            return {}
        bindings: dict[str, ImportBinding] = {}
        for alias in node.names:
            if alias.name == "*":
                bindings["*"] = ImportBinding(
                    local_name="*", module=base, symbol="*", node=node
                )
                continue
            local_name = alias.asname or alias.name
            submodule = f"{base}.{alias.name}"
            if submodule in self._modules:
                bindings[local_name] = ImportBinding(
                    local_name=local_name,
                    module=submodule,
                    symbol=None,
                    node=node,
                )
            else:
                bindings[local_name] = ImportBinding(
                    local_name=local_name,
                    module=base,
                    symbol=alias.name,
                    node=node,
                )
        return bindings


def _parse_python(path: Path) -> ast.Module:
    try:
        with tokenize.open(path) as stream:
            return ast.parse(stream.read(), filename=str(path), type_comments=True)
    except (OSError, SyntaxError, UnicodeError) as error:
        msg = f"Could not index Python module {path}: {error}"
        raise ModuleIndexError(msg) from error


def _definitions(
    body: Iterable[ast.stmt],
) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]:
    return {
        statement.name: statement
        for statement in body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _globals(body: Iterable[ast.stmt]) -> dict[str, ast.stmt]:
    found: dict[str, ast.stmt] = {}
    for statement in body:
        if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for name in _stored_names(statement):
                found[name] = statement
    return found


def _stored_names(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
    }


def _import_from_base(current: ModuleInfo, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module
    package = current.name if current.is_package else current.name.rpartition(".")[0]
    if not package:
        return None
    relative = f"{'.' * node.level}{node.module or ''}"
    try:
        return importlib.util.resolve_name(relative, package)
    except (ImportError, ValueError):
        return None
