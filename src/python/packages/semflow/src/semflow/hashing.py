"""Canonical AST normalization and semantic fingerprint generation."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import tokenize
from typing import TYPE_CHECKING, Literal, TypeVar, cast, override

from semflow.model import Fingerprint, HashPolicy, MemberFingerprint

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from semflow.model import EntrypointFingerprint

_NodeT = TypeVar("_NodeT", bound=ast.AST)


class HashingError(ValueError):
    """Raised when a configured source cannot be fingerprinted safely."""


class _SemanticNormalizer(ast.NodeTransformer):
    """Remove syntax declared non-semantic by a :class:`HashPolicy`."""

    def __init__(self, policy: HashPolicy) -> None:
        self.policy = policy
        self._scope: Literal["module", "class", "function"] = "module"

    @override
    def visit_Module(self, node: ast.Module) -> ast.Module:
        node = self._generic_visit(node)
        if self.policy.ignore_docstrings:
            node.body = _without_docstring(node.body)
        return node

    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        previous = self._scope
        self._scope = "class"
        node = self._generic_visit(node)
        self._scope = previous
        if self.policy.ignore_docstrings:
            node.body = _without_docstring(node.body)
        return node

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        previous = self._scope
        self._scope = "function"
        node = self._generic_visit(node)
        self._scope = previous
        if self.policy.ignore_docstrings:
            node.body = _without_docstring(node.body)
        if self.policy.ignore_annotations:
            _strip_function_annotations(node)
        return node

    @override
    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> ast.AsyncFunctionDef:
        previous = self._scope
        self._scope = "function"
        node = self._generic_visit(node)
        self._scope = previous
        if self.policy.ignore_docstrings:
            node.body = _without_docstring(node.body)
        if self.policy.ignore_annotations:
            _strip_function_annotations(node)
        return node

    @override
    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign | ast.Assign | None:
        node = self._generic_visit(node)
        if not self.policy.ignore_annotations or self._scope != "function":
            return node
        if node.value is None:
            return None
        return ast.Assign(targets=[node.target], value=node.value)

    @override
    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        node = self._generic_visit(node)
        if self.policy.ignore_annotations:
            node.type_comment = None
        return node

    @override
    def visit_For(self, node: ast.For) -> ast.For:
        node = self._generic_visit(node)
        if self.policy.ignore_annotations:
            node.type_comment = None
        return node

    @override
    def visit_AsyncFor(self, node: ast.AsyncFor) -> ast.AsyncFor:
        node = self._generic_visit(node)
        if self.policy.ignore_annotations:
            node.type_comment = None
        return node

    @override
    def visit_With(self, node: ast.With) -> ast.With:
        node = self._generic_visit(node)
        if self.policy.ignore_annotations:
            node.type_comment = None
        return node

    @override
    def visit_AsyncWith(self, node: ast.AsyncWith) -> ast.AsyncWith:
        node = self._generic_visit(node)
        if self.policy.ignore_annotations:
            node.type_comment = None
        return node

    @override
    def visit_If(self, node: ast.If) -> ast.If | list[ast.stmt] | None:
        if self.policy.ignore_type_checking_blocks and _is_type_checking(node.test):
            visited = [self.visit(statement) for statement in node.orelse]
            return [
                statement
                for result in visited
                for statement in _statement_results(result)
            ]
        return self._generic_visit(node)

    def _generic_visit(self, node: _NodeT) -> _NodeT:
        return cast("_NodeT", self.generic_visit(node))


def semantic_digest(source: str, policy: HashPolicy | None = None) -> str:
    """Hash executable Python structure under the selected normalization policy."""
    selected = policy or HashPolicy()
    try:
        tree = ast.parse(source, type_comments=True)
    except SyntaxError as error:
        msg = f"Cannot semantic-hash invalid Python: {error}"
        raise HashingError(msg) from error
    return semantic_node_digest(tree, selected)


def semantic_node_digest(
    node: ast.AST,
    policy: HashPolicy | None = None,
    *,
    preserve_annotations: bool = False,
) -> str:
    """Hash one AST node without mutating the caller's parsed tree."""
    normalized = normalize_semantic_node(
        node,
        policy,
        preserve_annotations=preserve_annotations,
    )
    canonical = ast.dump(
        normalized,
        annotate_fields=True,
        include_attributes=False,
        indent=None,
    )
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


def normalize_semantic_node(
    node: ast.AST,
    policy: HashPolicy | None = None,
    *,
    preserve_annotations: bool = False,
) -> ast.AST:
    """Return a normalized deep copy suitable for hashing and reachability."""
    selected = policy or HashPolicy()
    if preserve_annotations and selected.ignore_annotations:
        selected = HashPolicy(
            ignore_docstrings=selected.ignore_docstrings,
            ignore_annotations=False,
            ignore_type_checking_blocks=selected.ignore_type_checking_blocks,
        )
    normalized = _SemanticNormalizer(selected).visit(copy.deepcopy(node))
    ast.fix_missing_locations(normalized)
    return normalized


def fingerprint_source(
    source: Path,
    *,
    project_root: Path,
    policy: HashPolicy | None = None,
) -> Fingerprint:
    """Fingerprint one Python file or every Python file below a directory."""
    selected = policy or HashPolicy()
    resolved_source = source.resolve()
    resolved_root = project_root.resolve()
    if not resolved_source.is_relative_to(resolved_root):
        msg = f"Source escapes project root: {source}"
        raise HashingError(msg)

    if resolved_source.is_file():
        if resolved_source.suffix != ".py":
            msg = f"Only Python files can be semantic-hashed: {resolved_source}"
            raise HashingError(msg)
        files = (resolved_source,)
        kind = "file"
    elif resolved_source.is_dir():
        files = tuple(sorted(resolved_source.rglob("*.py")))
        kind = "directory"
        if not files:
            msg = f"Python dependency directory contains no .py files: {source}"
            raise HashingError(msg)
    else:
        msg = f"Python dependency does not exist: {source}"
        raise HashingError(msg)

    members = tuple(
        MemberFingerprint(
            path=path.relative_to(resolved_root).as_posix(),
            digest=semantic_digest(_read_python(path, resolved_root), selected),
        )
        for path in files
    )
    aggregate_payload = json.dumps(
        [member.to_dict() for member in members],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    aggregate = f"sha256:{hashlib.sha256(aggregate_payload).hexdigest()}"
    return Fingerprint(
        schema_version=1,
        source=resolved_source.relative_to(resolved_root).as_posix(),
        kind=kind,
        algorithm="sha256",
        policy=selected,
        digest=aggregate,
        members=members,
    )


def fingerprint_json(fingerprint: Fingerprint) -> str:
    """Serialize a fingerprint as canonical, review-friendly JSON."""
    return f"{json.dumps(fingerprint.to_dict(), indent=2, sort_keys=True)}\n"


def entrypoint_fingerprint_json(fingerprint: EntrypointFingerprint) -> str:
    """Serialize an entrypoint fingerprint with one reviewable line per member."""
    scalar_fields = {
        "algorithm": fingerprint.algorithm,
        "callback": fingerprint.callback,
        "digest": fingerprint.digest,
        "entrypoint": fingerprint.entrypoint,
    }
    members = [member.to_dict() for member in fingerprint.members]
    fallbacks = [fallback.to_dict() for fallback in fingerprint.fallbacks]
    lines = [
        "{",
        *(
            f"  {json.dumps(key)}: {json.dumps(value)},"
            for key, value in scalar_fields.items()
        ),
    ]
    lines.extend(_json_object_array("fallbacks", fallbacks, trailing=True))
    lines.extend(_json_object_array("members", members, trailing=True))
    lines.append(
        f'  "policy": {json.dumps(fingerprint.policy.to_dict(), sort_keys=True)},'
    )
    lines.append(f'  "schema_version": {fingerprint.schema_version}')
    lines.append("}")
    return "\n".join(lines) + "\n"


def _json_object_array(
    name: str,
    items: Sequence[object],
    *,
    trailing: bool,
) -> list[str]:
    lines = [f"  {json.dumps(name)}: ["]
    for index, item in enumerate(items):
        suffix = "," if index + 1 < len(items) else ""
        rendered = json.dumps(item, sort_keys=True, separators=(",", ":"))
        lines.append(f"    {rendered}{suffix}")
    lines.append(f"  ]{',' if trailing else ''}")
    return lines


def _without_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if body and _is_docstring(body[0]):
        return body[1:]
    return body


def _read_python(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(project_root):
        msg = f"Python member escapes project root: {path}"
        raise HashingError(msg)
    with tokenize.open(path) as stream:
        return stream.read()


def _is_docstring(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _strip_function_annotations(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> None:
    arguments = (
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
    )
    for argument in arguments:
        argument.annotation = None
    if node.args.vararg is not None:
        node.args.vararg.annotation = None
    if node.args.kwarg is not None:
        node.args.kwarg.annotation = None
    node.returns = None
    node.type_comment = None


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute)
        and test.attr == "TYPE_CHECKING"
        and isinstance(test.value, ast.Name)
        and test.value.id in {"typing", "typing_extensions"}
    )


def _statement_results(
    result: ast.AST | list[ast.AST] | None,
) -> tuple[ast.stmt, ...]:
    if result is None:
        return ()
    if isinstance(result, list):
        return tuple(item for item in result if isinstance(item, ast.stmt))
    return (result,) if isinstance(result, ast.stmt) else ()
