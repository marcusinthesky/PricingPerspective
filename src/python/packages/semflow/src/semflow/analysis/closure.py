"""Build conservative semantic closures from tracked command callbacks."""

from __future__ import annotations

import ast
import hashlib
import json
import tokenize
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from semflow.hashing import (
    normalize_semantic_node,
    semantic_digest,
    semantic_node_digest,
)
from semflow.model import (
    AnalysisFallback,
    EntrypointFingerprint,
    TrackedMember,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from semflow.analysis.index import ImportBinding, ModuleIndex, ModuleInfo
    from semflow.discovery import CommandTarget
    from semflow.model import HashPolicy
    from semflow.tracking import TrackSpec


class AnalysisError(ValueError):
    """Raised when a callback cannot be represented by the source index."""


@dataclass(frozen=True)
class _Reference:
    module: str
    symbol: str | None
    reason: str


class _ClosureBuilder:
    def __init__(
        self,
        *,
        index: ModuleIndex,
        policy: HashPolicy,
    ) -> None:
        self.index = index
        self.policy = policy
        self.members: dict[tuple[str, str, str], TrackedMember] = {}
        self.member_modules: dict[tuple[str, str, str], str] = {}
        self.fallbacks: dict[str, AnalysisFallback] = {}
        self._visited: set[tuple[str, str]] = set()
        self._bindings: set[tuple[str, str, str, str | None]] = set()

    def add_symbol(
        self,
        module_name: str,
        symbol: str,
        *,
        preserve_annotations: bool = False,
        reason: str = "reachable symbol",
    ) -> None:
        if module_name in self.fallbacks:
            return
        key = (module_name, symbol)
        if key in self._visited:
            return
        self._visited.add(key)
        module = self.index.module(module_name)
        if module is None:
            return

        owner, separator, attribute = symbol.partition(".")
        definition = module.definitions.get(owner)
        if definition is not None:
            self._add_definition(
                module,
                definition,
                preserve_annotations=preserve_annotations,
            )
            return

        global_node = module.globals.get(owner)
        if global_node is not None:
            self._add_node(module, owner, "global", global_node)
            self._follow_references(module, global_node)
            return

        binding = module.bindings.get(owner)
        if binding is not None:
            self._add_binding(module, binding)
            self._follow_binding(
                binding,
                attributes=tuple(attribute.split(".")) if separator else (),
                reason=f"re-export for {symbol}",
            )
            return

        self.fallback_module(module_name, f"{reason} could not resolve {symbol!r}")

    def fallback_module(self, module_name: str, reason: str) -> None:
        if module_name in self.fallbacks:
            return
        module = self.index.module(module_name)
        if module is None:
            return
        path = self.index.relative_path(module)
        fallback = AnalysisFallback(module=module_name, path=path, reason=reason)
        self.fallbacks[module_name] = fallback
        for key, owner in tuple(self.member_modules.items()):
            if owner == module_name:
                self.members.pop(key, None)
                self.member_modules.pop(key, None)
        self._add_member(
            module,
            symbol=f"{module_name}:<module>",
            kind="module",
            digest=semantic_digest(_read_source(module), self.policy),
        )
        normalized = normalize_semantic_node(module.tree, self.policy)
        for binding in self.index.all_bindings_in(module, normalized):
            self._follow_binding(binding, reason=f"fallback import from {module_name}")

    def _add_definition(
        self,
        module: ModuleInfo,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        *,
        preserve_annotations: bool,
    ) -> None:
        kind: Literal["function", "class"] = (
            "class" if isinstance(node, ast.ClassDef) else "function"
        )
        preserve = preserve_annotations or bool(node.decorator_list)
        self._add_node(
            module,
            node.name,
            kind,
            node,
            preserve_annotations=preserve,
        )
        self._add_future_imports(module)
        normalized = normalize_semantic_node(
            node,
            self.policy,
            preserve_annotations=preserve,
        )
        self._follow_references(module, normalized)

    def _add_future_imports(self, module: ModuleInfo) -> None:
        for node in module.future_imports:
            binding = next(iter(self.index.bindings_in(module, node).values()), None)
            if binding is not None:
                self._add_binding(module, binding)

    def _follow_references(self, module: ModuleInfo, node: ast.AST) -> None:
        for reference in _references(self.index, module, node, self._add_binding):
            if reference.symbol is None:
                continue
            self.add_symbol(
                reference.module,
                reference.symbol,
                reason=reference.reason,
            )

    def _follow_binding(
        self,
        binding: ImportBinding,
        *,
        reason: str,
        attributes: tuple[str, ...] = (),
    ) -> None:
        if binding.symbol == "*":
            self.fallback_module(binding.module, f"wildcard {reason}")
            return
        resolved = self.index.resolve_binding(binding, attributes)
        if resolved is None:
            return
        module_name, symbol = resolved
        if symbol is None:
            return
        self.add_symbol(module_name, symbol, reason=reason)

    def _add_binding(self, module: ModuleInfo, binding: ImportBinding) -> None:
        identity = (
            module.name,
            binding.local_name,
            binding.module,
            binding.symbol,
        )
        if identity in self._bindings or module.name in self.fallbacks:
            return
        self._bindings.add(identity)
        target = binding.module
        if binding.symbol is not None:
            target = f"{target}:{binding.symbol}"
        self._add_node(
            module,
            f"<binding:{binding.local_name}={target}>",
            "binding",
            binding.node,
        )

    def _add_node(
        self,
        module: ModuleInfo,
        symbol: str,
        kind: Literal["function", "class", "global", "binding"],
        node: ast.AST,
        *,
        preserve_annotations: bool = False,
    ) -> None:
        self._add_member(
            module,
            symbol=f"{module.name}:{symbol}",
            kind=kind,
            digest=semantic_node_digest(
                node,
                self.policy,
                preserve_annotations=preserve_annotations,
            ),
        )

    def _add_member(
        self,
        module: ModuleInfo,
        *,
        symbol: str,
        kind: Literal["function", "class", "global", "binding", "module"],
        digest: str,
    ) -> None:
        member = TrackedMember(
            path=self.index.relative_path(module),
            symbol=symbol,
            kind=kind,
            digest=digest,
        )
        key = (member.path, member.symbol, member.kind)
        self.members[key] = member
        self.member_modules[key] = module.name


def fingerprint_entrypoint(
    target: CommandTarget,
    *,
    index: ModuleIndex,
    policy: HashPolicy,
    track_spec: TrackSpec | None = None,
) -> EntrypointFingerprint:
    """Fingerprint one callback and its conservatively reachable first-party code."""
    callback_module = target.callback.__module__
    callback_symbol = target.callback.__qualname__
    if "<locals>" in callback_symbol:
        msg = f"Tracked callback must be module-level: {target.callback_name}"
        raise AnalysisError(msg)
    if index.module(callback_module) is None:
        msg = (
            "Tracked callback is outside configured source-roots: "
            f"{target.callback_name}"
        )
        raise AnalysisError(msg)

    builder = _ClosureBuilder(index=index, policy=policy)
    builder.add_symbol(
        callback_module,
        callback_symbol,
        preserve_annotations=True,
        reason="Typer callback",
    )
    for include in () if track_spec is None else track_spec.include:
        module_name, symbol = _parse_include(include)
        builder.add_symbol(module_name, symbol, reason="explicit @track include")

    members = tuple(sorted(builder.members.values(), key=_member_key))
    fallbacks = tuple(sorted(builder.fallbacks.values(), key=lambda item: item.module))
    payload = {
        "entrypoint": target.entrypoint_name,
        "callback": target.callback_name,
        "policy": policy.to_dict(),
        "members": [member.to_dict() for member in members],
        "fallbacks": [fallback.to_dict() for fallback in fallbacks],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = f"sha256:{hashlib.sha256(canonical).hexdigest()}"
    return EntrypointFingerprint(
        schema_version=2,
        entrypoint=target.entrypoint_name,
        callback=target.callback_name,
        algorithm="sha256",
        policy=policy,
        digest=digest,
        members=members,
        fallbacks=fallbacks,
    )


def _references(
    index: ModuleIndex,
    module: ModuleInfo,
    node: ast.AST,
    add_binding: Callable[[ModuleInfo, ImportBinding], None],
) -> tuple[_Reference, ...]:
    local_bindings = _binding_groups(index.all_bindings_in(module, node))
    found: set[_Reference] = set()
    handled_names: set[int] = set()

    _attribute_references(
        index,
        module,
        node,
        local_bindings,
        add_binding,
        found,
        handled_names,
    )
    _name_references(
        index,
        module,
        node,
        local_bindings,
        add_binding,
        found,
        handled_names,
    )
    return tuple(found)


def _attribute_references(
    index: ModuleIndex,
    module: ModuleInfo,
    node: ast.AST,
    local_bindings: dict[str, tuple[ImportBinding, ...]],
    add_binding: Callable[[ModuleInfo, ImportBinding], None],
    found: set[_Reference],
    handled_names: set[int],
) -> None:
    for child in ast.walk(node):
        if not isinstance(child, ast.Attribute) or not isinstance(child.ctx, ast.Load):
            continue
        parts = _attribute_parts(child)
        if parts is None:
            continue
        base, attributes, base_node = parts
        bindings = local_bindings.get(base)
        module_binding = module.bindings.get(base)
        if bindings is None and module_binding is not None:
            bindings = (module_binding,)
        if not bindings:
            continue
        handled_names.add(id(base_node))
        for binding in bindings:
            if binding is module_binding:
                add_binding(module, binding)
            resolved = index.resolve_binding(binding, attributes)
            if resolved is not None:
                found.add(
                    _Reference(
                        module=resolved[0],
                        symbol=resolved[1],
                        reason=(f"attribute reference {base}.{'.'.join(attributes)}"),
                    )
                )


def _name_references(
    index: ModuleIndex,
    module: ModuleInfo,
    node: ast.AST,
    local_bindings: dict[str, tuple[ImportBinding, ...]],
    add_binding: Callable[[ModuleInfo, ImportBinding], None],
    found: set[_Reference],
    handled_names: set[int],
) -> None:
    for child in ast.walk(node):
        if not isinstance(child, ast.Name) or not isinstance(child.ctx, ast.Load):
            continue
        if id(child) in handled_names:
            continue
        bindings = local_bindings.get(child.id)
        module_binding = module.bindings.get(child.id)
        if bindings is None and module_binding is not None:
            bindings = (module_binding,)
        if bindings:
            for binding in bindings:
                if binding is module_binding:
                    add_binding(module, binding)
                resolved = index.resolve_binding(binding)
                if resolved is not None:
                    found.add(
                        _Reference(
                            module=resolved[0],
                            symbol=resolved[1],
                            reason=f"imported reference {child.id}",
                        )
                    )
            continue
        if child.id in module.definitions or child.id in module.globals:
            found.add(
                _Reference(
                    module=module.name,
                    symbol=child.id,
                    reason=f"module reference {child.id}",
                )
            )


def _binding_groups(
    bindings: tuple[ImportBinding, ...],
) -> dict[str, tuple[ImportBinding, ...]]:
    grouped: dict[str, list[ImportBinding]] = {}
    for binding in bindings:
        grouped.setdefault(binding.local_name, []).append(binding)
    return {name: tuple(items) for name, items in grouped.items()}


def _attribute_parts(
    node: ast.Attribute,
) -> tuple[str, tuple[str, ...], ast.Name] | None:
    attributes: list[str] = [node.attr]
    value = node.value
    while isinstance(value, ast.Attribute):
        attributes.append(value.attr)
        value = value.value
    if not isinstance(value, ast.Name):
        return None
    attributes.reverse()
    return value.id, tuple(attributes), value


def _parse_include(value: str) -> tuple[str, str]:
    module_name, separator, symbol = value.partition(":")
    if not separator or not module_name or not symbol:
        msg = f"@track include must use module:symbol syntax: {value!r}"
        raise AnalysisError(msg)
    return module_name, symbol


def _member_key(member: TrackedMember) -> tuple[str, str, str]:
    return member.path, member.symbol, member.kind


def _read_source(module: ModuleInfo) -> str:
    with tokenize.open(module.path) as stream:
        return stream.read()
