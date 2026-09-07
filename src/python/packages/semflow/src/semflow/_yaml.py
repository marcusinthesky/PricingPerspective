"""Lossless scalar-location helpers for authored DVC YAML."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import yaml
from yaml.nodes import MappingNode, ScalarNode, SequenceNode

if TYPE_CHECKING:
    from yaml.nodes import Node


class YamlPatchError(ValueError):
    """Raised when authored DVC YAML cannot be located safely."""


@dataclass(frozen=True)
class DependencyNode:
    """Source span and scalar style for one item below a deps key."""

    value: str
    start: int
    end: int
    style: str | None


@dataclass(frozen=True)
class StageNode:
    """Authored stage name, command, and dependency scalar locations."""

    name: str
    command: str
    dependencies: tuple[DependencyNode, ...]


def stage_nodes(source: str) -> tuple[StageNode, ...]:
    """Locate direct stage commands and dependency scalars without round-tripping."""
    try:
        root = yaml.compose(source)
    except yaml.YAMLError as error:
        msg = f"Could not parse DVC YAML: {error}"
        raise YamlPatchError(msg) from error
    if not isinstance(root, MappingNode):
        msg = "DVC YAML root must be a mapping."
        raise YamlPatchError(msg)
    stages = _mapping_value(root, "stages")
    if not isinstance(stages, MappingNode):
        msg = "DVC YAML has no stages mapping."
        raise YamlPatchError(msg)

    found: list[StageNode] = []
    for name_node, value_node in stages.value:
        if not isinstance(name_node, ScalarNode) or not isinstance(
            value_node, MappingNode
        ):
            continue
        authored = _mapping_value(value_node, "do")
        body = authored if isinstance(authored, MappingNode) else value_node
        command = _mapping_value(body, "cmd")
        dependencies = _mapping_value(body, "deps")
        if not isinstance(command, ScalarNode) or not isinstance(
            dependencies, SequenceNode
        ):
            continue
        dependency_values = tuple(
            DependencyNode(
                value=item.value,
                start=item.start_mark.index,
                end=item.end_mark.index,
                style=item.style,
            )
            for item in dependencies.value
            if isinstance(item, ScalarNode)
        )
        found.append(
            StageNode(
                name=name_node.value,
                command=command.value,
                dependencies=dependency_values,
            )
        )
    return tuple(found)


def render_scalar(value: str, style: str | None) -> str:
    """Render a replacement using the authored scalar's quote style."""
    if style == "'":
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    if style == '"':
        return json.dumps(value)
    return value


def replace_spans(source: str, replacements: list[tuple[int, int, str]]) -> str:
    """Replace non-overlapping scalar spans while retaining every other byte."""
    ordered = sorted(replacements)
    for previous, current in pairwise(ordered):
        if current[0] < previous[1]:
            msg = "DVC dependency aliases produced overlapping YAML replacements."
            raise YamlPatchError(msg)
    result = source
    for start, end, value in reversed(ordered):
        result = f"{result[:start]}{value}{result[end:]}"
    return result


def dependency_line_span(
    source: str,
    dependency: DependencyNode,
    *,
    include_leading_comments: bool = False,
) -> tuple[int, int]:
    """Return one block-sequence line and its directly attached comments."""
    line_start = source.rfind("\n", 0, dependency.start) + 1
    newline = source.find("\n", dependency.end)
    line_end = len(source) if newline < 0 else newline + 1
    prefix = source[line_start : dependency.start]
    if prefix.strip() != "-" or "\n" in source[dependency.start : dependency.end]:
        msg = "Function-level patches require block-style, single-line DVC deps."
        raise YamlPatchError(msg)
    if include_leading_comments:
        indentation = prefix[: len(prefix) - len(prefix.lstrip())]
        line_start = _leading_comment_start(source, line_start, indentation)
    return line_start, line_end


def _leading_comment_start(source: str, line_start: int, indentation: str) -> int:
    start = line_start
    while start > 0:
        previous_end = start - 1
        previous_start = source.rfind("\n", 0, previous_end) + 1
        line = source[previous_start:previous_end]
        if not line.startswith(f"{indentation}#"):
            break
        start = previous_start
    return start


def _mapping_value(node: MappingNode, key_name: str) -> Node | None:
    for key, value in node.value:
        if isinstance(key, ScalarNode) and key.value == key_name:
            return value
    return None
