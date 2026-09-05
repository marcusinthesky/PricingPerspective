"""Resolve authored commands to their installed Typer callbacks."""

from __future__ import annotations

import inspect
import shlex
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

import typer
from typer.main import get_command, get_command_name
from typer.models import DefaultPlaceholder

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from importlib.metadata import EntryPoint


class DiscoveryError(ValueError):
    """Raised when an authored command cannot resolve to one callback."""


@dataclass(frozen=True)
class CommandTarget:
    """One installed application command and its original Python callback."""

    application: str
    command: tuple[str, ...]
    callback: Callable[..., object]

    @property
    def callback_name(self) -> str:
        """Return the stable module-qualified callback name."""
        return f"{self.callback.__module__}:{self.callback.__qualname__}"

    @property
    def entrypoint_name(self) -> str:
        """Return the authored application and command path."""
        return " ".join((self.application, *self.command))


def available_applications(overrides: Mapping[str, str]) -> frozenset[str]:
    """Return configured and installed console-script names."""
    installed = {item.name for item in entry_points(group="console_scripts")}
    return frozenset(installed | set(overrides))


def parse_authored_command(
    command: str,
    *,
    variables: Mapping[str, str],
    applications: frozenset[str],
) -> tuple[str, tuple[str, ...]] | None:
    """Expand DVC scalar vars and find an installed application command."""
    expanded = command
    for name, value in variables.items():
        expanded = expanded.replace(f"${{{name}}}", value)
    try:
        tokens = shlex.split(expanded)
    except ValueError as error:
        msg = f"Could not parse authored command {command!r}: {error}"
        raise DiscoveryError(msg) from error

    candidates = [
        index
        for index, token in enumerate(tokens)
        if token in applications
        and (index == 0 or tokens[index - 1] not in {"--package", "--extra"})
    ]
    if not candidates:
        return None
    index = candidates[0]
    if index + 1 >= len(tokens):
        msg = f"Application {tokens[index]!r} has no command in {command!r}."
        raise DiscoveryError(msg)
    command_tokens: list[str] = []
    for token in tokens[index + 1 :]:
        if token.startswith("-") or "${" in token:
            break
        command_tokens.append(token)
    if not command_tokens:
        msg = f"Application {tokens[index]!r} has no command path in {command!r}."
        raise DiscoveryError(msg)
    return tokens[index], tuple(command_tokens)


def discover_command(
    application: str,
    command: Sequence[str],
    *,
    overrides: Mapping[str, str],
) -> CommandTarget:
    """Load an application and resolve a Typer/Click command callback."""
    app = _load_application(application, overrides)
    if not isinstance(app, typer.Typer):
        msg = f"Console script {application!r} does not expose a Typer application."
        raise DiscoveryError(msg)
    registered = _registered_callback(app, tuple(command))
    if registered is not None:
        return CommandTarget(
            application=application,
            command=tuple(command),
            callback=inspect.unwrap(registered),
        )

    resolved = get_command(app)
    for position, token in enumerate(command):
        commands = getattr(resolved, "commands", None)
        if isinstance(commands, dict) and token in commands:
            resolved = commands[token]
            continue
        is_single_command = (
            position == 0
            and len(command) == 1
            and getattr(resolved, "name", None) == token
        )
        if not is_single_command:
            joined = " ".join(command)
            msg = f"Typer command {application} {joined!s} does not exist."
            raise DiscoveryError(msg)
    callback = getattr(resolved, "callback", None)
    if callback is None:
        joined = " ".join(command)
        msg = f"Typer command {application} {joined!s} has no callback."
        raise DiscoveryError(msg)
    original = inspect.unwrap(callback)
    return CommandTarget(
        application=application,
        command=tuple(command),
        callback=original,
    )


def _registered_callback(
    app: typer.Typer,
    command: tuple[str, ...],
) -> Callable[..., object] | None:
    """Resolve the authored callback before custom Click classes wrap it."""
    matches: list[Callable[..., object]] = []

    def visit(current: typer.Typer, prefix: tuple[str, ...]) -> None:
        for command_info in current.registered_commands:
            callback = command_info.callback
            if callback is None:
                continue
            name = command_info.name or get_command_name(callback.__name__)
            if (*prefix, name) == command:
                matches.append(callback)

        for group_info in current.registered_groups:
            child = group_info.typer_instance
            if not isinstance(child, typer.Typer):
                continue
            name = _group_name(group_info)
            visit(child, (*prefix, name) if name else prefix)

    visit(app, ())
    if len(matches) > 1:
        joined = " ".join(command)
        msg = (
            f"Typer command {joined!r} is registered more than once in the "
            "application tree."
        )
        raise DiscoveryError(msg)
    return matches[0] if matches else None


def _group_name(group_info: object) -> str | None:
    """Apply Typer's name precedence without constructing Click commands."""
    value = getattr(group_info, "name", None)
    if not isinstance(value, DefaultPlaceholder):
        return value if isinstance(value, str) else None

    child = getattr(group_info, "typer_instance", None)
    callback_info = getattr(child, "registered_callback", None)
    callback_name = getattr(callback_info, "name", None)
    if callback_info is not None and not isinstance(callback_name, DefaultPlaceholder):
        return callback_name if isinstance(callback_name, str) else None

    instance_info = getattr(child, "info", None)
    instance_name = getattr(instance_info, "name", None)
    if instance_info is not None and not isinstance(instance_name, DefaultPlaceholder):
        return instance_name if isinstance(instance_name, str) else None
    return value.value if isinstance(value.value, str) else None


def _load_application(application: str, overrides: Mapping[str, str]) -> object:
    override = overrides.get(application)
    if override is not None:
        return _load_object(override)
    matches = tuple(entry_points(group="console_scripts", name=application))
    if len(matches) != 1:
        msg = (
            f"Expected one console script named {application!r}; found {len(matches)}."
        )
        raise DiscoveryError(msg)
    return _load_entry_point(matches[0])


def _load_entry_point(entry_point: EntryPoint) -> object:
    try:
        return entry_point.load()
    except (AttributeError, ImportError, ModuleNotFoundError) as error:
        msg = f"Could not load console script {entry_point.name!r}: {error}"
        raise DiscoveryError(msg) from error


def _load_object(value: str) -> object:
    module_name, separator, attribute = value.partition(":")
    if not separator or not module_name or not attribute:
        msg = f"Application override must use module:attribute syntax: {value!r}."
        raise DiscoveryError(msg)
    try:
        module = import_module(module_name)
        return getattr(module, attribute)
    except (AttributeError, ImportError, ModuleNotFoundError) as error:
        msg = f"Could not load application override {value!r}: {error}"
        raise DiscoveryError(msg) from error
