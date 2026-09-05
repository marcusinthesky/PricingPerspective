"""Typed option models for Typer command boundaries."""

from __future__ import annotations

from contextvars import ContextVar
from copy import copy
from typing import TYPE_CHECKING, Protocol, cast

from pydantic import BaseModel, ConfigDict
from typer.core import TyperCommand
from typer.main import get_click_param
from typer.models import OptionInfo, ParameterInfo, ParamMeta, Required

if TYPE_CHECKING:
    import typer
    from typer.core import TyperArgument, TyperOption


class CliOptions(BaseModel):
    """Immutable, validated inputs for one CLI command."""

    model_config = ConfigDict(extra="forbid", frozen=True)


_CURRENT_OPTIONS: ContextVar[CliOptions | None] = ContextVar(
    "pipeline_cli_options", default=None
)


class _CommandInitializer(Protocol):
    """Describe the keyword-oriented constructor Typer uses internally."""

    def __call__(self, name: str | None, **kwargs: object) -> None: ...


class _CommandCallback(Protocol):
    """Describe the wrapper callback produced by Typer."""

    def __call__(self, **kwargs: object) -> object: ...


def _model_click_params(
    options_model: type[CliOptions],
) -> list[TyperArgument | TyperOption]:
    """Translate Pydantic fields carrying Typer metadata into Click parameters."""
    params: list[TyperArgument | TyperOption] = []
    for name, field in options_model.model_fields.items():
        parameter_metadata = [
            item for item in field.metadata if isinstance(item, ParameterInfo)
        ]
        if len(parameter_metadata) > 1:
            message = (
                f"{options_model.__name__}.{name} carries multiple "
                "typer.Option or typer.Argument annotations"
            )
            raise TypeError(message)

        parameter_info = (
            copy(parameter_metadata[0])
            if parameter_metadata
            else OptionInfo(help=field.description)
        )
        if field.is_required():
            parameter_info.default = Required
        elif field.default_factory is not None:
            parameter_info.default = field.default_factory
        else:
            parameter_info.default = field.default

        param, _convertor = get_click_param(
            ParamMeta(
                name=name,
                default=parameter_info,
                annotation=field.annotation,
            )
        )
        params.append(param)
    return params


def model_command(options_model: type[CliOptions]) -> type[TyperCommand]:
    """Build a Typer command class that validates options into ``options_model``."""
    model_params = _model_click_params(options_model)

    class ModelCommand(TyperCommand):
        """Parse model fields as options before invoking a one-argument callback."""

        def __init__(self, name: str | None, **kwargs: object) -> None:
            original_params = cast(
                "list[TyperArgument | TyperOption]", kwargs.get("params", [])
            )
            kwargs["params"] = [*model_params, *original_params]
            callback = cast("_CommandCallback", kwargs["callback"])

            def validated_callback(**values: object) -> object:
                token = _CURRENT_OPTIONS.set(options_model.model_validate(values))
                try:
                    return callback()
                finally:
                    _CURRENT_OPTIONS.reset(token)

            kwargs["callback"] = validated_callback
            initializer = cast("_CommandInitializer", super().__init__)
            initializer(name, **kwargs)

    return ModelCommand


def options_from_context[OptionsT: CliOptions](
    ctx: typer.Context,
    options_model: type[OptionsT],
) -> OptionsT:
    """Return the validated options installed by ``model_command``."""
    options = _CURRENT_OPTIONS.get()
    if not isinstance(options, options_model):
        message = (
            f"{options_model.__name__} was not installed for command {ctx.command_path}"
        )
        raise TypeError(message)
    return options


__all__ = ["CliOptions", "model_command", "options_from_context"]
