"""Loguru application logging with OpenTelemetry correlation and export."""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast, override

from loguru import logger
from opentelemetry import context, trace
from opentelemetry.exporter.otlp.json.file import FileSpanExporter
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import (
    BatchLogRecordProcessor,
    LogRecordExporter,
    LogRecordExportResult,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from opentelemetry.context import Context, Token
    from opentelemetry.sdk._logs import ReadableLogRecord
    from opentelemetry.trace import Span


@dataclass
class _LoggingState:
    """Mutable process-local logging and OpenTelemetry configuration."""

    configured: bool = False
    logger_provider: LoggerProvider | None = None
    tracer_provider: TracerProvider | None = None


_STATE = _LoggingState()


class _InterceptHandler(logging.Handler):
    """Re-emit standard-library logging records through Loguru."""

    @override
    def emit(self, record: logging.LogRecord) -> None:
        """Forward one record while preserving its originating logger name."""
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).patch(
            lambda loguru_record: loguru_record.update(name=record.name)
        ).log(level, record.getMessage())


class _OpenTelemetrySink:
    """Translate Loguru records into stdlib records for the OTel handler."""

    def __init__(self, handler: logging.Handler) -> None:
        """Store the configured OpenTelemetry logging handler."""
        self._handler = handler

    def write(self, message: object) -> None:
        """Export one Loguru message without re-entering the Loguru pipeline."""
        record = cast("dict[str, object]", getattr(message, "record"))  # noqa: B009
        level = record.get("level")
        level_no = int(getattr(level, "no", logging.INFO))
        name = str(record.get("name") or "")
        file_info = record.get("file")
        pathname = str(getattr(file_info, "path", ""))
        line_value = record.get("line", 0)
        line = int(line_value) if isinstance(line_value, (int, str)) else 0
        log_record = logging.LogRecord(
            name=name,
            level=level_no,
            pathname=pathname,
            lineno=line,
            msg=str(record.get("message", "")),
            args=(),
            exc_info=None,
        )
        time_info = record.get("time")
        timestamp = getattr(time_info, "timestamp", None)
        if callable(timestamp):
            log_record.created = float(cast("float", timestamp()))
        extra = record.get("extra")
        if isinstance(extra, dict):
            for key, value in extra.items():
                if isinstance(key, str):
                    log_record.__dict__[key] = value
        self._handler.emit(log_record)


class _JsonFileLogExporter(LogRecordExporter):
    """Persist OTel log records as local JSONL for later ingestion."""

    def __init__(self, path: Path) -> None:
        """Create an append-only exporter at ``path``."""
        self._path = path

    @override
    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        """Write a batch of OTel log records."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as stream:
                for data in batch:
                    record = data.log_record
                    stream.write(
                        json.dumps(
                            {
                                "resource": dict(data.resource.attributes),
                                "timestamp": record.timestamp,
                                "observed_timestamp": record.observed_timestamp,
                                "severity_text": record.severity_text,
                                "severity_number": (
                                    record.severity_number.name
                                    if record.severity_number is not None
                                    else None
                                ),
                                "body": record.body,
                                "attributes": dict(record.attributes or {}),
                                "trace_id": (
                                    f"{record.trace_id:032x}"
                                    if record.trace_id
                                    else None
                                ),
                                "span_id": (
                                    f"{record.span_id:016x}" if record.span_id else None
                                ),
                            },
                            default=str,
                        )
                        + "\n"
                    )
        except Exception:  # noqa: BLE001
            return LogRecordExportResult.FAILURE
        else:
            return LogRecordExportResult.SUCCESS

    @override
    def shutdown(self, timeout_millis: int = 30_000) -> None:
        """Close the exporter; each record is flushed synchronously."""
        del timeout_millis

    @override
    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        """Report success because writes are synchronous."""
        del timeout_millis
        return True


def _otel_trace_patcher(record: object) -> None:
    """Add current OTel trace identifiers to every Loguru record."""
    record_map = cast("dict[str, object]", record)
    extra = cast("dict[str, object]", record_map["extra"])
    extra["otelTraceID"] = "0"
    extra["otelSpanID"] = "0"
    extra["otelTraceSampled"] = False
    extra["otelServiceName"] = ""

    span = trace.get_current_span()
    span_context = span.get_span_context()
    if not span_context.is_valid:
        return
    extra["otelTraceID"] = format(span_context.trace_id, "032x")
    extra["otelSpanID"] = format(span_context.span_id, "016x")
    extra["otelTraceSampled"] = span_context.trace_flags.sampled
    provider = _STATE.tracer_provider
    if provider is not None:
        resource = provider.resource
        extra["otelServiceName"] = str(resource.attributes.get("service.name", ""))


def _configure_otel(app: str, oteldir: Path) -> logging.Handler:
    """Configure local OTel files and an optional OTLP log exporter."""
    oteldir.mkdir(parents=True, exist_ok=True)
    resource = Resource.create(
        {
            "service.name": app,
            "service.namespace": "pricing-perspective",
        }
    )
    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(
        SimpleLogRecordProcessor(_JsonFileLogExporter(oteldir / f"{app}.logs.jsonl"))
    )
    if endpoint := os.environ.get("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"):
        logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(OTLPLogExporter(endpoint=endpoint))
        )
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        SimpleSpanProcessor(FileSpanExporter(oteldir / f"{app}.traces.jsonl"))
    )
    trace.set_tracer_provider(tracer_provider)
    _STATE.logger_provider = logger_provider
    _STATE.tracer_provider = tracer_provider
    return LoggingHandler(level=logging.NOTSET, logger_provider=logger_provider)


def setup_logging(
    app: str,
    *,
    level: str = "INFO",
    logdir: Path | None = None,
    oteldir: Path | None = None,
) -> None:
    """Configure idempotent Loguru sinks and file-backed OTel providers."""
    if _STATE.configured:
        return

    if os.environ.get("TELEMETRY_DISABLE") == "1":
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            force=True,
        )
        _STATE.configured = True
        return

    otel_handler = _configure_otel(app, oteldir or Path("data/telemetry/otel"))
    logging.basicConfig(handlers=[_InterceptHandler()], level=level, force=True)
    logger.remove()
    logger.configure(extra={"app": app}, patcher=_otel_trace_patcher)
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> <level>{level: <7}</level> "
            "<cyan>{extra[app]}</cyan> <dim>{name}</dim> "
            "[trace_id={extra[otelTraceID]} span_id={extra[otelSpanID]}] - "
            "<level>{message}</level>"
        ),
    )
    resolved_logdir = logdir or Path("data/telemetry/logs")
    resolved_logdir.mkdir(parents=True, exist_ok=True)
    logger.add(
        resolved_logdir / f"{app}.log",
        level=level,
        rotation="20 MB",
        retention=5,
        enqueue=True,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} {level: <7} {extra[app]} {name} "
            "[trace_id={extra[otelTraceID]} span_id={extra[otelSpanID]}] - "
            "{message}"
        ),
    )
    logger.add(_OpenTelemetrySink(otel_handler).write, level=level)
    _STATE.configured = True


def _start_stage_span(app: str, stage: str) -> tuple[Span, Token[Context]]:
    """Start and attach one stage span to the current execution context."""
    if _STATE.tracer_provider is not None:
        tracer = _STATE.tracer_provider.get_tracer("dotell")
    else:
        tracer = trace.get_tracer("dotell")
    span = tracer.start_span("pricing-perspective.stage")
    token = context.attach(trace.set_span_in_context(span))
    span.set_attribute("pp.app", app)
    span.set_attribute("pp.stage", stage)
    return span, token


def _set_stage_span_attributes(span: Span, attributes: Mapping[str, object]) -> None:
    """Add scalar stage measurements to the active span."""
    for key, value in attributes.items():
        if value is None or not isinstance(value, (bool, float, int, str)):
            continue
        span.set_attribute(f"pp.{key}", value)


def _end_stage_span(span: Span, token: Token[Context]) -> None:
    """Detach and end a stage span."""
    context.detach(token)
    span.end()


def _force_flush_otel() -> None:
    """Flush providers without shutting them down for test reuse."""
    if _STATE.logger_provider is not None:
        _STATE.logger_provider.force_flush()
    if _STATE.tracer_provider is not None:
        _STATE.tracer_provider.force_flush()
