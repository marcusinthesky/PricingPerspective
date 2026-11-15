"""Focused tests for OpenTelemetry logging and stage correlation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _subprocess_environment() -> dict[str, str]:
    """Expose the checkout package to the child interpreter."""
    source = str(Path(__file__).parents[1] / "src")
    existing = os.environ.get("PYTHONPATH")
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(value for value in (source, existing) if value),
    }


def test_logging_and_stage_export_share_trace_context(tmp_path: Path) -> None:
    """Log records emitted inside a stage carry the stage span identifiers."""
    script = """
from pathlib import Path
import sys
from loguru import logger
from dotell import begin, setup_logging

root = Path(sys.argv[1])
setup_logging("pytest", logdir=root / "logs", oteldir=root / "otel")
finish = begin("otel-stage", app="pytest", sink=root / "runs.jsonl", interval=0.01)
logger.info("inside stage")
finish()
"""
    subprocess.run(  # noqa: S603
        [sys.executable, "-c", script, str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_environment(),
    )

    log_path = tmp_path / "otel" / "pytest.logs.jsonl"
    trace_path = tmp_path / "otel" / "pytest.traces.jsonl"
    log_records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert trace_path.read_text().strip()
    stage_log = next(
        record for record in log_records if record["body"] == "inside stage"
    )
    assert stage_log["resource"]["service.name"] == "pytest"
    assert stage_log["trace_id"] != "0" * 32
    assert stage_log["span_id"] != "0" * 16
