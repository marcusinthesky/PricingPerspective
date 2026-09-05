# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-aware DVCLive persistence for EttaX training metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast

from dvclive import Live

if TYPE_CHECKING:
    from types import TracebackType

type ParameterValue = (
    bool | float | int | str | list[ParameterValue] | dict[str, ParameterValue]
)

_EMA_DECAY = 0.98
_PLOTTED_METRICS = (
    "loss",
    "loss_ema",
    "gradient_norm",
    "learning_rate",
    "token_slots_per_second",
)


class TrainingTracker:
    """Persist one step-indexed metric series without creating DVC experiments."""

    def __init__(
        self,
        directory: str | Path,
        *,
        initial_step: int,
        parameters: dict[str, object],
    ) -> None:
        """Open or reconcile the metric series at the restored optimizer step."""
        self.directory = Path(directory)
        has_history = _reconcile_history(self.directory, initial_step)
        self._live = Live(
            dir=str(self.directory),
            resume=has_history,
            report=None,
            save_dvc_exp=False,
            dvcyaml=False,
        )
        if not has_history:
            self._live.log_params(
                cast("dict[str, ParameterValue]", parameters),
            )
        previous_ema = self._live.summary.get("loss_ema")
        self._loss_ema = (
            float(previous_ema) if isinstance(previous_ema, int | float) else None
        )
        previous_best = self._live.summary.get("best_loss_ema")
        self._best_loss_ema = (
            float(previous_best) if isinstance(previous_best, int | float) else None
        )
        previous_best_step = self._live.summary.get("best_step")
        self._best_step = (
            int(previous_best_step)
            if isinstance(previous_best_step, int | float)
            else initial_step
        )
        previous_probe_step = self._live.summary.get("checkpoint_probe_step")
        self._last_probe_step = (
            int(previous_probe_step)
            if isinstance(previous_probe_step, int | float)
            else None
        )
        previous_probe_best = self._live.summary.get("best_checkpoint_probe_loss")
        self._best_probe_loss = (
            float(previous_probe_best)
            if isinstance(previous_probe_best, int | float)
            else None
        )
        previous_probe_best_step = self._live.summary.get("best_checkpoint_probe_step")
        self._best_probe_step = (
            int(previous_probe_best_step)
            if isinstance(previous_probe_best_step, int | float)
            else initial_step
        )

    def __enter__(self) -> Self:
        """Return the active tracker."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Flush the latest summary even when training raises."""
        del exc_type, exc_value, traceback
        self._live.end()

    def record(
        self,
        *,
        step: int,
        loss: float,
        gradient_norm: float,
        learning_rate: float,
        token_slots_per_second: float,
    ) -> None:
        """Append one optimizer-step observation and refresh the summary."""
        self._loss_ema = (
            loss
            if self._loss_ema is None
            else _EMA_DECAY * self._loss_ema + (1.0 - _EMA_DECAY) * loss
        )
        if self._best_loss_ema is None or self._loss_ema < self._best_loss_ema:
            self._best_loss_ema = self._loss_ema
            self._best_step = step

        self._live.step = step
        values = {
            "loss": loss,
            "loss_ema": self._loss_ema,
            "gradient_norm": gradient_norm,
            "learning_rate": learning_rate,
            "token_slots_per_second": token_slots_per_second,
        }
        for name in _PLOTTED_METRICS:
            self._live.log_metric(name, values[name])
        self._live.summary.update(
            {
                "best_loss_ema": self._best_loss_ema,
                "best_step": self._best_step,
                "ema_decay": _EMA_DECAY,
            }
        )
        self._live.make_summary()

    def needs_checkpoint_probe(self, step: int) -> bool:
        """Return whether the restored step lacks its deterministic probe metric."""
        return self._last_probe_step != step

    def record_checkpoint_probe(self, *, step: int, loss: float) -> None:
        """Persist deterministic training-corpus loss at a checkpoint boundary."""
        if self._best_probe_loss is None or loss < self._best_probe_loss:
            self._best_probe_loss = loss
            self._best_probe_step = step
        self._last_probe_step = step
        self._live.step = step
        self._live.log_metric("checkpoint_probe_loss", loss)
        self._live.summary.update(
            {
                "best_checkpoint_probe_loss": self._best_probe_loss,
                "best_checkpoint_probe_step": self._best_probe_step,
                "checkpoint_probe_step": step,
            }
        )
        self._live.make_summary()

    def finalize(self, metrics: dict[str, float | int | str | bool]) -> None:
        """Merge terminal run metadata into DVCLive's DVC-readable summary."""
        self._live.summary.update(metrics)
        self._live.make_summary()


def _reconcile_history(directory: Path, checkpoint_step: int) -> bool:
    """Trim observations newer than the restored checkpoint before appending."""
    metrics_root = directory / "plots" / "metrics"
    histories: dict[str, list[tuple[int, float]]] = {}
    if metrics_root.is_dir():
        for path in sorted(metrics_root.rglob("*.tsv")):
            metric = path.relative_to(metrics_root).with_suffix("").as_posix()
            histories[metric] = _truncate_metric(path, metric, checkpoint_step)
    populated = {name: rows for name, rows in histories.items() if rows}
    if not populated:
        return False

    summary: dict[str, float | int] = {
        name: rows[-1][1] for name, rows in populated.items()
    }
    summary["step"] = max(rows[-1][0] for rows in populated.values())
    ema_rows = populated.get("loss_ema", [])
    if ema_rows:
        best_step, best_loss = min(ema_rows, key=lambda row: row[1])
        summary.update(
            {
                "best_loss_ema": best_loss,
                "best_step": best_step,
                "ema_decay": _EMA_DECAY,
            }
        )
    probe_rows = populated.get("checkpoint_probe_loss", [])
    if probe_rows:
        best_probe_step, best_probe_loss = min(probe_rows, key=lambda row: row[1])
        summary.update(
            {
                "best_checkpoint_probe_loss": best_probe_loss,
                "best_checkpoint_probe_step": best_probe_step,
                "checkpoint_probe_step": probe_rows[-1][0],
            }
        )
    _write_json_atomic(directory / "metrics.json", summary)
    return True


def _truncate_metric(
    path: Path,
    metric: str,
    checkpoint_step: int,
) -> list[tuple[int, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, dialect="excel-tab")
        rows = [
            (int(row["step"]), float(row[metric]))
            for row in reader
            if int(row["step"]) <= checkpoint_step
        ]
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, dialect="excel-tab", lineterminator="\n")
        writer.writerow(("step", metric))
        writer.writerows(rows)
    partial.replace(path)
    return rows


def _write_json_atomic(path: Path, values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(values, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)
