# SPDX-License-Identifier: Apache-2.0
"""Sequential 700M-content-token queue for the outcome-free frozen recipe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ettax.pilot.training import run_trial

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ettax.pilot.config import PilotArm, PilotConfig


def run_frozen_queue(
    config: PilotConfig, *, require_gpu: bool = True
) -> dict[str, object]:
    """Verify the freeze, forecast runtime, then train V0/V1/V3 sequentially."""
    freeze_path = config.run_root / "frozen-recipe.json"
    if not freeze_path.is_file():
        raise FileNotFoundError(f"missing outcome-free recipe freeze: {freeze_path}")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    _verify_freeze(config, freeze)
    run_id = str(freeze["run_id"])
    final_root = config.work_root.parent / "runs" / run_id
    forecast = runtime_forecast(config, freeze)
    _write_json_atomic(final_root / "runtime-forecast.json", forecast)

    summaries: dict[str, object] = {}
    for raw_arm in ("V0", "V1", "V3"):
        arm = cast("PilotArm", raw_arm)
        summaries[arm] = run_trial(
            config,
            str(freeze["trial"]),
            arm=arm,
            content_budget=int(freeze["final_content_tokens_per_arm"]),
            context_winner=int(freeze["maximum_context"]),
            require_gpu=require_gpu,
            output_root=final_root / arm,
            prepared_root=config.prepared_root,
        )
    report: dict[str, object] = {
        "arms": summaries,
        "forecast": forecast,
        "frozen_recipe": str(freeze_path),
        "frozen_recipe_sha256": _sha256(freeze_path),
        "run_id": run_id,
    }
    _write_json_atomic(final_root / "summary.json", report)
    return report


def runtime_forecast(
    config: PilotConfig, freeze: Mapping[str, object]
) -> dict[str, object]:
    """Forecast the full queue from measured pilot content throughput."""
    trial = str(freeze["trial"])
    rates: list[float] = []
    sources: list[str] = []
    for arm in ("V0", "V1"):
        path = config.trial_root(cast("PilotArm", arm), trial) / "summary.json"
        if path.is_file():
            summary = json.loads(path.read_text(encoding="utf-8"))
            rates.append(float(summary["content_tokens_per_second_this_run"]))
            sources.append(str(path))
    if not rates:
        raise FileNotFoundError(
            "no pilot throughput is available for the frozen recipe"
        )
    conservative_rate = min(rates)
    per_arm = int(cast("int", freeze["final_content_tokens_per_arm"]))
    seconds = 3 * per_arm / conservative_rate
    return {
        "arms": 3,
        "conservative_content_tokens_per_second": conservative_rate,
        "content_tokens_per_arm": per_arm,
        "estimated_gpu_hours": seconds / 3_600,
        "estimated_gpu_seconds": seconds,
        "pilot_sources": sources,
        "sequential": True,
    }


def source_tree_sha256(app_root: Path | None = None) -> str:
    """Hash runtime Python and TOML inputs independently of Git cleanliness."""
    root = app_root or Path(__file__).resolve().parents[3]
    candidates = sorted(
        (
            *(root / "src").rglob("*.py"),
            *(root / "configs").rglob("*.toml"),
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not candidates:
        raise FileNotFoundError(f"no EttaX source inputs found under {root}")
    digest = hashlib.sha256()
    for path in candidates:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _verify_freeze(config: PilotConfig, freeze: Mapping[str, object]) -> None:
    expected = {
        "pilot_id": config.pilot_id,
        "prepared_manifest_sha256": _sha256(config.prepared_root / "manifest.json"),
        "probe_manifest_sha256": _sha256(config.probe_manifest_path),
        "selection_report_sha256": _sha256(Path(str(freeze["selection_report"]))),
        "source_tree_sha256": source_tree_sha256(),
        "tokenizer_sha256": _sha256(config.tokenizer_path),
    }
    mismatches = {
        key: (freeze.get(key), value)
        for key, value in expected.items()
        if freeze.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"frozen recipe inputs changed: {mismatches}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.replace(path)
