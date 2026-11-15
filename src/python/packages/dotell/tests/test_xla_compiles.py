"""Unit tests for observed XLA compilation counting.

The regression these exist for: `statadic.numerics.linalg.dyadic` slices its design
matrix by an active-set mask, so every bootstrap draw handed the jitted solver a
different row count, busting the in-process jit cache and recompiling per draw
(197.1 ms/draw against 1.5 ms/draw at a fixed shape). Nothing in the telemetry
row could show it -- the `jax_cache_misses` field is a caller assertion and had
zero call sites. `test_varying_shapes_recompile_per_distinct_shape` is that bug
in miniature.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import pytest
from dotell import begin
from dotell._xla_compiles import (
    _BACKEND_COMPILE_EVENT,
    attach,
    attached,
    snapshot,
)

if TYPE_CHECKING:
    from pathlib import Path

# Deliberately tiny: this measures compilation *events*, not numerics.
_SHAPES = (3, 4, 5, 6)
_REPEATS = 5
#: Generous: the fan-out probe cold-starts two spawned interpreters and JAX in
#: each, so it is bounded well above its real cost rather than tuned to it.
_PROBE_TIMEOUT_S = 300


def _sum_of_squares(matrix: jnp.ndarray) -> jnp.ndarray:
    return (matrix * matrix).sum()


def test_backend_compile_event_name_is_current() -> None:
    """Guard the one JAX-internal string this module depends on.

    The listener API is public, but the event *name* is defined by
    ``jax._src.dispatch.BACKEND_COMPILE_EVENT``. If a JAX bump renames it the
    counter would silently report 0 forever, so pin it here rather than trust it.
    """
    from jax._src.dispatch import BACKEND_COMPILE_EVENT  # noqa: PLC0415

    assert BACKEND_COMPILE_EVENT == _BACKEND_COMPILE_EVENT


def test_varying_shapes_recompile_per_distinct_shape() -> None:
    """Shape churn compiles once per distinct shape; a fixed shape compiles once.

    This is the dyadic-bootstrap failure mode: same kernel, same work, one
    recompile per call because the row count moved.
    """
    assert attach()
    # Materialised up front: `jnp.ones` is itself jitted and compiles once per
    # new shape, which would otherwise be counted as the kernel's own churn.
    inputs = [jnp.ones((rows, 2)) for rows in _SHAPES]

    varying = jax.jit(_sum_of_squares)
    before_varying = snapshot()
    for operand in inputs:
        varying(operand).block_until_ready()
    varying_compiles = snapshot().compiles - before_varying.compiles

    fixed = jax.jit(_sum_of_squares)
    fixed(inputs[0]).block_until_ready()  # warm up
    before_fixed = snapshot()
    for _ in range(_REPEATS):
        fixed(inputs[0]).block_until_ready()
    fixed_compiles = snapshot().compiles - before_fixed.compiles

    assert varying_compiles == len(_SHAPES)
    assert fixed_compiles == 0


def test_compile_seconds_accumulate_with_compiles() -> None:
    """A counted compile also contributes non-negative observed compile time."""
    assert attach()
    before = snapshot()
    jax.jit(_sum_of_squares)(jnp.ones((7, 3))).block_until_ready()
    after = snapshot()

    assert after.compiles > before.compiles
    assert after.seconds >= before.seconds


def test_row_reports_observed_compiles_by_default(tmp_path: Path) -> None:
    """Compiles inside a stage land in the row without any opt-in env var."""
    sink = tmp_path / "runs.jsonl"
    finalize = begin("xla-stage", app="pytest", sink=sink, interval=0.01)
    # A freshly created jitted callable has its own cache, so this always
    # compiles regardless of what earlier tests warmed.
    jax.jit(_sum_of_squares)(jnp.ones((9, 2))).block_until_ready()
    finalize()

    row = json.loads(sink.read_text(encoding="utf-8").strip())
    assert attached()
    assert row["xla_compiles"] >= 1
    assert row["xla_compile_s"] >= 0.0
    assert row["xla_compiles_per_wall_s"] > 0.0


def test_row_reports_zero_compiles_for_a_jax_free_stage(tmp_path: Path) -> None:
    """A stage that compiles nothing reports 0, not a stale cumulative total."""
    sink = tmp_path / "runs.jsonl"
    finalize = begin("quiet-stage", app="pytest", sink=sink, interval=0.01)
    finalize()

    row = json.loads(sink.read_text(encoding="utf-8").strip())
    assert row["xla_compiles"] == 0
    assert row["xla_compiles_per_wall_s"] == pytest.approx(0.0)


def test_importing_dotell_does_not_import_jax() -> None:
    """The package must stay JAX-free at import: every stage imports it for logs."""
    probe = (
        "import sys, dotell;"
        " assert 'jax' not in sys.modules, sorted(sys.modules)[:0] or 'jax imported'"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)  # noqa: S603


_WORKER_PROBE = '''
import concurrent.futures as cf
import multiprocessing as mp
import sys
from pathlib import Path

from dotell import begin, configure_jax_compilation_cache


def work(size):
    """Compile one distinct shape inside a spawned worker."""
    configure_jax_compilation_cache()
    import jax
    import jax.numpy as jnp

    return float(jax.jit(lambda x: (x * x).sum())(jnp.ones((size, 2))))


if __name__ == "__main__":
    sink = Path(sys.argv[1])
    finalize = begin("fanned-out-stage", app="probe", sink=sink, interval=0.01)
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=2, mp_context=ctx) as pool:
        list(pool.map(work, [3, 4, 5, 6]))
    finalize()
'''


def test_row_reports_compiles_from_spawned_workers(tmp_path: Path) -> None:
    """A stage that fans all its work out still reports the pool's compiles.

    The Monte-Carlo stages run *everything* in ``spawn`` workers, so before the
    sidecar export their rows showed ~0 compiles no matter how much churn the
    pool suffered -- ``mc_gmm_wolak`` reported 2 compiles across 1073 seconds.
    """
    script = tmp_path / "fanout_probe.py"
    script.write_text(_WORKER_PROBE, encoding="utf-8")
    sink = tmp_path / "runs.jsonl"
    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(script), str(sink)],
        capture_output=True,
        text=True,
        check=False,
        timeout=_PROBE_TIMEOUT_S,
    )
    assert completed.returncode == 0, completed.stderr

    row = json.loads(sink.read_text(encoding="utf-8").strip())
    assert row["xla_worker_reports"] >= 1, (
        f"no worker exported its totals: {row}\n{completed.stderr}"
    )
    assert row["xla_compiles_workers"] >= 1, f"worker compiles went unrecorded: {row}"
    assert row["xla_compile_s_workers"] > 0.0


def test_row_reports_no_worker_activity_for_a_single_process_stage(
    tmp_path: Path,
) -> None:
    """``xla_worker_reports`` separates "pool compiled nothing" from "no pool"."""
    sink = tmp_path / "runs.jsonl"
    finalize = begin("solo-stage", app="pytest", sink=sink, interval=0.01)
    # Its own computation, not `_sum_of_squares`: JAX caches *backend
    # compilation* on the jaxpr and its avals process-globally, so re-jitting a
    # shape another test already compiled fires no event and this would read as
    # a regression in the parent counter.
    jax.jit(lambda x: (x - x.mean()).max())(jnp.ones((13, 11))).block_until_ready()
    finalize()

    row = json.loads(sink.read_text(encoding="utf-8").strip())
    assert row["xla_worker_reports"] == 0
    assert row["xla_compiles_workers"] == 0
    assert row["xla_compiles"] >= 1
