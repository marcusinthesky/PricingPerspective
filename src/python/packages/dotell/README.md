# `dotell` — application observability

Shared application logging, OpenTelemetry correlation/export, and per-stage
resource/timing telemetry for the DVC pipeline. `loguru` remains the
human-readable console/file API; the package translates those records into
OpenTelemetry logs and creates one OpenTelemetry stage span per instrumented
command. Local JSONL files remain the default, so no collector is required.

## What it does

| Concern | Function | Output |
|---|---|---|
| Application logging | `setup_logging(app)` | stdlib `logging` → Loguru; colourised console + rotating file sink at `data/telemetry/logs/<app>.log`; OTel JSONL at `data/telemetry/otel/<app>.logs.jsonl`. |
| Stage tracing | `begin(stage, app=…)` | One `pricing-perspective.stage` span at `data/telemetry/otel/<app>.traces.jsonl`, with scalar stage measurements attached at finalization. |
| Resource + timing | `begin(stage, app=…)` | Samples the **process tree** (command + forked workers) at low frequency; appends one JSON row per run to `data/telemetry/runs.jsonl`. |

All outputs live under `data/telemetry/`, which is git-ignored
(`/data/*`, `*.jsonl`) and is **not** a DVC output — telemetry never
invalidates the DAG.

## Integration (already wired)

Each app's `cli.py` has a Typer callback:

```python
from dotell import begin, setup_logging


@app.callback()
def _bootstrap(ctx: typer.Context) -> None:
    setup_logging("pipeline")
    if ctx.invoked_subcommand is not None:
        ctx.call_on_close(begin(ctx.invoked_subcommand, app="pipeline"))
```

`ctx.invoked_subcommand` is the DVC stage name; `ctx.call_on_close` fires the
finalizer when the command tears down. EttaX registers the same lifecycle at
its root callback and at the nested `vintage` callback, so `vintage.train`
and `vintage.run` remain distinct stages.

Every Loguru record emitted inside a stage carries the active `trace_id` and
`span_id` in console/file output and in the OTel log record. Set
`OTEL_EXPORTER_OTLP_LOGS_ENDPOINT` to additionally send OTel logs through the
standard OTLP/HTTP exporter; local JSONL export remains enabled.

> Coverage note: callbacks live inside the Python apps, so they cover the
> Python DVC stages in `pipeline`, `simulation`, and `ettax`. The LaTeX build
> (`tectonic` / `uv run --script`) stages are not instrumented.

## Querying runs

`runs.jsonl` is a DuckDB-native source:

```sql
SELECT stage, count(*) AS runs,
       median(wall_s)        AS median_wall_s,
       max(peak_rss_gb)      AS peak_gib,
       round(avg(mean_cpu_pct)) AS avg_cpu_pct
FROM read_json_auto('data/telemetry/runs.jsonl')
GROUP BY stage
ORDER BY median_wall_s DESC;
```

Rows carry `git_sha`, so you can trace how a stage's cost shifts across commits.

### `mean_cpu_pct` is not comparable across the process-tree fix

The sampler rebuilt its `psutil.Process` objects on every tick, and
`Process.cpu_percent(interval=None)` measures against the previous call *on that
object* — so it returned `0.0` for every child, on every sample, forever. Only
the parent, primed once at startup, was ever counted.

Any stage doing its work in child processes therefore logged near-idle CPU while
saturating cores. That is every `simulation` stage using
`harness.parallel.parallel_map`, plus `compute-long-short` via
`pipeline.stages.papers.paper2._hedge.tasks` — measured at **8.3 % against a
known 4-core burn, versus 380.7 % once the objects are held across samples**.

Rows written before the fix keep the error; nothing rewrites history. When
ranking with `mean_cpu_pct`, check whether the row predates it, and prefer the
`xla_compiles*` columns for the compute-vs-dispatch question they answer
directly. `peak_rss_gb` is unaffected — `memory_info()` needs no priming — and
stays joinable across the change.

## Knobs (environment variables)

| Variable | Effect |
|---|---|
| `TELEMETRY_DISABLE=1` | Disables sampling, spans, and OTel export; logging falls back to plain `basicConfig`. |
| `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT=<url>` | Opt-in OTLP/HTTP log export in addition to local OTel JSONL. |
| `PP_TELEMETRY_PROFILE=1` | Opt-in `cProfile` capture per stage (see below). Unset, the emitted row is byte-identical to today's. |
| `PP_TELEMETRY_JAX=1` | Opt-in caller-owned JAX diagnostic fields (see below). Unset, no JAX keys are emitted. |
| `PP_JAX_CACHE_DIR=<path>` | Persistent XLA compilation-cache location (default `.jax_cache`, resolved absolute). |
| `PP_JAX_CACHE_MAX_GIB=<n>` | LRU eviction ceiling for that cache (default `2`); `0` or negative disables eviction. |
| `PP_JAX_CACHE_STRICT=1` | Raise on a compilation-cache read/write error instead of warning and recompiling. Set for CI in `.github/workflows/ci-pipeline.yml`. |

## JAX compilation cache (`configure_jax_compilation_cache`)

Every DVC stage is a cold `uv run` process and the numerical stages fan out to
`spawn` workers that cold-import their stage module, so without a shared on-disk
cache each jitted kernel is recompiled per stage *and* per worker (t05.7 measured
`simulate-size-power` at 1029 s → 496 s from this alone). Both apps configure it
through the single site here — `pipeline.jax_cache` and `simulation.backend` are
thin adapters (t63); before that they were separate implementations that could
drift apart.

```python
from dotell import configure_jax_compilation_cache

configure_jax_compilation_cache()  # $PP_JAX_CACHE_DIR, else .jax_cache
configure_jax_compilation_cache(some_dir)  # explicit location wins
```

Call it before the first jit, and again inside every `spawn` worker — a spawned
interpreter inherits none of the parent's process-global JAX flags. The resolved
absolute path is republished in `PP_JAX_CACHE_DIR_RESOLVED`, an internal variable
a child prefers over its own default, so a parent and its workers share one
directory even though the two apps resolve it differently (pipeline through
pydantic settings, which also read `.env`; simulation from the environment only).
Writing back to `PP_JAX_CACHE_DIR` itself would invert the process-env-beats-
`.env` precedence `pipeline/ARCHITECTURE.md` guarantees.

A cached executable is keyed by HLO + platform + jaxlib version and is
bit-identical to a fresh compile, so this changes *when* code compiles, never
*what* it computes. The cache directory is gitignored and must never appear as a
`dvc.yaml` dep/out — its churn would bust stage hashes on every run.

Eviction needs `filelock` (JAX guards the LRU with a file lock and raises at
first compile without it); if it is missing, the cap degrades to unbounded
rather than failing the stage. Strict mode is off by default so a read-only or
full cache dir costs a recompile instead of the run — turn it on in CI, where a
silent full recompile would otherwise show up only as an elevated observed
`xla_compiles` in the telemetry rows (see below).

> **Why the cap needs a migration.** JAX writes a `<key>-atime` sidecar beside
> each entry only while eviction is enabled, but its eviction pass reads one for
> *every* `*-cache` file it finds. Turning the cap on over a directory populated
> without it therefore makes **every** cache write fail — as a warning by
> default, fatally under strict mode — and it never recovers, because nothing
> prunes the sidecar-less entries. `configure_jax_compilation_cache` backfills
> the missing sidecars first (and disables the cap if it cannot), so this is
> handled; it is documented because the failure mode is silent and looks like an
> ordinary cache miss.

## Observed XLA compilations (always on)

Every row carries what **JAX itself reported**, with no opt-in and no caller
cooperation:

| Field | Meaning |
|---|---|
| `xla_compiles` | Backend compilations JAX performed during the stage, in this process. |
| `xla_compile_s` | Seconds spent inside those compilations. |
| `xla_compiles_per_wall_s` | `xla_compiles / wall_s` of the same row — the recompilation-churn alarm. |

The counter is a `jax.monitoring.register_event_duration_secs_listener`
callback that matches JAX's `/jax/core/compile/backend_compile_duration` event
(counted by exact name, not by substring — the trace/MLIR and
`/jax/compilation_cache/*` events also contain "compile"). JAX fires it inside
`_cached_compilation`, which is `weakref_lru_cache`d, so **one event = one
in-process jit-cache miss**. That is exactly the shape-churn signature: a stage
that slices its design matrix per bootstrap draw compiles the *same* kernel once
per draw. The measured case (t65, `p1_dyadic_confound`) was 47 compiles over 15
draws at 197.1 ms/draw, versus 0 compiles and 1.5 ms/draw once the shape was
held fixed — a 136× difference that no existing field could surface.

Healthy stages compile a fixed set of kernels once, so `xla_compiles_per_wall_s`
decays toward 0 as the stage runs; a stage that holds a high rate for its whole
wall time is recompiling in a loop:

```sql
SELECT stage, max(xla_compiles) AS compiles, max(xla_compiles_per_wall_s) AS per_s
FROM read_json_auto('data/telemetry/runs.jsonl')
WHERE xla_compiles IS NOT NULL
GROUP BY stage ORDER BY per_s DESC;
```

Workers are counted, and counted **separately**:

- `xla_compiles` / `xla_compile_s` — this process.
- `xla_compiles_workers` / `xla_compile_s_workers` — its `spawn` workers, summed
  from sidecars each writes at interpreter exit.
- `xla_worker_reports` — how many workers actually reported.

A worker's listener lives in its own interpreter, so before this a stage that ran
*everything* in a pool showed its churn as ~0: `mc_gmm_wolak` reported 2 compiles
across 1073 s. Keep the two columns apart when reading — a fanned-out stage puts
its churn in the `_workers` pair and near-zero in the first, and worker compile
*seconds* are a sum over concurrent processes, so they are not comparable to
`wall_s` the way the parent's are.

Two honest limits, both readable off the row:

- **Exit-time export.** A worker killed outright never runs its hook, so
  `xla_worker_reports` is what arrived, not what ran. A stage known to fan out
  that reports `0` was not measured — it did not necessarily compile nothing.
- **`null` means "not observed", never "zero".** The listener attaches at
  `begin()` if JAX is already imported, else at `configure_jax_compilation_cache()`
  — the site every stage must reach before its first jit. If JAX was imported
  during a stage but the listener never attached, the three fields are `null`
  rather than a misleading `0`. Importing `dotell` still never imports JAX.

## JAX diagnostics (explicit, opt-in)

Beyond the compilation count above, JAX exposes no stable public callback that
attributes work to a *named kernel* as a trace, cache hit/miss, device
execution, or host synchronization. `dotell` does **not** infer those
states from private JAX internals or cache filenames. A stage that needs that
attribution can mark the boundaries it owns:

```python
from dotell import jax_diagnostics

diagnostics = jax_diagnostics()
with diagnostics.compile(
    "jcor.energy", x, static={"n_permutations": n_perm}, cache="miss"
):
    compiled = jitted_energy.lower(x).compile()
with diagnostics.execution("jcor.energy", x, static={"n_permutations": n_perm}):
    result = compiled(x)
with diagnostics.sync("jcor.energy", result):
    result.block_until_ready()
if irregular_groups:
    diagnostics.fallback("jcor.energy.irregular", x)
```

`compile` also records a raised compilation attempt as a failure. Record a cache
outcome only when the caller knows it; omit `cache=` otherwise. A parent process
that catches a failed worker may call `diagnostics.worker_failure(...)`.

The resulting optional row fields are compile/trace, execution, and sync seconds;
compile attempts/failures; confirmed cache hits/misses; fallback and worker-failure
counts; and `jax_events`. Each event contains a caller-chosen kernel identifier and
a stable signature made only from shape, dtype, and hashed static values. Array
values, static values, exceptions, and worker payloads are never written. Old rows
are null for these fields in the DuckDB telemetry marts.

The hook is local to the process running `begin()`: it does not automatically
cross process-pool boundaries, and it does not distinguish asynchronous device
execution without the explicit `sync` block. JAX remains an optional package
extra; importing `dotell` for logging and stage tracing does not import
JAX.

## Diagnosing a slow stage

1. Find regressed stages with the telemetry regression query in the `src/python/apps/harnessme`
   harness (compares `wall_s`/`peak_rss_gb` across `git_sha`s in `runs.jsonl`).
2. Rerun just that stage directly (bypassing `dvc repro`) with profiling on, e.g.
   for the `permutation_tests` stage (see `dvc.yaml`):

   ```bash
   PP_TELEMETRY_PROFILE=1 uv run --project src/python pipeline compute-permutation-tests
   ```

3. Render the resulting `.pstats` file:

   ```bash
   uvx gprof2dot -f pstats data/telemetry/profiles/pipeline__compute-permutation-tests__<ts>.pstats | dot -Tsvg -o profile.svg
   # or, text summary:
   uv run --project src/python python -m pstats data/telemetry/profiles/pipeline__compute-permutation-tests__<ts>.pstats
   ```

`cProfile` only captures the main process — the sampler thread's own overhead
is negligible but unprofiled, and any joblib/process-pool child processes are
**not** captured (profiling is inherently per-process). For whole-process-tree
sampling of a live run, use `py-spy` instead:

```bash
py-spy dump --pid <pid>                          # one-shot stack dump
py-spy record -s -o flame.svg --pid <pid>        # flamegraph over time
```
