"""Persistent on-disk XLA compilation-cache policy (t05.7, consolidated in t63).

Every DVC stage is a cold ``uv run`` process, and the numerical stages fan out
to ``spawn`` workers that cold-import their stage module -- so without a shared
on-disk cache each jitted kernel is recompiled per stage *and* per worker. A
cached executable is keyed by HLO + platform + jaxlib version and is
bit-identical to a fresh compile, so this changes *when* code compiles, never
*what* it computes (exempt from the t05 parity gate).

This module is the single config site for both apps. ``pipeline.jax_cache`` and
``simulation.backend`` are thin adapters over it; before t63 they were separate
implementations whose cache dirs could diverge (one resolved through pydantic
settings at call time, the other read the env var once at *module import*, so a
later ``monkeypatch.setenv`` silently no-opped).

It deliberately configures ONLY the compilation cache -- never global dtype
policy (``jax_enable_x64``), which is numerics-affecting and owned per stage.

The cache dir is kept OUTSIDE the DVC graph and gitignored -- never a
``dvc.yaml`` dep/out, whose churn would otherwise bust stage hashes every run.
"""

from __future__ import annotations

import os
import time
from importlib.util import find_spec
from pathlib import Path

from loguru import logger

from . import _xla_compiles

CACHE_DIR_ENV = "PP_JAX_CACHE_DIR"
MAX_GIB_ENV = "PP_JAX_CACHE_MAX_GIB"
STRICT_ENV = "PP_JAX_CACHE_STRICT"
# Internal, deliberately NOT the documented knob. The resolved absolute path is
# republished here for spawn children so a parent and its workers share one
# directory even when they resolve it differently (the pipeline app goes through
# pydantic settings, which also read `.env`; simulation reads the environment
# only). Writing back to `PP_JAX_CACHE_DIR` itself would invert the precedence
# `pipeline/ARCHITECTURE.md` guarantees -- process env beats `.env` -- so a
# child could never honour a `.env` value once a parent had exported one.
RESOLVED_DIR_ENV = "PP_JAX_CACHE_DIR_RESOLVED"

_DEFAULT_CACHE_DIR = Path(".jax_cache")
# Bounds unbounded growth across jaxlib bumps (each bump invalidates every
# entry by design, and the stale generation is never pruned otherwise). Far
# above observed steady-state use (~5 MiB / ~200 entries), so eviction should
# never fire in practice -- this is a ceiling, not a working-set target.
_DEFAULT_MAX_GIB = 2.0
# JAX's own default is 1.0s. Lowered because the fan-out's win comes from many
# individually cheap kernels compiled once per worker rather than once total.
_MIN_COMPILE_TIME_SECS = 0.5
# JAX sentinel: -1 disables eviction (unbounded). 0 would disable caching
# outright, so the GiB->bytes mapping below must never produce it.
_UNLIMITED = -1
_BYTES_PER_GIB = 2**30
_TRUTHY_ENV_VALUES = frozenset({"1", "true"})
# JAX's on-disk cache layout: one `<key>-cache` blob, and -- only once eviction
# has ever been enabled -- a sibling `<key>-atime` holding an 8-byte
# little-endian ns timestamp. See `jax._src.lru_cache`; the names are stable
# but private, so `_backfill_atimes` treats an unexpected layout as a no-op
# rather than asserting on it.
_CACHE_SUFFIX = "-cache"
_ATIME_SUFFIX = "-atime"
_ATIME_BYTES = 8


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY_ENV_VALUES


def _max_size_bytes() -> int:
    """Resolve the LRU eviction ceiling in bytes, or ``-1`` for unbounded."""
    raw = os.environ.get(MAX_GIB_ENV, "").strip()
    if not raw:
        gib = _DEFAULT_MAX_GIB
    else:
        try:
            gib = float(raw)
        except ValueError:
            logger.warning(
                "{}={!r} is not a number; falling back to {} GiB",
                MAX_GIB_ENV,
                raw,
                _DEFAULT_MAX_GIB,
            )
            gib = _DEFAULT_MAX_GIB
    if gib <= 0:
        return _UNLIMITED
    if find_spec("filelock") is None:
        # JAX guards LRU eviction with a file lock and raises at first compile
        # if `filelock` is absent. Degrade to unbounded rather than turning a
        # missing optional dependency into a failed stage.
        logger.warning("filelock is not installed; JAX cache size cap disabled")
        return _UNLIMITED
    return int(gib * _BYTES_PER_GIB)


def _backfill_atimes(cache_dir: Path) -> int:
    """Give every pre-eviction cache entry the access-time sidecar LRU needs.

    JAX writes a ``<key>-atime`` file beside each entry only while eviction is
    enabled, but ``LRUCache._evict_if_needed`` reads that sidecar for *every*
    ``*-cache`` file it finds. So a directory populated before the size cap was
    turned on (or by any process that opted out of it) makes every subsequent
    write raise ``FileNotFoundError`` -- caught and downgraded to a warning by
    default, fatal under strict mode. Either way the cache silently stops
    accepting new entries and never recovers, since nothing prunes the entries
    that are missing a sidecar.

    Stamping them as accessed now is exactly what JAX does on a cache hit, and
    is honest: they are live entries this process is about to use. Runs on
    every configure so a later opt-out cannot re-poison the directory.

    Returns:
        The number of sidecars written.

    """
    timestamp = time.time_ns().to_bytes(_ATIME_BYTES, "little")
    written = 0
    for entry in cache_dir.glob(f"*{_CACHE_SUFFIX}"):
        atime_path = entry.with_name(
            f"{entry.name.removesuffix(_CACHE_SUFFIX)}{_ATIME_SUFFIX}"
        )
        if atime_path.exists():
            continue
        # Written via a unique temp file and an atomic rename: spawned workers
        # backfill concurrently, and a torn 8-byte read would be parsed as
        # timestamp 0 and evicted first.
        staging = atime_path.with_name(f"{atime_path.name}.{os.getpid()}.tmp")
        staging.write_bytes(timestamp)
        staging.replace(atime_path)
        written += 1
    return written


def configure_jax_compilation_cache(cache_dir: Path | str | None = None) -> Path:
    """Point JAX at the shared on-disk compilation cache (idempotent).

    Must run before the first jit in the process. Call it again in every
    ``spawn`` worker: a spawned interpreter inherits none of the parent's
    process-global JAX flags.

    Args:
        cache_dir: Cache location. Defaults to the parent's already-resolved
            directory if this is a spawned child, else ``$PP_JAX_CACHE_DIR``,
            else ``.jax_cache``. Resolved to an absolute path at call time
            (never at import time) and republished in ``$PP_JAX_CACHE_DIR_RESOLVED``
            for children.

    Returns:
        The absolute cache directory now in effect.

    """
    # Imported inside the function so that importing ``dotell`` -- which
    # every stage does for logging alone -- does not itself pull in JAX.
    import jax  # noqa: PLC0415

    # This is the one site every stage is contractually required to reach
    # *before* its first jit, so it is where the observed-compilation listener
    # can be attached without missing events. Idempotent; per-process, so a
    # spawn worker counts only its own compiles (see `_xla_compiles`).
    _xla_compiles.attach()
    # ...and therefore also the one site guaranteed to run in every spawned
    # worker, which is what lets those per-worker counts be exported back to the
    # parent's row without either pool implementation knowing telemetry exists.
    # A no-op in the process that opened the sink.
    _xla_compiles.register_worker_export()

    resolved = Path(
        cache_dir
        if cache_dir is not None
        else os.environ.get(RESOLVED_DIR_ENV)
        or os.environ.get(CACHE_DIR_ENV)
        or _DEFAULT_CACHE_DIR
    ).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    os.environ[RESOLVED_DIR_ENV] = str(resolved)

    jax.config.update("jax_compilation_cache_dir", str(resolved))
    # NOT "no floor": JAX reads 0 as "you have not chosen", and `_initialize_cache`
    # replaces it with `default_min_cache_entry_size()` for the backing filesystem.
    # Set a positive value only to override that filesystem-aware default.
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    jax.config.update(
        "jax_persistent_cache_min_compile_time_secs", _MIN_COMPILE_TIME_SECS
    )
    max_size = _max_size_bytes()
    if max_size != _UNLIMITED:
        try:
            backfilled = _backfill_atimes(resolved)
        except OSError as error:
            # A cap that cannot be made consistent is worse than no cap: it
            # would block every cache write. Fall back to unbounded.
            logger.warning("JAX cache size cap disabled: {}", error)
            max_size = _UNLIMITED
        else:
            if backfilled:
                logger.debug(
                    "backfilled {} JAX cache access-time sidecars in {}",
                    backfilled,
                    resolved,
                )
    jax.config.update("jax_compilation_cache_max_size", max_size)
    # Off by default: a read-only or full cache dir degrades to a warning and a
    # full recompile rather than killing the stage. Opt in under CI to make
    # that silent slowdown -- otherwise visible only as an elevated observed
    # `xla_compiles` / `xla_compiles_per_wall_s` in the telemetry row -- a hard
    # failure instead.
    jax.config.update("jax_raise_persistent_cache_errors", val=_env_enabled(STRICT_ENV))
    return resolved
