"""Pipeline adapter over the shared XLA compilation-cache policy (t05.7/t63).

The policy itself -- which flags are set, why the cache dir stays outside the
DVC graph, why a cached executable is exempt from the t05 parity gate -- lives
in :mod:`dotell._jax_cache`, shared with the simulation app. This module
only supplies the pipeline's own resolution of the cache location: pydantic
``runtime_settings()``, which reads ``PP_JAX_CACHE_DIR`` from the environment or
``.env``.

Call :func:`configure_persistent_cache` at stage-module import time (mirroring
worker modules) so spawned workers re-run it
on cold import.
"""

from __future__ import annotations

from dotell import configure_jax_compilation_cache

from pipeline.io.settings import runtime_settings


def configure_persistent_cache() -> None:
    """Point JAX at the shared on-disk compilation cache (idempotent)."""
    configure_jax_compilation_cache(runtime_settings().jax_cache_dir)
