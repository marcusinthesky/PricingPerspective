"""JAX backend configuration for the simulation package.

This module provides a single entry-point, :func:`configure`, that must be
called **before any other JAX import or computation** in a process.  JAX
stores precision and platform choices in process-global flags; calling
``configure`` after the first JAX operation is undefined behaviour (JAX
will warn and may silently ignore the request).

Cross-backend honesty note
--------------------------
Results produced on CPU and GPU are **not bit-identical**.  XLA lowers
reductions differently across devices, and floating-point addition is not
associative.  Certificate stages (variance floor, hedging bounds) should
always run on CPU with float64 and a tolerance of ≥1e-9.  Statistical
stages (size, power) may safely use float32 on GPU; the target invariants
(violation rate, coverage) are tolerance-aware.
"""

from __future__ import annotations

from typing import Literal, NamedTuple, cast

from dotell import configure_jax_compilation_cache

# Persistent on-disk JAX compilation cache (t05.7). Every DVC stage is a cold
# `uv run` process; without a shared cache directory, every shape-specialized
# replication cell recompiles in each stage process. A cached executable is
# keyed by HLO + platform + jaxlib version and is
# bit-identical to a fresh compile, so this is exempt from the t05 parity gate.
# Kept OUTSIDE any DVC-tracked path and gitignored -- it must never become a
# `dvc.yaml` dep/out (its churn would otherwise bust stage hashes on every run).
#
# The policy (which flags, the size cap, strict mode) is shared with the
# pipeline app in `dotell._jax_cache`; t63 replaced this module's private
# copy with a delegation. Location comes from `PP_JAX_CACHE_DIR` -- overridable
# for sandboxed/CI runs where the repo root is ambiguous -- and is resolved on
# each call rather than once at import, so a test or caller that sets the env
# var after this module loads is still honoured.


def _configure_compilation_cache() -> None:
    """Point JAX's persistent compilation cache at a shared on-disk directory.

    Idempotent and cheap to call repeatedly from stage-process entry points.
    """
    configure_jax_compilation_cache()


class BackendInfo(NamedTuple):
    """Snapshot of the active JAX backend after :func:`configure` runs.

    Attributes:
        platform: JAX platform string, e.g. ``"cpu"`` or ``"gpu"``.
        dtype: Active default dtype string, ``"float32"`` or ``"float64"``.
        device_count: Number of devices visible to JAX on this platform.
        jax_version: ``jax.__version__`` string at configuration time.

    """

    platform: str
    dtype: str
    device_count: int
    jax_version: str


def configure(
    precision: Literal["float64", "float32", "auto"] = "auto",
    platform: str | None = None,
) -> BackendInfo:
    """Configure JAX backend precision and platform.

    Must be called **before** any other JAX computation in the process.
    JAX platform and x64 flags are process-global; a second call has no
    effect on already-compiled kernels.

    Args:
        precision: Floating-point precision to use.
            ``"float64"`` enables ``jax_enable_x64``.
            ``"float32"`` forces single precision.
            ``"auto"`` selects float64 on CPU and float32 on GPU.
        platform: If given, passed to ``jax.config.update("jax_platforms",
            platform)`` before device discovery.  Typical values:
            ``"cpu"``, ``"gpu"``, ``"tpu"``.

    Returns:
        BackendInfo: Snapshot of platform, dtype, device count, and JAX
            version after configuration is applied.

    """
    # Imported inside the function, not at module scope: this module is the
    # configuration site, so it must stay importable *before* JAX is first
    # imported (see the module docstring's ordering invariant).
    import jax
    import jax.numpy as jnp

    _configure_compilation_cache()

    if platform is not None:
        jax.config.update("jax_platforms", platform)

    # Discover active platform after optional override.
    devices = jax.devices()
    active_platform = devices[0].platform  # e.g. "cpu", "gpu", "tpu"
    device_count = len(devices)

    # Resolve "auto" based on discovered platform.
    if precision == "auto":
        resolved: Literal["float64", "float32"] = (
            "float64" if active_platform == "cpu" else "float32"
        )
    else:
        resolved = cast("Literal['float64', 'float32']", precision)

    if resolved == "float64":
        jax.config.update("jax_enable_x64", val=True)
    # For GPU: disable TF32 to avoid implicit precision reduction.
    if active_platform == "gpu":
        jax.config.update("jax_default_matmul_precision", "highest")

    dtype_str = str(jnp.zeros(1).dtype)

    return BackendInfo(
        platform=active_platform,
        dtype=dtype_str,
        device_count=device_count,
        jax_version=jax.__version__,
    )


# Precision/platform policy shared by every simulation DVC stage. Held here -- a
# tracked dependency of each simulation stage -- rather than as literals inside the
# thin ``cli.py`` dispatcher, so the dispatcher can be dropped from stage deps while
# this output-affecting choice stays version-controlled and change-detected. See
# the dependency-scoping section in the pipeline architecture.
DEFAULT_PRECISION: Literal["float64", "float32", "auto"] = "float64"
DEFAULT_PLATFORM: str = "cpu"


def configure_default() -> BackendInfo:
    """Configure the JAX backend with the simulation-wide precision policy."""
    return configure(precision=DEFAULT_PRECISION, platform=DEFAULT_PLATFORM)
