"""Chunked, transformable mapping over semantic Monte-Carlo replications.

Validation stages own their heterogeneous cell loops and static shapes.  This
module only supplies the common replication axis: replication ``i`` always
receives ``fold_in(cell_key, i)``.  Consequently reordering cells or extending
``n_sims`` cannot perturb an existing stream.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from dotell import jax_diagnostics

if TYPE_CHECKING:
    from collections.abc import Callable

    from jcor.core.typing import PRNGKey

__all__ = ["execute_replication_cell", "map_replications", "replication_keys"]

logger = logging.getLogger(__name__)


def replication_keys(cell_key: PRNGKey, n_sims: int) -> jax.Array:
    """Return the ordered threefry keys for one homogeneous simulation cell."""
    if n_sims < 1:
        msg = f"n_sims must be positive, got {n_sims}"
        raise ValueError(msg)
    indices = jnp.arange(n_sims, dtype=jnp.uint32)
    return jax.vmap(jax.random.fold_in, in_axes=(None, 0))(cell_key, indices)


def map_replications[ResultT](
    kernel: Callable[[PRNGKey], ResultT],
    cell_key: PRNGKey,
    n_sims: int,
    batch_size: int,
) -> ResultT:
    """Map ``kernel`` over stable replication keys, chunking only when needed.

    ``batch_size`` changes execution and peak memory, never key assignment or
    result order.  A cell that fits in one batch uses :func:`jax.vmap`; larger
    cells use :func:`jax.lax.map` with its native batched implementation.
    """
    if batch_size < 1:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)
    keys = replication_keys(cell_key, n_sims)
    if n_sims <= batch_size:
        return jax.vmap(kernel)(keys)
    return jax.lax.map(kernel, keys, batch_size=batch_size)


def execute_replication_cell[ResultT](
    driver: object,
    *args: object,
    name: str,
    replication_count: int,
    replication_batch_size: int,
    static: dict[str, object] | None = None,
    **driver_kwargs: object,
) -> ResultT:
    """Compile, execute, synchronize, and report one jitted cell driver.

    The driver owns shapes and static configuration. This host-only adapter
    records one structured start/end pair rather than emitting progress from
    individual replications; the optional JAX diagnostics stream receives
    separate compile, execution-dispatch, and synchronization events.
    """
    if replication_count < 1 or replication_batch_size < 1:
        msg = (
            "replication_count and replication_batch_size must be positive, got "
            f"{replication_count}, {replication_batch_size}"
        )
        raise ValueError(msg)
    batch_count = (
        replication_count + replication_batch_size - 1
    ) // replication_batch_size
    mode = "vmap" if replication_count <= replication_batch_size else "lax.map"
    metadata = {
        "n_sims": replication_count,
        "batch_size": replication_batch_size,
        "batch_count": batch_count,
        "mode": mode,
        **(static or {}),
    }
    logger.info(
        "replication_cell_start name=%s mode=%s n_sims=%d batch_size=%d batches=%d",
        name,
        mode,
        replication_count,
        replication_batch_size,
        batch_count,
    )
    diagnostics = jax_diagnostics()
    started = time.perf_counter()
    compile_started = time.perf_counter()
    with diagnostics.compile(name, *args, static=metadata):
        driver.lower(*args, **driver_kwargs).compile()  # type: ignore[attr-defined]
    compile_s = time.perf_counter() - compile_started
    execution_started = time.perf_counter()
    with diagnostics.execution(name, *args, static=metadata):
        result = driver(*args, **driver_kwargs)  # type: ignore[operator]
    with diagnostics.sync(name, result, static=metadata):
        result = jax.block_until_ready(result)
    execution_s = time.perf_counter() - execution_started
    logger.info(
        "replication_cell_end name=%s mode=%s n_sims=%d batch_size=%d "
        "batches=%d compile_s=%.6f execution_s=%.6f total_s=%.6f",
        name,
        mode,
        replication_count,
        replication_batch_size,
        batch_count,
        compile_s,
        execution_s,
        time.perf_counter() - started,
    )
    return result
