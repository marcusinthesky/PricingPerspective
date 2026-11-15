"""Explicit precision ownership for certified application entrypoints."""

import jax


def own_float64() -> None:
    """Select certified float64 arithmetic before constructing application arrays."""
    jax.config.update("jax_enable_x64", val=True)


def require_float64() -> None:
    """Reject imported certificate workflows when their caller disabled float64."""
    if not jax.config.x64_enabled:
        message = "Certified Hotelling workflows require JAX_ENABLE_X64=1."
        raise ValueError(message)
