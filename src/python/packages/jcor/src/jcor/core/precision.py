"""The one place jcor states a float64 requirement it cannot satisfy itself.

``jcor`` opens no ``jax.enable_x64`` scope and never touches the process-global
flag (``docs/numerical-host-boundaries.md`` §"Precision boundary: who owns
x64"). Precision is an input to this package. A handful of surfaces nonetheless
*need* float64 to be correct rather than merely accurate, and for those the only
honest options are to raise or to return a wrong answer.

The failure this guards against is not hypothetical and is not gradual. Run with
the caller flag off, ``model._dyadic`` rejected **every one** of 1999
node-bootstrap draws and ``inference.wolak`` rejected 1/1 solver lanes --- both
silently, both reported as ordinary domain errors far from the real cause. The
rank selector underneath the first runs on designs measured at condition number
>= 1e9 (worst 1e99), where float32 leaves its threshold test no signal at all.

So: raise at the door, name the flag, and let the caller fix it in one line.
A stage that cannot own x64 should call the ``*_kernel`` primitive and accept
the dtype it is given, not ask this package to paper over the difference.
"""

from __future__ import annotations

import jax


def require_x64(api_name: str) -> None:
    """Refuse to run a float64-critical surface under a float32 caller.

    Args:
        api_name: Public callable being protected, named in the error so the
            caller sees which surface refused rather than an internal frame.

    Raises:
        RuntimeError: When ``jax_enable_x64`` is not set by the caller.

    """
    if jax.config.read("jax_enable_x64"):
        return
    message = (
        f"{api_name} requires caller-owned jax_enable_x64=True: its float64 "
        "contract cannot be recovered once inputs are canonicalized. Enable the "
        "flag for the stage (see pipeline.precision.own_float64), or call the "
        "kernel primitive directly for dtype-preserving execution."
    )
    raise RuntimeError(message)
