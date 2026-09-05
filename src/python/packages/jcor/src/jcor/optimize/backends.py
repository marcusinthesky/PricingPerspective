"""Maintained JAX solver adapters with narrow, typed contracts.

Lineax and Optimistix remain implementation dependencies behind this module;
stage code does not depend on their result object layouts. The adapters expose
only fixed-shape array results and therefore compose with ``jit`` and ``vmap``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import lineax as lx
import optimistix as ox

from jcor.core.typing import Array, Float  # noqa: TC001  # runtime type contract

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["least_squares", "minimise_bfgs"]


def least_squares(
    design: Float[Array, "n p"],  # noqa: F722
    outcome: Float[Array, " n"],  # noqa: F722
) -> Float[Array, " p"]:  # noqa: F722
    """Solve a dense least-squares problem through Lineax's QR backend."""
    operator = lx.MatrixLinearOperator(design)
    return lx.linear_solve(operator, outcome, solver=lx.QR()).value


def minimise_bfgs(
    objective: Callable[[Array], Float[Array, ""]],  # noqa: F722
    initial: Float[Array, " p"],  # noqa: F722
    *,
    max_steps: int = 256,
    rtol: float = 1e-8,
    atol: float = 1e-8,
) -> Float[Array, " p"]:  # noqa: F722
    """Minimise a smooth scalar objective through Optimistix BFGS."""

    def wrapped(value: Array, args: None) -> Float[Array, ""]:  # noqa: F722
        del args
        return objective(value)

    solution = ox.minimise(
        wrapped,
        ox.BFGS(rtol=rtol, atol=atol),
        initial,
        max_steps=max_steps,
        throw=True,
    )
    return solution.value
