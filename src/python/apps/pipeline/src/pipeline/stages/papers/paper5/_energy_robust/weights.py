"""Strategy dispatch over the optimizers and the certified variance caps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pipeline.stages.papers.paper5._energy_robust.contracts import (
    RobustPortfolioError,
)
from pipeline.stages.papers.paper5._energy_robust.optimize import (
    higham_corner_weights,
    robust_corner_weights,
    robust_ridge_weights,
    sdp_corner_weights,
    wdro_weights,
)

if TYPE_CHECKING:
    from pipeline.stages.papers.paper5._energy_robust.contracts import (
        Float,
        RobustOptimizationRequest,
    )


def robust_weights(request: RobustOptimizationRequest) -> Float:
    """Single dispatch signature reaching every robust optimizer (T3 ladder).

    ``_cov_and_weights`` and :func:`backtest` route through this one
    function so callers (including MC-3 in t04.3 and the t05 backtest) need
    not know which optimizer implementation backs a given ``method`` name.

    Args:
        request: Method selection and method-specific covariance inputs.

    Returns:
        Long-only weights, shape ``(n,)``.

    """
    if request.method in {"corner", "higham_corner"}:
        if request.sigma_hi is None:
            message = f"{request.method} strategy requires sigma_hi"
            raise RobustPortfolioError(message)
        optimizer = (
            robust_corner_weights
            if request.method == "corner"
            else higham_corner_weights
        )
        return optimizer(request.sigma_hi, request.n_steps)
    if request.method == "ridge":
        if request.sigma_hat is None or request.radius is None:
            message = "ridge strategy requires sigma_hat and radius"
            raise RobustPortfolioError(message)
        return robust_ridge_weights(request.sigma_hat, request.radius, request.n_steps)
    if request.method == "sdp_corner":
        if request.sigma_lo is None or request.sigma_hi is None:
            message = "sdp_corner strategy requires sigma_lo and sigma_hi"
            raise RobustPortfolioError(message)
        return sdp_corner_weights(request.sigma_lo, request.sigma_hi, cap=request.cap)
    if request.method == "wdro":
        if request.sigma_hat is None or request.radius is None:
            message = "wdro strategy requires sigma_hat and radius"
            raise RobustPortfolioError(message)
        return wdro_weights(request.sigma_hat, request.radius, request.n_steps)
    message = f"robust_weights: unknown method {request.method!r}"
    raise RobustPortfolioError(message)


def certified_caps(
    w: Float, sigma_hi: Float, sigma_hat: Float, r: float
) -> dict[str, float]:
    """Both valid worst-case caps for a weight vector; the certified cap is their min.

    Args:
        w: Portfolio weights, shape ``(n,)``.
        sigma_hi: Box corner (upper covariance bracket).
        sigma_hat: Point covariance estimate (ridge-ball center).
        r: Frobenius-ball radius for this window.

    Returns:
        Dict ``box_cap``, ``ridge_cap``, ``certified_cap`` (the min, the
        valid worst-case guarantee under EITHER uncertainty model).

    """
    box_cap = float(w @ sigma_hi @ w)
    ridge_cap = float(w @ sigma_hat @ w + r * np.sum(w**2))
    return {
        "box_cap": box_cap,
        "ridge_cap": ridge_cap,
        "certified_cap": min(box_cap, ridge_cap),
    }
