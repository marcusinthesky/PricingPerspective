"""Materiality gate over the assembled brackets, and YAML/JSON plainification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jcor.core.random import cell_key
from jcor.model.covariance import ledoit_wolf_sample_kernel

from pipeline.stages.papers.paper5._energy_robust.contracts import MAX_MATERIALITY_RATIO

if TYPE_CHECKING:
    from pipeline.stages.papers.paper5._energy_robust.contracts import Float


def _materiality_gate(
    brackets_by_mode: dict[str, dict[str, object]],
    r_calib: Float,
    t_ref: int = 60,
    n_boot: int = 200,
    seed: int = 0,
) -> dict[str, object]:
    """Median bracket width (Sigma-scale) vs. a bootstrapped LW T=60 estimation band.

    The LW band is the median (across off-diagonal entries) of the central
    90% interval width of Ledoit-Wolf covariance entries over ``n_boot``
    circular contiguous blocks of length ``t_ref`` drawn from the
    calibration panel. Gate semantics follow the plan's PRE-REGISTERED
    threshold ("median bracket width < 2x LW's estimation-error band at
    T=60", anti-snooping protocol): PASS iff ``materiality_ratio =
    width/band < 2``; a wider bracket is flagged as failing materiality
    (vacuity kill-risk in the plan). ``materiality_ratio`` itself is
    reported so readers can apply stricter thresholds.
    """
    t_cal, n = r_calib.shape
    iu = np.triu_indices(n, k=1)
    if t_ref >= t_cal:
        lw_band = float("nan")
    else:
        key = cell_key(seed, "paper5_energy_robust_materiality", "lw_band", t_ref)
        starts = jax.random.randint(key, (n_boot,), 0, t_cal)
        offsets = jnp.arange(t_ref)
        panel = jnp.asarray(r_calib)

        def one_block(start: jax.Array) -> jax.Array:
            index = (start + offsets) % t_cal
            return ledoit_wolf_sample_kernel(panel[index]).sigma

        covariances = jax.jit(jax.vmap(one_block))(starts)
        ents = np.asarray(covariances[:, iu[0], iu[1]])
        band = np.quantile(ents, 0.95, axis=0) - np.quantile(ents, 0.05, axis=0)
        lw_band = float(np.median(band))
    modes: dict[str, object] = {}
    for mode, br in brackets_by_mode.items():
        width = cast("float", br["bracket_width_Sigma_median"])
        ratio = (width / lw_band) if lw_band > 0 else float("inf")
        modes[mode] = {
            "median_bracket_width_sigma": width,
            "materiality_ratio": ratio,
            "gate_verdict": (
                "PASS (pre-registered gate: bracket width < 2x LW T=60 band)"
                if ratio < MAX_MATERIALITY_RATIO
                else "FAIL (bracket width >= 2x LW T=60 estimation band)"
            ),
        }
    return {
        "lw_band_width_t60": lw_band,
        "t_ref": t_ref,
        "n_boot": n_boot,
        "modes": modes,
    }


def _to_plain(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        arr = cast("np.ndarray[Any, Any]", obj)
        if arr.ndim == 0:
            return _to_plain(arr.item())
        return [_to_plain(v) for v in arr]
    return obj
