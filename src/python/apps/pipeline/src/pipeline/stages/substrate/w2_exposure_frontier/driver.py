"""Artifact-producing driver for the H1 identification-frontier pilot."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import numpy as np
import yaml
from jcor.core.random import cell_key
from simulation.portfolios import weight_schemes

from pipeline.stages.substrate.energy_shared import _to_plain
from pipeline.stages.substrate.w2_exposure_frontier.bootstrap import (
    frontier_bootstrap_and_wolak,
)
from pipeline.stages.substrate.w2_exposure_frontier.contracts import (
    DEFAULT_H1_PILOT_CONFIG,
    FrontierBootstrapConfig,
)
from pipeline.stages.substrate.w2_exposure_frontier.diagnostics import (
    DISCLOSURE,
    _k_diagnostics,
    _weight_scheme_spread,
)
from pipeline.stages.substrate.w2_exposure_frontier.frontier import frontier_quantiles
from pipeline.stages.substrate.w2_exposure_frontier.render import (
    FrontierSeries,
    render_frontier,
)
from pipeline.stages.substrate.w2_exposure_frontier.returns_loadings import (
    load_return_panel,
    pick_k,
    sweep_k,
)
from pipeline.stages.substrate.w2_exposure_frontier.universe import load_universe

if TYPE_CHECKING:
    from pipeline.stages.substrate.w2_exposure_frontier.contracts import (
        H1ArtifactPaths,
        H1PilotConfig,
    )

logger = logging.getLogger(".".join((*__name__.split(".")[:-2], "pilot")))


def run_h1_pilot(
    paths: H1ArtifactPaths,
    config: H1PilotConfig = DEFAULT_H1_PILOT_CONFIG,
) -> dict[str, object]:
    """Run the H1 identification-frontier pilot and write the go/no-go verdict.

    Numerator: whitened-PCA loadings from the TRAIN return panel
    (``pipeline.returns_loadings``, K swept over ``ks`` and picked by
    ``k_target_explained_variance``). Denominator: text distance
    typed text W2 distance ``D``. Estimates the single
    lower/upper Lipschitz frontier over all 100-priced-ticker pairs,
    block-bootstraps the conservative floor ell_lo (F6), runs the
    externally-anchored pooled Wolak inequality test (F4), and converts
    ell_lo into a certified cap-reduction as a fraction of the reference
    (systematic) variance via the Lean-certified envelope. Writes
    ``verdict.yaml``, ``frontier_metrics.json``, and ``h1_frontier.pgf`` to
    ``output_dir``.

    The R14 kill is: degenerate frontier ``ell_lo <= 0`` or the Wolak test
    rejecting the lower-Lipschitz inequalities at the external anchor, OR an
    immaterial cap-reduction (< ``cap_kill_threshold``). Zones: kill
    ``< cap_kill_threshold``, marginal in between, go
    ``>= cap_marginal_threshold``.

    Args:
        paths: Per-ticker returns, typed-distance artifact, and output directory.
        config: Frontier calibration, resampling, and decision settings.

    Returns:
        The verdict dict (also serialised to ``verdict.yaml``).

    """
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    tickers, d, d2 = load_universe(paths.distance_artifact_dir, paths.returns_dir)
    n = len(tickers)
    logger.info("H1 pilot: %d priced tickers after return-panel alignment", n)

    _, returns_train = load_return_panel(
        paths.returns_dir, tickers, config.train_start, config.train_end
    )

    sweep = sweep_k(returns_train, config.ks)
    k_chosen = pick_k(sweep, config.k_target_explained_variance)
    k_diag = _k_diagnostics(sweep, k_chosen)
    betas = sweep[k_chosen].betas

    fq = frontier_quantiles(betas, d, config.tau_lo, config.tau_hi)
    bw = frontier_bootstrap_and_wolak(
        returns_train,
        d,
        d2,
        FrontierBootstrapConfig(
            k=k_chosen,
            tau_lo=config.tau_lo,
            n_boot=config.n_boot,
            n_clusters=config.n_clusters,
            n_mc=config.n_mc,
            seed=config.seed,
            wolak_ell0=config.wolak_ell0,
        ),
    )

    schemes = [
        (scheme.name, np.asarray(scheme.weights))
        for scheme in weight_schemes(
            cell_key(config.seed, "w2_exposure_frontier", "weight_schemes", "n", n),
            n,
            config.n_dirichlet,
        )
    ]
    cap = _weight_scheme_spread(schemes, betas, d2, bw.ell_lo)

    # --- Verdict logic (F3: single ell_lo drives both kills) ----------------
    kill_degenerate = bool(bw.ell_lo <= 0.0 or bw.wolak_p < config.wolak_alpha)
    headline_pct = cap["equal_weight"]
    kill_capreduction = bool(headline_pct < config.cap_kill_threshold)

    if kill_degenerate:
        verdict = "no-go"
        zone = "degenerate"
    elif headline_pct < config.cap_kill_threshold:
        verdict = "no-go"
        zone = "kill"
    elif headline_pct < config.cap_marginal_threshold:
        verdict = "marginal"
        zone = "marginal"
    else:
        verdict = "go"
        zone = "go"

    r14_fallback = (
        None
        if verdict == "go"
        else (
            "R14 fallback: recast as lead-lag information-diffusion paper, the "
            "pair-screen floor remainder or a methods-only result."
        )
    )

    render_frontier(
        FrontierSeries(
            weights=fq.weights,
            ratios=fq.ratios,
            ell_hat=fq.ell_hat,
            L_hat=fq.L_hat,
        ),
        paths.output_dir / "h1_frontier.pgf",
    )
    logger.info("Wrote %s", paths.output_dir / "h1_frontier.pgf")

    verdict_dict: dict[str, object] = {
        "verdict": verdict,
        "zone": zone,
        "kill_degenerate": kill_degenerate,
        "kill_capreduction": kill_capreduction,
        "r14_fallback": r14_fallback,
        "ell_hat": fq.ell_hat,
        "ell_lo": bw.ell_lo,
        "ell_hi": bw.ell_hi,
        "ell_se": bw.ell_se,
        "mde": bw.mde,
        "L_hat": fq.L_hat,
        "breach_rate": fq.breach,
        "wolak_stat": bw.wolak_stat,
        "wolak_p": bw.wolak_p,
        "wolak_ell0": bw.wolak_ell0,
        "n_clusters_eff": bw.n_clusters_eff,
        "block_length": bw.block_length,
        "cap_reduction_pct": cap,
        "headline_cap_reduction_pct": headline_pct,
        "k_diagnostics": k_diag,
        "universe": {
            "n_tickers": n,
            "tickers": tickers,
        },
        "config": {
            **config.model_dump(),
            "ks": list(config.ks),
        },
        "disclosure": DISCLOSURE,
    }

    plain = _to_plain(verdict_dict)
    with (paths.output_dir / "verdict.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(plain, fh, sort_keys=False, default_flow_style=False)
    logger.info("Wrote %s (verdict=%s)", paths.output_dir / "verdict.yaml", verdict)

    frontier_metrics = {
        "ell_hat": fq.ell_hat,
        "ell_lo": bw.ell_lo,
        "ell_hi": bw.ell_hi,
        "ell_se": bw.ell_se,
        "mde": bw.mde,
        "L_hat": fq.L_hat,
        "breach_rate": fq.breach,
        "wolak_stat": bw.wolak_stat,
        "wolak_p": bw.wolak_p,
        "wolak_ell0": bw.wolak_ell0,
        "k_diagnostics": k_diag,
        "cap_reduction_pct": cap,
    }
    with (paths.output_dir / "frontier_metrics.json").open("w") as fh:
        json.dump(_to_plain(frontier_metrics), fh, indent=2, sort_keys=False)
    logger.info("Wrote %s", paths.output_dir / "frontier_metrics.json")

    return verdict_dict
