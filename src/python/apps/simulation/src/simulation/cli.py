"""CLI entry-points for Monte-Carlo validation experiments.

Exposed as the ``simulation`` console-script via ``[project.scripts]``::

    simulation validate-variance-floor [--params-file PATH]
    simulation validate-hedging-error [--params-file PATH]
    simulation compare-pricefree      [--params-file PATH]
    simulation simulate-size-power     [--params-file PATH]
    simulation simulate-equivalence-calibration [--params-file PATH]
    simulation simulate-gmm-wolak-size-power [--params-file PATH]
    simulation audit-kappa-identity    [--params-file PATH]
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any

import typer
import yaml
from dotell import begin, setup_logging

from simulation.backend import configure_default
from simulation.io import write_csv_atomic, write_parquet_atomic, write_yaml_summary

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Monte-Carlo validation experiments for Lean theorem guarantees.",
    no_args_is_help=True,
)


@app.callback()
def _bootstrap(ctx: typer.Context) -> None:
    """Configure logging and per-stage resource telemetry for every command.

    Runs before any subcommand: routes stdlib logging through loguru and, for
    an actual subcommand invocation, samples the process tree and records a
    telemetry row on teardown (see the ``telemetry`` package).
    """
    setup_logging("simulation")
    if ctx.invoked_subcommand is not None:
        ctx.call_on_close(begin(ctx.invoked_subcommand, app="simulation"))


def _load_monte_carlo_params(params_file: Path) -> dict[str, Any]:
    """Load ``params["monte_carlo"]`` from a YAML file."""
    with Path(params_file).open(encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params["monte_carlo"]


def _load_test_size_power_params(params_file: Path) -> dict[str, Any]:
    """Load ``params["monte_carlo"]["test_size_power"]`` from a YAML file."""
    with Path(params_file).open(encoding="utf-8") as f:
        params = yaml.safe_load(f)
    monte_carlo = params["monte_carlo"]
    return {
        **monte_carlo["test_size_power"],
        "replication_batch_size": monte_carlo["replication_batch_size"],
    }


def _load_sub_params(params_file: Path, key: str) -> dict[str, Any]:
    """Load ``params["monte_carlo"][key]`` from a YAML file."""
    with Path(params_file).open(encoding="utf-8") as f:
        params = yaml.safe_load(f)
    monte_carlo = params["monte_carlo"]
    return {
        **monte_carlo[key],
        "replication_batch_size": monte_carlo["replication_batch_size"],
    }


# ---------------------------------------------------------------------------
# validate-variance-floor
# ---------------------------------------------------------------------------


@app.command("validate-variance-floor")
def cli_variance_floor(
    params_file: Path = typer.Option(Path("params.yaml"), help="Path to params.yaml"),
) -> None:
    """Validate Lean theorem portfolio_variance_lower_bound via Monte-Carlo."""
    configure_default()
    import jax.numpy as jnp

    from simulation.validation.variance_floor import CERTIFY_ARM, run_variance_floor

    cfg = _load_monte_carlo_params(params_file)
    logger.info("Monte-Carlo validation of portfolio_variance_lower_bound")
    logger.info("  n_assets configs: %s", cfg["n_assets"])
    logger.info("  embedding dims: %s", cfg["embedding_dim"])

    results_df, summary = run_variance_floor(cfg)

    output_dir = Path("data/mc/monte_carlo_validation")
    write_parquet_atomic(results_df, output_dir / "results_portfolio_floor.parquet")
    logger.info("Saved %d rows", len(results_df))

    # Violation summary CSV
    certify = results_df[results_df["arm"] == CERTIFY_ARM]
    violation_count = int(certify["violation"].sum())
    summary_df = (
        results_df.groupby(["n_assets", "embedding_dim", "lipschitz_scale", "arm"])
        .agg(
            n_configs=("violation", "count"),
            n_violations=("violation", "sum"),
            slack_mean=("slack", "mean"),
            slack_min=("slack", "min"),
            slack_p1=("slack", lambda x: float(jnp.percentile(x.to_numpy(), 1))),
            slack_p50=("slack", "median"),
        )
        .reset_index()
    )
    csv_path = output_dir / "violation_summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(summary_df, csv_path)
    logger.info("Saved violation summary to %s", csv_path)

    write_yaml_summary(summary, output_dir / "summary.yaml")
    logger.info("Saved summary")

    logger.info("\n=== VALIDATION RESULT ===")
    logger.info("Certify-arm configs tested: %d", len(certify))
    logger.info("Violations: %s", violation_count)
    logger.info("Mean slack (realized - floor): %.6e", certify["slack"].mean())
    logger.info("Min slack: %.6e", certify["slack"].min())

    # A pair with D_ij below the eps floor admits no finite Lipschitz constant,
    # so the theorem's hypothesis was never established for it — asserting the
    # conclusion there is not a certificate. Gated, unlike the probe arm below.
    n_skipped = int(summary["n_skipped_zero_D_pairs_total"])
    if n_skipped > 0:
        logger.error(
            "HYPOTHESIS INCOMPLETE: %d pair(s) fell below the D_ij floor, so no "
            "finite Lipschitz constant controls them and the certify arm is not "
            "a certificate.",
            n_skipped,
        )
        sys.exit(1)

    if violation_count > 0:
        logger.error("THEOREM VIOLATION DETECTED")
        sys.exit(1)

    # Reported, never gated: see the two-arm rationale in
    # `simulation.validation.variance_floor`'s module docstring.
    probe_rows = int(summary["probe_rows"])
    if probe_rows:
        probe_violations = int(summary["probe_violations"])
        binding = summary["probe_binding_scale"]
        logger.info(
            "Tightness probe (scales %s): %d/%d rows violate.",
            summary["probe_scale_factors"],
            probe_violations,
            probe_rows,
        )
        if binding is not None:
            logger.info(
                "Floor first binds at l = %.4g * l_tight, i.e. the certificate "
                "carries a ~%.1fx Lipschitz margin.",
                binding,
                1.0 / binding if binding else float("inf"),
            )
        if probe_violations == 0:
            logger.warning(
                "TIGHTNESS PROBE VACUOUS: the bound held even with a sub-tight "
                "Lipschitz constant, so zero violations in the certify arm is "
                "not evidence the floor is informative."
            )
    else:
        logger.warning(
            "No tightness probe configured (monte_carlo."
            "lipschitz_tightness_probe_factors is empty): zero violations on a "
            "grid where the hypothesis holds by construction certifies little."
        )

    logger.info("PASS: Zero violations. Theorem holds numerically across all configs.")


# ---------------------------------------------------------------------------
# validate-hedging-error
# ---------------------------------------------------------------------------


@app.command("validate-hedging-error")
def cli_hedging_error(
    params_file: Path = typer.Option(Path("params.yaml"), help="Path to params.yaml"),
) -> None:
    """Validate energy-barycenter hedging-error bound (Papers 2/3)."""
    configure_default()
    from simulation.validation.hedging_error import run_hedging_error

    cfg = _load_monte_carlo_params(params_file)
    logger.info("Monte-Carlo validation of energy-barycenter hedging-error bound")

    results_df, summary = run_hedging_error(cfg)

    output_dir = Path("data/mc/hedging_error_validation")
    write_parquet_atomic(results_df, output_dir / "results.parquet")
    write_yaml_summary(summary, output_dir / "summary.yaml")
    logger.info("Saved %d rows + summary", len(results_df))

    tolerance = cfg["tolerance"]
    bound_violations = int((results_df["bound_slack"] < -tolerance).sum())

    logger.info("\n=== HEDGING ERROR BOUND VALIDATION ===")
    logger.info("Total targets tested: %d", len(results_df))
    logger.info("Bound violations (slack < -%s): %s", tolerance, bound_violations)
    logger.info(
        "Median TE improvement (opt vs equal-weight): %.2f%%",
        summary["median_TE_improvement_opt_vs_equal_pct"],
    )

    if bound_violations > 0:
        logger.error("BOUND VIOLATION DETECTED")
        sys.exit(1)
    logger.info(
        "PASS: Zero bound violations. Bound holds numerically across all configs."
    )


# ---------------------------------------------------------------------------
# compare-pricefree
# ---------------------------------------------------------------------------


@app.command("compare-pricefree")
def cli_pricefree(
    params_file: Path = typer.Option(Path("params.yaml"), help="Path to params.yaml"),
) -> None:
    """Price-free portfolio construction vs sample-MVO comparison."""
    configure_default()
    from simulation.validation.pricefree import run_pricefree

    cfg = _load_monte_carlo_params(params_file)
    logger.info("PUNCHLINE: price-free portfolio construction vs sample-MVO")

    results_df, summary = run_pricefree(cfg)

    output_dir = Path("data/mc/pricefree_comparison")
    write_parquet_atomic(results_df, output_dir / "results.parquet")
    write_yaml_summary(summary, output_dir / "summary.yaml")
    logger.info("Saved %d rows + summary", len(results_df))

    total_cap_violations = int(results_df["cap_violation"].sum())
    logger.info("\n=== PUNCHLINE RESULT ===")
    logger.info("Cap violations: %s (expect 0)", total_cap_violations)
    logger.info("\nHEADLINE: %s", summary["headline"])

    if total_cap_violations > 0:
        logger.error("CAP VIOLATIONS detected")
        sys.exit(1)
    logger.info("PASS: Zero cap violations. Theorem floor <= realised variance holds.")


# ---------------------------------------------------------------------------
# simulate-size-power
# ---------------------------------------------------------------------------


@app.command("simulate-size-power")
def cli_size_power(
    params_file: Path = typer.Option(Path("params.yaml"), help="Path to params.yaml"),
) -> None:
    """Type I / Type II (size/power) validation of energy permutation test."""
    configure_default()
    from simulation.validation.size_power import run_size_power

    cfg = _load_test_size_power_params(params_file)
    logger.info("=== simulate_test_size_power ===")
    logger.info(
        "n_sims=%s, num_permutations=%s", cfg["n_sims"], cfg["num_permutations"]
    )

    logger.info(
        "Note: n_sims=%s; binomial SE at alpha=0.05 = %.4f",
        cfg["n_sims"],
        float(math.sqrt(0.05 * 0.95 / cfg["n_sims"])),
    )

    size_df, power_df, summary = run_size_power(cfg)

    output_dir = Path("data/mc/test_size_power")
    write_parquet_atomic(size_df, output_dir / "size_results.parquet")
    logger.info("Saved size results (%d rows)", len(size_df))

    write_parquet_atomic(power_df, output_dir / "power_results.parquet")
    logger.info("Saved power results (%d rows)", len(power_df))

    write_yaml_summary(summary, output_dir / "summary.yaml")
    logger.info("Saved summary")

    logger.info("\n=== SIZE REPORT (normal distribution) ===")
    size_normal = size_df[size_df["distribution"] == "normal"]
    for _, row in size_normal.iterrows():
        flag = "*** OUTSIDE 3SE ***" if row["flag_outside_3se"] else "ok"
        logger.info(
            "  n=%3d d=%s alpha=%.2f: rej=%.3f (SE=%.3f, z=%+.2f) %s",
            row["n"],
            row["d"],
            row["nominal_alpha"],
            row["empirical_rejection_rate"],
            row["binom_se"],
            row["z_score"],
            flag,
        )

    any_size_fail = bool(size_df["flag_outside_3se"].any())
    if any_size_fail:
        logger.warning(
            "SIZE CALIBRATION WARNING: empirical rejection rate outside ±3 SE "
            "for at least one combination. See size_results.parquet."
        )
    else:
        logger.info(
            "SIZE: all empirical rejection rates within ±3 SE of nominal. PASS."
        )


# ---------------------------------------------------------------------------
# simulate-equivalence-calibration (Paper 2, ROADMAP P0-2)
# ---------------------------------------------------------------------------


@app.command("simulate-equivalence-calibration")
def cli_equivalence_calibration(
    params_file: Path = typer.Option(Path("params.yaml"), help="Path to params.yaml"),
) -> None:
    """Calibrate energy equivalence at the boundary under familywise testing."""
    backend = configure_default()
    from simulation.validation.equivalence_calibration import (
        run_equivalence_calibration,
    )

    cfg = _load_sub_params(params_file, "equivalence_calibration")
    logger.info("=== Energy-equivalence boundary calibration (Paper 2 P0-2) ===")
    logger.info(
        "n_sims=%s, n_resamples=%s, M=%s",
        cfg["n_sims"],
        cfg["n_resamples"],
        cfg["m_values"],
    )

    results, boundary_design, summary = run_equivalence_calibration(cfg)
    output_dir = Path("data/mc/equivalence_calibration")
    write_parquet_atomic(results, output_dir / "results.parquet", backend_info=backend)
    write_parquet_atomic(
        boundary_design,
        output_dir / "boundary_design.parquet",
        backend_info=backend,
    )
    write_yaml_summary(summary, output_dir / "summary.yaml", backend_info=backend)
    logger.info(
        "Saved %s result rows and %s exact-boundary design rows",
        len(results),
        len(boundary_design),
    )
    logger.info(
        "Production scale: %s. Interpret empirical comparisons from outputs; "
        "the stage does not encode an expected winner.",
        summary["production_scale"],
    )


# ---------------------------------------------------------------------------
# simulate-gmm-wolak-size-power (Paper 3 — gates Wave 2)
# ---------------------------------------------------------------------------


@app.command("simulate-gmm-wolak-size-power")
def cli_gmm_wolak_size_power(
    params_file: Path = typer.Option(Path("params.yaml"), help="Path to params.yaml"),
) -> None:
    """Size/power of GMM Wald + Wolak inequality tests (calibrates χ̄² weights)."""
    configure_default()
    from simulation.validation.gmm_wolak_size_power import run_gmm_wolak_size_power

    cfg = _load_sub_params(params_file, "gmm_wolak_size_power")
    logger.info("=== GMM Wald + Wolak size/power (Paper 3, gates Wave 2) ===")
    logger.info(
        "n_sims=%s, T=%s, m=%s, n_mc=%s",
        cfg["n_sims"],
        cfg["T"],
        cfg["m"],
        cfg["n_mc"],
    )

    size_df, power_df, summary = run_gmm_wolak_size_power(cfg)

    output_dir = Path("data/mc/gmm_wolak_size_power")
    write_parquet_atomic(size_df, output_dir / "size_results.parquet")
    write_parquet_atomic(power_df, output_dir / "power_results.parquet")
    write_yaml_summary(summary, output_dir / "summary.yaml")
    logger.info("Saved %d size + %d power rows + summary", len(size_df), len(power_df))

    logger.info("\n=== SIZE REPORT ===")
    for _, row in size_df.iterrows():
        flag = "*** OUTSIDE 3SE ***" if row["flag_outside_3se"] else "ok"
        logger.info(
            "  %-5s alpha=%.2f: rej=%.3f (SE=%.3f, z=%+.2f) %s",
            row["test"],
            row["nominal_alpha"],
            row["empirical_rejection_rate"],
            row["binom_se"],
            row["z_score"],
            flag,
        )

    gate = summary["gate"]
    if not gate["all_sizes_within_3se"]:
        logger.warning(gate["note"])
    else:
        logger.info(gate["note"])


# ---------------------------------------------------------------------------
# audit-kappa-identity (ROADMAP §4 item 3)
# ---------------------------------------------------------------------------


@app.command("audit-kappa-identity")
def cli_kappa_audit(
    params_file: Path = typer.Option(Path("params.yaml"), help="Path to params.yaml"),
) -> None:
    """Audit which identity form (D², ½κ²D², ½κ_iκ_jD²) makes g_ij vanish."""
    configure_default()
    from simulation.validation.kappa_audit import run_kappa_audit

    cfg = _load_sub_params(params_file, "kappa_audit")
    logger.info("=== κ identity-form audit (ROADMAP §4 item 3) ===")

    results_df, summary = run_kappa_audit(cfg)

    output_dir = Path("data/papers/paper3/kappa_audit")
    write_parquet_atomic(results_df, output_dir / "results.parquet")
    write_yaml_summary(summary, output_dir / "summary.yaml")
    logger.info("Saved %d rows + summary", len(results_df))

    logger.info("\n=== RESIDUAL BY (kernel, form, regime) ===")
    for _, row in results_df.iterrows():
        logger.info(
            "  %-11s %-14s %-13s: mean|g|=%.3e max|g|=%.3e",
            row["kernel"],
            row["form"],
            row["kappa_regime"],
            row["mean_abs_residual"],
            row["max_abs_residual"],
        )
    logger.info("\nHEADLINE: %s", summary["headline"])


@app.command("simulate-window-size-power")
def cli_window_size_power(
    params_file: Path = typer.Option(Path("params.yaml"), help="Path to params.yaml"),
    output_dir: Path = typer.Option(
        Path("data/mc/window_size_power"),
        help="Output directory for size/power results + summary.",
    ),
) -> None:
    """SIZE/POWER of the overlap-robust Sharpe test vs evaluation-window T (t17 D6)."""
    configure_default()
    from simulation.validation.window_size_power import run_window_size_power

    cfg = _load_sub_params(params_file, "window_size_power")
    logger.info(
        "=== simulate_window_size_power === t_values=%s n_sims=%s n_boot=%s",
        cfg["t_values"],
        cfg["n_sims"],
        cfg["n_boot"],
    )
    size_df, power_df, summary = run_window_size_power(
        t_values=list(cfg["t_values"]),
        phi=float(cfg["phi"]),
        deltas=[float(d) for d in cfg["deltas"]],
        alpha_levels=[float(a) for a in cfg["alpha_levels"]],
        n_sims=int(cfg["n_sims"]),
        n_boot=int(cfg["n_boot"]),
        seed=int(cfg["seed"]),
        k_strategies=int(cfg["k_strategies"]),
        output_dir=str(output_dir),
        replication_batch_size=int(cfg["replication_batch_size"]),
    )
    logger.info(
        "Saved size (%d rows) + power (%d rows) + summary to %s",
        len(size_df),
        len(power_df),
        output_dir,
    )
    logger.info(
        "minimal adequate T (D6 readout): %s", summary.get("minimal_adequate_T")
    )


if __name__ == "__main__":
    app()
