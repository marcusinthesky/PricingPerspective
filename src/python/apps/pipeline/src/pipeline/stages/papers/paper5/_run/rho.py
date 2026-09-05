"""Fold-level SAR fitting and coefficient-stability disclosure."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pipeline.stages.papers.paper5 import estimate as est
from pipeline.stages.papers.paper5._run.contracts import (
    _CALIB_START,
    _RHO_SE_FALLBACK,
)
from pipeline.stages.substrate.panel import load_aligned_return_panel

if TYPE_CHECKING:
    from pipeline.stages.papers.paper5._run.contracts import _RhoFitInputs

logger = logging.getLogger(__name__)


def _resolve_rho_se(w_name: str, fold_name: str, row: pd.Series) -> float:
    """Resolve numerical-Hessian ``rho_se`` from the QMLE fit.

    Uses a loud, disclosed fallback to a fixed-width SE only when the curvature
    is unusable (never silent -- see t15).
    """
    rho_se = float(row.get("rho_se", float("nan")))
    if np.isfinite(rho_se) and rho_se > 0.0:
        return rho_se
    logger.warning(
        "rho_se: numerical-Hessian SE unusable for w_variant=%s fold=%s "
        "(rho_hat=%s); falling back to fixed-width SE=%.3f. This fallback "
        "widens/narrows the H3 (rho, W) bracket without a curvature-based "
        "justification and should be investigated before trusting coverage "
        "numbers for this cell.",
        w_name,
        fold_name,
        row.get("rho_hat"),
        _RHO_SE_FALLBACK,
    )
    return _RHO_SE_FALLBACK


def _fit_rho_models(
    inputs: _RhoFitInputs,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Load fold panels and fit every fixed and fold-specific SAR model."""
    rho_rows: list[dict[str, object]] = []
    fold_data: dict[str, dict[str, np.ndarray]] = {}
    n_tickers = len(inputs.tickers)
    for fold_name, (fold_start, fold_end) in inputs.folds.items():
        _, train_returns = load_aligned_return_panel(
            inputs.paths.returns_dir,
            inputs.tickers,
            _CALIB_START,
            f"{int(fold_start[:4]) - 1}-12-31",
            oos_returns_dir=inputs.paths.oos_returns_dir,
        )
        _, test_returns = load_aligned_return_panel(
            inputs.paths.returns_dir,
            inputs.tickers,
            fold_start,
            fold_end,
            oos_returns_dir=inputs.paths.oos_returns_dir,
        )
        fold_data[fold_name] = {"train": train_returns, "test": test_returns}
        fold_w_variants = {
            **inputs.frozen_w_variants,
            "correlation_knn": est.correlation_knn_w(train_returns, k=inputs.knn_k),
        }
        for variant_name, weights in fold_w_variants.items():
            ones = np.ones((n_tickers, 1))
            instruments = np.column_stack(
                [ones, weights @ ones, weights @ weights @ ones]
            )
            instrument_rank = int(np.linalg.matrix_rank(instruments))
            moran_i = est.moran_i(train_returns, weights)
            if variant_name == "rho_zero":
                rho_rows.append(
                    {
                        "fold": fold_name,
                        "w_variant": variant_name,
                        "method": "fixed",
                        "rho_hat": 0.0,
                        "alpha_hat": float(np.mean(train_returns)),
                        "moran_i": moran_i,
                        "abs_rho_lt_1": True,
                    }
                )
                continue
            qmle = est.sar_qmle(train_returns, weights)
            gmm = est.sar_gmm(train_returns, weights)
            rho_se = float(qmle["rho_se"])
            rho_hat = float(qmle["rho_hat"])
            rho_rows.extend(
                [
                    {
                        "fold": fold_name,
                        "w_variant": variant_name,
                        "method": "qmle",
                        "rho_hat": rho_hat,
                        "alpha_hat": qmle["alpha_hat"],
                        "sigma2_hat": qmle["sigma2_hat"],
                        "loglik": qmle["loglik"],
                        "rho_se": rho_se,
                        "rho_se_over_abs_rho": (
                            rho_se / abs(rho_hat) if abs(rho_hat) > 0 else float("nan")
                        ),
                        "rho_unstable": bool(
                            not np.isfinite(rho_se) or rho_se > 0.5 * abs(rho_hat)
                        ),
                        "moran_i": moran_i,
                        "abs_rho_lt_1": abs(rho_hat) < 1.0,
                        "instrument_rank": instrument_rank,
                    },
                    {
                        "fold": fold_name,
                        "w_variant": variant_name,
                        "method": "gmm",
                        "rho_hat": gmm["rho_hat"],
                        "alpha_hat": gmm["alpha_hat"],
                        "moran_i": moran_i,
                        "abs_rho_lt_1": abs(gmm["rho_hat"]) < 1.0,
                        "instrument_rank": instrument_rank,
                    },
                ]
            )
    return pd.DataFrame(rho_rows), fold_data


def _rho_stability_records(rho_df: pd.DataFrame) -> list[dict[str, object]]:
    """Log unstable spatial coefficients and return portable JSON records."""
    for _, row in rho_df[~rho_df["abs_rho_lt_1"]].iterrows():
        logger.warning(
            "rho_stability: |rho_hat| >= 1 for fold=%s w_variant=%s method=%s "
            "(rho_hat=%s) -- disclosed, not aborting the stage.",
            row["fold"],
            row["w_variant"],
            row["method"],
            row["rho_hat"],
        )
    columns = [
        name
        for name in (
            "fold",
            "w_variant",
            "method",
            "rho_hat",
            "rho_se",
            "abs_rho_lt_1",
            "rho_unstable",
        )
        if name in rho_df.columns
    ]
    return [
        {str(key): None if pd.isna(value) else value for key, value in record.items()}
        for record in rho_df[columns].to_dict(orient="records")
    ]
