"""S8 -- Sharpe decisions: is the performance real once you account for the search.

The overlap-robust *reporting* primitives a backtest overlay adds on top of the
long-run-covariance machinery (:mod:`jcor.operators.longrun`) and the
block-length selectors (:mod:`jcor.inference.block_length`). Everything here is
additive significance: it adds columns/objects that did not exist; it overwrites
no published point estimate and re-implements none of the bootstrap/HAC kernels.

Four families, two data-snooping controls (decision brief 2026-07-18, D4/D5;
references flagged for author staging, NOT yet @cited):

* **Sharpe ratio** -- one implementation, repo-wide
  (:func:`sharpe_ratio`). t46.8 retired ``simulation.hac._sharpe_ratio``
  against it at exact parity; see :mod:`._sharpe.ratio` for the measurement.
* **Deflated Sharpe Ratio** (Bailey & Lopez de Prado 2014): the Probabilistic
  Sharpe Ratio benchmarked against the *expected maximum* Sharpe under a null
  of ``n_trials`` independent zero-skill trials.
* **Paired Sharpe difference** (:func:`sharpe_difference_test`) and the
  stationary-bootstrap Sharpe CI (:func:`sharpe_ratio_ci`).
* **Multiple Sharpe test** (Romano-Wolf 2005 stepwise + White 2000 Reality
  Check) with familywise-error control.

All statistics are per-observation (non-annualized) unless the caller
annualizes.

Numerical policy (t48.4): scalar probabilities and fixed-index bootstrap
arithmetic run in JAX, with local float64 contexts that do not mutate the
process-global precision flag. The public functions remain eager adapters for
validation, block-length selection, and Python report construction. The
bootstrap kernels are compiled because they operate on ``(B, T[, k])`` panels;
one-off scalar Sharpe and moment reductions remain eager because compiling a
short return vector costs more than it saves.

Stage position: S8. The imports run S8 -> S7 (block length) and S8 -> S5
(circular block bootstrap), both legal under the t46 fence.

Step implementations live in the private :mod:`._sharpe` package (the merged
surface is ~490 lines, past the 250-line knee); this module owns the public
surface every caller addresses by name.

Corpus keys to stage (author acquires; do NOT fabricate @cite):
``bailey_deflated_2014``, ``romano_wolf_2005``, ``white_reality_2000``,
``hansen_spa_2005``, ``politis_stationary_1994``, ``politis_automatic_2004``,
``patton_correction_2009``, ``ledoit_wolf_2008``.
"""

from __future__ import annotations

from jcor.decision._sharpe.deflated import (
    DeflatedSharpeResult,
    deflated_sharpe_ratio,
    expected_maximum_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from jcor.decision._sharpe.difference import (
    SharpeDifferenceKernelResult,
    sharpe_difference_test,
    sharpe_difference_test_kernel,
    sharpe_ratio_ci,
)
from jcor.decision._sharpe.multiple import (
    MultipleSharpeKernelResult,
    MultipleSharpeResult,
    multiple_sharpe_test,
    multiple_sharpe_test_kernel,
)
from jcor.decision._sharpe.ratio import sharpe_ratio

__all__ = [
    "DeflatedSharpeResult",
    "MultipleSharpeKernelResult",
    "MultipleSharpeResult",
    "SharpeDifferenceKernelResult",
    "deflated_sharpe_ratio",
    "expected_maximum_sharpe_ratio",
    "multiple_sharpe_test",
    "multiple_sharpe_test_kernel",
    "probabilistic_sharpe_ratio",
    "sharpe_difference_test",
    "sharpe_difference_test_kernel",
    "sharpe_ratio",
    "sharpe_ratio_ci",
]
