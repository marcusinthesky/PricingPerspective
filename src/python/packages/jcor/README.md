# jcor: JAX Distance Correlation and Energy Statistics

High-performance JAX implementation of distance correlation,
energy statistics, and permutation tests with JIT compilation
and GPU acceleration.

## Quick Start

```bash
cd packages/jcor && uv sync
```

```python
from jcor.discrepancy.energy import UNIT_EXPONENT, energy_distance, energy_geometry
from jcor.ground.metrics import ANGULAR, EUCLIDEAN, cdist
from jcor.inference import energy_permutation_test
import jax.numpy as jnp

x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
y = jnp.array([[5.0, 6.0], [7.0, 8.0]])

dist = cdist(x, y, metric=EUCLIDEAN)
e_dist = energy_distance(x, y, metric=ANGULAR)
result = energy_permutation_test(
    x,
    y,
    energy_geometry(EUCLIDEAN, UNIT_EXPONENT),
    num_permutations=9999,
    seed=42,
)
print(f"p-value: {result.pvalue:.4f}")
```

## Reference and conformance checks

Independent-oracle checks are opt-in and do not enter the runtime wheel or the
default development environment:

```bash
uv sync --group reference
uv run --group reference pytest -m reference
uv run --group reference pytest -m conformance
```

Reference tests also carry `requires_optional("<distribution>")` and skip
cleanly when their oracle is unavailable. `reference` means an independent
oracle comparison; `conformance` is reserved for an externally promised
compatibility contract.

## Stages

Organised by **position in the inference chain**, not by mathematical
provenance (t46). A stage imports only *earlier* stages plus the rank-0
substrate; Ruff/Tach enforce package direction and review owns this internal
stage convention. **Import from the stage** — the
root re-exports a small headline set for convenience, but the stage path is what
survives a surface change.

| stage | rank | purpose |
|---|---|---|
| `core` | 0 | contracts: `DMat[Ax]` axiom lattice, jaxtyping aliases, `TestResult`/`EstimateResult`, `cell_key` |
| `optimize` | 0 | simplex-constrained PGD; traced PSD repairs — Qi-Sun nearest correlation/covariance, ridge, clip, diagonal shrink |
| `sample` | 1 | raw clouds: packing, reduction, alternative-hypothesis samplers |
| `ground` | 2 | `d: X x X -> R` on points — `cdist`, `pdist`, angular/cosine/euclidean |
| `discrepancy` | 3 ◄ **waist 1** | `d: P x P -> R` on measures — energy, MMD, transport, exact W1, metrization, energy balancing |
| `association` | 4 | distance matrix → coefficient: Mantel, DISCO |
| `geometry` | 4 | distance matrix → structure: PCoA, Weiszfeld medians, energy barycentres, neighbour stability |
| `operators` | 5 ◄ **waist 2** | structured operators: long-run/HAC and distance-implied covariance; rank-clean dyadic designs |
| `model` | 6 | covariance/shrinkage estimators; dyadic least squares and multinomial node bootstrap |
| `inference` | 7 | resampling tests and block length; GMM/GEL equality, Wolak/GMS inequality, clustering and boundary inference |
| `decision` | 8 | equivalence & TOST, multiplicity control, Sharpe |
| `experimental` | — | staging. **No semver promise**; stable stages may not import it |

Below waist 1 anything producing a distance matrix is interchangeable; above it,
nothing knows which produced it. Same at waist 2. That is why novel combinations
— energy divergences standing in for covariance, spatial operators built from
energy geometry — are free rather than boundary violations.

[`docs/index.md`](docs/index.md) indexes the same code by the **inferential
question** (independence · homogeneity · equivalence · estimation ·
calibration), which is how people actually arrive at the library.

> **Removed in t46**: `jcor.metrics`, `jcor.statistics`, `jcor.wasserstein`,
> `jcor.kernels`, `jcor.permutation`, `jcor.hypothesis`, `jcor.equivalence`,
> `jcor.block_length`, `jcor.mantel`, `jcor.seeds`, `jcor.pit_energy`,
> `jcor._energy_components`. Deleted, not deprecated — see `CHANGELOG.md` for
> the destination of each. No symbol was renamed by the move.
>
> **Caveat worth reading before you copy the snippet above.**
> `energy_distance` defaults to the angular pullback, which normalises before
> comparing and therefore *cannot see scale*: `energy_distance(X, 3X) == 0` on
> the default. Pass the `EUCLIDEAN` strategy wherever identification matters. The
> declared axioms and `Premetric` brand say so; `docs/index.md` has the full
> table.

## Validity-corrected hedge-quality inference (Paper 2)

The legacy `energy_kernel_permutation_test` / `mixture_approximation_test` read a
**high** permutation p-value as a **good** hedge — an inverted, double-dipping,
non-exchangeable, mis-scaled test. They have been removed; the corrections below
replace them with a composed, defensible statistic spread across the modules
above by topic (not one `corrected` module). **The core public API is
unchanged** (`energy_distance`, `energy_test_statistic`, `permutation_test`,
`energy_distance_kernel`); these are new entry points alongside it.

| Entry point | Module | Correction |
|---|---|---|
| `inverse_variance_m_eff` | `discrepancy.balancing` | #1 effective size `1/m_eff = Σ w_k²/m_k` (Kish; not `wᵀm`) |
| `equivalence_test` | `decision.equivalence` | #2 relevance null `H0: E ≥ δ`; reports `Ê` + bootstrap CI + equivalence curve |
| `corrected_mixture_test` | `inference.mixture` | #3 re-optimize `w*` in-permutation OR select-A/test-B split |
| `studentized_mixture_statistic` | `inference.studentized` | #4 studentized (Chung–Romano) mixture statistic |
| `fit_projection` | `sample.reduction` | #5 dim-reduction (fit on split A only) |
| `poor_hedge_margin` | `decision.equivalence` | δ-margin selection (economically-anchored) |
| `equivalence_multiplicity_control` | `decision.multiplicity` | FWER via `(k−1)` scaling / adaptive-Bonferroni (BH guarded) |
| `optimal_block_length`, `stationary_bootstrap_ci` | `inference.block_length` | OOS / CI block length: `scheme=stationary` → `D_SB=2g²(0)`; `circular` → `D_MB=(4/3)g²(0)` |

```python
import jcor, jax.numpy as jnp

# Equivalence-framed, studentized, re-optimized (all three at once):
result = jcor.equivalence_test(  # also jcor.decision.equivalence.equivalence_test
    x,
    candidates,
    delta=0.5,
    selection_split=None,
    reoptimize_in_permutation=True,  # full-sample route
)
print(result.energy_point_estimate, result.energy_ci)  # PRIMARY metric
print(result.equivalence, result.equivalence_curve)  # δ verdict + curve
```

The δ-margin is an **explicit, documented parameter** with no consensus standard
for energy distance — justify it (see `poor_hedge_margin`) rather than asserting
it. `m_eff` is the non-degenerate / studentization normalizer, **not** an
exact-null quantity (the null statistic is degenerate).

## Performance

With JIT + GPU: 0.03-0.10 ms distance metrics, 10k permutations in ~50ms.

## Documentation

- [AGENTS.md](AGENTS.md) — Agent conventions
- [notebooks/](notebooks/) — Monte Carlo simulations
- [docs/index.md](docs/index.md) — curated guide by inferential question

## References

- Székely & Rizzo (2013). *Energy statistics: A class of statistics based on distances.*
- Székely & Rizzo (2004). *Testing for equal distributions in high dimension.*
