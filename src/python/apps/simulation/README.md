---
description: "GPU-capable JAX Monte-Carlo simulation package for energy/Wasserstein metrics and portfolio optimisation."
title: Simulation
---

## simulation

GPU-capable JAX Monte-Carlo package consolidating the four ROADMAP §4 validation
scripts into a single, reusable, DVC-stage-consumable library. It provides
generators, distance metrics, portfolio optimizers, certified bounds, and a
permutation-test harness — all built on JAX for JIT-compilation and optional GPU
acceleration.

### Module map

| Module | One-line description |
|---|---|
| `backend.py` | `configure(precision, platform) → BackendInfo`; must be called first in every DVC wrapper |
| `generators/fields.py` | Cholesky-on-grid factor-field samplers (white, exponential, Matern-3/2 kernels) |
| `generators/assets.py` | Asset cluster-embedding draws (`cluster_embeddings`) |
| `generators/returns.py` | Return-panel generators (covariance draws, systematic-factor panels) |
| `radii.py` | Entity-neutral mean-embedding bootstrap radii and triangle-inequality pair composition |
| `metrics/energy.py` | `pairwise_energy_distance` — row-chunked via `lax.map` (avoids O(n·m·d) materialisation) |
| `metrics/__init__.py` | Unified `pairwise_distance_matrix(samples, metric)` — same `(n, n)` contract for all metrics; 1-D/sliced Wasserstein branches dispatch to `jcor.discrepancy.transport` |
| `optimize.py` | `pgd_min_variance`/`pgd_max_quadratic` — domain-named wrappers over `jcor.optimize.pgd_minimize_quadratic_form`/`jcor.optimize.pgd_maximize_quadratic_form`; `mixture_weights_qp`, `energy_barycentre_weights_qp` — `jcor.discrepancy.balancing` glue. `ridge_psd` moved to `jcor` outright. |
| `bounds.py` | `variance_floor`, `certified_floor`, `hedging_bound`, `lipschitz_constants`; `certified_floor_np` preserves float64 artifact arithmetic at the eager boundary |
| `portfolios.py` | Key-only named weight schemes (uniform, min-var, pf-max-spread, MVO, etc.) |
| `covariance.py` | Identity-stable compatibility facade: constructors are canonical in `jcor.operators.covariance`; covariance and shrinkage estimators in `jcor.model.covariance` |
| `psd_repair.py` | Identity-stable compatibility facade over `jcor.optimize.psd` (Higham, diagonal shrink, ridge/clip baselines and diagnostics) |
| `jcor.operators.longrun` (dependency) | Canonical Newey–West and Hall-centered HAC operators; the former `simulation.hac` compatibility module was removed in t46.8. Block-length/bootstrap inference lives in `jcor.inference`. |
| `inference/` | Identity-stable compatibility package over `jcor.inference`: GMM/GEL equality, Wolak/uncorrected indicator-GMS inequality, clustering and κ-boundary inference |
| `validation/variance_floor` | Portfolio variance floor Monte-Carlo validation |
| `validation/hedging_error` | Energy-barycenter hedging-error bound validation |
| `validation/pricefree` | Price-free portfolio construction vs sample-MVO |
| `validation/size_power` | Type I / II (size/power) energy permutation test |
| `validation/equivalence_calibration` | Paper-2 boundary size/FWER/power for energy equivalence under correlated claims |
| `validation/gmm_wolak_size_power` | Paper-3: GMM Wald + Wolak size/power MC (gates Wave 2; calibrates χ̄² weights) |
| `validation/kappa_audit` | Paper-3: κ identity-form audit (D² vs ½κ²D² vs ½κ_iκ_jD²) |
| `cli.py` | Typer CLI entry-point for all validation experiments |
| `harness/stats.py` | Rejection rate, SE, z-score, size/power summary helpers |
| `harness/replications.py` | Replication-key derivation plus memory-bounded `vmap`/`lax.map` execution |
| `io.py` | Parquet + YAML summary readers/writers matching DVC output contracts |

### Quick start

#### Installing (uv workspace)

```bash
# From the repo root — picks up simulation via [tool.uv.workspace] members:
uv sync --all-packages
```

Inside a DVC script (PEP 723 header):

```python
# /// script
# dependencies = ["simulation = { path = '../../apps/simulation' }"]
# ///
```

#### Justfile targets

Run from `src/python/apps/simulation/`:

```bash
just test        # fast tests only (excludes slow + gpu markers, <30 s)
just test-all    # full suite including slow statistical tests
```

#### CLI (validation experiments)

```bash
uv run simulation --help
uv run simulation validate-variance-floor
uv run simulation validate-hedging-error
uv run simulation compare-pricefree
uv run simulation simulate-size-power
uv run simulation simulate-equivalence-calibration
```

#### Backend first-call pattern

`simulation.configure` sets process-global JAX flags. Call it **before** any
other JAX computation; a second call after JIT-compiled kernels have run is
undefined behaviour.

```python
import simulation

info = simulation.configure(precision="float64", platform="cpu")
# BackendInfo(platform='cpu', dtype='float64', device_count=1, jax_version='...')
print(info)
```

Certificate stages always call `configure(precision="float64")`.
Statistical stages (size/power) may use `configure(precision="float32")` on GPU.

### Seeds policy

Seeded validators use `jcor.core.random.cell_key(base_seed, *labels)`:

- Each grid cell receives a **unique, independent** PRNGKey derived by MD5-hashing
  each label to a stable 32-bit integer and folding it into the running key via
  `jax.random.fold_in`.
- **Grid reordering never shifts streams**: the key depends only on
  `(base_seed, stage_name, semantic_cell_labels...)`, not on iteration order.
- Within a cell, replication `i` is always
  `jax.random.fold_in(cell_key, i)`. Increasing `n_sims` therefore preserves
  the existing prefix. Each kernel splits that replication key once, in an
  explicitly documented order, for distinct draws.
- Label ordering is order-sensitive by design — callers fix a canonical
  ordering such as `cell_key(seed, "mc_pricefree", n_assets, embed_dim, T)`.

**Breaking change from donor scripts**: t66 rerolls all simulation streams from
PCG64 to Threefry. Reroll acceptance is statistical: rates and means must remain
within three combined Monte-Carlo standard errors, while medians and quantiles
use preregistered fixed-method bootstrap intervals. Exact/dtype-aware parity is
reserved for comparing different execution schedules over the same Threefry
keys. Schemas and artifact paths remain stable.

### Precision policy

| Stage | Default dtype | Device | Rationale |
|---|---|---|---|
| Certificate (floor, hedging, pricefree) | float64 | CPU | Tolerance 1e-9 requires x64; `jax_enable_x64=True` |
| Statistical (size, power) | float32 | GPU-eligible | Invariants are tolerance-aware; GPU throughput worthwhile |

On GPU: TF32 is disabled via `jax_default_matmul_precision = "highest"` to
prevent implicit precision reduction in matrix multiplications.

Cross-backend honesty: CPU and GPU results are **not bit-identical** (XLA
reduction order differs across devices). `summary.yaml` always embeds
`backend: {platform, dtype, jax_version}` so results are traceable.
JAX threefry PRNG draws are backend-independent (same draw sequence; accumulation
order differs).

### Host-array boundary

`simulation` is the deliberate destination of `jcor`'s NumPy host door (t65), so
NumPy here is a host/report boundary that is *kept and measured*, not a random
number generator or a traced numerical dependency.
`docs/numpy-scipy-boundary-ledger.md` records the per-module census, the reasons
each surface stays, and why pytree registration is not yet the answer. The counts
are checked by Ruff, Deptry, Pyrefly, and focused package tests, so an unrecorded
narrowing fails the gate the same way growth does.

**SciPy is not a runtime dependency.** It lives in the `dev` group as a test-time
reference oracle only; `equivalence_calibration` uses stdlib `math.erf` and a
local `_bisect_increasing` in place of `scipy.special.erf` and
`scipy.optimize.brentq`.

### Architecture

See `ARCHITECTURE.md` for design rationale (comparison tables: why JAX, why
`apps/`, GPU trade-offs, no-POT/OTT decision, field samplers, config schemas,
optimizer choice, inference reservation).

<!-- insitu:begin gittree
id = "gittree"
path = "src/python/apps/simulation"
depth = 1
-->

```text
.
├── docs
├── src
└── tests
```

<!-- insitu:end -->
