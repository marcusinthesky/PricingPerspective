---
description: Formal verification of asset-pricing theorems in Lean 4 + mathlib — a Lake multi-package workspace.
---

# src/lean/

Lean 4 proofs backing the theoretical results in the papers. A
[Lake](https://github.com/leanprover/lean4/tree/master/src/lake) multi-package workspace:
the root `lakefile.lean` is a thin aggregator (`lean_lib «Main»`) that `require`s the four
packages below. Why this shape → `ARCHITECTURE.md`; rules for editing proofs → `AGENTS.md`.

## Quick start

```bash
just lean::cache     # fetch the mathlib build cache (first build only)
just lean::build     # build every package (runs cache first)
```

Recipes `cd` into `src/lean`, so run them from anywhere; bare `lake build` from
`src/lean/` also works.

## Structure

<!-- insitu:begin gittree
id = "gittree"
path = "src/lean"
depth = 1
-->

```text
.
├── blueprint
├── packages # Reusable Lean 4 libraries and formalizations.
├── scripts
└── tools
```

<!-- insitu:end -->

## Packages

| Member | Lake target | Purpose |
|---|---|---|
| `packages/PricingPerspective` | `lean_lib «PricingPerspective»` | Discrete + continuous APT, the spatial pricing core, and energy/Wasserstein lower bounds on systematic co-movement. Depends on the three foundations below. |
| `packages/PaperReconstructions` | `lean_lib «PaperReconstructions»` | Canonical factor-model definitions (`Ross1976/`) and reconstructions: Ross, Chamberlain–Rothschild, Ingersoll, Reisman, Shanken, Middleton–Satchell, Connor, Di Nunno. |
| `packages/EnergyStatistics` | `lean_lib «EnergyStatistics»` | Energy distance, distance covariance/correlation foundations (mathlib-publishable). |
| `packages/WassersteinGeometry` | `lean_lib «WassersteinGeometry»` | Optimal transport, finite gluing, Wasserstein geodesics, and Fréchet/MMOT barycenter geometry (mathlib-publishable). |

`mathlib`, `doc-gen4`, and `mdgen` are declared at the workspace root and pinned to
`v4.31.0`; see `ARCHITECTURE.md` for the dependency graph and the hoisted-pin rationale.

Within a package, files are organized by mathematical concept (mathlib's
`Defs.lean`/`Basic.lean` convention); a concept that outgrows its file is promoted to a
same-named directory — see `AGENTS.md` for the rule, `ARCHITECTURE.md` for the rationale.

## Proof status

The headline results are unconditional lower bounds tying systematic co-movement to
distance-covariance / energy geometry, machine-checked with **zero custom `axiom`
declarations** (economic assumptions are explicit hypotheses). The authoritative, live
inventory is machine-checked — trust it over any prose list:

```bash
bash scripts/sorry_gate.sh    # per-file sorry/axiom counts vs sorry-allowlist.txt
```

Per-theorem statements live in the source docstrings; open work is tracked in `TODO.md`
and specified under `.context/plan/{proofs,paper1,paper3,paper5,paper6}/`.
