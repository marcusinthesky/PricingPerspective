---
description: Reusable Lean 4 libraries and formalizations.
---

# src/lean/packages/

Lake packages containing formal proofs and reusable Lean 4 libraries.

## Contents

| Package | Description |
|---|---|
| `PricingPerspective/` | Energy distance lower bounds on systematic covariance and correlation; depends on the other three packages |
| `PaperReconstructions/` | Finite and approximate APT reconstructions; canonical factor model definitions in `Ross1976/` |
| `EnergyStatistics/` | Energy distance and distance covariance/correlation foundations (mathlib-publishable) |
| `WassersteinGeometry/` | Optimal-transport definitions, geodesics, Fréchet mean (mathlib-publishable) |

## Directory Tree

<!-- insitu:begin gittree
id = "gittree"
path = "src/lean/packages"
depth = 1
-->

```text
.
├── EnergyStatistics
├── PaperReconstructions # Lean 4 reconstruction of finite and approximate APT results.
├── PricingPerspective
└── WassersteinGeometry
```

<!-- insitu:end -->

Each package has its own `lakefile.lean` declaring dependencies and a
`lean_lib` target. The root `lakefile.lean` declares `require` entries that
point to these local paths, resolved via `lake-manifest.json`.

The planned packages are documentation scaffolds only until their `lakefile.lean`
and compiling Lean roots are added. Do not add them to the root workspace until
they build independently. See `../ARCHITECTURE.md` for dependency policy and
`../README.md` for the handoff protocol.
