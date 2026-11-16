---
description: Reusable Lean 4 libraries and formalizations.
---

# src/lean/packages/

Lake packages containing formal proofs and reusable Lean 4 libraries.

## Contents

| Package | Description |
|---|---|
| `PricingPerspective/` | Energy distance lower bounds on systematic covariance and correlation; depends on the energy and transport foundations |
| `EnergyStatistics/` | Energy distance, mean embeddings and empirical V-statistics |
| `WassersteinGeometry/` | Optimal-transport definitions, geodesics, finite gluing and multimarginal barycentres |

## Directory Tree

<!-- insitu:begin tree
id = "gittree"
depth = 1
directories_only = true
-->

```text
packages/
├── EnergyStatistics/
├── PricingPerspective/
└── WassersteinGeometry/
```

<!-- insitu:end -->

Each package has its own `lakefile.lean` declaring dependencies and a
`lean_lib` target. The root `lakefile.lean` declares `require` entries that
point to these local paths, resolved via `lake-manifest.json`.

See [the architecture](../ARCHITECTURE.md) for dependency policy and
[the workspace README](../README.md) for reproduction commands.
