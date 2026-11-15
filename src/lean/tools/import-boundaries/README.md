---
description: Native Lake import-graph checks for first-party Lean dependency direction.
---

# Import boundaries

Run `just lean::boundaries` after building. The check discovers all first-party
source modules and reads Lean's compiled import headers in Lake's environment,
including modules outside roll-ups. It retains direct edges without graph
presentation reductions or expensive unused-import analysis.
Foundations cannot depend on PricingPerspective. Operating models remain
independent of financial compositions, and Exposure, Spatial and Transmission
cannot import downstream Operating or Connections modules.

Exposure and Transmission may depend on each other's specific leaves. Broad
domain ordering would reject their legitimate floor and portfolio dependencies.
Lean itself checks module cycles. The tests protect dependency rules rather than
file locations or fixed module counts.
