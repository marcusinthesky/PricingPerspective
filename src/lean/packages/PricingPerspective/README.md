---
description: Economic exposure, operating, spatial and transmission results.
---

# PricingPerspective

This package owns the papers' economic mathematics. Reusable energy statistics,
transport geometry and finite factor-model types remain separate
Lake dependencies. See [workspace architecture](../../ARCHITECTURE.md) for
dependency direction and shared upstream ownership.

| Domain | Contents | Example import |
|---|---|---|
| Exposure | Covariance spaces, energy embeddings, portfolio bounds | `PricingPerspective.Exposure.Energy` |
| Operating | Variational inequalities, rational-share equilibrium and response, valuation, anchored adjustment | `PricingPerspective.Operating.RationalShare.Equilibrium` |
| Spatial | Spatial covariance, stability, Wasserstein weights and asymptotics | `PricingPerspective.Spatial.Stability` |
| Transmission | Finite-cloud and measure-level exposure transmission, envelopes and dispersion | `PricingPerspective.Transmission.Finite.Envelope` |

Operating's rational-share core lives in `RationalShare/`; its `FirmResponse`,
`EquilibriumResponse`, `Reallocation` and `SolverCertificates` subdirectories
own distinct model arguments. `Valuation` supplies independently importable
dated-response and financial readouts. The retained operating proofs support the
approved equilibrium, response and cash-flow claims.

Implementation files import specific leaves. Roll-ups expose concept families;
publication assembly selects annotated contributions separately. Namespaces follow
their conceptual owner; supporting filenames such as `Defs`, `Core` and `Model`
do not create artificial namespace levels. Superseded paths and global forwarding
declarations are removed.

Build with `just lean::check`; run `just lean::boundaries`,
`just lean::dependency-audit` and `just lean::certify` for maintained checks.
