import WassersteinGeometry.Real.Barycenter
import WassersteinGeometry.Real.Tangent

/-!
# Real-line Fréchet means and tangent maps

This public roll-up exposes the real-line API.  Generic coupling and distance definitions remain in
`WassersteinGeometry.Geodesics`; real barycenters require a finite nonempty family carrying explicit
quantile certificates, and deterministic logarithmic maps require an `OptimalTransportMap` witness.

The pinned mathlib provides CDFs and the package exports a
`CanonicalQuantileSpec` certificate for the generalized inverse. The real
quantile contract remains explicit in that certificate; no vacuous `True`
assumptions or unconditional deterministic-map claims are exported.

References: Panaretos & Zemel (2020), Chapters 2–4.
-/
