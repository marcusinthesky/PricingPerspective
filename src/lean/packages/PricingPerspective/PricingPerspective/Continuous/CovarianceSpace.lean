import PricingPerspective.Continuous.CovarianceSpace.Interchange

/-!
# Covariance space for continuous signed-measure exposures

This roll-up exposes the surviving covariance-space construction:

* variation-integrable signed exposures and their coherent real-module wrapper;
* the signed-measure field integral;
* signed-measure inner-product interchange and residual orthogonality.

## Retirement note (2026-08-21)

The null quotient, range identification, and complete closed-range Hilbert realization
(`CovarianceSpace.{Geometry, Completion, L2}`) were removed with the signed-measure
primitive they served. Nothing in the claim manifest referenced them, and the
measure-theoretic Hilbert geometry the programme is pivoting to is being built on
`WassersteinGeometry` instead — see `blueprint/src/chapters/random_exposure.tex`.

What remains is retained for one reason: `Continuous.Energy.SignedPopulation` needs the
inner-product interchange, which is proved here by Jordan decomposition. The full
identification with an iterated scalar kernel integral was never exported and would still
need an arbitrary continuous-linear-map interchange theorem.
-/
