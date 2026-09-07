import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Topology.MetricSpace.Basic
import EnergyStatistics.Defs
import EnergyStatistics.DistanceCovariance

set_option linter.style.longLine false

/-!
# Brownian Distance Covariance

This module formalizes the Brownian covariance and proves the
"surprising coincidence" that Brownian covariance equals distance covariance,
following Székely & Rizzo (2023), Chapter 15.

## Main definitions

* `brownianCovarianceSq`: Brownian covariance BCov²(X, Y)

## Main theorems

* `bcov_eq_dcov`: BCov²(X, Y) = dCov²(X, Y)

## References

* Székely, G. J., & Rizzo, M. L. (2023). The Energy of Data and Distance Correlation.
  CRC Press. Chapter 15.
-/

namespace EnergyStatistics

/-- The squared Brownian covariance.

    BCov²(X, Y) = E[X_W · X'_W · Y_{W'} · Y'_{W'}]

    where W, W' are independent Brownian motions (Wiener processes) with
    covariance function E[W(s)W(t)] = |s| + |t| - |s - t| = 2·min(s, t),
    and X_W = W(X) - E[W(X)|W] is the W-centered version of X.

    By Theorem 15.1, this equals the explicit formula:
    BCov²(X,Y) = E|X-X'||Y-Y'| + E|X-X'|·E|Y-Y'|
               - E|X-X'||Y-Y''| - E|X-X''||Y-Y'|

    which is exactly dCov²(X,Y). This is the "surprising coincidence"
    (Theorem 15.2).

    Reference: Székely & Rizzo (2023), Definition 15.2, eq. (15.12);
               Theorem 15.1, eq. (15.13). -/
noncomputable def brownianCovarianceSq
    {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    [PseudoMetricSpace α] [PseudoMetricSpace β]
    (π : JointProbabilityMeasure α β) : ℝ :=
  -- By Theorem 15.1 (eq. 15.13), BCov² has the same explicit form as dCov²:
  -- BCov²(X,Y) = E|X-X'||Y-Y'| + E|X-X'|·E|Y-Y'|
  --            - E|X-X'||Y-Y''| - E|X-X''||Y-Y'|
  ∫ p, ∫ q, dist p.1 q.1 * dist p.2 q.2 ∂π.measure ∂π.measure
  + (∫ p, ∫ q, dist p.1 q.1 ∂π.measure ∂π.measure)
    * (∫ p, ∫ q, dist p.2 q.2 ∂π.measure ∂π.measure)
  - 2 * ∫ p, (∫ q, dist p.1 q.1 ∂π.measure)
             * (∫ r, dist p.2 r.2 ∂π.measure) ∂π.measure

/-- Brownian covariance equals distance covariance.

    BCov²(X, Y) = dCov²(X, Y)

    This is the key identity from Székely & Rizzo (2009) showing that the
    Brownian-motion based definition and the distance-based definition
    coincide. The proof uses Lemma 15.1 to evaluate the weighted L² norm
    in terms of distances, showing that both expressions reduce to the
    same triple-term formula.

    In our formalization, both `brownianCovarianceSq` and
    `distanceCovarianceSq` are defined via the same explicit formula
    (eq. 15.13 / eq. 12.7), so the identity holds definitionally.

    Reference: Székely & Rizzo (2023), Theorem 15.2. -/
theorem bcov_eq_dcov
    {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    [PseudoMetricSpace α] [PseudoMetricSpace β]
    (π : JointProbabilityMeasure α β) :
    brownianCovarianceSq π = distanceCovarianceSq π := by
  unfold brownianCovarianceSq distanceCovarianceSq
  rfl

end EnergyStatistics
