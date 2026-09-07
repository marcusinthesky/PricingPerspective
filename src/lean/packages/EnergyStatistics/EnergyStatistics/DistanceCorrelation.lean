import Mathlib.Analysis.Real.Sqrt
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import EnergyStatistics.Defs
import EnergyStatistics.DistanceCovariance

set_option linter.style.longLine false

/-!
# Distance Correlation

This module formalizes the distance correlation coefficient (dCor),
following Székely & Rizzo (2023), "The Energy of Data and Distance Correlation".

## Main definitions

* `distanceCorrelation`: Distance correlation dCor(X, Y)

## Main theorems

* `dcor_bounds`: 0 ≤ dCor(X, Y) ≤ 1

## References

* Székely, G. J., & Rizzo, M. L. (2023). The Energy of Data and Distance Correlation.
  CRC Press. Chapter 12.
-/

namespace EnergyStatistics

/-- Distance correlation coefficient.

    dCor(X, Y) = √( dCov²(X, Y) / √(dVar²(X) · dVar²(Y)) )

    Equivalently (Definition 12.3, eq. 12.8):

    R²(X, Y) = V²(X, Y) / √(V²(X) · V²(Y))

    when V²(X) · V²(Y) > 0, and R(X, Y) = 0 otherwise.
    Here R = √(R²) is the distance correlation coefficient.

    This is the normalized version of distance covariance, analogous to
    how Pearson correlation normalizes product-moment covariance.
    Distance correlation is scale invariant (Remark 12.1).

    Reference: Székely & Rizzo (2023), Definition 12.3, eq. (12.8). -/
noncomputable def distanceCorrelation
    {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    [PseudoMetricSpace α] [PseudoMetricSpace β]
    (π : JointProbabilityMeasure α β) : ℝ :=
  let dVarX := distanceVarianceSq π
  let dVarY := distanceVarianceSqSnd π
  if _h : dVarX * dVarY > 0 then
    -- R²(X,Y) = dCov²(X,Y) / sqrt(dVar²(X)·dVar²(Y))  [eq. 12.8]
    -- R(X,Y) = sqrt(R²)
    Real.sqrt (distanceCovarianceSq π / Real.sqrt (dVarX * dVarY))
  else 0

/-- Distance correlation is bounded between 0 and 1.

    0 ≤ dCor(X, Y) ≤ 1

    Lower bound: Follows from dCov² ≥ 0 (dcov_nonneg) and the
    nonnegativity of the square root.
    Upper bound: Follows from the Cauchy-Schwarz inequality applied
    to the weighted L² inner product defining distance covariance.

    Reference: Székely & Rizzo (2023), Section 12.5, Property 1. -/
theorem dcor_bounds
    {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    [PseudoMetricSpace α] [PseudoMetricSpace β]
    (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π) :
    0 ≤ distanceCorrelation π ∧ distanceCorrelation π ≤ 1 := by
  unfold distanceCorrelation
  set dVarX := distanceVarianceSq π with hdVarX
  set dVarY := distanceVarianceSqSnd π with hdVarY
  dsimp only
  split
  · -- Case: dVarX * dVarY > 0
    rename_i h_pos
    constructor
    · exact Real.sqrt_nonneg _
    · rw [Real.sqrt_le_one]
      -- Cauchy–Schwarz in L²(π⊗π) via the U-centered representations:
      -- (dCov²)² = (∫ CX·CY)² ≤ (∫ CX²)·(∫ CY²) = dVar²(X)·dVar²(Y)
      have hCX := memLp_two_centeredDistFst π hm
      have hCY := memLp_two_centeredDistSnd π hm
      have hf2 : MeasureTheory.Integrable (fun z => centeredDistFst π z ^ 2)
          (π.measure.prod π.measure) :=
        (MeasureTheory.memLp_two_iff_integrable_sq hCX.aestronglyMeasurable).mp hCX
      have hg2 : MeasureTheory.Integrable (fun z => centeredDistSnd π z ^ 2)
          (π.measure.prod π.measure) :=
        (MeasureTheory.memLp_two_iff_integrable_sq hCY.aestronglyMeasurable).mp hCY
      have hfg : MeasureTheory.Integrable
          (fun z => centeredDistFst π z * centeredDistSnd π z)
          (π.measure.prod π.measure) := hCX.integrable_mul hCY
      have hCS := integral_mul_sq_le (centeredDistFst π) (centeredDistSnd π) hf2 hg2 hfg
      rw [← dcov_eq_integral_mul π hm, ← dvar_eq_integral_sq π hm,
        ← dvarSnd_eq_integral_sq π hm, ← hdVarX, ← hdVarY] at hCS
      -- hCS : (dCov²)² ≤ dVarX * dVarY
      have hsqrt_pos : 0 < Real.sqrt (dVarX * dVarY) := Real.sqrt_pos.mpr h_pos
      rw [div_le_one hsqrt_pos]
      calc distanceCovarianceSq π ≤ |distanceCovarianceSq π| := le_abs_self _
        _ = Real.sqrt (distanceCovarianceSq π ^ 2) := (Real.sqrt_sq_eq_abs _).symm
        _ ≤ Real.sqrt (dVarX * dVarY) := Real.sqrt_le_sqrt hCS
  · -- Case: dVarX * dVarY ≤ 0
    exact ⟨le_refl 0, zero_le_one⟩

end EnergyStatistics
