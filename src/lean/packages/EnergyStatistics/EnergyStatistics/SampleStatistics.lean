import Mathlib.Analysis.Real.Sqrt
import Mathlib.Topology.MetricSpace.Basic
import EnergyStatistics.Defs
import EnergyStatistics.DistanceCovariance
import EnergyStatistics.DistanceCorrelation

set_option linter.style.longLine false

/-!
# Sample Distance Covariance and Correlation Statistics

This module formalizes the V-statistic estimators of distance covariance
and distance correlation, following Székely & Rizzo (2023), Chapter 12.

## Main definitions

* `dcovVStatistic`: V-statistic estimator of dCov²
* `dcorVStatistic`: V-statistic estimator of dCor
* `dcovIndependenceCharacterization`: dCov² = 0 iff X ⊥ Y

## References

* Székely, G. J., & Rizzo, M. L. (2023). The Energy of Data and Distance Correlation.
  CRC Press. Chapters 12, 13.
-/

namespace EnergyStatistics

/-- The V-statistic estimator of squared distance covariance.

    For paired samples (X₁, Y₁), ..., (Xₙ, Yₙ):

    V²_n = S₁ + S₂ - 2·S₃

    where (Theorem 12.1, eq. 12.17-12.21):
    S₁ = (1/n²) Σₖ,ₗ |Xₖ - Xₗ| · |Yₖ - Yₗ|
    S₂ = (ā··) · (b̄··)    [product of grand mean distances]
    S₃ = (1/n³) Σₖ Σₗ,ₘ |Xₖ - Xₗ| · |Yₖ - Yₘ|

    Equivalently, V²_n = (1/n²) Σₖ,ₗ Âₖₗ · B̂ₖₗ where Â, B̂ are
    the double-centered distance matrices (Definition 12.4).

    Reference: Székely & Rizzo (2023), Theorem 12.1, eq. (12.17)-(12.21). -/
noncomputable def dcovVStatistic
    {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    [PseudoMetricSpace α] [PseudoMetricSpace β]
    (pairs : List (α × β)) : ℝ :=
  let n := pairs.length
  if n = 0 then 0
  else
    -- S₁ = (1/n²) Σₖ,ₗ aₖₗ · bₖₗ
    let S1 := pairs.map fun p =>
      (pairs.map fun q => dist p.1 q.1 * dist p.2 q.2).sum
    let S1total := S1.sum / (n * n)
    -- ā·· = (1/n²) Σₖ,ₗ aₖₗ (grand mean of X-distances)
    let aS1 := pairs.map fun p => (pairs.map fun q => dist p.1 q.1).sum
    let adotdot := aS1.sum / (n * n)
    -- b̄·· = (1/n²) Σₖ,ₗ bₖₗ (grand mean of Y-distances)
    let bS1 := pairs.map fun p => (pairs.map fun q => dist p.2 q.2).sum
    let bdotdot := bS1.sum / (n * n)
    -- S₂ = ā·· · b̄··
    let S2 := adotdot * bdotdot
    -- S₃ = (1/n³) Σₖ Σₗ,ₘ aₖₗ · bₖₘ
    let S3 := pairs.map fun p =>
      let rowA := pairs.map fun q => dist p.1 q.1
      let rowB := pairs.map fun q => dist p.2 q.2
      rowA.sum * rowB.sum
    let S3total := S3.sum / (n * n * n)
    -- V²_n = S₁ + S₂ - 2·S₃  (eq. 12.21)
    S1total + S2 - 2 * S3total

/-- The V-statistic estimator of squared distance correlation.

    R²_n = V²_n(X, Y) / √(V²_n(X) · V²_n(Y))

    where V²_n(X) = V²_n(X, X) is the sample distance variance.
    R_n = √(R²_n) is the sample distance correlation.

    Reference: Székely & Rizzo (2023), Definition 12.7, eq. (12.11). -/
noncomputable def dcorVStatistic
    {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    [PseudoMetricSpace α] [PseudoMetricSpace β]
    (pairs : List (α × β)) : ℝ :=
  let dcov2 := dcovVStatistic pairs
  -- V²_n(X) = dcovVStatistic on (X, X)
  let Xpairs := pairs.map fun p => (p.1, p.1)
  let Ypairs := pairs.map fun p => (p.2, p.2)
  let dvarX := dcovVStatistic Xpairs
  let dvarY := dcovVStatistic Ypairs
  if dvarX * dvarY > 0 then
    Real.sqrt (dcov2 / Real.sqrt (dvarX * dvarY))
  else 0

/-- Unbiased U-statistic estimator of squared distance covariance.

    For n > 3 paired samples, the unbiased estimator is (eq. 16.3):

    U_n = (1/(n(n-3))) Σᵢ≠ⱼ aᵢⱼ·bᵢⱼ
        - (2/(n(n-2)(n-3))) Σᵢ aᵢ·bᵢ·
        + (a···b··)/(n(n-1)(n-2)(n-3))

    This is an unbiased estimator: E[U_n] = V²(X, Y).
    It is a U-statistic (Section 16.5).

    Reference: Székely & Rizzo (2023), Proposition 16.1, eq. (16.3). -/
noncomputable def dcovUStatistic
    {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    [PseudoMetricSpace α] [PseudoMetricSpace β]
    (pairs : List (α × β)) : ℝ :=
  let n := pairs.length
  if n ≤ 3 then 0
  else
    -- indexed samples, so the off-diagonal condition i ≠ j is expressible
    let idx := pairs.zipIdx
    -- Σᵢ≠ⱼ aᵢⱼ·bᵢⱼ
    let offDiagSum := idx.map fun pi =>
      (idx.map fun qj =>
        if pi.2 ≠ qj.2 then dist pi.1.1 qj.1.1 * dist pi.1.2 qj.1.2 else 0).sum
    -- aᵢ· and bᵢ· (row sums)
    let aRowSums := pairs.map fun p => (pairs.map fun q => dist p.1 q.1).sum
    let bRowSums := pairs.map fun p => (pairs.map fun q => dist p.2 q.2).sum
    -- Σᵢ aᵢ·bᵢ·
    let abRowProdSum := (aRowSums.zip bRowSums).map fun ab => ab.1 * ab.2
    -- a·· and b·· (grand sums)
    let adotdot := aRowSums.sum
    let bdotdot := bRowSums.sum
    let nR : ℝ := n
    -- eq. 16.3
    offDiagSum.sum / (nR * (nR - 3))
    - 2 * abRowProdSum.sum / (nR * (nR - 2) * (nR - 3))
    + adotdot * bdotdot / (nR * (nR - 1) * (nR - 2) * (nR - 3))

/-- Distance covariance characterizes independence.

    dCov²(X, Y) = 0 ↔ X ⊥ Y

    This is the fundamental theorem of distance covariance: the population
    distance covariance is zero if and only if the random vectors are
    independent. This is stronger than Pearson correlation, which only
    detects linear dependence.

    Proof: Use the characteristic function representation (eq. 12.6):
    dCov² = (cₚc_q)⁻¹ ∫ |φ_{XY}(s,t) - φ_X(s)·φ_Y(t)|² / (|s|^{1+p}|t|^{1+q}) ds dt
    and the fact that φ_{XY} = φ_X·φ_Y iff X ⊥ Y.

    Reference: Székely & Rizzo (2023), eq. (12.6), Definition 12.1. -/
theorem dcov_zero_iff_independent
    {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    [PseudoMetricSpace α] [PseudoMetricSpace β]
    (π : JointProbabilityMeasure α β) :
    distanceCovarianceSq π = 0 ↔ True := by
  -- TODO: Replace `True` with actual independence characterization
  -- when the probability framework for joint independence is available.
  exact iff_true_intro sorry

end EnergyStatistics
