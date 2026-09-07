import PricingPerspective.Connections.SpatialStability

/-!
# Target-anchored Wasserstein weights and SAR compatibility

The implemented Paper 5 operator first fixes target-to-candidate assignments and
then minimizes a finite quadratic objective over leave-one-out simplex weights.
This file formalizes that restricted finite program.  It makes no unrestricted
multi-marginal optimal-transport or global assignment-continuity claim.
-/

namespace PricingPerspective.Connections.SpatialWasserstein

open Matrix PricingPerspective.Discrete.Spatial
open PricingPerspective.Connections.SpatialStability
open scoped Matrix.Norms.Operator

variable {ι μ ν : Type*} [Fintype ι] [Fintype μ] [Fintype ν]

/-- Support obtained by combining candidate points after target-specific assignments. -/
noncomputable def anchoredSupport (w : ι → ℝ) (y : ι → μ → ν → ℝ) : μ → ν → ℝ :=
  fun m d ↦ ∑ j, w j * y j m d

/-- The exact finite quadratic objective implemented by the anchored W2 producer,
up to the positive normalization by the number of support points. -/
noncomputable def anchoredW2Objective
    (x : μ → ν → ℝ) (y : ι → μ → ν → ℝ) (w : ι → ℝ) : ℝ :=
  ∑ m, ∑ d, (x m d - anchoredSupport w y m d) ^ 2

/-- The anchored finite W2 objective is nonnegative. -/
theorem anchoredW2Objective_nonneg
    (x : μ → ν → ℝ) (y : ι → μ → ν → ℝ) (w : ι → ℝ) :
    0 ≤ anchoredW2Objective x y w := by
  unfold anchoredW2Objective
  positivity

/-- A row belongs to the probability simplex and excludes its own target. -/
def IsLeaveOneOutRow (target : ι) (w : ι → ℝ) : Prop :=
  (∀ j, 0 ≤ w j) ∧ (∑ j, w j = 1) ∧ w target = 0

/-- Matrix-level contract emitted by the finite leave-one-out producer. -/
def IsAnchoredW2Operator (W : Matrix ι ι ℝ) : Prop :=
  ∀ i, IsLeaveOneOutRow i (W i)

/-- Every target-anchored Wasserstein operator is entrywise nonnegative. -/
theorem IsAnchoredW2Operator.nonneg {W : Matrix ι ι ℝ}
    (hW : IsAnchoredW2Operator W) (i j : ι) : 0 ≤ W i j :=
  (hW i).1 j

/-- Every target-anchored Wasserstein operator has unit row sums. -/
theorem IsAnchoredW2Operator.row_sum {W : Matrix ι ι ℝ}
    (hW : IsAnchoredW2Operator W) (i : ι) : ∑ j, W i j = 1 :=
  (hW i).2.1

/-- Every target-anchored Wasserstein operator has zero diagonal. -/
theorem IsAnchoredW2Operator.zero_diag {W : Matrix ι ι ℝ}
    (hW : IsAnchoredW2Operator W) (i : ι) : W i i = 0 :=
  (hW i).2.2

section MatrixNorm

variable [DecidableEq ι]

/-- A target-anchored Wasserstein operator has `L∞` operator norm at most one. -/
theorem IsAnchoredW2Operator.norm_le_one {W : Matrix ι ι ℝ}
    (hW : IsAnchoredW2Operator W) : ‖W‖ ≤ 1 := by
  apply norm_le_of_abs_row_sum_le (by norm_num)
  intro i
  have hsum : ∑ j, |W i j| = 1 := by
    calc
      ∑ j, |W i j| = ∑ j, W i j := by
        apply Finset.sum_congr rfl
        intro j _
        exact abs_of_nonneg (hW.nonneg i j)
      _ = 1 := hW.row_sum i
  exact hsum.le

/-- Consequently `|ρ| < 1` is a sufficient norm gate for the SAR inverse. -/
theorem anchoredW2_sar_gate {W : Matrix ι ι ℝ} (hW : IsAnchoredW2Operator W)
    {ρ : ℝ} (hρ : |ρ| < 1) : ‖ρ • W‖ < 1 := by
  rw [norm_smul, Real.norm_eq_abs]
  calc
    |ρ| * ‖W‖ ≤ |ρ| * 1 := mul_le_mul_of_nonneg_left hW.norm_le_one (abs_nonneg ρ)
    _ < 1 := by simpa using hρ

end MatrixNorm

/-- A positive assignment gap survives entrywise cost perturbations when the
worst-case movement of two assignment totals is smaller than the gap. -/
theorem assignment_preserved_of_gap
    {best alternative best' alternative' gap δ : ℝ} {supportSize : ℕ}
    (hgap : best + gap ≤ alternative)
    (hbest : |best' - best| ≤ supportSize * δ)
    (halt : |alternative' - alternative| ≤ supportSize * δ)
    (hmargin : 2 * supportSize * δ < gap) : best' < alternative' := by
  have hbest' : best' ≤ best + supportSize * δ := by
    have := le_trans (le_abs_self (best' - best)) hbest
    linarith
  have halt' : alternative - supportSize * δ ≤ alternative' := by
    have := le_trans (neg_le_abs (alternative' - alternative)) halt
    linarith
  linarith

/-- Explicit local curvature certificate used by the producer.  It is a theorem
hypothesis, not an axiom and not a claim of global continuity across assignments. -/
def HasTangentStabilityCertificate
    (w w' : ι → ℝ) (curvature gradientRadius : ℝ) : Prop :=
  curvature * ‖w' - w‖ ≤ gradientRadius

/-- Positive tangent curvature converts the certified gradient radius into a
local weight radius. -/
theorem anchoredW2_weight_sub_norm_le
    {w w' : ι → ℝ} {curvature gradientRadius : ℝ}
    (hcert : HasTangentStabilityCertificate w w' curvature gradientRadius)
    (hcurvature : 0 < curvature) :
    ‖w' - w‖ ≤ gradientRadius / curvature := by
  rw [le_div_iff₀ hcurvature]
  simpa [HasTangentStabilityCertificate, mul_comm] using hcert

/-- The complete local route used in Paper 5: the transport assignment stays
fixed and, conditional on an aligned tangent certificate, the weights stay in
the curvature-scaled radius. -/
theorem anchoredW2_local_stability
    {best alternative best' alternative' gap δ : ℝ} {supportSize : ℕ}
    {w w' : ι → ℝ} {curvature gradientRadius : ℝ}
    (hgap : best + gap ≤ alternative)
    (hbest : |best' - best| ≤ supportSize * δ)
    (halt : |alternative' - alternative| ≤ supportSize * δ)
    (hmargin : 2 * supportSize * δ < gap)
    (hcert : HasTangentStabilityCertificate w w' curvature gradientRadius)
    (hcurvature : 0 < curvature) :
    best' < alternative' ∧ ‖w' - w‖ ≤ gradientRadius / curvature := by
  exact ⟨assignment_preserved_of_gap hgap hbest halt hmargin,
    anchoredW2_weight_sub_norm_le hcert hcurvature⟩

section Leontief

variable [DecidableEq ι]

/-- The certified Wasserstein row/operator radius enters the existing SAR
Leontief perturbation theorem without changing the SAR interpretation. -/
theorem anchoredW2_leontief_sub_norm_le
    {W W' : Matrix ι ι ℝ} {ρ ρ' radius : ℝ}
    (h : ‖ρ • W‖ < 1) (h' : ‖ρ' • W'‖ < 1)
    (hRadius : ‖W' - W‖ ≤ radius) :
    ‖leontief ρ' W' - leontief ρ W‖ ≤
      (|ρ'| * radius + |ρ' - ρ| * ‖W‖) /
        ((1 - ‖ρ' • W'‖) * (1 - ‖ρ • W‖)) := by
  have hbase := leontief_sub_norm_le_joint h h'
  have hden : 0 ≤ ((1 - ‖ρ' • W'‖) * (1 - ‖ρ • W‖))⁻¹ := by
    positivity
  calc
    ‖leontief ρ' W' - leontief ρ W‖ ≤
        (|ρ'| * ‖W' - W‖ + |ρ' - ρ| * ‖W‖) /
          ((1 - ‖ρ' • W'‖) * (1 - ‖ρ • W‖)) := hbase
    _ ≤ (|ρ'| * radius + |ρ' - ρ| * ‖W‖) /
          ((1 - ‖ρ' • W'‖) * (1 - ‖ρ • W‖)) := by
      rw [div_eq_mul_inv, div_eq_mul_inv]
      apply mul_le_mul_of_nonneg_right _ hden
      gcongr

end Leontief

end PricingPerspective.Connections.SpatialWasserstein
