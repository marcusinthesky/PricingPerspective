import WassersteinGeometry.Multimarginal.Pairwise
import WassersteinGeometry.Hilbert.Polarization

set_option linter.style.longLine false

open scoped InnerProductSpace
open Finset

/-!
# Hilbert identities for multi-marginal dispersion

The weighted polarization identity is the deterministic algebra behind the
portfolio variance envelope: squared norm of the weighted mean equals weighted
second moments minus weighted pairwise dispersion.
-/

namespace WassersteinGeometry.Multimarginal

variable {A H : Type*} [Fintype A]
  [NormedAddCommGroup H] [InnerProductSpace ℝ H]

/-- The squared norm of a weighted Hilbert sum is its covariance double sum. -/
lemma normSq_weightedSum (q : ProbabilityWeight A) (z : A → H) :
    ‖∑ a, q a • z a‖ ^ 2 =
      ∑ a, ∑ b, q a * q b * ⟪z a, z b⟫_ℝ := by
  calc
    ‖∑ a, q a • z a‖ ^ 2 = ⟪∑ a, q a • z a, ∑ b, q b • z b⟫_ℝ := by
      rw [real_inner_self_eq_norm_sq]
    _ = ∑ a, ⟪q a • z a, ∑ b, q b • z b⟫_ℝ := by rw [sum_inner]
    _ = ∑ a, ∑ b, ⟪q a • z a, q b • z b⟫_ℝ := by simp_rw [inner_sum]
    _ = ∑ a, ∑ b, q a * q b * ⟪z a, z b⟫_ℝ := by
      refine Finset.sum_congr rfl fun a _ ↦ ?_
      refine Finset.sum_congr rfl fun b _ ↦ ?_
      rw [real_inner_smul_left, real_inner_smul_right]
      ring

/-- Weighted Hilbert polarization for a finite probability weight.

`‖∑qᵢzᵢ‖² = ∑qᵢ‖zᵢ‖² − ½∑ᵢ∑ⱼqᵢqⱼ‖zᵢ−zⱼ‖²`.
-/
theorem weighted_polarization (q : ProbabilityWeight A) (z : A → H) :
    ‖∑ a, q a • z a‖ ^ 2 =
      ∑ a, q a * ‖z a‖ ^ 2 -
        (1 / 2 : ℝ) * ∑ a, ∑ b, q a * q b * ‖z a - z b‖ ^ 2 := by
  have hleft :
      ∑ a, ∑ b, q a * q b * ‖z a‖ ^ 2 = ∑ a, q a * ‖z a‖ ^ 2 := by
    calc
      ∑ a, ∑ b, q a * q b * ‖z a‖ ^ 2 =
          ∑ a, (q a * ‖z a‖ ^ 2) * ∑ b, q b := by
        refine Finset.sum_congr rfl fun a _ ↦ ?_
        rw [Finset.mul_sum]
        refine Finset.sum_congr rfl fun b _ ↦ ?_
        ring
      _ = ∑ a, q a * ‖z a‖ ^ 2 := by rw [q.mass_one]; simp
  have hright :
      ∑ a, ∑ b, q a * q b * ‖z b‖ ^ 2 = ∑ b, q b * ‖z b‖ ^ 2 := by
    rw [Finset.sum_comm]
    simpa [mul_comm] using hleft
  have hdist :
      ∑ a, ∑ b, q a * q b * ‖z a - z b‖ ^ 2 =
        2 * (∑ a, q a * ‖z a‖ ^ 2) -
          2 * (∑ a, ∑ b, q a * q b * ⟪z a, z b⟫_ℝ) := by
    calc
      ∑ a, ∑ b, q a * q b * ‖z a - z b‖ ^ 2 =
          ∑ a, ∑ b,
            (q a * q b * ‖z a‖ ^ 2 -
              2 * (q a * q b * ⟪z a, z b⟫_ℝ) +
                q a * q b * ‖z b‖ ^ 2) := by
        refine Finset.sum_congr rfl fun a _ ↦ ?_
        refine Finset.sum_congr rfl fun b _ ↦ ?_
        rw [norm_sub_sq_real]
        ring
      _ = 2 * (∑ a, q a * ‖z a‖ ^ 2) -
          2 * (∑ a, ∑ b, q a * q b * ⟪z a, z b⟫_ℝ) := by
        simp_rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
        rw [hleft, hright]
        ring
  rw [normSq_weightedSum q z, hdist]
  ring

/-- Exact weighted variance decomposition around the weighted Hilbert mean.

For every comparison point `y`, the weighted squared radius about `y` is the weighted
variance about `∑ a, q a • z a` plus the squared distance from that mean to `y`.  No
strict positivity of the weights is required.
-/
theorem weighted_variance_decomposition (q : ProbabilityWeight A) (z : A → H) (y : H) :
    ∑ a, q a * ‖z a - y‖ ^ 2 =
      ∑ a, q a * ‖z a - ∑ b, q b • z b‖ ^ 2 +
        ‖(∑ b, q b • z b) - y‖ ^ 2 := by
  let m : H := ∑ b, q b • z b
  have hcenter : ∑ a, q a • (z a - m) = 0 := by
    simp_rw [smul_sub]
    rw [Finset.sum_sub_distrib]
    change m - ∑ a, q a • m = 0
    rw [← Finset.sum_smul, q.mass_one, one_smul, sub_self]
  have hcross : ∑ a, q a * ⟪z a - m, m - y⟫_ℝ = 0 := by
    simp_rw [← real_inner_smul_left]
    rw [← sum_inner]
    simpa using congrArg (fun x : H ↦ ⟪x, m - y⟫_ℝ) hcenter
  have hcross' : ∑ a, q a * (2 * ⟪z a - m, m - y⟫_ℝ) = 0 := by
    calc
      ∑ a, q a * (2 * ⟪z a - m, m - y⟫_ℝ) =
          2 * ∑ a, q a * ⟪z a - m, m - y⟫_ℝ := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro a _
        ring
      _ = 0 := by rw [hcross, mul_zero]
  change ∑ a, q a * ‖z a - y‖ ^ 2 =
    ∑ a, q a * ‖z a - m‖ ^ 2 + ‖m - y‖ ^ 2
  have hpoint (a : A) : z a - y = (z a - m) + (m - y) := by abel
  simp_rw [hpoint, norm_add_sq_real, mul_add, Finset.sum_add_distrib]
  rw [hcross', add_zero, ← Finset.sum_mul, q.mass_one, one_mul]

/-- Weighted variance about the Hilbert mean equals half the weighted pairwise sum. -/
theorem weighted_variance_eq_pairwise (q : ProbabilityWeight A) (z : A → H) :
    ∑ a, q a * ‖z a - ∑ c, q c • z c‖ ^ 2 =
      (1 / 2 : ℝ) * ∑ a, ∑ b, q a * q b * ‖z a - z b‖ ^ 2 := by
  have hdecomp := weighted_variance_decomposition q z (0 : H)
  have hpolar := weighted_polarization q z
  simp only [sub_zero] at hdecomp
  linarith

end WassersteinGeometry.Multimarginal
