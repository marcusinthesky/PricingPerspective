import Architect
import PricingPerspective.Operating.RationalShare.FirmResponse.IncumbentResponse
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Tactic

/-!
# Finite-capacity normalization certificate

The moving-opportunity population gate compares normalized activity compositions computed from
an approximate nonnegative capacity vector and its exact counterpart.  This module turns an
Euclidean capacity error into an explicit Euclidean composition error.  The denominator condition
ensures that the exact vector retains positive mass after the worst-case finite-dimensional mass
perturbation.

All claims use exact real arithmetic.  The numerical pipeline remains responsible for supplying a
valid capacity error bound and evaluating the displayed expression conservatively in floating
point.
-/

namespace PricingPerspective.Operating.RationalShare.SolverCertificates.NormalizationCertificate

open PricingPerspective.Operating.VariationalInequality
  PricingPerspective.Operating.RationalShare.FirmResponse
  PricingPerspective.Operating.RationalShare.FirmResponse.Response
  PricingPerspective.Operating.RationalShare.FirmResponse.IncumbentResponse

open scoped RealInnerProductSpace
open RationalShare

variable {α : Type*} [Fintype α]

/-- Total mass of a finite Euclidean capacity vector. -/
def totalMass (q : EuclideanSpace ℝ α) : ℝ := ∑ a, q a

/-- Normalize a finite capacity vector by its total mass.  The definition remains total because
real division is total; the certificate below derives strict positivity of both masses. -/
noncomputable def normalized (q : EuclideanSpace ℝ α) : EuclideanSpace ℝ α :=
  (totalMass q)⁻¹ • q

/-- Coordinate form of finite-capacity normalization. -/
@[simp]
theorem normalized_apply (q : EuclideanSpace ℝ α) (a : α) :
    normalized q a = q a / totalMass q := by
  simp [normalized, div_eq_mul_inv, mul_comm]

/-- A nonnegative finite capacity vector has Euclidean norm at most its total mass. -/
theorem norm_le_totalMass_of_nonneg {q : EuclideanSpace ℝ α} (hq : ∀ a, 0 ≤ q a) :
    ‖q‖ ≤ totalMass q := by
  classical
  have hrepr : q = ∑ a, EuclideanSpace.single a (q a) := by
    ext b
    simp
  calc
    ‖q‖ = ‖∑ a, EuclideanSpace.single a (q a)‖ := congrArg norm hrepr
    _ ≤
        ∑ a, ‖EuclideanSpace.single a (q a)‖ := norm_sum_le _ _
    _ = ∑ a, q a := by
      apply Finset.sum_congr rfl
      intro a _
      simp [Real.norm_eq_abs, abs_of_nonneg (hq a)]
    _ = totalMass q := rfl

/-- The total-mass error is at most `sqrt(card α)` times the Euclidean vector error. -/
theorem abs_totalMass_sub_le_sqrt_card_mul_norm
    (q r : EuclideanSpace ℝ α) :
    |totalMass q - totalMass r| ≤ Real.sqrt (Fintype.card α) * ‖q - r‖ := by
  let one : EuclideanSpace ℝ α := WithLp.toLp 2 (fun _ ↦ (1 : ℝ))
  have hinner : ⟪one, q - r⟫ = totalMass q - totalMass r := by
    simp [one, PiLp.inner_apply, totalMass, Finset.sum_sub_distrib]
  have hone : ‖one‖ = Real.sqrt (Fintype.card α) := by
    rw [EuclideanSpace.norm_eq]
    simp [one]
  rw [← hinner]
  calc
    |⟪one, q - r⟫| ≤ ‖one‖ * ‖q - r‖ := abs_real_inner_le_norm one (q - r)
    _ = Real.sqrt (Fintype.card α) * ‖q - r‖ := by rw [hone]

/-- If the computed mass exceeds its finite-dimensional error radius, the exact nonnegative
capacity vector has strictly positive total mass. -/
theorem totalMass_pos_of_norm_error
    {qhat qstar : EuclideanSpace ℝ α} {e : ℝ}
    (hqstar : ∀ a, 0 ≤ qstar a) (herror : ‖qhat - qstar‖ ≤ e)
    (hmass : Real.sqrt (Fintype.card α) * e < totalMass qhat) :
    0 < totalMass qstar := by
  have hmass_error :
      |totalMass qhat - totalMass qstar| ≤ Real.sqrt (Fintype.card α) * e :=
    (abs_totalMass_sub_le_sqrt_card_mul_norm qhat qstar).trans
      (mul_le_mul_of_nonneg_left herror (Real.sqrt_nonneg _))
  have hlower :
      totalMass qhat - Real.sqrt (Fintype.card α) * e ≤ totalMass qstar := by
    have hforward :=
      (le_abs_self (totalMass qhat - totalMass qstar)).trans hmass_error
    linarith
  have hstar_nonneg : 0 ≤ totalMass qstar :=
    Finset.sum_nonneg fun a _ ↦ hqstar a
  have hstar_ne : totalMass qstar ≠ 0 := by
    intro hzero
    rw [hzero] at hlower
    linarith
  exact lt_of_le_of_ne hstar_nonneg (Ne.symm hstar_ne)

/-- A sharper intermediate certificate: if both masses are positive, composition error is at
most the vector error plus its induced mass error, divided by the exact vector's mass. -/
theorem norm_normalized_sub_normalized_le_add_mass_error
    {qhat qstar : EuclideanSpace ℝ α}
    (hqhat : ∀ a, 0 ≤ qhat a)
    (hhat : 0 < totalMass qhat) (hstar : 0 < totalMass qstar) :
    ‖normalized qhat - normalized qstar‖ ≤
      (‖qhat - qstar‖ + |totalMass qhat - totalMass qstar|) / totalMass qstar := by
  let S := totalMass qhat
  let T := totalMass qstar
  have hS : 0 < S := by simpa [S] using hhat
  have hT : 0 < T := by simpa [T] using hstar
  have hnorm : ‖qhat‖ ≤ S := norm_le_totalMass_of_nonneg hqhat
  have hid : normalized qhat - normalized qstar =
      T⁻¹ • (qhat - qstar) + (S⁻¹ - T⁻¹) • qhat := by
    ext a
    simp only [normalized, PiLp.sub_apply, PiLp.smul_apply, smul_eq_mul]
    change S⁻¹ * qhat a - T⁻¹ * qstar a =
      T⁻¹ * (qhat a - qstar a) + (S⁻¹ - T⁻¹) * qhat a
    ring
  have hinv : |S⁻¹ - T⁻¹| = |S - T| / (S * T) := by
    rw [inv_sub_inv hS.ne' hT.ne', abs_div, abs_sub_comm]
    simp only [abs_mul, abs_of_pos hS, abs_of_pos hT]
  have hsecond : |S - T| / (S * T) * ‖qhat‖ ≤ |S - T| / T := by
    rw [div_mul_eq_mul_div]
    apply (div_le_div_iff₀ (mul_pos hS hT) hT).2
    have hbase := mul_le_mul_of_nonneg_left hnorm (abs_nonneg (S - T))
    calc
      |S - T| * ‖qhat‖ * T ≤ |S - T| * S * T :=
        mul_le_mul_of_nonneg_right hbase hT.le
      _ = |S - T| * (S * T) := by ring
  rw [hid]
  calc
    ‖T⁻¹ • (qhat - qstar) + (S⁻¹ - T⁻¹) • qhat‖ ≤
        ‖T⁻¹ • (qhat - qstar)‖ + ‖(S⁻¹ - T⁻¹) • qhat‖ := norm_add_le _ _
    _ = ‖qhat - qstar‖ / T + |S - T| / (S * T) * ‖qhat‖ := by
      rw [norm_smul, norm_smul]
      change |T⁻¹| * ‖qhat - qstar‖ + |S⁻¹ - T⁻¹| * ‖qhat‖ = _
      rw [hinv, abs_of_pos (inv_pos.mpr hT)]
      simp [div_eq_inv_mul, mul_comm]
    _ ≤ ‖qhat - qstar‖ / T + |S - T| / T := add_le_add le_rfl hsecond
    _ = (‖qhat - qstar‖ + |S - T|) / T := by ring
    _ = (‖qhat - qstar‖ + |totalMass qhat - totalMass qstar|) /
        totalMass qstar := rfl

/-- **Population-gate normalization certificate.**

Let `qhat` be a computed nonnegative capacity block and `qstar` its exact nonnegative block.  If
their Euclidean distance is at most `e` and the computed mass exceeds the worst-case mass error
`sqrt(card α) * e`, then both normalizations are well-defined with positive denominators and their
Euclidean distance is at most
`2 * sqrt(card α) * e / (totalMass qhat - sqrt(card α) * e)`.

This is the exact real-arithmetic inequality used to propagate a joint equilibrium-distance bound
through each firm's finite normalization map. -/
theorem norm_normalized_sub_normalized_le
    {qhat qstar : EuclideanSpace ℝ α} {e : ℝ}
    (hcard : 0 < Fintype.card α)
    (hqhat : ∀ a, 0 ≤ qhat a) (hqstar : ∀ a, 0 ≤ qstar a)
    (herror : ‖qhat - qstar‖ ≤ e)
    (hmass : Real.sqrt (Fintype.card α) * e < totalMass qhat) :
    ‖normalized qhat - normalized qstar‖ ≤
      2 * Real.sqrt (Fintype.card α) * e /
        (totalMass qhat - Real.sqrt (Fintype.card α) * e) := by
  let k := Real.sqrt (Fintype.card α)
  let d := k * e
  let S := totalMass qhat
  let T := totalMass qstar
  have hcard_real : (0 : ℝ) < Fintype.card α := by exact_mod_cast hcard
  have hk_pos : 0 < k := by simpa [k] using Real.sqrt_pos.mpr hcard_real
  have hk_one : 1 ≤ k := by
    dsimp [k]
    rw [Real.one_le_sqrt]
    exact_mod_cast hcard
  have he : 0 ≤ e := (norm_nonneg (qhat - qstar)).trans herror
  have hd : 0 ≤ d := mul_nonneg hk_pos.le he
  have hmass_error : |S - T| ≤ d := by
    calc
      |S - T| = |totalMass qhat - totalMass qstar| := rfl
      _ ≤ Real.sqrt (Fintype.card α) * ‖qhat - qstar‖ :=
        abs_totalMass_sub_le_sqrt_card_mul_norm qhat qstar
      _ ≤ Real.sqrt (Fintype.card α) * e :=
        mul_le_mul_of_nonneg_left herror (Real.sqrt_nonneg _)
      _ = d := rfl
  have hmargin : 0 < S - d := by simpa [S, d, k] using sub_pos.mpr hmass
  have hlower : S - d ≤ T := by
    have hforward : S - T ≤ d := (le_abs_self (S - T)).trans hmass_error
    linarith
  have hT : 0 < T := by
    simpa [T] using totalMass_pos_of_norm_error hqstar herror hmass
  have hS : 0 < S := lt_of_lt_of_le hmargin (sub_le_self S hd)
  have he_le_d : e ≤ d := by
    dsimp [d]
    nlinarith [mul_le_mul_of_nonneg_right hk_one he]
  have hnum : ‖qhat - qstar‖ + |S - T| ≤ 2 * d := by linarith
  calc
    ‖normalized qhat - normalized qstar‖ ≤
        (‖qhat - qstar‖ + |totalMass qhat - totalMass qstar|) / totalMass qstar :=
      norm_normalized_sub_normalized_le_add_mass_error hqhat
        (by simpa [S] using hS) (by simpa [T] using hT)
    _ = (‖qhat - qstar‖ + |S - T|) / T := rfl
    _ ≤ 2 * d / (S - d) := div_le_div₀ (mul_nonneg (by positivity) hd) hnum hmargin hlower
    _ = 2 * Real.sqrt (Fintype.card α) * e /
        (totalMass qhat - Real.sqrt (Fintype.card α) * e) := by
      simp only [d, k, S]
      ring

/-! ### Joint-profile to firm-composition bridge -/

variable {ι : Type*} [Fintype ι]

/-- Extracting one firm's activity block cannot increase Euclidean error.

This is the missing projection step between a joint rational-share equilibrium-distance
certificate on the firm-by-activity profile and the local capacity error consumed by the
normalization theorem. -/
theorem norm_incumbentBlock_sub_le_norm_profile_sub
    (qhat qstar : Profile ι α) (i : ι) :
    ‖IncumbentResponse.incumbentBlock qhat i -
        IncumbentResponse.incumbentBlock qstar i‖ ≤ ‖qhat - qstar‖ := by
  apply (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).mp
  rw [RationalShare.norm_sq_eq_sum, RationalShare.norm_sq_eq_sum,
    Fintype.sum_prod_type, Fintype.sum_prod_type]
  simp only [Fintype.sum_unique, PiLp.sub_apply]
  exact Finset.single_le_sum
    (fun j _ ↦ Finset.sum_nonneg fun a _ ↦ sq_nonneg (qhat (j, a) - qstar (j, a)))
    (Finset.mem_univ i)

/-- **End-to-end joint-error normalization certificate.**

A joint rational-share profile error `‖qhat - qstar‖ ≤ e` supplies the local firm-block error
required by `norm_normalized_sub_normalized_le`.  Thus the population gate may use the certified
joint equilibrium-distance bound directly while retaining the same activity-count and computed
firm-mass denominator as the finite-capacity theorem. -/
@[blueprint "eq:normalization-bound"
  (statement := /--
    Joint capacity error bounds normalized firm composition error. Assumptions: Nonnegative
    capacities and computed mass exceeding the error-dependent threshold.
  -/)]
theorem norm_normalized_incumbentBlock_sub_normalized_incumbentBlock_le
    (qhat qstar : Profile ι α) (i : ι) {e : ℝ}
    (hcard : 0 < Fintype.card α)
    (hqhat : ∀ p, 0 ≤ qhat p) (hqstar : ∀ p, 0 ≤ qstar p)
    (herror : ‖qhat - qstar‖ ≤ e)
    (hmass : Real.sqrt (Fintype.card α) * e <
      totalMass (IncumbentResponse.incumbentBlock qhat i)) :
    ‖normalized (IncumbentResponse.incumbentBlock qhat i) -
        normalized (IncumbentResponse.incumbentBlock qstar i)‖ ≤
      2 * Real.sqrt (Fintype.card α) * e /
        (totalMass (IncumbentResponse.incumbentBlock qhat i) -
          Real.sqrt (Fintype.card α) * e) := by
  have hblock_error :
      ‖IncumbentResponse.incumbentBlock qhat i -
          IncumbentResponse.incumbentBlock qstar i‖ ≤ e :=
    (norm_incumbentBlock_sub_le_norm_profile_sub qhat qstar i).trans herror
  have hblock_hat : ∀ p, 0 ≤ IncumbentResponse.incumbentBlock qhat i p := by
    rintro ⟨u, a⟩
    cases u
    exact hqhat (i, a)
  have hblock_star : ∀ p, 0 ≤ IncumbentResponse.incumbentBlock qstar i p := by
    rintro ⟨u, a⟩
    cases u
    exact hqstar (i, a)
  have hcard_block : 0 < Fintype.card (Unit × α) := by
    simpa using hcard
  simpa using norm_normalized_sub_normalized_le hcard_block hblock_hat hblock_star
    hblock_error (by simpa using hmass)

end PricingPerspective.Operating.RationalShare.SolverCertificates.NormalizationCertificate
