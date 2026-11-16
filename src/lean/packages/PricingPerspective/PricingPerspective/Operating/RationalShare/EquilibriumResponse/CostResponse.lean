import Architect
import PricingPerspective.Operating.RationalShare.EquilibriumResponse.EquilibriumDerivative

/-!
# The equilibrium response to a cost disturbance

Linear activity costs enter the negative marginal-profit field additively, so a cost change
`δc` shifts the field by exactly `δc`. Under strict complementarity the equilibrium therefore
responds through the same restricted linear system as an opportunity change, with forcing `-δc`
in place of the opportunity impulse:

* **Additive shift.** `pseudoGradient N c' H y - pseudoGradient N c H y = c' - c` at every profile.
* **Continuity.** Equilibria at two cost schedules are at most `‖c' - c‖ / mu` apart.
* **First-order expansion.** On the stable active set, `q*(c') - q*(c) - u = o(‖c' - c‖)`, where
  `u` solves `P (J u) = P (-(c' - c))` with `P u = u`.

The argument follows `equilibrium_sub_linearResponse_le`. Because the shift does not depend on
the profile, the impulse-continuity remainder of the opportunity case is absent.
-/

namespace PricingPerspective.Operating.RationalShare.EquilibriumResponse

open PricingPerspective.Operating.VariationalInequality

open Filter Topology

variable {κ ι α : Type*} [Fintype κ] [Fintype ι] [Fintype α]

/-- A change in linear activity costs, as a profile. -/
noncomputable def costShift (c c' : ι → α → ℝ) : Profile ι α :=
  WithLp.toLp 2 fun p ↦ c' p.1 p.2 - c p.1 p.2

/-- The Euclidean distance between two linear-cost schedules. -/
noncomputable def costDist (c c' : ι → α → ℝ) : ℝ := ‖costShift c c'‖

/-- **Costs shift the field additively.** -/
theorem pseudoGradient_cost_sub (N : Nodes κ α) (c c' : ι → α → ℝ) (H : ι → α → α → ℝ)
    (x : Profile ι α) :
    pseudoGradient N c' H x - pseudoGradient N c H x = costShift c c' := by
  ext ⟨i, a⟩
  simp [costShift]

section Derivative

variable [DecidableEq ι]

/-- **Equilibria move Lipschitz-continuously in costs.** -/
theorem orthantEquilibrium_dist_le_costShift (N : Nodes κ α) (c c' : ι → α → ℝ)
    {H : ι → α → α → ℝ} {mu : ℝ} (hsymm : ∀ i a b, H i a b = H i b a) (hmu : 0 < mu)
    (hH : ∀ i, ∀ u : α → ℝ, mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (hc : ∀ i a, 0 ≤ c i a) (hc' : ∀ i a, 0 ≤ c' i a) {x y : Profile ι α}
    (hx : IsNashOrthant N c H x) (hy : IsNashOrthant N c' H y) :
    ‖y - x‖ ≤ costDist c c' / mu := by
  have h := orthantEquilibrium_dist_le_operatorPerturbation N N c c' hsymm hsymm hmu hmu hH hH
    hc hc' hx hy
  rwa [pseudoGradient_cost_sub] at h

/-- **The cost response is the derivative of the equilibrium map in costs.**

Let `x` be the equilibrium at costs `c`, with strict complementarity. Then for every `ε > 0`
there is a neighbourhood of `c` on which the equilibrium `y` at nonnegative costs `c'` satisfies
`‖y - x - u‖ ≤ ε ‖c' - c‖`. Here `u` solves the linear response system on the active set of `x`
with forcing `-(c' - c)`. -/
@[blueprint "eq:cost-response"
  (statement := /--
    Under strict complementarity the equilibrium responds to a change in linear activity costs
    through the restricted linear system with forcing minus the cost change. Assumptions:
    Assumption 1 primitives, an equilibrium with strict complementarity, the active set of
    positive capacities, and nonnegative nearby linear costs.
  -/)]
theorem equilibrium_sub_costResponse_le (N : Nodes κ α) (c : ι → α → ℝ)
    {H : ι → α → α → ℝ} {mu : ℝ} (hsymm : ∀ i a b, H i a b = H i b a) (hmu : 0 < mu)
    (hH : ∀ i, ∀ u : α → ℝ, mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (hc : ∀ i a, 0 ≤ c i a) {x : Profile ι α} (hxN : IsNashOrthant N c H x)
    (hsc : ∀ p, 0 < x p ∨ 0 < pseudoGradient N c H x p) {ε : ℝ} (hε : 0 < ε) :
    ∃ d > 0, ∀ c' : ι → α → ℝ, (∀ i a, 0 ≤ c' i a) → costDist c c' < d →
      ∀ y, IsNashOrthant N c' H y →
      ∀ u, SolvesRestricted (jac N H x) (restrictL fun p ↦ 0 < x p) (-costShift c c') u →
        ‖y - x - u‖ ≤ ε * costDist c c' := by
  classical
  set F := pseudoGradient N c H with hF
  set act : ι × α → Prop := fun p ↦ 0 < x p with hact
  set P := restrictL (ι := ι) (α := α) act with hP
  have hPproj : IsOrthoProjection P := restrictL_isOrthoProjection act
  have hx0 : ∀ p, 0 ≤ x p := hxN.1
  have hDpos : ∀ l, 0 < N.denom x l := N.denom_pos hx0
  have hxvi := solvesVI_capacityOrthant_of_isNashOrthant N c hsymm hmu hH hc hxN
  have hxcomp := complementarity_of_solvesVI hxvi
  -- Constants.
  set K := mu⁻¹ with hK
  have hK0 : 0 ≤ K := inv_nonneg.mpr hmu.le
  set Lip := lipConst N H with hLip
  have hLip0 : 0 ≤ Lip := add_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _)
  -- The strict-complementarity gap.
  obtain ⟨m, hm, hgap⟩ : ∃ m > 0, ∀ p, m ≤ max (x p) (F x p) := by
    rcases (Finset.univ : Finset (ι × α)).eq_empty_or_nonempty with he | hne
    · exact ⟨1, one_pos, fun p ↦ absurd (Finset.mem_univ p) (by simp [he])⟩
    · obtain ⟨p0, -, hp0⟩ := Finset.exists_min_image Finset.univ
        (fun p ↦ max (x p) (F x p)) hne
      refine ⟨max (x p0) (F x p0), ?_, fun p ↦ hp0 p (Finset.mem_univ p)⟩
      rcases hsc p0 with h | h
      · exact lt_max_of_lt_left h
      · exact lt_max_of_lt_right h
  -- Fréchet expansion of the field.
  set eta := ε * mu ^ 2 with heta
  have heta0 : 0 < eta := by positivity
  have hfd := (hasFDerivAt_pseudoGradient N c H x hDpos).isLittleO.def heta0
  obtain ⟨d1, hd1, hball1⟩ := Metric.eventually_nhds_iff.mp hfd
  set d := min d1 (m / 2) / (K + Lip * K + 1 + 1) with hd
  have hden : 0 < K + Lip * K + 1 + 1 := by positivity
  have hdpos : 0 < d := div_pos (lt_min hd1 (by positivity)) hden
  refine ⟨d, hdpos, fun c' hc' hρ y hy u hu ↦ ?_⟩
  set ρ := costDist c c' with hρdef
  have hρ0 : 0 ≤ ρ := norm_nonneg _
  set Fc := pseudoGradient N c' H with hFc
  have hy0 : ∀ p, 0 ≤ y p := hy.1
  have hyvi := solvesVI_capacityOrthant_of_isNashOrthant N c' hsymm hmu hH hc' hy
  have hycomp := complementarity_of_solvesVI hyvi
  -- Cost continuity.
  have hcont : ‖y - x‖ ≤ K * ρ := by
    have h := orthantEquilibrium_dist_le_costShift N c c' hsymm hmu hH hc hc' hxN hy
    calc ‖y - x‖ ≤ ρ / mu := h
      _ = K * ρ := by rw [hK, div_eq_inv_mul]
  have hsmall : (K + Lip * K + 1 + 1) * ρ < min d1 (m / 2) := by
    have := (lt_div_iff₀' hden).mp (hd ▸ hρ)
    linarith
  have hLKρ : 0 ≤ Lip * (K * ρ) := mul_nonneg hLip0 (mul_nonneg hK0 hρ0)
  have hKρ0 : 0 ≤ K * ρ := mul_nonneg hK0 hρ0
  have hexpand : (K + Lip * K + 1 + 1) * ρ = K * ρ + Lip * (K * ρ) + ρ + ρ := by ring
  have hKρ1 : K * ρ < d1 := by
    linarith [min_le_left d1 (m / 2)]
  have hgapρ : K * ρ + Lip * (K * ρ) + ρ < m := by
    linarith [min_le_right d1 (m / 2)]
  -- The field at the new equilibrium is close to the field at the old one.
  have hshift : Fc y - F y = costShift c c' := pseudoGradient_cost_sub N c c' H y
  have hfield : ‖Fc y - F x‖ ≤ Lip * (K * ρ) + ρ := by
    have hsplit : Fc y - F x = (Fc y - F y) + (F y - F x) := by abel
    have h2 : ‖F y - F x‖ ≤ Lip * (K * ρ) := by
      have hbox := lipschitzOn_pseudoGradient N c H (lo := fun _ ↦ (0 : ℝ))
        (hi := fun p ↦ max (x p) (y p)) (fun _ ↦ le_refl 0) y
        (fun p ↦ ⟨hy0 p, le_max_right _ _⟩) x (fun p ↦ ⟨hx0 p, le_max_left _ _⟩)
      exact hbox.trans (mul_le_mul_of_nonneg_left hcont hLip0)
    rw [hsplit, hshift]
    exact (norm_add_le _ _).trans (by rw [← costDist]; linarith)
  -- The active set is unchanged.
  have hcoord : ∀ (v : Profile ι α) p, |v p| ≤ ‖v‖ := fun v p ↦ by
    simpa [Real.norm_eq_abs] using PiLp.norm_apply_le v p
  have hactive : ∀ p, act p → 0 < y p := by
    intro p hp
    have hxp : m ≤ x p := by
      have := hgap p
      rwa [max_eq_left ((hxcomp p).2 hp ▸ hp.le)] at this
    have := hcoord (y - x) p
    rw [PiLp.sub_apply] at this
    have := abs_le.mp (this.trans hcont)
    linarith
  have hinactive : ∀ p, ¬ act p → y p = 0 := by
    intro p hp
    have hxp0 : x p = 0 := le_antisymm (not_lt.mp hp) (hx0 p)
    have hFp : m ≤ F x p := by
      have := hgap p
      rwa [hxp0, max_eq_right (hxcomp p).1] at this
    have hFy : 0 < Fc y p := by
      have := hcoord (Fc y - F x) p
      rw [PiLp.sub_apply] at this
      have := abs_le.mp (this.trans hfield)
      linarith [this.1]
    by_contra hne
    have hpos : 0 < y p := lt_of_le_of_ne (hy0 p) (Ne.symm hne)
    exact absurd ((hycomp p).2 hpos) hFy.ne'
  -- Supports and vanishing active fields.
  have hPyx : P (y - x) = y - x := by
    ext p
    by_cases hp : act p
    · simp [hP, hp]
    · simp [hP, hp, hinactive p hp, le_antisymm (not_lt.mp hp) (hx0 p)]
  have hPFc : P (Fc y) = 0 := by
    ext p
    by_cases hp : act p
    · simpa [hP, hp] using (hycomp p).2 (hactive p hp)
    · simp [hP, hp]
  have hPF : P (F x) = 0 := by
    ext p
    by_cases hp : act p
    · simpa [hP, hp] using (hxcomp p).2 hp
    · simp [hP, hp]
  -- First-order expansion on the active set.
  set R1 := F y - F x - jac N H x (y - x) with hR1
  have hR1le : ‖R1‖ ≤ eta * ‖y - x‖ := by
    have := hball1 (by rw [dist_eq_norm]; exact hcont.trans_lt hKρ1)
    simpa [hR1] using this
  have hexp : Fc y - F x = jac N H x (y - x) + R1 + costShift c c' := by
    rw [hR1, ← hshift]; abel
  have hw : SolvesRestricted (jac N H x) P (-R1) (y - x - u) := by
    refine ⟨by rw [map_sub, hPyx, hu.1], ?_⟩
    have h0 : P (Fc y - F x) = 0 := by rw [map_sub, hPFc, hPF, sub_zero]
    rw [hexp, map_add, map_add] at h0
    rw [map_sub (jac N H x), map_sub P, hu.2, map_neg, map_neg]
    rw [← sub_eq_zero, ← h0]
    abel
  have hbound := hw.norm_le hPproj hmu (fun v _ ↦ jac_coercive N H x hH hx0 v)
  have hPle := hPproj.norm_le (-R1)
  calc ‖y - x - u‖ ≤ ‖P (-R1)‖ / mu := hbound
    _ ≤ ‖R1‖ / mu :=
        div_le_div_of_nonneg_right (hPle.trans (norm_neg R1).le) hmu.le
    _ ≤ eta * (K * ρ) / mu :=
        div_le_div_of_nonneg_right
          (hR1le.trans (mul_le_mul_of_nonneg_left hcont heta0.le)) hmu.le
    _ = ε * ρ := by
        rw [heta, hK]; field_simp

end Derivative

end PricingPerspective.Operating.RationalShare.EquilibriumResponse
