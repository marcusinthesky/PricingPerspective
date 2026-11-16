import Architect
import PricingPerspective.Operating.RationalShare.EquilibriumResponse.LinearDynamics
import PricingPerspective.Operating.RationalShare.EquilibriumResponse.OrthantComparativeStatics

/-!
# The linear response is the derivative of the equilibrium map

`LinearResponse.lean` defines the response to an opportunity change as the solution of the
equilibrium conditions linearised on a fixed active set. This module shows that the linearised
response is the derivative of the equilibrium map under strict complementarity. It does not use
an implicit-function theorem.

* **Differentiability.** The negative marginal-profit field is Fréchet differentiable wherever
  every share denominator is positive, and its derivative is `jac`. This follows from uniqueness
  of the derivative along lines.
* **Complementarity.** At an orthant equilibrium every field coordinate is nonnegative, and it is
  zero wherever capacity is positive.
* **Stable active set.** Suppose every coordinate has positive capacity or a strictly positive
  field value (strict complementarity). Equilibria move Lipschitz-continuously in opportunity
  values, so for small changes the active set does not change.
* **First-order expansion.** On the stable active set, the error `q*(r) - q*(r₀) - χ δr` solves
  the linearised system with a forcing that is `o(‖δr‖)`. The restricted coercivity bound turns
  this into `o(‖δr‖)` for the error itself.
-/

namespace PricingPerspective.Operating.RationalShare.EquilibriumResponse

open PricingPerspective.Operating.VariationalInequality

open Filter Topology
open scoped RealInnerProductSpace

variable {κ ι α : Type*} [Fintype κ] [Fintype ι] [Fintype α]

/-! ### Differentiability of the field -/

section Differentiability

variable (N : Nodes κ α) (c : ι → α → ℝ) (H : ι → α → α → ℝ) (x : Profile ι α)

theorem differentiableAt_pseudoGradient (hx : ∀ l, N.denom x l ≠ 0) :
    DifferentiableAt ℝ (pseudoGradient N c H) x := by
  rw [differentiableAt_piLp]
  intro p
  simp only [pseudoGradient, Nodes.rival, Nodes.denom, Nodes.appeal, PiLp.toLp_apply]
  fun_prop (disch := simp_all [Nodes.denom, Nodes.appeal])

/-- The Fréchet derivative of the field is the Jacobian `jac`. -/
theorem fderiv_pseudoGradient_apply (hx : ∀ l, 0 < N.denom x l) (u : Profile ι α) :
    fderiv ℝ (pseudoGradient N c H) x u = jac N H x u := by
  have hdiff := (differentiableAt_pseudoGradient N c H x fun l ↦ (hx l).ne').hasFDerivAt
  have hline : HasDerivAt (fun t : ℝ ↦ x + t • u) u 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const u).const_add x
  have hcomp : HasDerivAt (fun t : ℝ ↦ pseudoGradient N c H (x + t • u))
      (fderiv ℝ (pseudoGradient N c H) x u) 0 := by
    have hdiff' : HasFDerivAt (pseudoGradient N c H) (fderiv ℝ (pseudoGradient N c H) x)
        (x + (0 : ℝ) • u) := by simpa using hdiff
    simpa [Function.comp_def] using hdiff'.comp_hasDerivAt (0 : ℝ) hline
  ext p
  have hcoord := ((PiLp.proj 2 (fun _ : ι × α ↦ ℝ) p).hasFDerivAt.comp_hasDerivAt (0 : ℝ) hcomp)
  exact hcoord.unique (hasDerivAt_pseudoGradient_line N H x c hx u p)

/-- **The field is Fréchet differentiable with derivative `jac`.** -/
@[blueprint "app:equilibrium-derivative"
  (statement := /--
    The negative marginal-profit field is Fréchet differentiable with derivative equal to the
    Jacobian blocks. Assumptions: A profile with positive share denominators.
  -/)]
theorem hasFDerivAt_pseudoGradient (hx : ∀ l, 0 < N.denom x l) :
    HasFDerivAt (pseudoGradient N c H) (LinearMap.toContinuousLinearMap (jac N H x)) x := by
  have hdiff := (differentiableAt_pseudoGradient N c H x fun l ↦ (hx l).ne').hasFDerivAt
  have heq : fderiv ℝ (pseudoGradient N c H) x = LinearMap.toContinuousLinearMap (jac N H x) := by
    ext1 u
    rw [fderiv_pseudoGradient_apply N c H x hx u]
    rfl
  rwa [heq] at hdiff

theorem continuousAt_opportunityLoading (hx : ∀ l, N.denom x l ≠ 0) (i : ι) (a : α) (l : κ) :
    ContinuousAt (fun z : Profile ι α ↦ N.opportunityLoading z i a l) x := by
  simp only [Nodes.opportunityLoading, Nodes.rival, Nodes.denom, Nodes.appeal]
  fun_prop (disch := simp_all [Nodes.denom, Nodes.appeal])

end Differentiability

/-! ### Complementarity at an orthant equilibrium -/

/-- At a solution of the orthant variational inequality, each coordinate of the field is
nonnegative, and it vanishes wherever the coordinate is positive. -/
theorem complementarity_of_solvesVI {F : Profile ι α → Profile ι α} {y : Profile ι α}
    (hy : SolvesVI F capacityOrthant y) (p : ι × α) :
    0 ≤ F y p ∧ (0 < y p → F y p = 0) := by
  classical
  have hyp : 0 ≤ y p := hy.1 p
  have hup : 0 ≤ F y p := by
    have hz : y + EuclideanSpace.single p 1 ∈ (capacityOrthant : Set (Profile ι α)) := by
      intro q
      by_cases hq : q = p
      · subst hq; simp [hy.1 q, add_nonneg]
      · simp [hq, hy.1 q]
    have h := hy.2 _ hz
    rwa [add_sub_cancel_left, EuclideanSpace.inner_single_right, one_mul] at h
  refine ⟨hup, fun hpos ↦ le_antisymm ?_ hup⟩
  have hz : y - EuclideanSpace.single p (y p) ∈ (capacityOrthant : Set (Profile ι α)) := by
    intro q
    by_cases hq : q = p
    · subst hq; simp
    · simp [hq, hy.1 q]
  have h := hy.2 _ hz
  rw [sub_sub_cancel_left, inner_neg_right, EuclideanSpace.inner_single_right] at h
  simp only [conj_trivial] at h
  nlinarith

/-! ### The derivative of the equilibrium map -/

section Derivative

variable [DecidableEq ι]

/-- The Euclidean distance between two opportunity-value vectors. -/
noncomputable def opportunityDist (r r' : κ → ℝ) : ℝ := Real.sqrt (∑ l, (r' l - r l) ^ 2)

omit [DecidableEq ι] in
/-- The opportunity impulse moves continuously with the profile, uniformly in the size of the
opportunity change. -/
theorem opportunityImpulse_sub_le (N : Nodes κ α) {x : Profile ι α} (hx : ∀ l, N.denom x l ≠ 0)
    {eta : ℝ} (heta : 0 < eta) :
    ∃ d > 0, ∀ z : Profile ι α, ‖z - x‖ < d → ∀ δ : κ → ℝ,
      ‖opportunityImpulse N z δ - opportunityImpulse N x δ‖ ≤
        eta * Real.sqrt (∑ l, δ l ^ 2) := by
  set Ψ : Profile ι α → ℝ := fun z ↦
    ∑ p : ι × α, ∑ l, (N.opportunityLoading z p.1 p.2 l - N.opportunityLoading x p.1 p.2 l) ^ 2
    with hΨ
  have hcont : ContinuousAt Ψ x := by
    simp only [hΨ, Nodes.opportunityLoading, Nodes.rival, Nodes.denom, Nodes.appeal]
    fun_prop (disch := simp_all [Nodes.denom, Nodes.appeal])
  have hΨx : Ψ x = 0 := by simp [hΨ]
  obtain ⟨d, hd, hball⟩ := Metric.continuousAt_iff.mp hcont (eta ^ 2) (by positivity)
  refine ⟨d, hd, fun z hz δ ↦ ?_⟩
  have hΨz : Ψ z < eta ^ 2 := by
    have := hball (by rwa [dist_eq_norm])
    rw [hΨx, Real.dist_eq, sub_zero] at this
    exact (le_abs_self _).trans_lt this
  have hsq : ‖opportunityImpulse N z δ - opportunityImpulse N x δ‖ ^ 2 ≤
      (∑ l, δ l ^ 2) * Ψ z := by
    rw [EuclideanSpace.norm_sq_eq, hΨ, Finset.mul_sum]
    refine Finset.sum_le_sum fun p _ ↦ ?_
    have hcoord : (opportunityImpulse N z δ - opportunityImpulse N x δ) p =
        ∑ l, δ l * (N.opportunityLoading z p.1 p.2 l - N.opportunityLoading x p.1 p.2 l) := by
      simp only [opportunityImpulse, PiLp.sub_apply, ← Finset.sum_sub_distrib,
        mul_sub]
    rw [Real.norm_eq_abs, sq_abs, hcoord]
    exact Finset.sum_mul_sq_le_sq_mul_sq _ _ _
  have hS : 0 ≤ ∑ l, δ l ^ 2 := Finset.sum_nonneg fun _ _ ↦ sq_nonneg _
  have hle : ‖opportunityImpulse N z δ - opportunityImpulse N x δ‖ ^ 2 ≤
      (eta * Real.sqrt (∑ l, δ l ^ 2)) ^ 2 := by
    rw [mul_pow, Real.sq_sqrt hS]
    calc _ ≤ (∑ l, δ l ^ 2) * Ψ z := hsq
      _ ≤ (∑ l, δ l ^ 2) * eta ^ 2 := mul_le_mul_of_nonneg_left hΨz.le hS
      _ = eta ^ 2 * ∑ l, δ l ^ 2 := by ring
  exact (pow_le_pow_iff_left₀ (norm_nonneg _) (by positivity) two_ne_zero).mp hle

/-- **The linear response is the derivative of the equilibrium map.**

Let `x` be the equilibrium at opportunity values `N.revenue`, with strict complementarity: every
coordinate has positive capacity or a strictly positive field value. Let the active set be the
coordinates with positive capacity. Then for every `ε > 0` there is a neighbourhood of
`N.revenue` on which the equilibrium `y` at nonnegative values `r` satisfies
`‖y - x - χ (r - N.revenue)‖ ≤ ε ‖r - N.revenue‖`. Here `χ δr` is the linear response of
`LinearResponse.lean` on that active set. -/
@[blueprint "eq:equilibrium-derivative"
  (statement := /--
    Under strict complementarity the linear response is the derivative of the equilibrium map in
    opportunity values. Assumptions: Assumption 1 primitives, an equilibrium with strict
    complementarity, the active set of positive capacities, and nonnegative nearby opportunity
    values.
  -/)]
theorem equilibrium_sub_linearResponse_le (N : Nodes κ α) (c : ι → α → ℝ)
    {H : ι → α → α → ℝ} {mu : ℝ} (hsymm : ∀ i a b, H i a b = H i b a) (hmu : 0 < mu)
    (hH : ∀ i, ∀ u : α → ℝ, mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (hc : ∀ i a, 0 ≤ c i a) {x : Profile ι α} (hxN : IsNashOrthant N c H x)
    (hsc : ∀ p, 0 < x p ∨ 0 < pseudoGradient N c H x p) {ε : ℝ} (hε : 0 < ε) :
    ∃ d > 0, ∀ (r : κ → ℝ) (hr : ∀ l, 0 ≤ r l), opportunityDist N.revenue r < d →
      ∀ y, IsNashOrthant (N.withRevenue r hr) c H y →
      ∀ u, SolvesRestricted (jac N H x) (restrictL fun p ↦ 0 < x p)
          (opportunityImpulse N x fun l ↦ r l - N.revenue l) u →
        ‖y - x - u‖ ≤ ε * opportunityDist N.revenue r := by
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
  set M := Real.sqrt (opportunityModulusSq ι N) with hM
  have hM0 : 0 ≤ M := Real.sqrt_nonneg _
  set K := M / mu with hK
  have hK0 : 0 ≤ K := div_nonneg hM0 hmu.le
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
  -- Fréchet expansion of the field and continuity of the impulse.
  set eta := ε * mu / (K + 1) with heta
  have heta0 : 0 < eta := by positivity
  have hfd := (hasFDerivAt_pseudoGradient N c H x hDpos).isLittleO.def heta0
  obtain ⟨d1, hd1, hball1⟩ := Metric.eventually_nhds_iff.mp hfd
  obtain ⟨d2, hd2, hball2⟩ := opportunityImpulse_sub_le N (fun l ↦ (hDpos l).ne') heta0
  set d := min (min d1 d2) (m / 2) / (K + Lip * K + M + 1) with hd
  have hden : 0 < K + Lip * K + M + 1 := by positivity
  have hdpos : 0 < d := div_pos (lt_min (lt_min hd1 hd2) (by positivity)) hden
  refine ⟨d, hdpos, fun r hr hρ y hy u hu ↦ ?_⟩
  set δ : κ → ℝ := fun l ↦ r l - N.revenue l with hδ
  set ρ := opportunityDist N.revenue r with hρdef
  have hρ0 : 0 ≤ ρ := Real.sqrt_nonneg _
  set Fr := pseudoGradient (N.withRevenue r hr) c H with hFr
  have hy0 : ∀ p, 0 ≤ y p := hy.1
  have hyvi := solvesVI_capacityOrthant_of_isNashOrthant (N.withRevenue r hr) c hsymm hmu hH hc hy
  have hycomp := complementarity_of_solvesVI hyvi
  -- Opportunity continuity.
  have hcont : ‖y - x‖ ≤ K * ρ := by
    have h := orthantEquilibrium_dist_le_opportunityShift N r hr c hsymm hmu hH hc hxN hy
    calc ‖y - x‖ ≤ M * ρ / mu := h
      _ = K * ρ := by rw [hK]; ring
  have hsmall : (K + Lip * K + M + 1) * ρ < min (min d1 d2) (m / 2) := by
    have := (lt_div_iff₀' hden).mp (hd ▸ hρ)
    linarith
  have hexpand : (K + Lip * K + M + 1) * ρ = K * ρ + Lip * (K * ρ) + M * ρ + ρ := by ring
  have hLKρ : 0 ≤ Lip * (K * ρ) := mul_nonneg hLip0 (mul_nonneg hK0 hρ0)
  have hMρ : 0 ≤ M * ρ := mul_nonneg hM0 hρ0
  have hKρ0 : 0 ≤ K * ρ := mul_nonneg hK0 hρ0
  have hKρS : K * ρ ≤ (K + Lip * K + M + 1) * ρ := by linarith
  have hKρ1 : K * ρ < d1 := by
    linarith [min_le_left (min d1 d2) (m / 2), min_le_left d1 d2, hKρS]
  have hKρ2 : K * ρ < d2 := by
    linarith [min_le_left (min d1 d2) (m / 2), min_le_right d1 d2, hKρS]
  have hgapρ : K * ρ + Lip * (K * ρ) + M * ρ < m := by
    have := min_le_right (min d1 d2) (m / 2)
    linarith
  -- The field at the new equilibrium is close to the field at the old one.
  have hfield : ‖Fr y - F x‖ ≤ Lip * (K * ρ) + M * ρ := by
    have hsplit : Fr y - F x = (Fr y - F y) + (F y - F x) := by abel
    have h1 : ‖Fr y - F y‖ ≤ M * ρ := by
      have hsq := norm_sq_pseudoGradient_withRevenue_sub_le N r hr c H hy0
      have hle : ‖Fr y - F y‖ ≤ Real.sqrt (opportunityModulusSq ι N * ∑ l, (r l - N.revenue l) ^ 2) := by
        rw [← Real.sqrt_sq (norm_nonneg (Fr y - F y))]
        exact Real.sqrt_le_sqrt hsq
      rwa [Real.sqrt_mul (opportunityModulusSq_nonneg N)] at hle
    have h2 : ‖F y - F x‖ ≤ Lip * (K * ρ) := by
      have hbox := lipschitzOn_pseudoGradient N c H (lo := fun _ ↦ (0 : ℝ))
        (hi := fun p ↦ max (x p) (y p)) (fun _ ↦ le_refl 0) y
        (fun p ↦ ⟨hy0 p, le_max_right _ _⟩) x (fun p ↦ ⟨hx0 p, le_max_left _ _⟩)
      exact hbox.trans (mul_le_mul_of_nonneg_left hcont hLip0)
    rw [hsplit]
    exact (norm_add_le _ _).trans (by linarith)
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
    have hFy : 0 < Fr y p := by
      have := hcoord (Fr y - F x) p
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
  have hPFr : P (Fr y) = 0 := by
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
  set R2 := opportunityImpulse N y δ - opportunityImpulse N x δ with hR2
  have hshift : Fr y - F y = -opportunityImpulse N y δ :=
    pseudoGradient_withRevenue_sub_eq_neg_impulse N H y r hr c
  have hR1le : ‖R1‖ ≤ eta * ‖y - x‖ := by
    have := hball1 (by rw [dist_eq_norm]; exact hcont.trans_lt hKρ1)
    simpa [hR1] using this
  have hR2le : ‖R2‖ ≤ eta * ρ := hball2 y (hcont.trans_lt hKρ2) δ
  have hexp : Fr y - F x = jac N H x (y - x) + R1 - opportunityImpulse N x δ - R2 := by
    rw [hR1, hR2]
    have : Fr y = F y - opportunityImpulse N y δ := by rw [sub_eq_add_neg, ← hshift]; abel
    rw [this]; abel
  have hw : SolvesRestricted (jac N H x) P (R2 - R1) (y - x - u) := by
    refine ⟨by rw [map_sub, hPyx, hu.1], ?_⟩
    have h0 : P (Fr y - F x) = 0 := by rw [map_sub, hPFr, hPF, sub_zero]
    rw [hexp] at h0
    rw [map_sub (jac N H x), map_sub P, hu.2, map_sub P, ← sub_eq_zero]
    rw [map_sub, map_sub, map_add] at h0
    rw [← h0]
    abel
  have hbound := hw.norm_le hPproj hmu (fun v _ ↦ jac_coercive N H x hH hx0 v)
  have hPle := hPproj.norm_le (R2 - R1)
  calc ‖y - x - u‖ ≤ ‖P (R2 - R1)‖ / mu := hbound
    _ ≤ (‖R2‖ + ‖R1‖) / mu :=
        div_le_div_of_nonneg_right (hPle.trans (norm_sub_le _ _)) hmu.le
    _ ≤ (eta * ρ + eta * (K * ρ)) / mu := by
        refine div_le_div_of_nonneg_right ?_ hmu.le
        linarith [mul_le_mul_of_nonneg_left hcont heta0.le]
    _ = ε * ρ := by
        rw [heta]; field_simp; ring

end Derivative

end PricingPerspective.Operating.RationalShare.EquilibriumResponse
