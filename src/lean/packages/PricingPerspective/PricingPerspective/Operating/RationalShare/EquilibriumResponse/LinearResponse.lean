import Architect
import PricingPerspective.Operating.RationalShare.EquilibriumResponse.ResponseOperator

/-!
# The linear response of the rational-share equilibrium, and its rival feedback

This module machine-checks ESCAPE's local response to opportunity values in its *linearised*
form. It does not invoke an implicit-function theorem. Fix an equilibrium and a set of active
capacity coordinates. The response `δq` to an opportunity change `δr` is defined as the unique
solution, supported on the active set, of the equilibrium conditions linearised there:

    P (J δq) = P (𝒢 δr),    P δq = δq,

where `J` is the Jacobian of the negative marginal-profit field and `𝒢 δr` is the exact change in
that field when opportunities move. Under strict complementarity with a locally fixed active set,
this is the derivative of the equilibrium map; that identification is standard and is *not*
checked here.

**Generic part.** Over any finite-dimensional real inner-product space, a linear map `T` that is
`mu`-coercive on the range of a self-adjoint idempotent `P` makes the restricted system uniquely
solvable, with `‖u‖ ≤ ‖P g‖ / mu`. Split `T = Td + C` into a coercive "direct" part and a
remainder. Two consequences follow:

* the difference between the full and direct responses is the full response to the forcing
  `-C u_d`. This is the identity `χ_full - χ_direct = -J⁻¹ C χ_direct`;
* when `‖P C‖ ≤ L < mu`, the Jacobi iteration `P Td v_{h+1} = P (g - C v_h)` converges to the full
  response at rate `L / mu`. Started from `v₀ = 0`, the iterate `v_{h+1}` is the `h`-th partial sum
  of the Neumann series `Σ (-Td⁻¹ C)^k u_d`, and `v₁` is the direct response.

**Rational-share part.** The Jacobian splits into own blocks and cross blocks:

    J_ii = H_i + Σ_l 2 r_l B_il / D_l³ k_l k_lᵀ,    J_ij = Σ_l r_l (B_il - A_il) / D_l³ k_l k_lᵀ.

It is the directional derivative of the field along any line, at any profile with positive
denominators. Both `J` and its own-block part are `mu`-coercive at every nonnegative profile. For
`J` this comes from uniqueness of derivatives: the existing segment derivative of the revenue
field and its nonnegativity carry over directly.
-/

namespace PricingPerspective.Operating.RationalShare.EquilibriumResponse

open PricingPerspective.Operating.VariationalInequality

open scoped RealInnerProductSpace

section Restricted

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- A self-adjoint idempotent linear map: an orthogonal projection, such as restriction to a set
of coordinates. -/
structure IsOrthoProjection (P : E →ₗ[ℝ] E) : Prop where
  idem : ∀ u, P (P u) = P u
  symm : ∀ u v, ⟪P u, v⟫ = ⟪u, P v⟫

/-- `u` solves the linear system `T u = g` restricted to the range of `P`. -/
def SolvesRestricted (T P : E →ₗ[ℝ] E) (g u : E) : Prop :=
  P u = u ∧ P (T u) = P g

variable {T P : E →ₗ[ℝ] E} {mu : ℝ}

/-- Differences of restricted solutions solve the restricted system for the difference of
forcings. -/
theorem SolvesRestricted.sub {g g' u u' : E} (hu : SolvesRestricted T P g u)
    (hu' : SolvesRestricted T P g' u') : SolvesRestricted T P (g - g') (u - u') :=
  ⟨by rw [map_sub, hu.1, hu'.1], by rw [map_sub, map_sub, hu.2, hu'.2, map_sub]⟩

/-- **The restricted response is bounded by the restricted forcing.** -/
theorem SolvesRestricted.norm_le (hP : IsOrthoProjection P) (hmu : 0 < mu)
    (hT : ∀ v, P v = v → mu * ‖v‖ ^ 2 ≤ ⟪T v, v⟫) {g u : E}
    (hu : SolvesRestricted T P g u) : ‖u‖ ≤ ‖P g‖ / mu := by
  have key : ⟪T u, u⟫ = ⟪P g, u⟫ := by
    calc ⟪T u, u⟫ = ⟪T u, P u⟫ := by rw [hu.1]
      _ = ⟪P (T u), u⟫ := (hP.symm _ _).symm
      _ = ⟪P g, u⟫ := by rw [hu.2]
  have h1 := hT u hu.1
  have h2 : ⟪P g, u⟫ ≤ ‖P g‖ * ‖u‖ := real_inner_le_norm _ _
  rw [le_div_iff₀ hmu]
  rcases eq_or_lt_of_le (norm_nonneg u) with h0 | hpos
  · rw [← h0, zero_mul]; exact norm_nonneg _
  · refine le_of_mul_le_mul_right ?_ hpos
    nlinarith

/-- **Uniqueness of the restricted response.** -/
theorem SolvesRestricted.unique (hP : IsOrthoProjection P) (hmu : 0 < mu)
    (hT : ∀ v, P v = v → mu * ‖v‖ ^ 2 ≤ ⟪T v, v⟫) {g u u' : E}
    (hu : SolvesRestricted T P g u) (hu' : SolvesRestricted T P g u') : u = u' := by
  have h := (hu.sub hu').norm_le hP hmu hT
  rw [sub_self, map_zero, norm_zero, zero_div] at h
  exact sub_eq_zero.mp (norm_le_zero_iff.mp h)

/-- **Existence of the restricted response.** The map `u ↦ P T P u + (u - P u)` is injective by
the norm bound, hence surjective in finite dimension, and its preimage of `P g` is the solution. -/
theorem exists_solvesRestricted [FiniteDimensional ℝ E] (hP : IsOrthoProjection P)
    (hmu : 0 < mu) (hT : ∀ v, P v = v → mu * ‖v‖ ^ 2 ≤ ⟪T v, v⟫) (g : E) :
    ∃ u, SolvesRestricted T P g u := by
  let L : E →ₗ[ℝ] E := P ∘ₗ T ∘ₗ P + (LinearMap.id - P)
  have hL : ∀ u, L u = P (T (P u)) + (u - P u) := fun u ↦ rfl
  -- Projecting `L u` recovers the restricted equation.
  have hPL : ∀ u, P (L u) = P (T (P u)) := by
    intro u
    rw [hL, map_add, map_sub, hP.idem, hP.idem, sub_self, add_zero]
  -- Any preimage of a projected vector is supported on the range of `P`.
  have hsolve : ∀ u h, L u = P h → SolvesRestricted T P h u := by
    intro u h hu
    have hproj : P (T (P u)) = P h := by rw [← hPL, hu, hP.idem]
    have hfix : P u = u := by
      have : u - P u = 0 := by
        have := hu
        rw [hL, hproj] at this
        simpa using this
      exact (sub_eq_zero.mp this).symm
    exact ⟨hfix, by rwa [hfix] at hproj⟩
  have hinj : Function.Injective L := by
    rw [← LinearMap.ker_eq_bot, LinearMap.ker_eq_bot']
    intro u hu
    have hsol : SolvesRestricted T P 0 u := hsolve u 0 (by rw [hu, map_zero])
    have h := hsol.norm_le hP hmu hT
    rw [map_zero, norm_zero, zero_div] at h
    exact norm_le_zero_iff.mp h
  obtain ⟨u, hu⟩ := LinearMap.injective_iff_surjective.mp hinj (P g)
  exact ⟨u, hsolve u g hu⟩

/-- **Existence and uniqueness.** -/
theorem existsUnique_solvesRestricted [FiniteDimensional ℝ E] (hP : IsOrthoProjection P)
    (hmu : 0 < mu) (hT : ∀ v, P v = v → mu * ‖v‖ ^ 2 ≤ ⟪T v, v⟫) (g : E) :
    ∃! u, SolvesRestricted T P g u := by
  obtain ⟨u, hu⟩ := exists_solvesRestricted hP hmu hT g
  exact ⟨u, hu, fun u' hu' ↦ hu'.unique hP hmu hT hu⟩

/-- **The feedback identity.** Split `T = Td + C`. The full response minus the direct response is
the full response to the forcing `-C u_d` generated by the direct response. This is
`χ_full - χ_direct = -J⁻¹ C χ_direct` on the range of `P`. -/
theorem feedback_solvesRestricted {Td C : E →ₗ[ℝ] E} {g uf ud : E}
    (hf : SolvesRestricted (Td + C) P g uf) (hd : SolvesRestricted Td P g ud) :
    SolvesRestricted (Td + C) P (-(C ud)) (uf - ud) := by
  refine ⟨by rw [map_sub, hf.1, hd.1], ?_⟩
  rw [map_sub, map_sub, hf.2, LinearMap.add_apply, map_add, hd.2, map_neg]
  abel

/-- **Convergence of the rival-feedback iteration.** Suppose the direct part is `mu`-coercive and
the remainder satisfies `‖P (C v)‖ ≤ L ‖v‖` on the range of `P`. Then the Jacobi iterates
`P Td v_{h+1} = P (g - C v_h)` approach the full response at rate `L / mu`.

Started from `v₀ = 0`, the iterate `v_{h+1}` is the `h`-th partial sum of the Neumann series of
repeated rival feedback. -/
theorem jacobi_iterate_dist_le (hP : IsOrthoProjection P) (hmu : 0 < mu) {Td C : E →ₗ[ℝ] E}
    (hTd : ∀ v, P v = v → mu * ‖v‖ ^ 2 ≤ ⟪Td v, v⟫) {L : ℝ} (hL : 0 ≤ L)
    (hC : ∀ v, P v = v → ‖P (C v)‖ ≤ L * ‖v‖) {g uf : E}
    (hf : SolvesRestricted (Td + C) P g uf) (v : ℕ → E) (hv0 : P (v 0) = v 0)
    (hv : ∀ h, SolvesRestricted Td P (g - C (v h)) (v (h + 1))) (h : ℕ) :
    ‖v h - uf‖ ≤ (L / mu) ^ h * ‖v 0 - uf‖ := by
  -- The full response is a fixed point of the iteration.
  have hfix : SolvesRestricted Td P (g - C uf) uf := by
    refine ⟨hf.1, ?_⟩
    have h2 := hf.2
    rw [LinearMap.add_apply, map_add] at h2
    rw [map_sub, ← h2]
    abel
  have hsupp : ∀ h, P (v h) = v h := by
    intro h
    cases h with
    | zero => exact hv0
    | succ h => exact (hv h).1
  have hrho : 0 ≤ L / mu := div_nonneg hL hmu.le
  induction h with
  | zero => simp
  | succ h ih =>
    have herr := (hv h).sub hfix
    have hforce : g - C (v h) - (g - C uf) = -(C (v h - uf)) := by
      rw [map_sub]; abel
    rw [hforce] at herr
    have hb := herr.norm_le hP hmu hTd
    rw [map_neg, norm_neg] at hb
    have hCe := hC (v h - uf) (by rw [map_sub, hsupp h, hf.1])
    calc ‖v (h + 1) - uf‖ ≤ ‖P (C (v h - uf))‖ / mu := hb
      _ ≤ L * ‖v h - uf‖ / mu := div_le_div_of_nonneg_right hCe hmu.le
      _ = L / mu * ‖v h - uf‖ := by ring
      _ ≤ L / mu * ((L / mu) ^ h * ‖v 0 - uf‖) := mul_le_mul_of_nonneg_left ih hrho
      _ = (L / mu) ^ (h + 1) * ‖v 0 - uf‖ := by ring

end Restricted

open Finset

variable {κ ι α : Type*} [Fintype κ] [Fintype ι] [Fintype α]

/-! ### Coordinate restriction -/

/-- Restriction to a set of active firm-activity coordinates. -/
noncomputable def restrictL (act : ι × α → Prop) [DecidablePred act] :
    Profile ι α →ₗ[ℝ] Profile ι α where
  toFun u := WithLp.toLp 2 fun p ↦ if act p then u p else 0
  map_add' u v := by
    ext p
    by_cases h : act p <;> simp [h]
  map_smul' c u := by
    ext p
    by_cases h : act p <;> simp [h]

@[simp] theorem restrictL_apply (act : ι × α → Prop) [DecidablePred act] (u : Profile ι α)
    (p : ι × α) : restrictL act u p = if act p then u p else 0 := rfl

theorem restrictL_isOrthoProjection (act : ι × α → Prop) [DecidablePred act] :
    IsOrthoProjection (restrictL (ι := ι) (α := α) act) := by
  refine ⟨fun u ↦ ?_, fun u v ↦ ?_⟩
  · ext p
    by_cases h : act p <;> simp [h]
  · rw [inner_eq_sum, inner_eq_sum]
    exact Finset.sum_congr rfl fun p _ ↦ by by_cases h : act p <;> simp [h]

/-! ### Linearity of appeal in the direction -/

theorem appeal_add (N : Nodes κ α) (u v : Profile ι α) (i : ι) (l : κ) :
    N.appeal (u + v) i l = N.appeal u i l + N.appeal v i l := by
  simp only [Nodes.appeal, PiLp.add_apply, mul_add, Finset.sum_add_distrib]

theorem appeal_smul (N : Nodes κ α) (c : ℝ) (u : Profile ι α) (i : ι) (l : κ) :
    N.appeal (c • u) i l = c * N.appeal u i l := by
  simp only [Nodes.appeal, PiLp.smul_apply, smul_eq_mul, Finset.mul_sum]
  exact Finset.sum_congr rfl fun a _ ↦ by ring

theorem dirSum_add (N : Nodes κ α) (u v : Profile ι α) (l : κ) :
    dirSum N (u + v) l = dirSum N u l + dirSum N v l := by
  simp only [dirSum, appeal_add, Finset.sum_add_distrib]

theorem dirSum_smul (N : Nodes κ α) (c : ℝ) (u : Profile ι α) (l : κ) :
    dirSum N (c • u) l = c * dirSum N u l := by
  simp only [dirSum, appeal_smul, Finset.mul_sum]

/-! ### The Jacobian blocks -/

variable (N : Nodes κ α) (H : ι → α → α → ℝ) (x : Profile ι α)

/-- Own-block Jacobian: cost curvature plus `Σ_l 2 r_l B_il / D_l³ k_l k_lᵀ`, applied to firm
`i`'s own direction. -/
noncomputable def ownJac : Profile ι α →ₗ[ℝ] Profile ι α where
  toFun u := WithLp.toLp 2 fun p ↦ ∑ b, H p.1 p.2 b * u (p.1, b) +
    ∑ l, N.revenue l * N.kernel l p.2 *
      (2 * N.rival x p.1 l / N.denom x l ^ 3 * N.appeal u p.1 l)
  map_add' u v := by
    ext p
    simp only [PiLp.add_apply, appeal_add, mul_add, Finset.sum_add_distrib]
    ring
  map_smul' c u := by
    ext p
    simp only [PiLp.smul_apply, smul_eq_mul, appeal_smul, RingHom.id_apply]
    rw [mul_add, Finset.mul_sum, Finset.mul_sum]
    congr 1
    · exact Finset.sum_congr rfl fun b _ ↦ by ring
    · exact Finset.sum_congr rfl fun l _ ↦ by ring

/-- Cross-block Jacobian: `Σ_l r_l (B_il - A_il) / D_l³ k_l k_lᵀ` applied to the rivals' total
direction. -/
noncomputable def crossJac : Profile ι α →ₗ[ℝ] Profile ι α where
  toFun u := WithLp.toLp 2 fun p ↦ ∑ l, N.revenue l * N.kernel l p.2 *
      ((N.rival x p.1 l - N.appeal x p.1 l) / N.denom x l ^ 3 *
        (dirSum N u l - N.appeal u p.1 l))
  map_add' u v := by
    ext p
    simp only [PiLp.add_apply, appeal_add, dirSum_add, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun l _ ↦ by ring
  map_smul' c u := by
    ext p
    simp only [PiLp.smul_apply, smul_eq_mul, appeal_smul, dirSum_smul, Finset.mul_sum,
      RingHom.id_apply]
    exact Finset.sum_congr rfl fun l _ ↦ by ring

/-- The Jacobian of the negative marginal-profit field. -/
noncomputable def jac : Profile ι α →ₗ[ℝ] Profile ι α := ownJac N H x + crossJac N x

theorem ownJac_apply (u : Profile ι α) (i : ι) (a : α) :
    ownJac N H x u (i, a) = ∑ b, H i a b * u (i, b) +
      ∑ l, N.revenue l * N.kernel l a *
        (2 * N.rival x i l / N.denom x l ^ 3 * N.appeal u i l) := rfl

theorem crossJac_apply (u : Profile ι α) (i : ι) (a : α) :
    crossJac N x u (i, a) = ∑ l, N.revenue l * N.kernel l a *
      ((N.rival x i l - N.appeal x i l) / N.denom x l ^ 3 *
        (dirSum N u l - N.appeal u i l)) := rfl

/-! ### The Jacobian is the derivative of the field -/

/-- One node's contribution along a line: the quotient rule with affine numerator and
denominator. -/
theorem node_line_hasDerivAt {r k D S A a : ℝ} (hD : D ≠ 0) :
    HasDerivAt (fun t : ℝ ↦ r * ((D + t * S) - (A + t * a)) / (D + t * S) ^ 2 * k)
      (r * k * ((S - a) / D ^ 2 - 2 * (D - A) * S / D ^ 3)) 0 := by
  have hn : HasDerivAt (fun t : ℝ ↦ r * ((D + t * S) - (A + t * a))) (r * (S - a)) 0 := by
    have h1 : HasDerivAt (fun t : ℝ ↦ D + t * S) S 0 := by
      simpa using ((hasDerivAt_id (0 : ℝ)).mul_const S).const_add D
    have h2 : HasDerivAt (fun t : ℝ ↦ A + t * a) a 0 := by
      simpa using ((hasDerivAt_id (0 : ℝ)).mul_const a).const_add A
    exact (h1.sub h2).const_mul r
  have hd : HasDerivAt (fun t : ℝ ↦ (D + t * S) ^ 2) (2 * D * S) 0 := by
    have h1 : HasDerivAt (fun t : ℝ ↦ D + t * S) S 0 := by
      simpa using ((hasDerivAt_id (0 : ℝ)).mul_const S).const_add D
    have hfun : (fun t : ℝ ↦ (D + t * S) ^ 2) = fun t ↦ (D + t * S) * (D + t * S) := by
      funext t; ring
    rw [hfun]
    refine (h1.mul h1).congr_deriv ?_
    simp only [zero_mul, add_zero]
    ring
  have hne : (D + 0 * S) ^ 2 ≠ 0 := by simpa using pow_ne_zero 2 hD
  refine ((hn.div hd hne).mul_const k).congr_deriv ?_
  simp only [zero_mul, add_zero]
  field_simp

/-- **The Jacobian is the directional derivative of the field.** Along any line through a
profile with positive denominators, each coordinate of the negative marginal-profit field has
derivative given by `jac`. -/
@[blueprint "app:response-jacobian"
  (statement := /--
    The own and rival Jacobian blocks are the directional derivative of the negative
    marginal-profit field. Assumptions: A profile with positive share denominators and any
    direction.
  -/)]
theorem hasDerivAt_pseudoGradient_line (c : ι → α → ℝ) (hx : ∀ l, 0 < N.denom x l)
    (u : Profile ι α) (p : ι × α) :
    HasDerivAt (fun t : ℝ ↦ pseudoGradient N c H (x + t • u) p) (jac N H x u p) 0 := by
  obtain ⟨i, a⟩ := p
  have reshape : (fun t : ℝ ↦ pseudoGradient N c H (x + t • u) (i, a)) =
      fun t : ℝ ↦ (c i a + ∑ b, H i a b * (x (i, b) + t * u (i, b))) -
        ∑ l, N.revenue l * ((N.denom x l + t * dirSum N u l) -
          (N.appeal x i l + t * N.appeal u i l)) /
          (N.denom x l + t * dirSum N u l) ^ 2 * N.kernel l a := by
    funext t
    simp only [pseudoGradient_apply, Nodes.rival, denom_add_smul, appeal_add_smul,
      PiLp.add_apply, PiLp.smul_apply, smul_eq_mul]
  have hcost : HasDerivAt (fun t : ℝ ↦ c i a + ∑ b, H i a b * (x (i, b) + t * u (i, b)))
      (∑ b, H i a b * u (i, b)) 0 := by
    refine HasDerivAt.const_add _ (HasDerivAt.fun_sum fun b _ ↦ ?_)
    have h : HasDerivAt (fun t : ℝ ↦ x (i, b) + t * u (i, b)) (u (i, b)) 0 := by
      simpa using ((hasDerivAt_id (0 : ℝ)).mul_const (u (i, b))).const_add (x (i, b))
    exact h.const_mul (H i a b)
  have hrev := HasDerivAt.fun_sum (u := Finset.univ) (x := (0 : ℝ))
    (A := fun l t ↦ N.revenue l * ((N.denom x l + t * dirSum N u l) -
      (N.appeal x i l + t * N.appeal u i l)) /
      (N.denom x l + t * dirSum N u l) ^ 2 * N.kernel l a)
    (fun l _ ↦ node_line_hasDerivAt (hx l).ne')
  rw [reshape]
  refine (hcost.sub hrev).congr_deriv ?_
  simp only [jac, LinearMap.add_apply, PiLp.add_apply, ownJac_apply, crossJac_apply,
    Nodes.rival]
  rw [add_assoc, ← Finset.sum_add_distrib, sub_eq_add_neg, ← Finset.sum_neg_distrib]
  congr 1
  refine Finset.sum_congr rfl fun l _ ↦ ?_
  have hD := (hx l).ne'
  field_simp
  ring

/-! ### Coercivity -/

/-- The cost quadratic form carries the modulus. -/
theorem cost_quadratic_ge {mu : ℝ}
    (hH : ∀ i, ∀ u : α → ℝ, mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (u : Profile ι α) :
    mu * ‖u‖ ^ 2 ≤ ∑ p : ι × α, (∑ b, H p.1 p.2 b * u (p.1, b)) * u p := by
  rw [norm_sq_eq_sum, Fintype.sum_prod_type, Fintype.sum_prod_type, Finset.mul_sum]
  refine Finset.sum_le_sum fun i _ ↦ (hH i fun a ↦ u (i, a)).trans (le_of_eq ?_)
  refine Finset.sum_congr rfl fun a _ ↦ ?_
  rw [Finset.sum_mul]
  exact Finset.sum_congr rfl fun b _ ↦ by ring

/-- **The Jacobian is `mu`-coercive at every nonnegative profile.** The pairing `⟪J u, u⟫` is the
derivative of `t ↦ ⟪F (x + t u), u⟫` at zero. By uniqueness of derivatives, it equals the cost
quadratic form plus the revenue segment derivative, which is nonnegative. -/
theorem jac_coercive {mu : ℝ}
    (hH : ∀ i, ∀ u : α → ℝ, mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (hx : ∀ p, 0 ≤ x p) (u : Profile ι α) :
    mu * ‖u‖ ^ 2 ≤ ⟪jac N H x u, u⟫ := by
  set c : ι → α → ℝ := fun _ _ ↦ 0
  have hpos : ∀ l, 0 < N.denom x l := N.denom_pos hx
  -- Derivative through the Jacobian.
  have hg1 : HasDerivAt (fun t : ℝ ↦ ⟪pseudoGradient N c H (x + t • u), u⟫)
      ⟪jac N H x u, u⟫ 0 := by
    have h := HasDerivAt.fun_sum (u := Finset.univ) (x := (0 : ℝ))
      (A := fun p t ↦ pseudoGradient N c H (x + t • u) p * u p)
      (fun p _ ↦ (hasDerivAt_pseudoGradient_line N H x c hpos u p).mul_const (u p))
    have hfun : (fun t : ℝ ↦ ⟪pseudoGradient N c H (x + t • u), u⟫) =
        fun t ↦ ∑ p, pseudoGradient N c H (x + t • u) p * u p := by
      funext t; rw [inner_eq_sum]
    rw [hfun, inner_eq_sum]
    exact h
  -- Derivative through the cost/revenue split.
  set K : ℝ := ∑ p : ι × α, (∑ b, H p.1 p.2 b * u (p.1, b)) * u p with hK
  have hcost : (fun t : ℝ ↦ ⟪costField (ι := ι) c H (x + t • u), u⟫) =
      fun t ↦ ⟪costField (ι := ι) c H x, u⟫ + t * K := by
    funext t
    rw [inner_eq_sum, inner_eq_sum, hK, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun p _ ↦ ?_
    have hsplit : ∑ b, H p.1 p.2 b * (x + t • u) (p.1, b) =
        ∑ b, H p.1 p.2 b * x (p.1, b) + t * ∑ b, H p.1 p.2 b * u (p.1, b) := by
      rw [Finset.mul_sum, ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun b _ ↦ by
        simp only [PiLp.add_apply, PiLp.smul_apply, smul_eq_mul]; ring
    simp only [costField, hsplit]
    ring
  have hcostD : HasDerivAt (fun t : ℝ ↦ ⟪costField (ι := ι) c H (x + t • u), u⟫) K 0 := by
    rw [hcost]
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const K).const_add
      ⟪costField (ι := ι) c H x, u⟫
  have hrevD := revenue_segment_hasDerivAt N x u 0 (by simpa using hpos)
  have hg2 : HasDerivAt (fun t : ℝ ↦ ⟪pseudoGradient N c H (x + t • u), u⟫)
      (K + ∑ l, N.revenue l
        * (N.denom (x + (0 : ℝ) • u) l * (dirSq N u l + dirSum N u l ^ 2)
            - 2 * dirSum N u l * dirCross N (x + (0 : ℝ) • u) u l)
        / N.denom (x + (0 : ℝ) • u) l ^ 3) 0 := by
    have hfun : (fun t : ℝ ↦ ⟪pseudoGradient N c H (x + t • u), u⟫) =
        fun t ↦ ⟪costField (ι := ι) c H (x + t • u), u⟫ + ⟪revenueField N (x + t • u), u⟫ := by
      funext t
      rw [pseudoGradient_eq_costField_add_revenueField, inner_add_left]
    rw [hfun]
    exact hcostD.add hrevD
  rw [hg1.unique hg2]
  have hrev := revenue_derivative_nonneg N (z := x + (0 : ℝ) • u) (by simpa using hx) u
  linarith [cost_quadratic_ge H hH u]

/-- **The own-block Jacobian is `mu`-coercive at every nonnegative profile.** Its demand part is
`Σ_l Σ_i 2 r_l B_il / D_l³ A_i(u)_l²`, which is nonnegative. -/
theorem ownJac_coercive {mu : ℝ}
    (hH : ∀ i, ∀ u : α → ℝ, mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (hx : ∀ p, 0 ≤ x p) (u : Profile ι α) :
    mu * ‖u‖ ^ 2 ≤ ⟪ownJac N H x u, u⟫ := by
  have hdemand : 0 ≤ ∑ p : ι × α, (∑ l, N.revenue l * N.kernel l p.2 *
      (2 * N.rival x p.1 l / N.denom x l ^ 3 * N.appeal u p.1 l)) * u p := by
    rw [Fintype.sum_prod_type]
    refine Finset.sum_nonneg fun i _ ↦ ?_
    have hswap : ∑ a, (∑ l, N.revenue l * N.kernel l a *
          (2 * N.rival x i l / N.denom x l ^ 3 * N.appeal u i l)) * u (i, a) =
        ∑ l, N.revenue l * (2 * N.rival x i l / N.denom x l ^ 3) * N.appeal u i l ^ 2 := by
      simp only [Finset.sum_mul]
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun l _ ↦ ?_
      have hA : N.appeal u i l = ∑ a, N.kernel l a * u (i, a) := rfl
      calc ∑ a, N.revenue l * N.kernel l a *
              (2 * N.rival x i l / N.denom x l ^ 3 * N.appeal u i l) * u (i, a)
          = N.revenue l * (2 * N.rival x i l / N.denom x l ^ 3) * N.appeal u i l *
              ∑ a, N.kernel l a * u (i, a) := by
            rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun a _ ↦ by ring
        _ = _ := by rw [← hA]; ring
    rw [hswap]
    refine Finset.sum_nonneg fun l _ ↦ ?_
    have hD := N.denom_pos hx l
    have hB := N.rival_nonneg hx i l
    exact mul_nonneg (mul_nonneg (N.revenue_nonneg l) (by positivity)) (sq_nonneg _)
  have hsplit : ⟪ownJac N H x u, u⟫ =
      ∑ p : ι × α, (∑ b, H p.1 p.2 b * u (p.1, b)) * u p +
        ∑ p : ι × α, (∑ l, N.revenue l * N.kernel l p.2 *
          (2 * N.rival x p.1 l / N.denom x l ^ 3 * N.appeal u p.1 l)) * u p := by
    rw [inner_eq_sum, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun p _ ↦ by
      obtain ⟨i, a⟩ := p
      rw [ownJac_apply]; ring
  rw [hsplit]
  linarith [cost_quadratic_ge H hH u]

/-! ### The response operator -/

/-- The opportunity impulse `𝒢 δ`: the exact decrease in the negative marginal-profit field when
opportunity values move by `δ` (`pseudoGradient_withRevenue_sub`). -/
noncomputable def opportunityImpulse (δ : κ → ℝ) : Profile ι α :=
  WithLp.toLp 2 fun p ↦ ∑ l, δ l * N.opportunityLoading x p.1 p.2 l

/-- The field moves by exactly minus the impulse when opportunity values change. -/
theorem pseudoGradient_withRevenue_sub_eq_neg_impulse (r : κ → ℝ) (hr : ∀ l, 0 ≤ r l)
    (c : ι → α → ℝ) :
    pseudoGradient (N.withRevenue r hr) c H x - pseudoGradient N c H x =
      -opportunityImpulse N x (fun l ↦ r l - N.revenue l) := by
  ext ⟨i, a⟩
  rw [pseudoGradient_withRevenue_sub]
  rfl

/-- **The linear response operator.** At a nonnegative profile and for any set of active
coordinates, the linearised equilibrium conditions `P (J δq) = P (𝒢 δr)`, `P δq = δq` have a
unique solution, and `‖δq‖ ≤ ‖P (𝒢 δr)‖ / mu`. -/
@[blueprint "prop:response"
  (statement := /--
    The equilibrium conditions linearized on any active set have a unique solution. Assumptions:
    A nonnegative profile, uniformly positive cost curvature, and any set of active coordinates.
  -/)]
theorem linearResponse_existsUnique {mu : ℝ} (hmu : 0 < mu)
    (hH : ∀ i, ∀ u : α → ℝ, mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (hx : ∀ p, 0 ≤ x p) (act : ι × α → Prop) [DecidablePred act] (δ : κ → ℝ) :
    ∃! u, SolvesRestricted (jac N H x) (restrictL act) (opportunityImpulse N x δ) u :=
  existsUnique_solvesRestricted (restrictL_isOrthoProjection act) hmu
    (fun v _ ↦ jac_coercive N H x hH hx v) _

/-- The response is bounded by the restricted opportunity impulse. -/
@[blueprint "eq:linear-response"
  (statement := /--
    The linearized response is bounded by the restricted opportunity impulse divided by nu.
    Assumptions: A nonnegative profile, uniformly positive cost curvature, and any set of active
    coordinates.
  -/)]
theorem linearResponse_norm_le {mu : ℝ} (hmu : 0 < mu)
    (hH : ∀ i, ∀ u : α → ℝ, mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (hx : ∀ p, 0 ≤ x p) (act : ι × α → Prop) [DecidablePred act] {δ : κ → ℝ} {u : Profile ι α}
    (hu : SolvesRestricted (jac N H x) (restrictL act) (opportunityImpulse N x δ) u) :
    ‖u‖ ≤ ‖restrictL act (opportunityImpulse N x δ)‖ / mu :=
  hu.norm_le (restrictL_isOrthoProjection act) hmu (fun v _ ↦ jac_coercive N H x hH hx v)

/-- **The direct response exists and is unique.** Holding rivals at the equilibrium field, only
the own blocks respond. -/
theorem directResponse_existsUnique {mu : ℝ} (hmu : 0 < mu)
    (hH : ∀ i, ∀ u : α → ℝ, mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (hx : ∀ p, 0 ≤ x p) (act : ι × α → Prop) [DecidablePred act] (δ : κ → ℝ) :
    ∃! u, SolvesRestricted (ownJac N H x) (restrictL act) (opportunityImpulse N x δ) u :=
  existsUnique_solvesRestricted (restrictL_isOrthoProjection act) hmu
    (fun v _ ↦ ownJac_coercive N H x hH hx v) _

/-- **Direct response versus rival feedback.** The full response minus the direct response is the
full response to the rival forcing `-C u_d` that the direct response generates. -/
@[blueprint "eq:feedback"
  (statement := /--
    Full minus direct response is the full response to the rival forcing generated by the direct
    response. Assumptions: Full and direct responses on the same active set.
  -/)]
theorem feedback_eq_response_to_rival_forcing (act : ι × α → Prop) [DecidablePred act]
    {δ : κ → ℝ} {uf ud : Profile ι α}
    (hf : SolvesRestricted (jac N H x) (restrictL act) (opportunityImpulse N x δ) uf)
    (hd : SolvesRestricted (ownJac N H x) (restrictL act) (opportunityImpulse N x δ) ud) :
    SolvesRestricted (jac N H x) (restrictL act) (-(crossJac N x ud)) (uf - ud) :=
  feedback_solvesRestricted hf hd

/-- **Convergence of repeated rival feedback.** If the cross blocks satisfy
`‖P (C v)‖ ≤ L ‖v‖` with `L < mu`, the rival-feedback iterates converge to the full response at
rate `L / mu`. -/
theorem rivalFeedback_iterate_dist_le {mu : ℝ} (hmu : 0 < mu)
    (hH : ∀ i, ∀ u : α → ℝ, mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (hx : ∀ p, 0 ≤ x p) (act : ι × α → Prop) [DecidablePred act] {L : ℝ} (hL : 0 ≤ L)
    (hC : ∀ v, restrictL act v = v → ‖restrictL act (crossJac N x v)‖ ≤ L * ‖v‖)
    {δ : κ → ℝ} {uf : Profile ι α}
    (hf : SolvesRestricted (jac N H x) (restrictL act) (opportunityImpulse N x δ) uf)
    (v : ℕ → Profile ι α) (hv0 : restrictL act (v 0) = v 0)
    (hv : ∀ h, SolvesRestricted (ownJac N H x) (restrictL act)
      (opportunityImpulse N x δ - crossJac N x (v h)) (v (h + 1))) (h : ℕ) :
    ‖v h - uf‖ ≤ (L / mu) ^ h * ‖v 0 - uf‖ :=
  jacobi_iterate_dist_le (restrictL_isOrthoProjection act) hmu
    (fun w _ ↦ ownJac_coercive N H x hH hx w) hL hC hf v hv0 hv h

end PricingPerspective.Operating.RationalShare.EquilibriumResponse
