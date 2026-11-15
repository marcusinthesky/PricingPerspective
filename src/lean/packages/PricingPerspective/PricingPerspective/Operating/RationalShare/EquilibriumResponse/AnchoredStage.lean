import Architect
import PricingPerspective.Operating.RationalShare.EquilibriumResponse.AnchoredAdjustment
import PricingPerspective.Operating.RationalShare.EquilibriumResponse.OrthantComparativeStatics
import PricingPerspective.Operating.VariationalInequality.Existence

/-!
# Existence of the anchored stage equilibrium

`AnchoredAdjustment.lean` proves that anchored stage equilibria are unique and that a path of
them contracts toward the static equilibrium. It assumes each stage equilibrium exists. This
module removes that assumption for the rational-share economy.

The anchored stage map is `G y = F y + kappa (y - a)`. It is `(nu + kappa)`-strongly monotone on
the capacity orthant, and on every box it is Lipschitz with a constant that does not depend on the
box. The existing existence theorem therefore solves the variational inequality on a box. Testing
that solution against the zero profile bounds its norm by `‖G 0‖ / (nu + kappa)`. A box wider than
this bound never binds from above, so the box solution also solves the variational inequality on
the whole orthant.

The argument never uses the sign of the linear costs. It does not need the shifted cost
`c - kappa a` to be nonnegative, which is what reusing the orthant Nash theorem would require.
-/

namespace PricingPerspective.Operating.RationalShare.EquilibriumResponse

open PricingPerspective.Operating.VariationalInequality
  PricingPerspective.Operating.RationalShare.EquilibriumResponse.AnchoredAdjustment
  PricingPerspective.Operating.RationalShare.Reallocation

open scoped RealInnerProductSpace

variable {κ ι α : Type*} [Fintype κ] [Fintype ι] [Fintype α] [DecidableEq ι]

/-- A step size meeting both Zarantonello conditions exists for any positive modulus. -/
theorem exists_zarantonello_step {mu L : ℝ} (hmu : 0 < mu) :
    ∃ gamma : ℝ, 0 < gamma ∧ 1 - 2 * gamma * mu + gamma ^ 2 * L ^ 2 < 1 ∧
      0 ≤ 1 - 2 * gamma * mu + gamma ^ 2 * L ^ 2 := by
  set gamma := mu / (2 * (1 + mu ^ 2 + L ^ 2)) with hgdef
  have hden : (0 : ℝ) < 2 * (1 + mu ^ 2 + L ^ 2) := by positivity
  have hgamma : 0 < gamma := div_pos hmu hden
  have h2gm : 2 * gamma * mu < 1 := by
    have hrw : 2 * gamma * mu = mu ^ 2 / (1 + mu ^ 2 + L ^ 2) := by
      rw [hgdef]; field_simp
    rw [hrw, div_lt_one (by positivity)]
    nlinarith [sq_nonneg L]
  have hstep : gamma * L ^ 2 < 2 * mu := by
    rw [hgdef, div_mul_eq_mul_div, div_lt_iff₀ hden]
    nlinarith [sq_nonneg L, hmu, sq_nonneg mu]
  refine ⟨gamma, hgamma, ?_, ?_⟩
  · have hlt : gamma ^ 2 * L ^ 2 < 2 * gamma * mu := by
      have hfac : gamma ^ 2 * L ^ 2 = gamma * (gamma * L ^ 2) := by ring
      rw [hfac]
      calc gamma * (gamma * L ^ 2) < gamma * (2 * mu) := mul_lt_mul_of_pos_left hstep hgamma
        _ = 2 * gamma * mu := by ring
    linarith
  · nlinarith [h2gm, sq_nonneg (gamma * L)]

variable (N : Nodes κ α) (c : ι → α → ℝ) (H : ι → α → α → ℝ)

/-- The anchored stage map: the field plus the gradient of `kappa / 2 * ‖y - a‖ ^ 2`. -/
noncomputable def anchoredField (kappa : ℝ) (a y : Profile ι α) : Profile ι α :=
  pseudoGradient N c H y + kappa • (y - a)

omit [DecidableEq ι] in
theorem box_subset_capacityOrthant {hi : ι × α → ℝ} :
    box (fun _ ↦ (0 : ℝ)) hi ⊆ (capacityOrthant : Set (Profile ι α)) :=
  fun _ hx p ↦ (hx p).1

omit [DecidableEq ι] in
/-- The anchor adds `kappa` to the monotonicity modulus. -/
theorem stronglyMonotoneOn_anchoredField {mu kappa : ℝ}
    (hH : ∀ i, ∀ u : α → ℝ, mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (a : Profile ι α) :
    StronglyMonotoneOn (anchoredField N c H kappa a) capacityOrthant (mu + kappa) := by
  intro x hx y hy
  have h := stronglyMonotoneOn_pseudoGradient_capacityOrthant N c hH x hx y hy
  have hsplit : anchoredField N c H kappa a x - anchoredField N c H kappa a y =
      (pseudoGradient N c H x - pseudoGradient N c H y) + kappa • (x - y) := by
    simp only [anchoredField, smul_sub]; abel
  rw [hsplit, inner_add_left, real_inner_smul_left, real_inner_self_eq_norm_sq]
  linarith

/-- On any box in the orthant, the anchor adds `kappa` to the Lipschitz constant. -/
theorem lipschitzOn_anchoredField {kappa : ℝ} (hk : 0 ≤ kappa) (a : Profile ι α)
    {hi : ι × α → ℝ} :
    ∀ x ∈ box (fun _ ↦ (0 : ℝ)) hi, ∀ y ∈ box (fun _ ↦ (0 : ℝ)) hi,
      ‖anchoredField N c H kappa a x - anchoredField N c H kappa a y‖ ≤
        (lipConst N H + kappa) * ‖x - y‖ := by
  intro x hx y hy
  have h := lipschitzOn_pseudoGradient N c H (lo := fun _ ↦ (0 : ℝ)) (hi := hi)
    (fun _ ↦ le_refl 0) x hx y hy
  have hsplit : anchoredField N c H kappa a x - anchoredField N c H kappa a y =
      (pseudoGradient N c H x - pseudoGradient N c H y) + kappa • (x - y) := by
    simp only [anchoredField, smul_sub]; abel
  rw [hsplit, add_mul]
  refine (norm_add_le _ _).trans (add_le_add h ?_)
  rw [norm_smul, Real.norm_of_nonneg hk]

/-- **Existence of the anchored stage equilibrium.** For every anchor, the rational-share stage
game with adjustment cost `kappa / 2 * ‖y - a‖ ^ 2` and contemporaneous rivals has an equilibrium
on the capacity orthant. No sign condition on the linear costs is needed. -/
theorem exists_proxSolvesVI_capacityOrthant {mu kappa : ℝ} (hmu : 0 < mu) (hk : 0 ≤ kappa)
    (hH : ∀ i, ∀ u : α → ℝ, mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (a : Profile ι α) :
    ∃ y, ProxSolvesVI (pseudoGradient N c H) capacityOrthant kappa a y := by
  set G := anchoredField N c H kappa a with hG
  set m := mu + kappa with hm
  have hmpos : 0 < m := by rw [hm]; linarith
  set R : ℝ := ‖G 0‖ / m + 1 with hR
  have hRpos : 0 < R := by rw [hR]; positivity
  set S : Set (Profile ι α) := box (fun _ ↦ (0 : ℝ)) (fun _ ↦ R) with hS
  have hne : S.Nonempty := ⟨0, fun _ ↦ ⟨le_refl _, by simp [hRpos.le]⟩⟩
  have hmono : StronglyMonotoneOn G S m := fun x hx y hy ↦
    stronglyMonotoneOn_anchoredField N c H hH a x (box_subset_capacityOrthant hx) y
      (box_subset_capacityOrthant hy)
  obtain ⟨gamma, hgamma, hq, hq0⟩ :=
    exists_zarantonello_step (mu := m) (L := lipConst N H + kappa) hmpos
  obtain ⟨y, hyS, hyvi⟩ := exists_solvesVI hne (isComplete_box _ _) (convex_box _ _) hmono
    (lipschitzOn_anchoredField N c H hk a) hgamma hq hq0
  -- Testing against zero bounds the solution strictly inside the upper faces.
  have h0S : (0 : Profile ι α) ∈ S := fun _ ↦ ⟨le_refl _, by simp [hRpos.le]⟩
  have hnorm : ‖y‖ < R := by
    have htest := hyvi 0 h0S
    have hsm := hmono y hyS 0 h0S
    rw [sub_zero, inner_sub_left] at hsm
    rw [zero_sub, inner_neg_right] at htest
    have hcs : -⟪G 0, y⟫ ≤ ‖G 0‖ * ‖y‖ := by
      have := real_inner_le_norm (-(G 0)) y
      rwa [inner_neg_left, norm_neg] at this
    have hle : m * ‖y‖ ^ 2 ≤ ‖G 0‖ * ‖y‖ := by linarith
    have hy_le : ‖y‖ ≤ ‖G 0‖ / m := by
      rw [le_div_iff₀ hmpos]
      rcases eq_or_lt_of_le (norm_nonneg y) with h0 | hpos
      · rw [← h0]; simp [norm_nonneg]
      · nlinarith
    rw [hR]; linarith
  refine ⟨y, box_subset_capacityOrthant hyS, fun z hz ↦ ?_⟩
  -- Step toward `z` far enough to stay inside the box.
  set δ := R - ‖y‖ with hδ
  have hδpos : 0 < δ := by rw [hδ]; linarith
  set t : ℝ := min 1 (δ / (1 + ‖z - y‖)) with ht
  have htpos : 0 < t := lt_min one_pos (by positivity)
  have ht1 : t ≤ 1 := min_le_left _ _
  have htδ : t * ‖z - y‖ < δ := by
    have hle : t ≤ δ / (1 + ‖z - y‖) := min_le_right _ _
    have hpos : 0 < 1 + ‖z - y‖ := by positivity
    calc t * ‖z - y‖ ≤ δ / (1 + ‖z - y‖) * ‖z - y‖ :=
          mul_le_mul_of_nonneg_right hle (norm_nonneg _)
      _ < δ / (1 + ‖z - y‖) * (1 + ‖z - y‖) :=
          mul_lt_mul_of_pos_left (by linarith) (by positivity)
      _ = δ := by field_simp
  have hwS : y + t • (z - y) ∈ S := by
    intro p
    have hyp : 0 ≤ y p := (hyS p).1
    have hzp : 0 ≤ z p := hz p
    have hcoord : (y + t • (z - y)) p = (1 - t) * y p + t * z p := by
      simp only [PiLp.add_apply, PiLp.smul_apply, PiLp.sub_apply, smul_eq_mul]; ring
    refine ⟨by rw [hcoord]; nlinarith, ?_⟩
    have hyb : y p ≤ ‖y‖ := (le_abs_self _).trans (by
      simpa [Real.norm_eq_abs] using PiLp.norm_apply_le y p)
    have hdb : t * (z p - y p) ≤ t * ‖z - y‖ := by
      refine mul_le_mul_of_nonneg_left ?_ htpos.le
      have := PiLp.norm_apply_le (z - y) p
      rw [PiLp.sub_apply, Real.norm_eq_abs] at this
      exact (le_abs_self _).trans this
    have hw : (y + t • (z - y)) p = y p + t * (z p - y p) := by
      simp only [PiLp.add_apply, PiLp.smul_apply, PiLp.sub_apply, smul_eq_mul]
    show (y + t • (z - y)) p ≤ R
    rw [hw]; linarith
  have hvi := hyvi _ hwS
  rw [add_sub_cancel_left, real_inner_smul_right] at hvi
  have : 0 ≤ ⟪G y, z - y⟫ := nonneg_of_mul_nonneg_right (by linarith) htpos
  simpa [hG, anchoredField] using this

/-- **The anchored stage equilibrium exists and is unique.** -/
@[blueprint "app:anchored-stage"
  (statement := /--
    Every anchored stage game of the rational-share economy has a unique equilibrium on the
    capacity orthant. Assumptions: Uniformly positive cost curvature, a nonnegative adjustment
    weight, contemporaneous rivals, and any anchor.
  -/)]
theorem existsUnique_proxSolvesVI_capacityOrthant {mu kappa : ℝ} (hmu : 0 < mu)
    (hk : 0 ≤ kappa)
    (hH : ∀ i, ∀ u : α → ℝ, mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (a : Profile ι α) :
    ∃! y, ProxSolvesVI (pseudoGradient N c H) capacityOrthant kappa a y := by
  obtain ⟨y, hy⟩ := exists_proxSolvesVI_capacityOrthant N c H hmu hk hH a
  exact ⟨y, hy, fun z hz ↦ AnchoredAdjustment.jointProx_unique hk (by linarith)
    (stronglyMonotoneOn_pseudoGradient_capacityOrthant N c hH) hz hy⟩

/-- **The anchored path exists and converges to the static equilibrium.** From any initial
profile, successive anchored stage equilibria form a path whose distance to the unique Nash
equilibrium falls at least geometrically, at rate `kappa / (kappa + mu)`. -/
theorem exists_anchoredPath_dist_le {mu kappa : ℝ} (hsymm : ∀ i a b, H i a b = H i b a)
    (hmu : 0 < mu) (hk : 0 ≤ kappa)
    (hH : ∀ i, ∀ u : α → ℝ, mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (hc : ∀ i a, 0 ≤ c i a) {qstar : Profile ι α} (hstar : IsNashOrthant N c H qstar)
    (q0 : Profile ι α) :
    ∃ q : ℕ → Profile ι α, q 0 = q0 ∧
      (∀ t, ProxSolvesVI (pseudoGradient N c H) capacityOrthant kappa (q t) (q (t + 1))) ∧
      ∀ t, ‖q t - qstar‖ ≤ (kappa / (kappa + mu)) ^ t * ‖q0 - qstar‖ := by
  choose T hT using fun a : Profile ι α ↦ exists_proxSolvesVI_capacityOrthant N c H hmu hk hH a
  let q : ℕ → Profile ι α := fun t ↦ Nat.rec q0 (fun _ x ↦ T x) t
  have hstep : ∀ t, ProxSolvesVI (pseudoGradient N c H) capacityOrthant kappa (q t) (q (t + 1)) :=
    fun t ↦ hT (q t)
  refine ⟨q, rfl, hstep, fun t ↦ ?_⟩
  have h := AnchoredAdjustment.jointProx_iterate_dist_le hk (by linarith)
    (stronglyMonotoneOn_pseudoGradient_capacityOrthant N c hH)
    (solvesVI_capacityOrthant_of_isNashOrthant N c hsymm hmu hH hc hstar) q hstep t
  simpa [q] using h

end PricingPerspective.Operating.RationalShare.EquilibriumResponse
