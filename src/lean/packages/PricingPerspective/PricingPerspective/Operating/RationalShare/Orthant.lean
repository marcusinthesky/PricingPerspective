import PricingPerspective.Operating.RationalShare.Monotonicity

set_option linter.style.longLine false

namespace PricingPerspective.Operating.RationalShare

open PricingPerspective.Operating.VariationalInequality

open Finset
open scoped RealInnerProductSpace

variable {κ ι α : Type*} [Fintype κ] [Fintype ι] [Fintype α] [DecidableEq ι]

/-! ### The orthant the simulation optimizes over

`box` is a finite rectangle; the Python solver passes `bounds=[(0.0, None)]`, which is the
nonnegative orthant with no cap. A theorem about the rectangle is not a theorem about the
orthant unless the cap is shown to be slack, so both directions are proved here and the
equivalence is what downstream results use.

The bridge is a bound on unpenalized best responses: revenue is at most the total available, own
curvature charges the square of the block, and the zero block earns exactly nothing. So no
equilibrium and no best response can be large, and a cap beyond that radius removes nothing. The
cap condition is written as `2 * totalRevenue <= eta * hi p ^ 2` rather than with a square root,
which is the same condition and avoids carrying `Real.sqrt` through every proof.

This bounds **equilibria and best responses**. It says nothing about whether a proximal
adjustment trajectory stays in the same rectangle; that is a question about dynamics and is not
answered anywhere in this module. -/

/-- Total revenue available across the demand nodes: the most any one firm could ever earn. -/
def totalRevenue (N : Nodes κ α) : ℝ := ∑ l, N.revenue l

omit [DecidableEq ι] in
theorem totalRevenue_nonneg (N : Nodes κ α) : 0 ≤ totalRevenue N :=
  Finset.sum_nonneg fun l _ => N.revenue_nonneg l

/-- Nash equilibrium on the uncapped nonnegative orthant -- the game the solver actually runs. -/
def IsNashOrthant (N : Nodes κ α) (c : ι → α → ℝ) (H : ι → α → α → ℝ) (x : Profile ι α) :
    Prop :=
  (∀ p, 0 ≤ x p) ∧
    ∀ i : ι, ∀ q : α → ℝ, (∀ a, 0 ≤ q a) →
      payoff N c H i (replaceFirm x i q) ≤ payoff N c H i x

omit [DecidableEq ι] in
/-- **Revenue is bounded by what exists.** Each node's share is at most one, because a firm's
appeal is part of its own denominator. -/
theorem revenue_le_totalRevenue (N : Nodes κ α) {z : Profile ι α} (hz : ∀ p, 0 ≤ z p) (i : ι) :
    (∑ l, N.revenue l * N.appeal z i l / N.denom z l) ≤ totalRevenue N := by
  refine Finset.sum_le_sum fun l _ => ?_
  have hD : 0 < N.denom z l := N.denom_pos hz l
  have hle : N.appeal z i l ≤ N.denom z l := by
    have := N.rival_nonneg hz i l
    rw [Nodes.rival] at this
    linarith
  have hshare : N.appeal z i l / N.denom z l ≤ 1 := (div_le_one hD).2 hle
  have hnn : 0 ≤ N.appeal z i l / N.denom z l :=
    div_nonneg (N.appeal_nonneg hz i l) hD.le
  calc N.revenue l * N.appeal z i l / N.denom z l
      = N.revenue l * (N.appeal z i l / N.denom z l) := by ring
    _ ≤ N.revenue l * 1 := by
        exact mul_le_mul_of_nonneg_left hshare (N.revenue_nonneg l)
    _ = N.revenue l := mul_one _

omit [DecidableEq ι] in
/-- **The best-response radius.** Revenue is capped and own curvature is charged, so a large
block cannot pay for itself. -/
theorem payoff_le_totalRevenue_sub (N : Nodes κ α) (c : ι → α → ℝ) {H : ι → α → α → ℝ}
    {eta : ℝ} (hc : ∀ i a, 0 ≤ c i a)
    (hH : ∀ i, ∀ u : α → ℝ, eta * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    {z : Profile ι α} (hz : ∀ p, 0 ≤ z p) (i : ι) :
    payoff N c H i z ≤ totalRevenue N - eta / 2 * ∑ a, z (i, a) ^ 2 := by
  have hrev := revenue_le_totalRevenue N hz i
  have hlin : 0 ≤ ∑ a, c i a * z (i, a) :=
    Finset.sum_nonneg fun a _ => mul_nonneg (hc i a) (hz (i, a))
  have hquad := hH i fun a => z (i, a)
  rw [payoff]
  linarith

/-- **The zero block earns exactly nothing.** Its appeal vanishes at every node, so every share
numerator does, and both cost terms vanish with it. -/
theorem payoff_replaceFirm_zero (N : Nodes κ α) (c : ι → α → ℝ) (H : ι → α → α → ℝ) (i : ι)
    (x : Profile ι α) :
    payoff N c H i (replaceFirm x i (fun _ => 0)) = 0 := by
  have happ : ∀ l, N.appeal (replaceFirm x i fun _ => 0) i l = 0 := by
    intro l
    simp [Nodes.appeal]
  rw [payoff]
  have hrev : (∑ l, N.revenue l * N.appeal (replaceFirm x i fun _ => 0) i l
      / N.denom (replaceFirm x i fun _ => 0) l) = 0 :=
    Finset.sum_eq_zero fun l _ => by rw [happ l]; ring
  have hlin : (∑ a, c i a * replaceFirm x i (fun _ => 0) (i, a)) = 0 :=
    Finset.sum_eq_zero fun a _ => by simp
  have hquad : (∑ a, ∑ b, H i a b * replaceFirm x i (fun _ => 0) (i, a)
      * replaceFirm x i (fun _ => 0) (i, b)) = 0 :=
    Finset.sum_eq_zero fun a _ => Finset.sum_eq_zero fun b _ => by simp
  rw [hrev, hlin, hquad]
  ring

omit [DecidableEq ι] in
/-- A profile's coordinate is bounded by the square root of its own block energy, without ever
naming a square root. -/
theorem sq_le_block_sum (z : Profile ι α) (i : ι) (a : α) :
    z (i, a) ^ 2 ≤ ∑ b, z (i, b) ^ 2 :=
  Finset.single_le_sum (f := fun b => z (i, b) ^ 2) (fun _ _ => sq_nonneg _)
    (Finset.mem_univ a)

/-- **Every orthant equilibrium lies in the cap.** This is the direction that makes uniqueness on
the rectangle a statement about the orthant rather than only about the rectangle. -/
theorem mem_box_of_isNashOrthant (N : Nodes κ α) (c : ι → α → ℝ) {H : ι → α → α → ℝ}
    {hi : ι × α → ℝ} {eta : ℝ} (heta : 0 < eta) (hc : ∀ i a, 0 ≤ c i a)
    (hH : ∀ i, ∀ u : α → ℝ, eta * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (hhi : ∀ p, 0 ≤ hi p) (hcap : ∀ p, 2 * totalRevenue N ≤ eta * hi p ^ 2)
    {x : Profile ι α} (hx : IsNashOrthant N c H x) :
    x ∈ box (fun _ => 0) hi := by
  rintro ⟨i, a⟩
  refine ⟨hx.1 (i, a), ?_⟩
  have hzero := hx.2 i (fun _ => 0) fun _ => le_refl 0
  rw [payoff_replaceFirm_zero] at hzero
  have hupper := payoff_le_totalRevenue_sub N c hc hH hx.1 i
  have hblock : eta * ∑ b, x (i, b) ^ 2 ≤ 2 * totalRevenue N := by linarith
  have hsingle : eta * x (i, a) ^ 2 ≤ eta * ∑ b, x (i, b) ^ 2 :=
    mul_le_mul_of_nonneg_left (sq_le_block_sum x i a) heta.le
  have hsq : x (i, a) ^ 2 ≤ hi (i, a) ^ 2 := by
    have := hcap (i, a)
    nlinarith
  nlinarith [hx.1 (i, a), hhi (i, a)]

/-- **The cap removes no deviation.** A block outside it is dominated by the zero block, which is
always available, so a rectangle equilibrium is an orthant equilibrium. -/
theorem isNashOrthant_of_isNash (N : Nodes κ α) (c : ι → α → ℝ) {H : ι → α → α → ℝ}
    {hi : ι × α → ℝ} {eta : ℝ} (heta : 0 < eta) (hc : ∀ i a, 0 ≤ c i a)
    (hH : ∀ i, ∀ u : α → ℝ, eta * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (hhi : ∀ p, 0 ≤ hi p) (hcap : ∀ p, 2 * totalRevenue N ≤ eta * hi p ^ 2)
    {x : Profile ι α} (hx : IsNash N c H (fun _ => 0) hi x) :
    IsNashOrthant N c H x := by
  have hxnn : ∀ p, 0 ≤ x p := nonneg_of_mem_box (fun _ => le_refl 0) hx.1
  refine ⟨hxnn, fun i q hq => ?_⟩
  by_cases hsmall : ∀ a, q a ≤ hi (i, a)
  · exact hx.2 i q fun a => ⟨hq a, hsmall a⟩
  · push_neg at hsmall
    obtain ⟨a, ha⟩ := hsmall
    have hfeas : ∀ p, 0 ≤ replaceFirm x i q p := by
      rintro ⟨j, b⟩
      by_cases hj : j = i
      · subst hj; simpa using hq b
      · simpa [replaceFirm_of_ne x hj] using hxnn (j, b)
    have hupper := payoff_le_totalRevenue_sub N c hc hH hfeas i
    have hq_a : replaceFirm x i q (i, a) = q a := by simp
    have hsingle : q a ^ 2 ≤ ∑ b, replaceFirm x i q (i, b) ^ 2 := by
      have := sq_le_block_sum (replaceFirm x i q) i a
      rwa [hq_a] at this
    have hbig : hi (i, a) ^ 2 < q a ^ 2 := by
      have h0 := hhi (i, a)
      nlinarith
    have hstrict : 2 * totalRevenue N < eta * ∑ b, replaceFirm x i q (i, b) ^ 2 := by
      have hc1 := hcap (i, a)
      nlinarith
    have hneg : payoff N c H i (replaceFirm x i q) < 0 := by linarith
    have hzero := hx.2 i (fun _ => 0) fun b => ⟨le_refl 0, hhi (i, b)⟩
    rw [payoff_replaceFirm_zero] at hzero
    linarith

/-- **The two equilibrium notions coincide under a slack cap.** Existence transfers one way,
uniqueness the other, so a theorem proved on the rectangle is a theorem about the orthant. -/
theorem isNash_iff_isNashOrthant (N : Nodes κ α) (c : ι → α → ℝ) {H : ι → α → α → ℝ}
    {hi : ι × α → ℝ} {eta : ℝ} (heta : 0 < eta) (hc : ∀ i a, 0 ≤ c i a)
    (hH : ∀ i, ∀ u : α → ℝ, eta * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (hhi : ∀ p, 0 ≤ hi p) (hcap : ∀ p, 2 * totalRevenue N ≤ eta * hi p ^ 2)
    {x : Profile ι α} :
    IsNash N c H (fun _ => 0) hi x ↔ IsNashOrthant N c H x := by
  refine ⟨isNashOrthant_of_isNash N c heta hc hH hhi hcap, fun hx => ?_⟩
  refine ⟨mem_box_of_isNashOrthant N c heta hc hH hhi hcap hx, fun i q hq => ?_⟩
  exact hx.2 i q fun a => (hq a).1

end PricingPerspective.Operating.RationalShare
