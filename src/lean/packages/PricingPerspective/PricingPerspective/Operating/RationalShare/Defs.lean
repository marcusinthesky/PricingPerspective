import PricingPerspective.Operating.VariationalInequality.Existence
import Mathlib.Analysis.InnerProductSpace.PiL2

set_option linter.style.longLine false

namespace PricingPerspective.Operating.RationalShare

open PricingPerspective.Operating.VariationalInequality

open Finset
open scoped RealInnerProductSpace

/-! ### The scalar identity

Everything in this module rests on one piece of field algebra, isolated here so that the rest is
bookkeeping over finite sums. -/

/-- **The exact unilateral share difference.** Moving a firm's appeal at one node from `a` to
`a + s`, with the rival contribution `B` fixed and `D = B + a` the denominator before the move,
changes revenue by a linear term and a quadratic-order remainder, exactly.

This is the rational-share analogue of the completed square in `Game.payoff_replace_eq`. It is an
identity, not an approximation: there is no `O(s ^ 2)` and no Taylor hypothesis. -/
theorem share_difference {r B a s D : ℝ} (hD : D = B + a) (hDpos : D ≠ 0)
    (hshift : D + s ≠ 0) :
    r * (a + s) / (D + s) - r * a / D
      = r * B * s / D ^ 2 - r * B * s ^ 2 / (D ^ 2 * (D + s)) := by
  subst hD
  field_simp
  ring

/-- The remainder is nonnegative whenever the rival contribution and the revenue weight are, and
the post-deviation denominator is positive. That is what makes it play the curvature term's role
rather than fighting it. -/
theorem remainder_nonneg {r B s D : ℝ} (hr : 0 ≤ r) (hB : 0 ≤ B) (hD : 0 < D)
    (hshift : 0 < D + s) :
    0 ≤ r * B * s ^ 2 / (D ^ 2 * (D + s)) :=
  div_nonneg (by positivity) (by positivity)

/-! ### Demand primitives, profiles, and firm blocks -/

variable {κ ι α : Type*} [Fintype κ] [Fintype ι] [Fintype α] [DecidableEq ι]

/-- The finite demand side: a revenue weight, a nonnegative appeal kernel, and a strictly
positive outside option at every node.

The outside option is what keeps every denominator away from zero without any assumption on the
capacities, which is why no interiority condition appears anywhere below. -/
structure Nodes (κ α : Type*) [Fintype κ] [Fintype α] where
  /-- Revenue available at the node. -/
  revenue : κ → ℝ
  /-- How one unit of activity `a` converts into appeal at the node. -/
  kernel : κ → α → ℝ
  /-- The node's outside option, strictly positive. -/
  outside : κ → ℝ
  revenue_nonneg : ∀ l, 0 ≤ revenue l
  kernel_nonneg : ∀ l a, 0 ≤ kernel l a
  outside_pos : ∀ l, 0 < outside l

/-- A capacity profile: one real action per firm **and activity**. This is the difference from
the quadratic game, where each firm had a single scalar. -/
abbrev Profile (ι α : Type*) [Fintype ι] [Fintype α] := EuclideanSpace ℝ (ι × α)

omit [DecidableEq ι] in
theorem inner_eq_sum (x y : Profile ι α) : ⟪x, y⟫ = ∑ p, x p * y p := by
  simp [PiLp.inner_apply, mul_comm]

namespace Nodes

variable (N : Nodes κ α)

/-- Firm `i`'s appeal at node `l`: a nonnegative linear image of its own block. -/
def appeal (x : Profile ι α) (i : ι) (l : κ) : ℝ := ∑ a, N.kernel l a * x (i, a)

/-- The node's share denominator. -/
def denom (x : Profile ι α) (l : κ) : ℝ := N.outside l + ∑ i, N.appeal x i l

/-- Everything in the denominator that firm `i` does not control. -/
def rival (x : Profile ι α) (i : ι) (l : κ) : ℝ := N.denom x l - N.appeal x i l

omit [DecidableEq ι] in
theorem denom_eq_rival_add (x : Profile ι α) (i : ι) (l : κ) :
    N.denom x l = N.rival x i l + N.appeal x i l := by
  rw [rival]; ring

end Nodes

/-- The nonnegative capacity orthant, capped firmwise. Feasibility here does one job: it keeps
every appeal nonnegative, so that every denominator stays at or above the outside option. -/
def box (lo hi : ι × α → ℝ) : Set (Profile ι α) := {x | ∀ p, lo p ≤ x p ∧ x p ≤ hi p}

omit [DecidableEq ι] in
theorem mem_box {lo hi : ι × α → ℝ} {x : Profile ι α} :
    x ∈ box lo hi ↔ ∀ p, lo p ≤ x p ∧ x p ≤ hi p := Iff.rfl

omit [DecidableEq ι] in
theorem convex_box (lo hi : ι × α → ℝ) : Convex ℝ (box lo hi) := by
  intro x hx y hy s t hs ht hst p
  have hxp := hx p
  have hyp := hy p
  refine ⟨?_, ?_⟩
  · have h1 := mul_le_mul_of_nonneg_left hxp.1 hs
    have h2 := mul_le_mul_of_nonneg_left hyp.1 ht
    calc lo p = s * lo p + t * lo p := by rw [← add_mul, hst, one_mul]
      _ ≤ s * x p + t * y p := by linarith
      _ = (s • x + t • y) p := by simp
  · have h1 := mul_le_mul_of_nonneg_left hxp.2 hs
    have h2 := mul_le_mul_of_nonneg_left hyp.2 ht
    calc (s • x + t • y) p = s * x p + t * y p := by simp
      _ ≤ s * hi p + t * hi p := by linarith
      _ = hi p := by rw [← add_mul, hst, one_mul]

omit [DecidableEq ι] in
theorem isClosed_box (lo hi : ι × α → ℝ) : IsClosed (box lo hi) := by
  have : box lo hi
      = ⋂ p, {x : Profile ι α | lo p ≤ x p} ∩ {x : Profile ι α | x p ≤ hi p} := by
    ext x
    simp [box, Set.mem_iInter, forall_and]
  rw [this]
  refine isClosed_iInter fun p => IsClosed.inter ?_ ?_
  · exact isClosed_le continuous_const (EuclideanSpace.proj p).continuous
  · exact isClosed_le (EuclideanSpace.proj p).continuous continuous_const

omit [DecidableEq ι] in
theorem isComplete_box (lo hi : ι × α → ℝ) : IsComplete (box lo hi) :=
  (isClosed_box lo hi).isComplete

/-- Replace firm `i`'s whole block, leaving every rival coordinate alone. -/
def replaceFirm (x : Profile ι α) (i : ι) (q : α → ℝ) : Profile ι α :=
  WithLp.toLp 2 fun p => if p.1 = i then q p.2 else x p

@[simp] theorem replaceFirm_self (x : Profile ι α) (i : ι) (q : α → ℝ) (a : α) :
    replaceFirm x i q (i, a) = q a := by simp [replaceFirm]

theorem replaceFirm_of_ne (x : Profile ι α) {i j : ι} (h : j ≠ i) (q : α → ℝ) (a : α) :
    replaceFirm x i q (j, a) = x (j, a) := by simp [replaceFirm, h]

theorem replaceFirm_mem_box {lo hi : ι × α → ℝ} {x : Profile ι α} (hx : x ∈ box lo hi) {i : ι}
    {q : α → ℝ} (hq : ∀ a, lo (i, a) ≤ q a ∧ q a ≤ hi (i, a)) :
    replaceFirm x i q ∈ box lo hi := by
  rintro ⟨j, a⟩
  by_cases hj : j = i
  · subst hj; simpa using hq a
  · simpa [replaceFirm_of_ne x hj] using hx (j, a)

/-! ### What a unilateral deviation moves, and what it leaves alone

The three lemmas here are the whole reason the scalar identity is enough: replacing one firm's
block shifts each node's denominator by that firm's own appeal change and leaves the rival
contribution exactly fixed, so every node reduces to the two-variable algebra above. -/

namespace Nodes

variable (N : Nodes κ α)

/-- The appeal a standalone block would generate at a node. -/
def blockAppeal (q : α → ℝ) (l : κ) : ℝ := ∑ a, N.kernel l a * q a

omit [DecidableEq ι] in
theorem appeal_eq_blockAppeal (x : Profile ι α) (i : ι) (l : κ) :
    N.appeal x i l = N.blockAppeal (fun a => x (i, a)) l := rfl

theorem appeal_replaceFirm_self (x : Profile ι α) (i : ι) (q : α → ℝ) (l : κ) :
    N.appeal (replaceFirm x i q) i l = N.blockAppeal q l :=
  Finset.sum_congr rfl fun a _ => by rw [replaceFirm_self]

theorem appeal_replaceFirm_of_ne (x : Profile ι α) {i j : ι} (h : j ≠ i) (q : α → ℝ) (l : κ) :
    N.appeal (replaceFirm x i q) j l = N.appeal x j l :=
  Finset.sum_congr rfl fun a _ => by rw [replaceFirm_of_ne x h]

/-- **The rival contribution is untouched.** This is the statement that makes the deviation a
one-variable problem at each node. -/
theorem rival_replaceFirm (x : Profile ι α) (i : ι) (q : α → ℝ) (l : κ) :
    N.rival (replaceFirm x i q) i l = N.rival x i l := by
  have hsplit : ∀ y : Profile ι α, N.rival y i l
      = N.outside l + ∑ j ∈ Finset.univ.erase i, N.appeal y j l := by
    intro y
    rw [rival, denom, ← Finset.add_sum_erase Finset.univ (fun j => N.appeal y j l)
      (Finset.mem_univ i)]
    ring
  rw [hsplit, hsplit]
  refine congrArg _ (Finset.sum_congr rfl fun j hj => ?_)
  exact N.appeal_replaceFirm_of_ne x (Finset.ne_of_mem_erase hj) q l

/-- **The denominator shifts by the firm's own appeal change, and by nothing else.** -/
theorem denom_replaceFirm (x : Profile ι α) (i : ι) (q : α → ℝ) (l : κ) :
    N.denom (replaceFirm x i q) l
      = N.denom x l + (N.blockAppeal q l - N.appeal x i l) := by
  rw [N.denom_eq_rival_add (replaceFirm x i q) i, N.rival_replaceFirm x i q l,
    N.appeal_replaceFirm_self x i q l, N.denom_eq_rival_add x i]
  ring

/-! Positivity, from the outside option alone. -/

omit [DecidableEq ι] in
theorem appeal_nonneg {x : Profile ι α} (hx : ∀ p, 0 ≤ x p) (i : ι) (l : κ) :
    0 ≤ N.appeal x i l :=
  Finset.sum_nonneg fun a _ => mul_nonneg (N.kernel_nonneg l a) (hx (i, a))

omit [DecidableEq ι] in
theorem blockAppeal_nonneg {q : α → ℝ} (hq : ∀ a, 0 ≤ q a) (l : κ) :
    0 ≤ N.blockAppeal q l :=
  Finset.sum_nonneg fun a _ => mul_nonneg (N.kernel_nonneg l a) (hq a)

omit [DecidableEq ι] in
/-- Every denominator is at least the outside option, hence strictly positive. No interiority
assumption is used, and none is available: the capacities may all be zero. -/
theorem denom_pos {x : Profile ι α} (hx : ∀ p, 0 ≤ x p) (l : κ) : 0 < N.denom x l :=
  lt_of_lt_of_le (N.outside_pos l)
    (le_add_of_nonneg_right (Finset.sum_nonneg fun i _ => N.appeal_nonneg hx i l))

omit [DecidableEq ι] in
/-- The rival contribution is itself nonnegative: it is the outside option plus the other
firms' appeals. Stated without `erase`, so that no decidability on the firm index is needed. -/
theorem rival_nonneg {x : Profile ι α} (hx : ∀ p, 0 ≤ x p) (i : ι) (l : κ) :
    0 ≤ N.rival x i l := by
  have hle : N.appeal x i l ≤ ∑ j, N.appeal x j l :=
    Finset.single_le_sum (fun j _ => N.appeal_nonneg hx j l) (Finset.mem_univ i)
  have hout := (N.outside_pos l).le
  rw [rival, denom]
  linarith

end Nodes

/-! ### The payoff and its pseudo-gradient -/

/-- Firm `i`'s payoff: share revenue over the nodes, net of an affine cost and a convex own-cost
quadratic. The revenue term is a **ratio**, which is the entire difference from `Game.lean`. -/
noncomputable def payoff (N : Nodes κ α) (c : ι → α → ℝ) (H : ι → α → α → ℝ) (i : ι)
    (x : Profile ι α) : ℝ :=
  (∑ l, N.revenue l * N.appeal x i l / N.denom x l)
    - (∑ a, c i a * x (i, a))
    - (1 / 2) * ∑ a, ∑ b, H i a b * x (i, a) * x (i, b)

/-- The negative pseudo-gradient, written in closed form. As in `Game.lean` it is **defined**
rather than differentiated, and `payoff_replaceFirm_eq` is what certifies the formula. -/
noncomputable def pseudoGradient (N : Nodes κ α) (c : ι → α → ℝ) (H : ι → α → α → ℝ)
    (x : Profile ι α) : Profile ι α :=
  WithLp.toLp 2 fun p =>
    (c p.1 p.2 + ∑ b, H p.1 p.2 b * x (p.1, b))
      - ∑ l, N.revenue l * N.rival x p.1 l / N.denom x l ^ 2 * N.kernel l p.2

omit [DecidableEq ι] in
@[simp] theorem pseudoGradient_apply (N : Nodes κ α) (c : ι → α → ℝ) (H : ι → α → α → ℝ)
    (x : Profile ι α) (i : ι) (a : α) :
    pseudoGradient N c H x (i, a)
      = (c i a + ∑ b, H i a b * x (i, b))
        - ∑ l, N.revenue l * N.rival x i l / N.denom x l ^ 2 * N.kernel l a := rfl

/-- The appeal shift a deviation produces at each node. -/
def shift (N : Nodes κ α) (x : Profile ι α) (i : ι) (q : α → ℝ) (l : κ) : ℝ :=
  N.blockAppeal q l - N.appeal x i l

omit [DecidableEq ι] in
/-- The shift is the kernel image of the coordinate deviation, which is what lets the node sums
and the activity sums be exchanged. -/
theorem shift_eq_sum (N : Nodes κ α) (x : Profile ι α) (i : ι) (q : α → ℝ) (l : κ) :
    shift N x i q l = ∑ a, N.kernel l a * (q a - x (i, a)) := by
  rw [shift, Nodes.blockAppeal, Nodes.appeal, ← Finset.sum_sub_distrib]
  exact Finset.sum_congr rfl fun a _ => by ring

/-- The nonnegative quadratic-order remainder the ratio contributes, one term per node. -/
noncomputable def remainder (N : Nodes κ α) (x : Profile ι α) (i : ι) (q : α → ℝ) : ℝ :=
  ∑ l, N.revenue l * N.rival x i l * shift N x i q l ^ 2
    / (N.denom x l ^ 2 * (N.denom x l + shift N x i q l))

omit [DecidableEq ι] in
/-- **The remainder is nonnegative on the feasible set.** Every factor is, and the denominator
stays positive because the post-deviation appeal is still nonnegative and the outside option is
strictly positive. This is the property that lets it play the curvature term's role. -/
theorem remainder_nonneg' (N : Nodes κ α) {x : Profile ι α} {q : α → ℝ} (i : ι)
    (hx : ∀ p, 0 ≤ x p) (hq : ∀ a, 0 ≤ q a) : 0 ≤ remainder N x i q := by
  refine Finset.sum_nonneg fun l _ => ?_
  have hpos : 0 < N.denom x l + shift N x i q l := by
    have : N.denom x l + shift N x i q l = N.rival x i l + N.blockAppeal q l := by
      rw [shift, N.denom_eq_rival_add x i]; ring
    rw [this]
    have h2 := N.blockAppeal_nonneg hq l
    have h3 := N.outside_pos l
    have hriv : N.outside l ≤ N.rival x i l := by
      have hle : N.appeal x i l ≤ ∑ j, N.appeal x j l :=
        Finset.single_le_sum (fun j _ => N.appeal_nonneg hx j l) (Finset.mem_univ i)
      rw [Nodes.rival, Nodes.denom]
      linarith
    linarith
  have hnum : 0 ≤ N.revenue l * N.rival x i l * shift N x i q l ^ 2 :=
    mul_nonneg (mul_nonneg (N.revenue_nonneg l) (N.rival_nonneg hx i l)) (sq_nonneg _)
  exact div_nonneg hnum (by positivity)

end PricingPerspective.Operating.RationalShare
