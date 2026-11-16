import PricingPerspective.Operating.RationalShare.Defs

set_option linter.style.longLine false

namespace PricingPerspective.Operating.RationalShare

open PricingPerspective.Operating.VariationalInequality

open Finset
open scoped RealInnerProductSpace

variable {κ ι α : Type*} [Fintype κ] [Fintype ι] [Fintype α] [DecidableEq ι]

/-! ### The exact unilateral expansion

This is the rational-share analogue of `Game.payoff_replace_eq`, and it is what every first-order
argument below runs on. -/

/-- **Revenue, expanded exactly.** The node sum of shares splits into its value before the
deviation, a term linear in the appeal shift, and the nonnegative remainder. -/
theorem revenue_replaceFirm_eq (N : Nodes κ α) {x : Profile ι α} {q : α → ℝ} (i : ι)
    (hx : ∀ p, 0 ≤ x p) (hq : ∀ a, 0 ≤ q a) :
    (∑ l, N.revenue l * N.appeal (replaceFirm x i q) i l / N.denom (replaceFirm x i q) l)
      = (∑ l, N.revenue l * N.appeal x i l / N.denom x l)
        + (∑ l, N.revenue l * N.rival x i l * shift N x i q l / N.denom x l ^ 2)
        - remainder N x i q := by
  have hpoint : ∀ l ∈ (Finset.univ : Finset κ),
      N.revenue l * N.appeal (replaceFirm x i q) i l / N.denom (replaceFirm x i q) l
        = N.revenue l * N.appeal x i l / N.denom x l
          + (N.revenue l * N.rival x i l * shift N x i q l / N.denom x l ^ 2
            - N.revenue l * N.rival x i l * shift N x i q l ^ 2
              / (N.denom x l ^ 2 * (N.denom x l + shift N x i q l))) := by
    intro l _
    have hDpos : (0 : ℝ) < N.denom x l := N.denom_pos hx l
    have hsplit : N.denom x l + shift N x i q l = N.rival x i l + N.blockAppeal q l := by
      rw [shift, N.denom_eq_rival_add x i]; ring
    have hshiftpos : (0 : ℝ) < N.denom x l + shift N x i q l := by
      have h2 := N.blockAppeal_nonneg hq l
      have hriv := N.rival_nonneg hx i l
      have hout := N.outside_pos l
      have hle : N.appeal x i l ≤ ∑ j, N.appeal x j l :=
        Finset.single_le_sum (fun j _ => N.appeal_nonneg hx j l) (Finset.mem_univ i)
      have hrivpos : (0 : ℝ) < N.rival x i l := by
        rw [Nodes.rival, Nodes.denom]; linarith
      rw [hsplit]; linarith
    have happ : N.appeal (replaceFirm x i q) i l = N.appeal x i l + shift N x i q l := by
      rw [N.appeal_replaceFirm_self x i q l, shift]; ring
    have hden : N.denom (replaceFirm x i q) l = N.denom x l + shift N x i q l := by
      rw [N.denom_replaceFirm x i q l, shift]
    rw [happ, hden]
    have := share_difference (r := N.revenue l) (B := N.rival x i l)
      (a := N.appeal x i l) (s := shift N x i q l) (D := N.denom x l)
      (N.denom_eq_rival_add x i l) (ne_of_gt hDpos) (ne_of_gt hshiftpos)
    linarith [this]
  rw [Finset.sum_congr rfl hpoint, Finset.sum_add_distrib, Finset.sum_sub_distrib, remainder]
  ring

omit [DecidableEq ι] in
/-- **Exchanging the node sum with the activity sum.** The pseudo-gradient's revenue term is a
node sum read activity by activity; the expansion's linear term is the same numbers read node by
node. Pairing either against the deviation gives the same scalar.

This is the only place the two index sets are swapped, so it is isolated rather than inlined. -/
theorem sum_kernel_swap (N : Nodes κ α) (x : Profile ι α) (i : ι) (q : α → ℝ) (w : κ → ℝ) :
    (∑ a, (∑ l, w l * N.kernel l a) * (q a - x (i, a))) = ∑ l, w l * shift N x i q l := by
  have expand : ∀ a, (∑ l, w l * N.kernel l a) * (q a - x (i, a))
      = ∑ l, w l * N.kernel l a * (q a - x (i, a)) := by
    intro a
    rw [Finset.sum_mul]
  have swap : (∑ a, ∑ l, w l * N.kernel l a * (q a - x (i, a)))
      = ∑ l, ∑ a, w l * N.kernel l a * (q a - x (i, a)) := Finset.sum_comm
  have collect : ∀ l, (∑ a, w l * N.kernel l a * (q a - x (i, a)))
      = w l * shift N x i q l := by
    intro l
    rw [shift_eq_sum, Finset.mul_sum]
    exact Finset.sum_congr rfl fun a _ => by ring
  calc (∑ a, (∑ l, w l * N.kernel l a) * (q a - x (i, a)))
      = ∑ a, ∑ l, w l * N.kernel l a * (q a - x (i, a)) :=
        Finset.sum_congr rfl fun a _ => expand a
    _ = ∑ l, ∑ a, w l * N.kernel l a * (q a - x (i, a)) := swap
    _ = ∑ l, w l * shift N x i q l := Finset.sum_congr rfl fun l _ => collect l

omit [DecidableEq ι] [Fintype ι] [Fintype κ] in
/-- **The own-cost quadratic, expanded about the current block.** Symmetry is what collapses the
two cross terms into one, and it is the only thing symmetry is used for. -/
theorem quadratic_expand {H : α → α → ℝ} (hsymm : ∀ a b, H a b = H b a) (v d : α → ℝ) :
    (∑ a, ∑ b, H a b * (v a + d a) * (v b + d b))
      = (∑ a, ∑ b, H a b * v a * v b)
        + 2 * ∑ a, d a * ∑ b, H a b * v b
        + ∑ a, ∑ b, H a b * d a * d b := by
  have hinner : ∀ a, (∑ b, H a b * (v a + d a) * (v b + d b))
      = ∑ b, (H a b * v a * v b + H a b * v a * d b
          + H a b * d a * v b + H a b * d a * d b) :=
    fun a => Finset.sum_congr rfl fun b _ => by ring
  rw [Finset.sum_congr rfl fun a _ => hinner a]
  simp only [Finset.sum_add_distrib]
  have hcross1 : (∑ a, ∑ b, H a b * d a * v b) = ∑ a, d a * ∑ b, H a b * v b :=
    Finset.sum_congr rfl fun a _ => by
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun b _ => by ring
  have hcross2 : (∑ a, ∑ b, H a b * v a * d b) = ∑ a, d a * ∑ b, H a b * v b := by
    have swap : (∑ a, ∑ b, H a b * v a * d b) = ∑ b, ∑ a, H a b * v a * d b :=
      Finset.sum_comm
    rw [swap]
    refine Finset.sum_congr rfl fun b _ => ?_
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun a _ => by rw [hsymm a b]; ring
  rw [hcross1, hcross2]
  ring

/-- **The exact unilateral expansion of the rational-share payoff.**

Replacing firm `i`'s whole block changes its payoff by a term linear in the pseudo-gradient,
minus the own-cost quadratic in the deviation, minus a remainder that is nonnegative on the
feasible set. Nothing is approximated: this is an identity between two real numbers, and no
derivative is taken anywhere in its proof.

It is the rational-share counterpart of `Game.payoff_replace_eq`, and it is the reason the
first-order arguments below are algebra rather than analysis. -/
theorem payoff_replaceFirm_eq (N : Nodes κ α) (c : ι → α → ℝ) {H : ι → α → α → ℝ}
    (hsymm : ∀ i a b, H i a b = H i b a) {x : Profile ι α} {q : α → ℝ} (i : ι)
    (hx : ∀ p, 0 ≤ x p) (hq : ∀ a, 0 ≤ q a) :
    payoff N c H i (replaceFirm x i q)
      = payoff N c H i x
        - (∑ a, pseudoGradient N c H x (i, a) * (q a - x (i, a)))
        - (1 / 2) * ∑ a, ∑ b, H i a b * (q a - x (i, a)) * (q b - x (i, b))
        - remainder N x i q := by
  have hrev := revenue_replaceFirm_eq N (x := x) (q := q) i hx hq
  have hrevlin : (∑ l, N.revenue l * N.rival x i l * shift N x i q l / N.denom x l ^ 2)
      = ∑ l, N.revenue l * N.rival x i l / N.denom x l ^ 2 * shift N x i q l :=
    Finset.sum_congr rfl fun l _ => by ring
  have hlin : (∑ a, c i a * replaceFirm x i q (i, a))
      = (∑ a, c i a * x (i, a)) + ∑ a, c i a * (q a - x (i, a)) := by
    rw [← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun a _ => by rw [replaceFirm_self]; ring
  have hblock : ∀ a b, H i a b * replaceFirm x i q (i, a) * replaceFirm x i q (i, b)
      = H i a b * (x (i, a) + (q a - x (i, a))) * (x (i, b) + (q b - x (i, b))) := by
    intro a b
    rw [replaceFirm_self, replaceFirm_self]
    ring
  have hquadeq : (∑ a, ∑ b, H i a b * replaceFirm x i q (i, a) * replaceFirm x i q (i, b))
      = ∑ a, ∑ b, H i a b * (x (i, a) + (q a - x (i, a)))
          * (x (i, b) + (q b - x (i, b))) :=
    Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun b _ => hblock a b
  have hquad := quadratic_expand (hsymm i) (fun a => x (i, a)) (fun a => q a - x (i, a))
  have hgrad : (∑ a, pseudoGradient N c H x (i, a) * (q a - x (i, a)))
      = (∑ a, c i a * (q a - x (i, a)))
        + (∑ a, (q a - x (i, a)) * ∑ b, H i a b * x (i, b))
        - ∑ l, N.revenue l * N.rival x i l / N.denom x l ^ 2 * shift N x i q l := by
    have hsplit : ∀ a, pseudoGradient N c H x (i, a) * (q a - x (i, a))
        = c i a * (q a - x (i, a))
          + (q a - x (i, a)) * (∑ b, H i a b * x (i, b))
          - (∑ l, N.revenue l * N.rival x i l / N.denom x l ^ 2 * N.kernel l a)
            * (q a - x (i, a)) := by
      intro a
      rw [pseudoGradient_apply]
      ring
    rw [Finset.sum_congr rfl fun a _ => hsplit a]
    simp only [Finset.sum_sub_distrib, Finset.sum_add_distrib]
    rw [sum_kernel_swap N x i q fun l => N.revenue l * N.rival x i l / N.denom x l ^ 2]
  rw [payoff, payoff, hrev, hrevlin, hlin, hquadeq, hquad, hgrad]
  ring

/-! ### Nash equilibrium, and the equivalence

The feasible set is a box with a nonnegative floor. Nonnegativity is not cosmetic: it is what
keeps every appeal nonnegative and so every denominator at or above the outside option, which is
the only positivity the expansion above ever needs. -/

/-- Nash equilibrium of the rational-share game: no firm gains by moving its **whole block** to
another feasible one. -/
def IsNash (N : Nodes κ α) (c : ι → α → ℝ) (H : ι → α → α → ℝ) (lo hi : ι × α → ℝ)
    (x : Profile ι α) : Prop :=
  x ∈ box lo hi ∧
    ∀ i : ι, ∀ q : α → ℝ, (∀ a, lo (i, a) ≤ q a ∧ q a ≤ hi (i, a)) →
      payoff N c H i (replaceFirm x i q) ≤ payoff N c H i x

omit [DecidableEq ι] in
/-- Membership of a nonnegatively-floored box gives the coordinate nonnegativity every
positivity argument in this module runs on. -/
theorem nonneg_of_mem_box {lo hi : ι × α → ℝ} (hlo : ∀ p, 0 ≤ lo p) {x : Profile ι α}
    (hx : x ∈ box lo hi) (p : ι × α) : 0 ≤ x p :=
  le_trans (hlo p) (hx p).1

/-- **Variational inequality implies Nash.** Testing the inequality at one block deviation leaves
that firm's coordinates; own convexity and the nonnegative remainder absorb everything else. -/
theorem isNash_of_solvesVI (N : Nodes κ α) (c : ι → α → ℝ) {H : ι → α → α → ℝ}
    {lo hi : ι × α → ℝ} {x : Profile ι α}
    (hsymm : ∀ i a b, H i a b = H i b a)
    (hpsd : ∀ i, ∀ u : α → ℝ, 0 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (hlo : ∀ p, 0 ≤ lo p)
    (h : SolvesVI (pseudoGradient N c H) (box lo hi) x) :
    IsNash N c H lo hi x := by
  have hx := nonneg_of_mem_box hlo h.1
  refine ⟨h.1, fun i q hq => ?_⟩
  have hqnn : ∀ a, 0 ≤ q a := fun a => le_trans (hlo (i, a)) (hq a).1
  have hfeas : replaceFirm x i q ∈ box lo hi := replaceFirm_mem_box h.1 hq
  have hvi := h.2 _ hfeas
  rw [inner_eq_sum] at hvi
  have hzero : ∀ p : ι × α, p.1 ≠ i →
      pseudoGradient N c H x p * (replaceFirm x i q - x) p = 0 := by
    rintro ⟨j, a⟩ hj
    have : (replaceFirm x i q - x) (j, a) = 0 := by
      simp [replaceFirm_of_ne x hj]
    rw [this, mul_zero]
  have hcollapse : (∑ p, pseudoGradient N c H x p * (replaceFirm x i q - x) p)
      = ∑ a, pseudoGradient N c H x (i, a) * (q a - x (i, a)) := by
    rw [Fintype.sum_prod_type]
    refine (Finset.sum_eq_single i (fun j _ hj => ?_) (fun hj =>
      absurd (Finset.mem_univ i) hj)).trans ?_
    · exact Finset.sum_eq_zero fun a _ => hzero (j, a) hj
    · exact Finset.sum_congr rfl fun a _ => by simp
  rw [hcollapse] at hvi
  rw [payoff_replaceFirm_eq N c hsymm i hx hqnn]
  have hquadnn := hpsd i fun a => q a - x (i, a)
  have hremnn := remainder_nonneg' N (q := q) i hx hqnn
  linarith

/-! ### The feasible chord, and a step-independent bound on the remainder

The converse direction needs the remainder to shrink at least quadratically along a shortening
step. It does, and the constant is explicit: every denominator along the chord is at least the
outside option, so replacing it by the outside option bounds the whole node sum by `s ^ 2` times
a fixed number. No continuity argument is used. -/

/-- Firm `i`'s block moved a fraction `s` of the way toward `y`. -/
def chord (x y : Profile ι α) (i : ι) (s : ℝ) : α → ℝ :=
  fun a => x (i, a) + s * (y (i, a) - x (i, a))

omit [DecidableEq ι] in
theorem chord_mem {lo hi : ι × α → ℝ} {x y : Profile ι α} (hx : x ∈ box lo hi)
    (hy : y ∈ box lo hi) {s : ℝ} (hs0 : 0 ≤ s) (hs1 : s ≤ 1) (i : ι) (a : α) :
    lo (i, a) ≤ chord x y i s a ∧ chord x y i s a ≤ hi (i, a) := by
  have hxa := hx (i, a)
  have hya := hy (i, a)
  simp only [chord]
  constructor
  · nlinarith [mul_nonneg hs0 (by linarith [hya.1] : (0 : ℝ) ≤ y (i, a) - lo (i, a)),
      mul_nonneg (by linarith : (0 : ℝ) ≤ 1 - s)
        (by linarith [hxa.1] : (0 : ℝ) ≤ x (i, a) - lo (i, a))]
  · nlinarith [mul_nonneg hs0 (by linarith [hya.2] : (0 : ℝ) ≤ hi (i, a) - y (i, a)),
      mul_nonneg (by linarith : (0 : ℝ) ≤ 1 - s)
        (by linarith [hxa.2] : (0 : ℝ) ≤ hi (i, a) - x (i, a))]

omit [DecidableEq ι] in
theorem chord_nonneg {x y : Profile ι α} (hx : ∀ p, 0 ≤ x p) (hy : ∀ p, 0 ≤ y p) {s : ℝ}
    (hs0 : 0 ≤ s) (hs1 : s ≤ 1) (i : ι) (a : α) : 0 ≤ chord x y i s a := by
  have h1 : chord x y i s a = (1 - s) * x (i, a) + s * y (i, a) := by rw [chord]; ring
  rw [h1]
  exact add_nonneg (mul_nonneg (by linarith) (hx (i, a))) (mul_nonneg hs0 (hy (i, a)))

omit [DecidableEq ι] in
/-- **The appeal shift is linear in the step.** This is what turns the remainder's node sum into
`s ^ 2` times a constant. -/
theorem shift_chord (N : Nodes κ α) (x y : Profile ι α) (i : ι) (s : ℝ) (l : κ) :
    shift N x i (chord x y i s) l
      = s * N.blockAppeal (fun a => y (i, a) - x (i, a)) l := by
  rw [shift, Nodes.blockAppeal, Nodes.blockAppeal, Nodes.appeal, ← Finset.sum_sub_distrib,
    Finset.mul_sum]
  exact Finset.sum_congr rfl fun a _ => by rw [chord]; ring

/-- The step-independent constant. Replacing each post-deviation denominator by the outside
option it dominates gives a bound that no longer mentions the step. -/
noncomputable def remainderBound (N : Nodes κ α) (x : Profile ι α) (i : ι) (d : α → ℝ) : ℝ :=
  ∑ l, N.revenue l * N.rival x i l * N.blockAppeal d l ^ 2
    / (N.denom x l ^ 2 * N.outside l)

omit [DecidableEq ι] in
theorem remainderBound_nonneg (N : Nodes κ α) {x : Profile ι α} (hx : ∀ p, 0 ≤ x p) (i : ι)
    (d : α → ℝ) : 0 ≤ remainderBound N x i d := by
  refine Finset.sum_nonneg fun l _ => ?_
  have hnum : 0 ≤ N.revenue l * N.rival x i l * N.blockAppeal d l ^ 2 :=
    mul_nonneg (mul_nonneg (N.revenue_nonneg l) (N.rival_nonneg hx i l)) (sq_nonneg _)
  have hden : 0 < N.denom x l ^ 2 * N.outside l := by
    have := N.denom_pos hx l
    have := N.outside_pos l
    positivity
  exact div_nonneg hnum hden.le

omit [DecidableEq ι] in
/-- **The remainder is quadratic in the step, with an explicit constant.** -/
theorem remainder_le_sq (N : Nodes κ α) {x y : Profile ι α} (hx : ∀ p, 0 ≤ x p)
    (hy : ∀ p, 0 ≤ y p) {s : ℝ} (hs0 : 0 ≤ s) (hs1 : s ≤ 1) (i : ι) :
    remainder N x i (chord x y i s)
      ≤ s ^ 2 * remainderBound N x i fun a => y (i, a) - x (i, a) := by
  rw [remainder, remainderBound, Finset.mul_sum]
  refine Finset.sum_le_sum fun l _ => ?_
  set sigma := N.blockAppeal (fun a => y (i, a) - x (i, a)) l with hsig
  have hDpos : (0 : ℝ) < N.denom x l := N.denom_pos hx l
  have hout := N.outside_pos l
  have hchordnn : ∀ a, 0 ≤ chord x y i s a := chord_nonneg hx hy hs0 hs1 i
  have hlower : N.outside l ≤ N.denom x l + shift N x i (chord x y i s) l := by
    have hsplit : N.denom x l + shift N x i (chord x y i s) l
        = N.rival x i l + N.blockAppeal (chord x y i s) l := by
      rw [shift, N.denom_eq_rival_add x i]; ring
    have hriv : N.outside l ≤ N.rival x i l := by
      have hle : N.appeal x i l ≤ ∑ j, N.appeal x j l :=
        Finset.single_le_sum (fun j _ => N.appeal_nonneg hx j l) (Finset.mem_univ i)
      rw [Nodes.rival, Nodes.denom]; linarith
    have hblk := N.blockAppeal_nonneg hchordnn l
    rw [hsplit]; linarith
  have hnum : 0 ≤ N.revenue l * N.rival x i l * sigma ^ 2 :=
    mul_nonneg (mul_nonneg (N.revenue_nonneg l) (N.rival_nonneg hx i l)) (sq_nonneg _)
  have hrw : N.revenue l * N.rival x i l * shift N x i (chord x y i s) l ^ 2
      = s ^ 2 * (N.revenue l * N.rival x i l * sigma ^ 2) := by
    rw [shift_chord, ← hsig]; ring
  rw [hrw, mul_div_assoc]
  refine mul_le_mul_of_nonneg_left ?_ (sq_nonneg s)
  gcongr

/-- **Nash implies the variational inequality.** This is the direction that needs an argument.

Nash bounds a linear term *plus* an own-cost quadratic *plus* the share remainder, and the
variational inequality wants the linear term alone. Both obstructions are of quadratic order in
the step -- the first exactly, the second by `remainder_le_sq` -- so a short enough feasible step
makes them jointly smaller than the linear deficit they would have to cover.

The step is chosen explicitly, exactly as in `Game.solvesVI_of_isNash`: on a box the chord is
feasible for every `s` in `(0, 1]`, and `s = min 1 (B / (2A + 1))` does it. No limit is taken and
no derivative exists anywhere in the argument. -/
theorem solvesVI_of_isNash (N : Nodes κ α) (c : ι → α → ℝ) {H : ι → α → α → ℝ}
    {lo hi : ι × α → ℝ} {x : Profile ι α}
    (hsymm : ∀ i a b, H i a b = H i b a)
    (hpsd : ∀ i, ∀ u : α → ℝ, 0 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (hlo : ∀ p, 0 ≤ lo p)
    (h : IsNash N c H lo hi x) :
    SolvesVI (pseudoGradient N c H) (box lo hi) x := by
  have hx := nonneg_of_mem_box hlo h.1
  refine ⟨h.1, fun y hy => ?_⟩
  have hynn := nonneg_of_mem_box hlo hy
  rw [inner_eq_sum, Fintype.sum_prod_type]
  refine Finset.sum_nonneg fun i _ => ?_
  have hcoord : ∀ a : α, (y - x) (i, a) = y (i, a) - x (i, a) := fun a => by simp
  simp only [hcoord]
  by_contra hcon
  push_neg at hcon
  set G := ∑ a, pseudoGradient N c H x (i, a) * (y (i, a) - x (i, a)) with hG
  set Q := ∑ a, ∑ b, H i a b * (y (i, a) - x (i, a)) * (y (i, b) - x (i, b)) with hQ
  set K := remainderBound N x i (fun a => y (i, a) - x (i, a)) with hK
  have hQnn : 0 ≤ Q := hpsd i _
  have hKnn : 0 ≤ K := remainderBound_nonneg N hx i _
  set A := Q / 2 + K with hA
  set B := -G with hB
  have hBpos : 0 < B := by rw [hB]; linarith
  have hAnn : 0 ≤ A := by rw [hA]; linarith
  have h2A : (0 : ℝ) < 2 * A + 1 := by linarith
  set s := min 1 (B / (2 * A + 1)) with hs
  have hspos : 0 < s := lt_min one_pos (div_pos hBpos h2A)
  have hsle : s ≤ 1 := min_le_left _ _
  have hchordnn : ∀ a, 0 ≤ chord x y i s a := chord_nonneg hx hynn hspos.le hsle i
  have hnash := h.2 i (chord x y i s) (chord_mem h.1 hy hspos.le hsle i)
  rw [payoff_replaceFirm_eq N c hsymm i hx hchordnn] at hnash
  have hlinterm : (∑ a, pseudoGradient N c H x (i, a) * (chord x y i s a - x (i, a)))
      = s * G := by
    have hstep : ∀ a, pseudoGradient N c H x (i, a) * (chord x y i s a - x (i, a))
        = s * (pseudoGradient N c H x (i, a) * (y (i, a) - x (i, a))) := by
      intro a; rw [chord]; ring
    rw [Finset.sum_congr rfl fun a _ => hstep a, ← Finset.mul_sum, hG]
  have hquadterm : (∑ a, ∑ b, H i a b * (chord x y i s a - x (i, a))
      * (chord x y i s b - x (i, b))) = s ^ 2 * Q := by
    have hinner : ∀ a, (∑ b, H i a b * (chord x y i s a - x (i, a))
        * (chord x y i s b - x (i, b)))
        = s ^ 2 * ∑ b, H i a b * (y (i, a) - x (i, a)) * (y (i, b) - x (i, b)) := by
      intro a
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun b _ => by rw [chord, chord]; ring
    rw [Finset.sum_congr rfl fun a _ => hinner a, ← Finset.mul_sum, hQ]
  rw [hlinterm, hquadterm] at hnash
  have hrem := remainder_le_sq N hx hynn hspos.le hsle i
  rw [← hK] at hrem
  have hstepwise : s * B ≤ s ^ 2 * A := by rw [hB, hA]; nlinarith [hnash, hrem]
  have hBleAs : B ≤ A * s := by nlinarith [hstepwise, hspos]
  have hsmall : s ≤ B / (2 * A + 1) := min_le_right _ _
  have hlt : A * s < B := by
    have h1 : A * s ≤ A * (B / (2 * A + 1)) := mul_le_mul_of_nonneg_left hsmall hAnn
    have h2 : A * (B / (2 * A + 1)) < B := by
      rw [mul_div_assoc', div_lt_iff₀ h2A]
      nlinarith [hBpos, hAnn]
    linarith
  linarith

/-- **The equivalence, for the game the simulation actually runs.** On a nonnegatively-floored
box, with convex own costs, the variational inequality for the rational-share pseudo-gradient is
exactly Nash equilibrium of the multi-activity share game.

`Game.isNash_iff_solvesVI` says this for a quadratic companion with appeal held fixed. This says
it for the ratio, with appeal a function of capacity, and it is the statement the operating arm
of the paper actually needs. -/
theorem isNash_iff_solvesVI (N : Nodes κ α) (c : ι → α → ℝ) {H : ι → α → α → ℝ}
    {lo hi : ι × α → ℝ} {x : Profile ι α}
    (hsymm : ∀ i a b, H i a b = H i b a)
    (hpsd : ∀ i, ∀ u : α → ℝ, 0 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (hlo : ∀ p, 0 ≤ lo p) :
    IsNash N c H lo hi x ↔ SolvesVI (pseudoGradient N c H) (box lo hi) x :=
  ⟨solvesVI_of_isNash N c hsymm hpsd hlo, isNash_of_solvesVI N c hsymm hpsd hlo⟩

end PricingPerspective.Operating.RationalShare
