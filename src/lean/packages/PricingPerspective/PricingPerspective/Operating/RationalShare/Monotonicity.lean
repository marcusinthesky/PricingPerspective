import PricingPerspective.Operating.RationalShare.NashVI
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.Calculus.Deriv.Pow

set_option linter.style.longLine false

namespace PricingPerspective.Operating.RationalShare

open PricingPerspective.Operating.VariationalInequality

open Finset
open scoped RealInnerProductSpace

variable {κ ι α : Type*} [Fintype κ] [Fintype ι] [Fintype α] [DecidableEq ι]

/-! ### Strong monotonicity, from the demand-node inequality

`Operating/Appeal.lean` reads `node_block_nonneg` as a statement about a *fixed* appeal matrix.
That reading understates it. The array it bounds,

    B l i j  =  D l * (if i = j then 1 else 0)  +  D l  -  appeal i l  -  appeal j l

is the **unscaled symmetric numerator** of the rational-share revenue Jacobian in appeal
coordinates: the symmetric part of that Jacobian is `(revenue l / denom l ^ 3) • B l`, and the
scaling is state-dependent rather than constant. So the inequality already proved there is the
inequality this ratio game needs, and what was missing was the connection, not the mathematics.

The consequence is that the revenue half is monotone on the whole nonnegative orthant with no
hypothesis beyond nonnegative appeal, and the strictly positive modulus comes from own-cost
curvature. It cannot come from the revenue side: the demand bound's slack is `outside l` times
the squared *appeal* direction, and the activity-to-appeal kernel need not be injective, so a
nonzero capacity direction can have zero appeal image. That is why `mu` below is the cost
modulus alone. -/

/-- The revenue half of the negative pseudo-gradient. -/
noncomputable def revenueField (N : Nodes κ α) (x : Profile ι α) : Profile ι α :=
  WithLp.toLp 2 fun p =>
    -∑ l, N.revenue l * N.rival x p.1 l / N.denom x l ^ 2 * N.kernel l p.2

/-- The cost half: affine, and the only source of a strictly positive modulus. -/
noncomputable def costField (c : ι → α → ℝ) (H : ι → α → α → ℝ) (x : Profile ι α) :
    Profile ι α :=
  WithLp.toLp 2 fun p => c p.1 p.2 + ∑ b, H p.1 p.2 b * x (p.1, b)

omit [DecidableEq ι] in
theorem pseudoGradient_eq_costField_add_revenueField (N : Nodes κ α) (c : ι → α → ℝ)
    (H : ι → α → α → ℝ) (x : Profile ι α) :
    pseudoGradient N c H x = costField (ι := ι) c H x + revenueField N x := by
  ext p
  simp [pseudoGradient, costField, revenueField, sub_eq_add_neg]

/-! The three directional quantities every node contributes. They are the appeal image of a
capacity direction, its firmwise sum, and its pairing with the current appeal. -/

/-- The firmwise sum of a direction's appeal image at a node: `S l` in the derivative below. -/
def dirSum (N : Nodes κ α) (d : Profile ι α) (l : κ) : ℝ := ∑ i, N.appeal d i l

/-- The squared appeal image of a direction at a node: `Q l` below. -/
def dirSq (N : Nodes κ α) (d : Profile ι α) (l : κ) : ℝ := ∑ i, N.appeal d i l ^ 2

/-- The current appeal paired against a direction at a node: `C l` below. -/
def dirCross (N : Nodes κ α) (x d : Profile ι α) (l : κ) : ℝ :=
  ∑ i, N.appeal x i l * N.appeal d i l

omit [DecidableEq ι] in
theorem appeal_add_smul (N : Nodes κ α) (y d : Profile ι α) (t : ℝ) (i : ι) (l : κ) :
    N.appeal (y + t • d) i l = N.appeal y i l + t * N.appeal d i l := by
  simp only [Nodes.appeal, PiLp.add_apply, PiLp.smul_apply, smul_eq_mul, Finset.mul_sum,
    ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun a _ => by ring

omit [DecidableEq ι] in
theorem denom_add_smul (N : Nodes κ α) (y d : Profile ι α) (t : ℝ) (l : κ) :
    N.denom (y + t • d) l = N.denom y l + t * dirSum N d l := by
  simp only [Nodes.denom, dirSum, appeal_add_smul]
  rw [Finset.sum_add_distrib, ← Finset.mul_sum]
  ring

omit [DecidableEq ι] in
theorem dirCross_add_smul (N : Nodes κ α) (y d : Profile ι α) (t : ℝ) (l : κ) :
    dirCross N (y + t • d) d l = dirCross N y d l + t * dirSq N d l := by
  simp only [dirCross, dirSq, appeal_add_smul, Finset.mul_sum, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun i _ => by ring

omit [DecidableEq ι] in
/-- **The revenue pairing, node by node.** Pairing the revenue field against a direction turns
the activity sum inside out: what is left is one scalar ratio per node, in the three directional
quantities above. -/
theorem inner_revenueField (N : Nodes κ α) (x d : Profile ι α) :
    ⟪revenueField N x, d⟫
      = -∑ l, N.revenue l
          * (N.denom x l * dirSum N d l - dirCross N x d l) / N.denom x l ^ 2 := by
  have expand : ∀ p : ι × α, revenueField N x p * d p
      = -∑ l, N.revenue l * N.rival x p.1 l / N.denom x l ^ 2 * (N.kernel l p.2 * d p) := by
    rintro ⟨i, a⟩
    simp only [revenueField, neg_mul, Finset.sum_mul]
    exact congrArg Neg.neg (Finset.sum_congr rfl fun l _ => by ring)
  have node : ∀ l : κ, (∑ p : ι × α,
        N.revenue l * N.rival x p.1 l / N.denom x l ^ 2 * (N.kernel l p.2 * d p))
      = N.revenue l * (N.denom x l * dirSum N d l - dirCross N x d l) / N.denom x l ^ 2 := by
    intro l
    have split : (∑ p : ι × α,
          N.revenue l * N.rival x p.1 l / N.denom x l ^ 2 * (N.kernel l p.2 * d p))
        = ∑ i, ∑ a,
          N.revenue l * N.rival x i l / N.denom x l ^ 2 * (N.kernel l a * d (i, a)) := by
      rw [Fintype.sum_prod_type]
    rw [split]
    have inner_a : ∀ i : ι, (∑ a,
          N.revenue l * N.rival x i l / N.denom x l ^ 2 * (N.kernel l a * d (i, a)))
        = N.revenue l / N.denom x l ^ 2 * (N.rival x i l * N.appeal d i l) := by
      intro i
      rw [Nodes.appeal, Finset.mul_sum, Finset.mul_sum]
      exact Finset.sum_congr rfl fun a _ => by ring
    rw [Finset.sum_congr rfl fun i _ => inner_a i, ← Finset.mul_sum]
    have rival_sum : (∑ i, N.rival x i l * N.appeal d i l)
        = N.denom x l * dirSum N d l - dirCross N x d l := by
      simp only [Nodes.rival, dirSum, dirCross, sub_mul, Finset.sum_sub_distrib, Finset.mul_sum]
    rw [rival_sum]
    ring
  rw [inner_eq_sum, Finset.sum_congr rfl fun p _ => expand p, Finset.sum_neg_distrib,
    Finset.sum_comm, Finset.sum_congr rfl fun l _ => node l]

/-! The derivative of the pairing along a segment, and why it is nonnegative. Both are isolated:
the first is field algebra in one real variable, the second is `node_block_nonneg` applied at the
segment point. -/

/-- **The scalar quotient rule this module needs.** Both numerator and denominator are affine in
the step, so the derivative is elementary and the only hypothesis is that the denominator has not
vanished. -/
theorem node_ratio_hasDerivAt {r D0 S C0 Q t : ℝ} (hD : 0 < D0 + t * S) :
    HasDerivAt (fun u : ℝ => r * (((D0 + u * S) * S - (C0 + u * Q)) / (D0 + u * S) ^ 2))
      (-(r * ((D0 + t * S) * (Q + S ^ 2) - 2 * S * (C0 + t * Q)) / (D0 + t * S) ^ 3)) t := by
  have h0 : D0 + t * S ≠ 0 := ne_of_gt hD
  have hd : HasDerivAt (fun u : ℝ => D0 + u * S) S t := by
    simpa only [one_mul, id] using ((hasDerivAt_id t).mul_const S).const_add D0
  have hq : HasDerivAt (fun u : ℝ => C0 + u * Q) Q t := by
    simpa only [one_mul, id] using ((hasDerivAt_id t).mul_const Q).const_add C0
  have hn : HasDerivAt (fun u : ℝ => (D0 + u * S) * S - (C0 + u * Q)) (S * S - Q) t :=
    HasDerivAt.sub (HasDerivAt.mul_const hd S) hq
  have hsq : HasDerivAt (fun u : ℝ => (D0 + u * S) ^ 2) (2 * (D0 + t * S) * S) t := by
    have hfun : (fun u : ℝ => (D0 + u * S) ^ 2)
        = fun u : ℝ => (D0 + u * S) * (D0 + u * S) := by
      funext u; ring
    have hval : (2 : ℝ) * (D0 + t * S) * S
        = S * (D0 + t * S) + (D0 + t * S) * S := by ring
    rw [hfun, hval]
    exact hd.mul hd
  have hne : (D0 + t * S) ^ 2 ≠ 0 := pow_ne_zero 2 h0
  have main := HasDerivAt.const_mul r (HasDerivAt.div hn hsq hne)
  have hval : r * (((S * S - Q) * (D0 + t * S) ^ 2
        - ((D0 + t * S) * S - (C0 + t * Q)) * (2 * (D0 + t * S) * S))
      / ((D0 + t * S) ^ 2) ^ 2)
      = -(r * ((D0 + t * S) * (Q + S ^ 2) - 2 * S * (C0 + t * Q)) / (D0 + t * S) ^ 3) := by
    field_simp
    ring
  rw [← hval]
  exact main

omit [DecidableEq ι] in
/-- **The segment derivative, in closed form.** This is the expression `node_block_nonneg`
bounds, scaled node by node by `revenue l / denom l ^ 3`. -/
theorem revenue_segment_hasDerivAt (N : Nodes κ α) (y d : Profile ι α) (t : ℝ)
    (hpos : ∀ l, 0 < N.denom (y + t • d) l) :
    HasDerivAt (fun u : ℝ => ⟪revenueField N (y + u • d), d⟫)
      (∑ l, N.revenue l
        * (N.denom (y + t • d) l * (dirSq N d l + dirSum N d l ^ 2)
            - 2 * dirSum N d l * dirCross N (y + t • d) d l)
        / N.denom (y + t • d) l ^ 3) t := by
  have reshape : (fun u : ℝ => ⟪revenueField N (y + u • d), d⟫)
      = fun u : ℝ => -∑ l, N.revenue l
          * (((N.denom y l + u * dirSum N d l) * dirSum N d l
              - (dirCross N y d l + u * dirSq N d l))
            / (N.denom y l + u * dirSum N d l) ^ 2) := by
    funext u
    rw [inner_revenueField]
    refine congrArg Neg.neg (Finset.sum_congr rfl fun l _ => ?_)
    rw [denom_add_smul, dirCross_add_smul]
    ring
  have hterm : ∀ l : κ, HasDerivAt
      (fun u : ℝ => N.revenue l
        * (((N.denom y l + u * dirSum N d l) * dirSum N d l
            - (dirCross N y d l + u * dirSq N d l))
          / (N.denom y l + u * dirSum N d l) ^ 2))
      (-(N.revenue l
        * ((N.denom y l + t * dirSum N d l) * (dirSq N d l + dirSum N d l ^ 2)
            - 2 * dirSum N d l * (dirCross N y d l + t * dirSq N d l))
        / (N.denom y l + t * dirSum N d l) ^ 3)) t := by
    intro l
    have := hpos l
    rw [denom_add_smul] at this
    exact node_ratio_hasDerivAt this
  have hsum : HasDerivAt
      (fun u : ℝ => ∑ l, N.revenue l
        * (((N.denom y l + u * dirSum N d l) * dirSum N d l
            - (dirCross N y d l + u * dirSq N d l))
          / (N.denom y l + u * dirSum N d l) ^ 2))
      (∑ l, -(N.revenue l
        * ((N.denom y l + t * dirSum N d l) * (dirSq N d l + dirSum N d l ^ 2)
            - 2 * dirSum N d l * (dirCross N y d l + t * dirSq N d l))
        / (N.denom y l + t * dirSum N d l) ^ 3)) t :=
    HasDerivAt.fun_sum fun l _ => hterm l
  have goal_eq : (∑ l, N.revenue l
        * (N.denom (y + t • d) l * (dirSq N d l + dirSum N d l ^ 2)
            - 2 * dirSum N d l * dirCross N (y + t • d) d l)
        / N.denom (y + t • d) l ^ 3)
      = -∑ l, -(N.revenue l
        * ((N.denom y l + t * dirSum N d l) * (dirSq N d l + dirSum N d l ^ 2)
            - 2 * dirSum N d l * (dirCross N y d l + t * dirSq N d l))
        / (N.denom y l + t * dirSum N d l) ^ 3) := by
    rw [Finset.sum_neg_distrib, neg_neg]
    exact Finset.sum_congr rfl fun l _ => by rw [denom_add_smul, dirCross_add_smul]
  rw [reshape, goal_eq]
  exact hsum.neg

omit [DecidableEq ι] in
/-- **The demand block is nonnegative at every feasible profile.** `node_block_nonneg` supplies
the bracket; the `revenue l / denom l ^ 3` scaling is positive, and the `outside l` slack the
inequality leaves over is discarded because the activity-to-appeal kernel need not be injective. -/
theorem revenue_derivative_nonneg (N : Nodes κ α) {z : Profile ι α} (hz : ∀ p, 0 ≤ z p)
    (d : Profile ι α) :
    0 ≤ ∑ l, N.revenue l
      * (N.denom z l * (dirSq N d l + dirSum N d l ^ 2)
          - 2 * dirSum N d l * dirCross N z d l)
      / N.denom z l ^ 3 := by
  refine Finset.sum_nonneg fun l _ => ?_
  have hD : 0 < N.denom z l := N.denom_pos hz l
  have hnode := node_block_nonneg (fun i => N.appeal z i l) (fun i => N.appeal d i l)
    (N.outside l) (N.denom z l) (fun i => N.appeal_nonneg hz i l) rfl
  have hsq : (0 : ℝ) ≤ dirSq N d l :=
    Finset.sum_nonneg fun i _ => sq_nonneg _
  have hslack : (0 : ℝ) ≤ N.outside l * (dirSq N d l + dirSum N d l ^ 2) :=
    mul_nonneg (N.outside_pos l).le (by positivity)
  have hnum : (0 : ℝ) ≤ N.denom z l * (dirSq N d l + dirSum N d l ^ 2)
      - 2 * dirSum N d l * dirCross N z d l := by
    simp only [dirSq, dirSum, dirCross] at hnode hslack ⊢
    linarith
  exact div_nonneg (mul_nonneg (N.revenue_nonneg l) hnum) (by positivity)

omit [DecidableEq ι] in
/-- **The revenue half is monotone on the whole feasible orthant.** No curvature assumption, no
viability floor, and no bound on cross-sensitivity: nonnegative appeal is the entire hypothesis,
exactly as `node_block_nonneg` advertises. -/
theorem stronglyMonotoneOn_revenueField (N : Nodes κ α) {lo hi : ι × α → ℝ}
    (hlo : ∀ p, 0 ≤ lo p) :
    StronglyMonotoneOn (revenueField N) (box lo hi) 0 := by
  refine stronglyMonotoneOn_of_segment_deriv fun x hx y hy => ?_
  have hmem : ∀ t ∈ Set.Icc (0 : ℝ) 1, y + t • (x - y) ∈ box lo hi := by
    intro t ht
    have hsplit : y + t • (x - y) = (1 - t) • y + t • x := by
      rw [sub_smul, smul_sub, one_smul]; abel
    rw [hsplit]
    exact convex_box lo hi hy hx (by linarith [ht.2]) ht.1 (by ring)
  refine ⟨fun t => ∑ l, N.revenue l
      * (N.denom (y + t • (x - y)) l * (dirSq N (x - y) l + dirSum N (x - y) l ^ 2)
          - 2 * dirSum N (x - y) l * dirCross N (y + t • (x - y)) (x - y) l)
      / N.denom (y + t • (x - y)) l ^ 3, fun t ht => ?_, fun t ht => ?_⟩
  · exact revenue_segment_hasDerivAt N y (x - y) t fun l =>
      N.denom_pos (nonneg_of_mem_box hlo (hmem t ht)) l
  · simpa using revenue_derivative_nonneg N (nonneg_of_mem_box hlo (hmem t ht)) (x - y)

omit [DecidableEq ι] in
/-- The squared norm of a profile, coordinatewise. -/
theorem norm_sq_eq_sum (u : Profile ι α) : ‖u‖ ^ 2 = ∑ p, u p ^ 2 := by
  rw [← real_inner_self_eq_norm_sq, inner_eq_sum]
  exact Finset.sum_congr rfl fun p _ => by ring

omit [DecidableEq ι] in
/-- **The cost half carries the modulus.** The map is affine, so this is the own-cost quadratic
form read off firm by firm -- no calculus, and no appeal to the demand side. -/
theorem stronglyMonotoneOn_costField (c : ι → α → ℝ) {H : ι → α → α → ℝ} {mu : ℝ}
    (s : Set (Profile ι α))
    (hH : ∀ i, ∀ u : α → ℝ, mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b) :
    StronglyMonotoneOn (costField (ι := ι) c H) s mu := by
  intro x _ y _
  set u : Profile ι α := x - y with hu
  have pair : ⟪costField (ι := ι) c H x - costField (ι := ι) c H y, x - y⟫
      = ∑ i, ∑ a, ∑ b, H i a b * u (i, a) * u (i, b) := by
    rw [inner_eq_sum, Fintype.sum_prod_type]
    refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun a _ => ?_
    have coord : (costField (ι := ι) c H x - costField (ι := ι) c H y) (i, a)
        = ∑ b, H i a b * u (i, b) := by
      have expand : ∀ b : α, H i a b * u (i, b)
          = H i a b * x (i, b) - H i a b * y (i, b) := by
        intro b
        simp only [hu, PiLp.sub_apply]
        ring
      rw [Finset.sum_congr rfl fun b _ => expand b, Finset.sum_sub_distrib]
      simp only [costField, PiLp.sub_apply]
      ring
    rw [coord, Finset.sum_mul]
    exact Finset.sum_congr rfl fun b _ => by ring
  rw [pair, norm_sq_eq_sum, Fintype.sum_prod_type, Finset.mul_sum]
  exact Finset.sum_le_sum fun i _ => hH i fun a => u (i, a)

omit [DecidableEq ι] in
/-- **Strong monotonicity of the rational-share pseudo-gradient, from the primitives.**
This is the hypothesis `existsUnique_isNash_of_moduli` carries, discharged. It needs no
restriction on the demand side beyond the structure `Nodes` already imposes -- nonnegative
revenue and kernel, strictly positive outside option -- and takes its modulus from own-cost
curvature, which the registered economy supplies as a strictly positive `own_curvature`.

The generic caveat survives: a merely positive-semidefinite `H` gives `mu = 0` and no
uniqueness. Strict positivity is a property of the registered instance, not of every game this
definition admits. -/
theorem stronglyMonotoneOn_pseudoGradient (N : Nodes κ α) (c : ι → α → ℝ)
    {H : ι → α → α → ℝ} {lo hi : ι × α → ℝ} {mu : ℝ} (hlo : ∀ p, 0 ≤ lo p)
    (hH : ∀ i, ∀ u : α → ℝ, mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b) :
    StronglyMonotoneOn (pseudoGradient N c H) (box lo hi) mu := by
  have hsum := (stronglyMonotoneOn_costField c (box lo hi) hH).add
    (stronglyMonotoneOn_revenueField N hlo)
  have hfun : (fun z => costField (ι := ι) c H z + revenueField N z)
      = pseudoGradient N c H :=
    funext fun z => (pseudoGradient_eq_costField_add_revenueField N c H z).symm
  rw [hfun, add_zero] at hsum
  exact hsum

end PricingPerspective.Operating.RationalShare
