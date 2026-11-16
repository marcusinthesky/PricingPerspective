import PricingPerspective.Operating.RationalShare.Orthant

set_option linter.style.longLine false

namespace PricingPerspective.Operating.RationalShare

open PricingPerspective.Operating.VariationalInequality

open Finset
open scoped RealInnerProductSpace

variable {κ ι α : Type*} [Fintype κ] [Fintype ι] [Fintype α] [DecidableEq ι]

/-! ### The Lipschitz bound, with its norms named

`existsUnique_solvesVI` needs a modulus of continuity as well as a modulus of monotonicity. The
constant below is explicit and deliberately not sharp. Two conventions are fixed so that it means
something definite:

* the profile norm is the Euclidean norm on `EuclideanSpace ℝ (ι × α)`, and
* the matrix norm is **Frobenius**, which dominates the Euclidean operator norm, so a Frobenius
  bound is a valid Lipschitz constant.

With those, the constant is `costFrobenius H + demandFrobenius N`: the Frobenius norm of the
block-diagonal own-cost operator `diag(H 1, ..., H n)`, plus a demand term built from
`revenue l / outside l ^ 2` and the kernel's outer products. Only its finiteness and the
positivity of the outside option are used downstream, to choose a step size. -/

/-- **The ratio's modulus.** Both the rival contribution and the denominator move by at most the
total appeal change, and every denominator is at or above the outside option, so the share
derivative moves by a bounded multiple of them. -/
theorem ratio_diff_abs_le {o Dx Dy Rx Ry : ℝ} (ho : 0 < o) (hx : o ≤ Dx) (hy : o ≤ Dy)
    (hRy : 0 ≤ Ry) (hRyD : Ry ≤ Dy) :
    |Rx / Dx ^ 2 - Ry / Dy ^ 2| ≤ (|Rx - Ry| + 2 * |Dx - Dy|) / o ^ 2 := by
  have hDx : 0 < Dx := lt_of_lt_of_le ho hx
  have hDy : 0 < Dy := lt_of_lt_of_le ho hy
  have hsplit : Rx / Dx ^ 2 - Ry / Dy ^ 2
      = (Rx - Ry) / Dx ^ 2 + Ry * (Dy - Dx) * (Dy + Dx) / (Dx ^ 2 * Dy ^ 2) := by
    field_simp
    ring
  have hfirst : |(Rx - Ry) / Dx ^ 2| ≤ |Rx - Ry| / o ^ 2 := by
    rw [abs_div, abs_of_pos (by positivity : (0:ℝ) < Dx ^ 2)]
    exact div_le_div_of_nonneg_left (abs_nonneg _) (by positivity) (by nlinarith)
  have hsecond : |Ry * (Dy - Dx) * (Dy + Dx) / (Dx ^ 2 * Dy ^ 2)| ≤ 2 * |Dx - Dy| / o ^ 2 := by
    rw [abs_div, abs_of_pos (by positivity : (0:ℝ) < Dx ^ 2 * Dy ^ 2)]
    rw [div_le_div_iff₀ (by positivity) (by positivity)]
    have hnum : |Ry * (Dy - Dx) * (Dy + Dx)| = Ry * |Dx - Dy| * (Dy + Dx) := by
      rw [abs_mul, abs_mul, abs_of_nonneg hRy,
        abs_of_pos (by linarith : (0:ℝ) < Dy + Dx), abs_sub_comm]
    rw [hnum]
    -- `o ^ 2` is dominated by `Dx ^ 2` and by `Dx * Dy`, one for each half of `Dy + Dx`.
    have h1 : o ^ 2 ≤ Dx ^ 2 := by nlinarith
    have h2 : o ^ 2 ≤ Dx * Dy := by nlinarith
    have hpos : (0:ℝ) ≤ (Dy + Dx) * o ^ 2 := by positivity
    have e1 : Ry * ((Dy + Dx) * o ^ 2) ≤ Dy * ((Dy + Dx) * o ^ 2) :=
      mul_le_mul_of_nonneg_right hRyD hpos
    have e3 : Dy * (Dy * o ^ 2) ≤ Dy * (Dy * Dx ^ 2) := by nlinarith
    have e4 : Dy * (Dx * o ^ 2) ≤ Dy * (Dx * (Dx * Dy)) := by nlinarith
    have hkey : Ry * (Dy + Dx) * o ^ 2 ≤ 2 * (Dx ^ 2 * Dy ^ 2) := by nlinarith
    have hleft : Ry * |Dx - Dy| * (Dy + Dx) * o ^ 2
        = |Dx - Dy| * (Ry * (Dy + Dx) * o ^ 2) := by ring
    have hright : 2 * |Dx - Dy| * (Dx ^ 2 * Dy ^ 2)
        = |Dx - Dy| * (2 * (Dx ^ 2 * Dy ^ 2)) := by ring
    rw [hleft, hright]
    exact mul_le_mul_of_nonneg_left hkey (abs_nonneg (Dx - Dy))
  calc |Rx / Dx ^ 2 - Ry / Dy ^ 2|
      = |(Rx - Ry) / Dx ^ 2 + Ry * (Dy - Dx) * (Dy + Dx) / (Dx ^ 2 * Dy ^ 2)| := by rw [hsplit]
    _ ≤ |(Rx - Ry) / Dx ^ 2| + |Ry * (Dy - Dx) * (Dy + Dx) / (Dx ^ 2 * Dy ^ 2)| :=
        abs_add_le _ _
    _ ≤ |Rx - Ry| / o ^ 2 + 2 * |Dx - Dy| / o ^ 2 := by linarith
    _ = (|Rx - Ry| + 2 * |Dx - Dy|) / o ^ 2 := by ring

omit [DecidableEq ι] in
/-- The appeal image of a profile difference, bounded coordinatewise. -/
theorem abs_appeal_sub_le (N : Nodes κ α) (x y : Profile ι α) (i : ι) (l : κ) :
    |N.appeal x i l - N.appeal y i l| ≤ ∑ a, N.kernel l a * |(x - y) (i, a)| := by
  have hrw : N.appeal x i l - N.appeal y i l = ∑ a, N.kernel l a * (x - y) (i, a) := by
    simp only [Nodes.appeal, PiLp.sub_apply, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun a _ => by ring
  rw [hrw]
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun a _ => ?_)
  rw [abs_mul, abs_of_nonneg (N.kernel_nonneg l a)]

omit [DecidableEq ι] in
/-- The total appeal change at a node, which bounds both the denominator and the rival move. -/
theorem abs_denom_sub_le (N : Nodes κ α) (x y : Profile ι α) (l : κ) :
    |N.denom x l - N.denom y l| ≤ ∑ q : ι × α, N.kernel l q.2 * |(x - y) q| := by
  have hrw : N.denom x l - N.denom y l = ∑ j, (N.appeal x j l - N.appeal y j l) := by
    rw [Finset.sum_sub_distrib]
    simp only [Nodes.denom]
    ring
  rw [hrw, Fintype.sum_prod_type]
  exact (Finset.abs_sum_le_sum_abs _ _).trans
    (Finset.sum_le_sum fun j _ => abs_appeal_sub_le N x y j l)

omit [DecidableEq ι] in
theorem abs_rival_sub_le (N : Nodes κ α) (x y : Profile ι α) (i : ι) (l : κ) :
    |N.rival x i l - N.rival y i l| ≤ 2 * ∑ q : ι × α, N.kernel l q.2 * |(x - y) q| := by
  have hrw : N.rival x i l - N.rival y i l
      = (N.denom x l - N.denom y l) + -(N.appeal x i l - N.appeal y i l) := by
    simp only [Nodes.rival]
    ring
  have hden := abs_denom_sub_le N x y l
  have happ : |N.appeal x i l - N.appeal y i l| ≤ ∑ q : ι × α, N.kernel l q.2 * |(x - y) q| := by
    refine (abs_appeal_sub_le N x y i l).trans ?_
    rw [Fintype.sum_prod_type]
    exact Finset.single_le_sum
      (f := fun j => ∑ a, N.kernel l a * |(x - y) (j, a)|)
      (fun j _ => Finset.sum_nonneg fun a _ =>
        mul_nonneg (N.kernel_nonneg l a) (abs_nonneg _)) (Finset.mem_univ i)
  rw [hrw]
  refine (abs_add_le _ _).trans ?_
  rw [abs_neg]
  linarith

/-- The Frobenius norm of the block-diagonal own-cost operator `diag(H 1, ..., H n)`. -/
noncomputable def costFrobenius (H : ι → α → α → ℝ) : ℝ :=
  Real.sqrt (∑ p : ι × α, ∑ q : ι × α, (if q.1 = p.1 then H p.1 p.2 q.2 else 0) ^ 2)

/-- The demand side's coefficient, node by node: `4 * revenue / outside ^ 2` against the
kernel's outer product. -/
noncomputable def demandCoeff (N : Nodes κ α) (p q : ι × α) : ℝ :=
  ∑ l, 4 * N.revenue l / N.outside l ^ 2 * (N.kernel l p.2 * N.kernel l q.2)

/-- The Frobenius norm of the demand coefficient array. -/
noncomputable def demandFrobenius (ι : Type*) [Fintype ι] (N : Nodes κ α) : ℝ :=
  Real.sqrt (∑ p : ι × α, ∑ q : ι × α, demandCoeff N p q ^ 2)

/-- The explicit Lipschitz constant: own-cost plus demand, both in Frobenius norm. -/
noncomputable def lipConst (N : Nodes κ α) (H : ι → α → α → ℝ) : ℝ :=
  costFrobenius H + demandFrobenius ι N

omit [DecidableEq ι] in
theorem demandCoeff_nonneg (N : Nodes κ α) (p q : ι × α) : 0 ≤ demandCoeff N p q :=
  Finset.sum_nonneg fun l _ =>
    mul_nonneg (div_nonneg (by linarith [N.revenue_nonneg l]) (sq_nonneg _))
      (mul_nonneg (N.kernel_nonneg l p.2) (N.kernel_nonneg l q.2))

omit [DecidableEq ι] in
/-- **A coefficient bound gives a norm bound.** Cauchy-Schwarz row by row, then summing: the
Frobenius norm of the coefficient array is a Lipschitz constant for any map it dominates. -/
theorem norm_le_frobenius {M : (ι × α) → (ι × α) → ℝ} {v u : Profile ι α}
    (hv : ∀ p, |v p| ≤ ∑ q : ι × α, M p q * |u q|) :
    ‖v‖ ≤ Real.sqrt (∑ p : ι × α, ∑ q : ι × α, M p q ^ 2) * ‖u‖ := by
  have hFnn : (0 : ℝ) ≤ ∑ p : ι × α, ∑ q : ι × α, M p q ^ 2 :=
    Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _
  have hCnn : (0 : ℝ) ≤ Real.sqrt (∑ p : ι × α, ∑ q : ι × α, M p q ^ 2) := Real.sqrt_nonneg _
  have hCsq : Real.sqrt (∑ p : ι × α, ∑ q : ι × α, M p q ^ 2) ^ 2
      = ∑ p : ι × α, ∑ q : ι × α, M p q ^ 2 := Real.sq_sqrt hFnn
  have habs : (∑ q : ι × α, u q ^ 2) = ∑ q : ι × α, |u q| ^ 2 :=
    Finset.sum_congr rfl fun q _ => (sq_abs _).symm
  have key : ‖v‖ ^ 2
      ≤ (Real.sqrt (∑ p : ι × α, ∑ q : ι × α, M p q ^ 2) * ‖u‖) ^ 2 := by
    rw [norm_sq_eq_sum, mul_pow, hCsq, norm_sq_eq_sum, habs, Finset.sum_mul]
    refine Finset.sum_le_sum fun p _ => ?_
    have hsq : v p ^ 2 ≤ (∑ q, M p q * |u q|) ^ 2 := by
      rw [← sq_abs (v p)]
      exact pow_le_pow_left₀ (abs_nonneg _) (hv p) 2
    exact hsq.trans (Finset.sum_mul_sq_le_sq_mul_sq _ _ _)
  have hroot := Real.sqrt_le_sqrt key
  rwa [Real.sqrt_sq (norm_nonneg v), Real.sqrt_sq (mul_nonneg hCnn (norm_nonneg u))] at hroot

omit [DecidableEq ι] in
/-- The cost field's difference, coordinate by coordinate: the firm's own block only. -/
theorem costField_sub_apply (c : ι → α → ℝ) (H : ι → α → α → ℝ) (x y : Profile ι α)
    (p : ι × α) :
    (costField (ι := ι) c H x - costField (ι := ι) c H y) p
      = ∑ b, H p.1 p.2 b * (x - y) (p.1, b) := by
  have expand : ∀ b : α, H p.1 p.2 b * (x - y) (p.1, b)
      = H p.1 p.2 b * x (p.1, b) - H p.1 p.2 b * y (p.1, b) := by
    intro b
    simp only [PiLp.sub_apply]
    ring
  rw [Finset.sum_congr rfl fun b _ => expand b, Finset.sum_sub_distrib]
  simp only [costField, PiLp.sub_apply]
  ring

/-- The cost half is dominated by the block-diagonal coefficient array. -/
theorem abs_costField_sub_le (c : ι → α → ℝ) (H : ι → α → α → ℝ) (x y : Profile ι α)
    (p : ι × α) :
    |(costField (ι := ι) c H x - costField (ι := ι) c H y) p|
      ≤ ∑ q : ι × α, |if q.1 = p.1 then H p.1 p.2 q.2 else 0| * |(x - y) q| := by
  have hrow : (∑ q : ι × α, |if q.1 = p.1 then H p.1 p.2 q.2 else 0| * |(x - y) q|)
      = ∑ b, |H p.1 p.2 b| * |(x - y) (p.1, b)| := by
    rw [Fintype.sum_prod_type]
    refine (Finset.sum_eq_single p.1 (fun j _ hj => ?_) (fun hj =>
      absurd (Finset.mem_univ p.1) hj)).trans ?_
    · exact Finset.sum_eq_zero fun b _ => by simp [hj]
    · exact Finset.sum_congr rfl fun b _ => by simp
  rw [costField_sub_apply, hrow]
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun b _ => ?_)
  rw [abs_mul]

omit [DecidableEq ι] in
/-- The revenue field's difference, coordinate by coordinate. -/
theorem revenueField_sub_apply (N : Nodes κ α) (x y : Profile ι α) (p : ι × α) :
    (revenueField N x - revenueField N y) p
      = ∑ l, N.revenue l * N.kernel l p.2
          * (N.rival y p.1 l / N.denom y l ^ 2 - N.rival x p.1 l / N.denom x l ^ 2) := by
  simp only [revenueField, PiLp.sub_apply, neg_sub_neg]
  rw [← Finset.sum_sub_distrib]
  exact Finset.sum_congr rfl fun l _ => by ring

omit [DecidableEq ι] in
/-- **The demand half is dominated by the demand coefficient array.** The ratio modulus is
applied node by node, then the appeal bound turns each node's factor into a row of the array. -/
theorem abs_revenueField_sub_le (N : Nodes κ α) {x y : Profile ι α} (hx : ∀ p, 0 ≤ x p)
    (hy : ∀ p, 0 ≤ y p) (p : ι × α) :
    |(revenueField N x - revenueField N y) p|
      ≤ ∑ q : ι × α, demandCoeff N p q * |(x - y) q| := by
  have hnode : ∀ l : κ,
      |N.revenue l * N.kernel l p.2
        * (N.rival y p.1 l / N.denom y l ^ 2 - N.rival x p.1 l / N.denom x l ^ 2)|
      ≤ 4 * N.revenue l / N.outside l ^ 2 * N.kernel l p.2
          * ∑ q : ι × α, N.kernel l q.2 * |(x - y) q| := by
    intro l
    have hox : N.outside l ≤ N.denom x l :=
      le_add_of_nonneg_right (Finset.sum_nonneg fun j _ => N.appeal_nonneg hx j l)
    have hoy : N.outside l ≤ N.denom y l :=
      le_add_of_nonneg_right (Finset.sum_nonneg fun j _ => N.appeal_nonneg hy j l)
    have hRx : 0 ≤ N.rival x p.1 l := N.rival_nonneg hx p.1 l
    have hRxD : N.rival x p.1 l ≤ N.denom x l := by
      have := N.appeal_nonneg hx p.1 l
      rw [Nodes.rival]; linarith
    have hratio := ratio_diff_abs_le (Rx := N.rival y p.1 l) (N.outside_pos l) hoy hox hRx hRxD
    have hR := abs_rival_sub_le N y x p.1 l
    have hD := abs_denom_sub_le N y x l
    set W : ℝ := ∑ q : ι × α, N.kernel l q.2 * |(x - y) q| with hW
    have hswap : (∑ q : ι × α, N.kernel l q.2 * |(y - x) q|) = W := by
      rw [hW]
      exact Finset.sum_congr rfl fun q _ => by rw [show (y - x) q = -((x - y) q) by
        simp only [PiLp.sub_apply]; ring, abs_neg]
    rw [hswap] at hR hD
    have hWnn : 0 ≤ W :=
      Finset.sum_nonneg fun q _ => mul_nonneg (N.kernel_nonneg l q.2) (abs_nonneg _)
    have hbound : |N.rival y p.1 l / N.denom y l ^ 2 - N.rival x p.1 l / N.denom x l ^ 2|
        ≤ 4 * W / N.outside l ^ 2 := by
      refine hratio.trans ?_
      have hnum : |N.rival y p.1 l - N.rival x p.1 l|
          + 2 * |N.denom y l - N.denom x l| ≤ 4 * W := by linarith
      have hpos : (0:ℝ) < N.outside l ^ 2 := pow_pos (N.outside_pos l) 2
      gcongr
    have hfac : (0:ℝ) ≤ N.revenue l * N.kernel l p.2 :=
      mul_nonneg (N.revenue_nonneg l) (N.kernel_nonneg l p.2)
    rw [abs_mul, abs_of_nonneg hfac]
    have := mul_le_mul_of_nonneg_left hbound hfac
    calc N.revenue l * N.kernel l p.2
          * |N.rival y p.1 l / N.denom y l ^ 2 - N.rival x p.1 l / N.denom x l ^ 2|
        ≤ N.revenue l * N.kernel l p.2 * (4 * W / N.outside l ^ 2) := this
      _ = 4 * N.revenue l / N.outside l ^ 2 * N.kernel l p.2 * W := by ring
  rw [revenueField_sub_apply]
  refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
  refine (Finset.sum_le_sum fun l _ => hnode l).trans ?_
  have expand : ∀ l : κ, 4 * N.revenue l / N.outside l ^ 2 * N.kernel l p.2
        * (∑ q : ι × α, N.kernel l q.2 * |(x - y) q|)
      = ∑ q : ι × α, 4 * N.revenue l / N.outside l ^ 2
          * (N.kernel l p.2 * N.kernel l q.2) * |(x - y) q| := by
    intro l
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun q _ => by ring
  rw [Finset.sum_congr rfl fun l _ => expand l, Finset.sum_comm]
  refine le_of_eq (Finset.sum_congr rfl fun q _ => ?_)
  rw [demandCoeff, Finset.sum_mul]

end PricingPerspective.Operating.RationalShare
