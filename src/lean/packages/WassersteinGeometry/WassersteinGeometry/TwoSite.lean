import Architect
import Mathlib.Analysis.Calculus.Deriv.Abs
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Data.Fin.VecNotation
import Mathlib.Order.ConditionallyCompleteLattice.Basic
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Optimal transport on two sites

This file solves the finite transport problem between the probability vectors
`(p, 1 - p)` and `(q, 1 - q)`.  A coupling is an explicitly nonnegative `2 × 2` array
with those row and column marginals.  Moving mass between the two distinct sites costs
`d²` per unit, while diagonal mass costs zero.

The optimizer matches as much diagonal mass as possible.  Consequently the infimum cost is
`d² |p - q|`.  The final two lemmas differentiate this infimum along a source-share path while
holding the target share fixed, on either smooth branch away from `p = q`.
-/

namespace WassersteinGeometry

/-- A finite coupling of the two-site probability vectors `(p, 1 - p)` and `(q, 1 - q)`.

The `mass i j` entries are genuine transport masses: they are nonnegative, the two rows have
source marginals `(p, 1 - p)`, and the two columns have target marginals `(q, 1 - q)`.
-/
structure TwoSiteCoupling (p q : ℝ) where
  mass : Fin 2 → Fin 2 → ℝ
  nonneg : ∀ i j, 0 ≤ mass i j
  row_zero : mass 0 0 + mass 0 1 = p
  row_one : mass 1 0 + mass 1 1 = 1 - p
  col_zero : mass 0 0 + mass 1 0 = q
  col_one : mass 0 1 + mass 1 1 = 1 - q

/-- The squared two-site transport cost: `d²` times the off-diagonal transported mass. -/
def twoSiteCost {p q : ℝ} (d : ℝ) (π : TwoSiteCoupling p q) : ℝ :=
  d ^ 2 * (π.mass 0 1 + π.mass 1 0)

/-- Every feasible two-site coupling transports at least `|p - q|` mass off the diagonal.

This is the finite transport lower bound.  It follows from the row and column marginals,
which identify `p - q` with the difference of the two off-diagonal masses.
-/
theorem abs_sub_le_offDiagonal {p q : ℝ} (π : TwoSiteCoupling p q) :
    |p - q| ≤ π.mass 0 1 + π.mass 1 0 := by
  have hdiff : p - q = π.mass 0 1 - π.mass 1 0 := by
    linarith [π.row_zero, π.col_zero]
  rw [hdiff]
  calc
    |π.mass 0 1 - π.mass 1 0| ≤ |π.mass 0 1| + |π.mass 1 0| := abs_sub _ _
    _ = π.mass 0 1 + π.mass 1 0 := by
      rw [abs_of_nonneg (π.nonneg 0 1), abs_of_nonneg (π.nonneg 1 0)]

/-- Every feasible coupling has cost at least `d² |p - q|`. -/
theorem twoSiteCost_lower_bound {d p q : ℝ} (π : TwoSiteCoupling p q) :
    d ^ 2 * |p - q| ≤ twoSiteCost d π := by
  exact mul_le_mul_of_nonneg_left (abs_sub_le_offDiagonal π) (sq_nonneg d)

/-- The monotone two-site coupling, which matches the common mass at each site first.

For `p ≤ q` its rows are `(p, 0)` and `(q - p, 1 - q)`; for `q < p` they are
`(q, p - q)` and `(0, 1 - p)`.  The interval assumptions make every entry nonnegative.
-/
noncomputable def monotoneTwoSiteCoupling {p q : ℝ}
    (hp : p ∈ Set.Icc (0 : ℝ) 1) (hq : q ∈ Set.Icc (0 : ℝ) 1) :
    TwoSiteCoupling p q := by
  by_cases hpq : p ≤ q
  · refine
      { mass := ![![p, 0], ![q - p, 1 - q]]
        nonneg := ?_
        row_zero := by simp
        row_one := by simp
        col_zero := by simp
        col_one := by simp }
    intro i j
    fin_cases i <;> fin_cases j <;> simp <;> linarith [hp.1, hq.2]
  · have hqp : q ≤ p := le_of_lt (lt_of_not_ge hpq)
    refine
      { mass := ![![q, p - q], ![0, 1 - p]]
        nonneg := ?_
        row_zero := by simp
        row_one := by simp
        col_zero := by simp
        col_one := by simp }
    intro i j
    fin_cases i <;> fin_cases j <;> simp <;> linarith [hq.1, hp.2]

/-- The monotone coupling transports exactly `|p - q|` mass off the diagonal. -/
theorem monotoneTwoSiteCoupling_offDiagonal {p q : ℝ}
    (hp : p ∈ Set.Icc (0 : ℝ) 1) (hq : q ∈ Set.Icc (0 : ℝ) 1) :
    (monotoneTwoSiteCoupling hp hq).mass 0 1 +
        (monotoneTwoSiteCoupling hp hq).mass 1 0 = |p - q| := by
  by_cases hpq : p ≤ q
  · simp [monotoneTwoSiteCoupling, hpq, abs_of_nonpos (sub_nonpos.mpr hpq)]
  · have hqp : q ≤ p := le_of_lt (lt_of_not_ge hpq)
    simp [monotoneTwoSiteCoupling, hpq, abs_of_nonneg (sub_nonneg.mpr hqp)]

/-- The monotone coupling attains cost `d² |p - q|`. -/
theorem monotoneTwoSiteCoupling_cost {d p q : ℝ}
    (hp : p ∈ Set.Icc (0 : ℝ) 1) (hq : q ∈ Set.Icc (0 : ℝ) 1) :
    twoSiteCost d (monotoneTwoSiteCoupling hp hq) = d ^ 2 * |p - q| := by
  simp only [twoSiteCost, monotoneTwoSiteCoupling_offDiagonal hp hq]

/-- The monotone coupling minimizes cost among all feasible two-site couplings. -/
theorem monotoneTwoSiteCoupling_minimal {d p q : ℝ}
    (hp : p ∈ Set.Icc (0 : ℝ) 1) (hq : q ∈ Set.Icc (0 : ℝ) 1)
    (π : TwoSiteCoupling p q) :
    twoSiteCost d (monotoneTwoSiteCoupling hp hq) ≤ twoSiteCost d π := by
  rw [monotoneTwoSiteCoupling_cost hp hq]
  exact twoSiteCost_lower_bound π

/-- The set of squared costs of all finite couplings with the specified two-site marginals. -/
def twoSiteCosts (d p q : ℝ) : Set ℝ :=
  {c | ∃ π : TwoSiteCoupling p q, c = twoSiteCost d π}

/-- The two-site squared transport distance, defined as the real infimum over feasible costs. -/
noncomputable def twoSiteDistanceSq (d p q : ℝ) : ℝ :=
  sInf (twoSiteCosts d p q)

/-- Feasible two-site transport costs are bounded below by `d² |p - q|`. -/
theorem twoSiteCosts_bddBelow (d p q : ℝ) : BddBelow (twoSiteCosts d p q) := by
  refine ⟨d ^ 2 * |p - q|, ?_⟩
  rintro c ⟨π, rfl⟩
  exact twoSiteCost_lower_bound π

/-- For probability shares, the finite two-site cost set is nonempty. -/
theorem twoSiteCosts_nonempty {d p q : ℝ}
    (hp : p ∈ Set.Icc (0 : ℝ) 1) (hq : q ∈ Set.Icc (0 : ℝ) 1) :
    (twoSiteCosts d p q).Nonempty := by
  refine ⟨twoSiteCost d (monotoneTwoSiteCoupling hp hq), ?_⟩
  exact ⟨monotoneTwoSiteCoupling hp hq, rfl⟩

/-- **Exact two-site transport formula.**

For probability vectors `(p, 1 - p)` and `(q, 1 - q)` at sites separated by `d`, the
finite transport infimum is attained by the monotone coupling and equals `d² |p - q|`.
-/
@[blueprint "eq:two-site-transport"
  (statement := /--
    The finite two-site squared optimal transport infimum is d squared times the absolute
    probability-share difference. Assumptions: Both shares lie in the unit interval; diagonal
    transport costs zero and cross-site transport costs d squared.
  -/)]
theorem twoSiteDistanceSq_eq {d p q : ℝ}
    (hp : p ∈ Set.Icc (0 : ℝ) 1) (hq : q ∈ Set.Icc (0 : ℝ) 1) :
    twoSiteDistanceSq d p q = d ^ 2 * |p - q| := by
  apply le_antisymm
  · unfold twoSiteDistanceSq
    apply csInf_le (twoSiteCosts_bddBelow d p q)
    exact ⟨monotoneTwoSiteCoupling hp hq, (monotoneTwoSiteCoupling_cost hp hq).symm⟩
  · unfold twoSiteDistanceSq
    apply le_csInf (twoSiteCosts_nonempty hp hq)
    rintro c ⟨π, rfl⟩
    exact twoSiteCost_lower_bound π

/-- On the branch `p < q`, the exact two-site distance is `d² (q - p)`. -/
theorem twoSiteDistanceSq_eq_of_lt {d p q : ℝ}
    (hp : p ∈ Set.Icc (0 : ℝ) 1) (hq : q ∈ Set.Icc (0 : ℝ) 1) (hpq : p < q) :
    twoSiteDistanceSq d p q = d ^ 2 * (q - p) := by
  rw [twoSiteDistanceSq_eq hp hq, abs_of_neg (sub_neg.mpr hpq)]
  ring

/-- On the branch `q < p`, the exact two-site distance is `d² (p - q)`. -/
theorem twoSiteDistanceSq_eq_of_gt {d p q : ℝ}
    (hp : p ∈ Set.Icc (0 : ℝ) 1) (hq : q ∈ Set.Icc (0 : ℝ) 1) (hqp : q < p) :
    twoSiteDistanceSq d p q = d ^ 2 * (p - q) := by
  rw [twoSiteDistanceSq_eq hp hq, abs_of_pos (sub_pos.mpr hqp)]

/-- Below a frozen target share, two-site squared distance has derivative `-d²v` along any
source-share path with derivative `v`.  No sign restriction on `v` is required.
-/
theorem hasDerivAt_twoSiteDistanceSq_frozen_of_lt {d q s₀ v : ℝ} {p : ℝ → ℝ}
    (hp : ∀ s, p s ∈ Set.Icc (0 : ℝ) 1) (hq : q ∈ Set.Icc (0 : ℝ) 1)
    (hderiv : HasDerivAt p v s₀) (hlt : p s₀ < q) :
    HasDerivAt (fun s ↦ twoSiteDistanceSq d (p s) q) (-(d ^ 2 * v)) s₀ := by
  have habs : HasDerivAt (fun s ↦ |p s - q|) (-v) s₀ := by
    simpa using (hderiv.sub_const q).hasFDerivAt.abs_of_neg (sub_neg.mpr hlt) |>.hasDerivAt
  have hformula : (fun s ↦ twoSiteDistanceSq d (p s) q) =
      fun s ↦ d ^ 2 * |p s - q| := by
    funext s
    exact twoSiteDistanceSq_eq (hp s) hq
  rw [hformula]
  simpa only [mul_neg] using HasDerivAt.const_mul (d ^ 2) habs

/-- Above a frozen target share, two-site squared distance has derivative `d²v` along any
source-share path with derivative `v`.  No sign restriction on `v` is required.
-/
theorem hasDerivAt_twoSiteDistanceSq_frozen_of_gt {d q s₀ v : ℝ} {p : ℝ → ℝ}
    (hp : ∀ s, p s ∈ Set.Icc (0 : ℝ) 1) (hq : q ∈ Set.Icc (0 : ℝ) 1)
    (hderiv : HasDerivAt p v s₀) (hgt : q < p s₀) :
    HasDerivAt (fun s ↦ twoSiteDistanceSq d (p s) q) (d ^ 2 * v) s₀ := by
  have habs : HasDerivAt (fun s ↦ |p s - q|) v s₀ := by
    simpa using (hderiv.sub_const q).hasFDerivAt.abs_of_pos (sub_pos.mpr hgt) |>.hasDerivAt
  have hformula : (fun s ↦ twoSiteDistanceSq d (p s) q) =
      fun s ↦ d ^ 2 * |p s - q| := by
    funext s
    exact twoSiteDistanceSq_eq (hp s) hq
  rw [hformula]
  simpa using HasDerivAt.const_mul (d ^ 2) habs

end WassersteinGeometry
