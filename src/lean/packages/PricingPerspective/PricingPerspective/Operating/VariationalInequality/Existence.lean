import PricingPerspective.Operating.VariationalInequality.Monotone
import Mathlib.Analysis.InnerProductSpace.Projection.Minimal
import Mathlib.Topology.MetricSpace.Contracting

set_option linter.style.longLine false

/-!
# Existence: the projected map has a fixed point

`Monotone.lean` proves that a strongly monotone variational inequality has **at most one**
solution. That is not existence, and a numerically computed profile is not a formal witness
of an exact equilibrium. This module supplies the missing half without a fixed-point theorem
for set-valued maps: on a nonempty complete convex set, the metric projection exists and is
nonexpansive, and the projected residual map

  `T x = proj_s (x - gamma * F x)`

is a genuine contraction for small enough `gamma` whenever `F` is strongly monotone and
Lipschitz. Banach then gives a fixed point, and a fixed point of `T` is exactly a solution of
the variational inequality.

## What this does and does not settle

It settles existence **for any game whose negative pseudo-gradient meets the stated moduli on
a nonempty closed convex strategy set**, and it settles convergence of the projected
iteration at the same factor. Together with `eq_of_solvesVI` that is an `ExistsUnique`.

It does **not** verify that the registered Gaussian demand game meets those moduli. That
assembly needs the payoff's derivative in Lean and is not delivered here; what the arm has
for it is `node_block_nonneg` and a numerical curvature certificate. The gap is therefore a
named hypothesis rather than an unstated assumption.

## Main results

* `exists_isProj` — the Hilbert projection exists on a nonempty complete convex set.
* `isProj_normSq_le`, `isProj_dist_le` — the projection is nonexpansive.
* `isProj_sub_smul_iff_solvesVI` — fixed points of the projected map are the VI solutions.
* `projected_step_dist_le` — the Zarantonello estimate.
* `exists_solvesVI` — existence, and with `eq_of_solvesVI`, `existsUnique_solvesVI`.
-/

namespace PricingPerspective.Operating.VariationalInequality

open scoped RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable {F : E → E} {s : Set E} {mu L gamma : ℝ} {u v x y : E}

/-- `v` is the metric projection of `u` onto `s`, in variational form. -/
def IsProj (s : Set E) (u v : E) : Prop :=
  v ∈ s ∧ ∀ w ∈ s, ⟪u - v, w - v⟫ ≤ 0

/-- **The Hilbert projection theorem, in variational form.** -/
theorem exists_isProj (hne : s.Nonempty) (hc : IsComplete s) (hcv : Convex ℝ s) (u : E) :
    ∃ v, IsProj s u v := by
  obtain ⟨v, hv, hmin⟩ := exists_norm_eq_iInf_of_complete_convex hne hc hcv u
  exact ⟨v, hv, (norm_eq_iInf_iff_real_inner_le_zero hcv hv).mp hmin⟩

/-- **The projection contracts in the squared norm.** Testing each projection's inequality
with the other point is the same device that gives uniqueness of the variational inequality;
here it produces the nonexpansiveness the contraction estimate needs. -/
theorem isProj_normSq_le (h1 : IsProj s u v) (h2 : IsProj s x y) :
    ‖v - y‖ ^ 2 ≤ ⟪u - x, v - y⟫ := by
  have e1 := h1.2 y h2.1
  have e2 := h2.2 v h1.1
  have hsum : ⟪u - v, y - v⟫ + ⟪x - y, v - y⟫ ≤ 0 := by linarith
  have hleft : ⟪u - v, y - v⟫ = -⟪u - v, v - y⟫ := by
    rw [show y - v = -(v - y) by abel, inner_neg_right]
  have hshape : ((x - y) - (u - v) : E) = (v - y) - (u - x) := by abel
  have hexpand : ⟪(x - y) - (u - v), v - y⟫ = ⟪x - y, v - y⟫ - ⟪u - v, v - y⟫ := by
    rw [inner_sub_left]
  rw [hshape, inner_sub_left, real_inner_self_eq_norm_sq] at hexpand
  rw [hleft] at hsum
  linarith

/-- The projection is nonexpansive. -/
theorem isProj_dist_le (h1 : IsProj s u v) (h2 : IsProj s x y) : ‖v - y‖ ≤ ‖u - x‖ := by
  have hsq := isProj_normSq_le h1 h2
  have hcs : ⟪u - x, v - y⟫ ≤ ‖u - x‖ * ‖v - y‖ := real_inner_le_norm _ _
  rcases eq_or_lt_of_le (norm_nonneg (v - y)) with h | h
  · rw [← h]; exact norm_nonneg _
  · have : ‖v - y‖ * ‖v - y‖ ≤ ‖u - x‖ * ‖v - y‖ := by nlinarith
    exact le_of_mul_le_mul_right (by linarith [this]) h

/-- **Fixed points of the projected map are exactly the solutions.** The projection step is
a reformulation, not an approximation: the displacement `-gamma * F v` is orthogonal-in-the-
inequality-sense to the feasible directions exactly when `v` solves the game. -/
theorem isProj_sub_smul_iff_solvesVI (hgamma : 0 < gamma) (hv : v ∈ s) :
    IsProj s (v - gamma • F v) v ↔ SolvesVI F s v := by
  constructor
  · rintro ⟨-, h⟩
    refine ⟨hv, fun w hw => ?_⟩
    have := h w hw
    rw [show v - gamma • F v - v = -(gamma • F v) by abel, inner_neg_left,
      real_inner_smul_left] at this
    nlinarith [this]
  · rintro ⟨-, h⟩
    refine ⟨hv, fun w hw => ?_⟩
    have := h w hw
    rw [show v - gamma • F v - v = -(gamma • F v) by abel, inner_neg_left,
      real_inner_smul_left]
    nlinarith [this]

/-- **The Zarantonello estimate.** One projected step contracts the distance by the factor
`1 - 2 gamma mu + gamma^2 L^2`, in squared form.

Strong monotonicity supplies the negative cross term and the Lipschitz bound caps the
quadratic one, so the factor dips below one exactly when `gamma L^2 < 2 mu`. -/
theorem projected_step_dist_le
    (hF : StronglyMonotoneOn F s mu)
    (hLip : ∀ a ∈ s, ∀ b ∈ s, ‖F a - F b‖ ≤ L * ‖a - b‖)
    (hgamma : 0 ≤ gamma) (hx : x ∈ s) (hy : y ∈ s)
    (h1 : IsProj s (x - gamma • F x) u) (h2 : IsProj s (y - gamma • F y) v) :
    ‖u - v‖ ^ 2 ≤ (1 - 2 * gamma * mu + gamma ^ 2 * L ^ 2) * ‖x - y‖ ^ 2 := by
  have hstep : ‖u - v‖ ≤ ‖(x - gamma • F x) - (y - gamma • F y)‖ := isProj_dist_le h1 h2
  have hrw : (x - gamma • F x) - (y - gamma • F y) = (x - y) - gamma • (F x - F y) := by
    rw [smul_sub]; abel
  have hexpand : ‖(x - y) - gamma • (F x - F y)‖ ^ 2
      = ‖x - y‖ ^ 2 - 2 * gamma * ⟪F x - F y, x - y⟫ + gamma ^ 2 * ‖F x - F y‖ ^ 2 := by
    rw [← real_inner_self_eq_norm_sq, inner_sub_sub_self, real_inner_smul_left,
      real_inner_smul_right, real_inner_smul_left, real_inner_smul_right,
      real_inner_self_eq_norm_sq, real_inner_self_eq_norm_sq,
      real_inner_comm (F x - F y) (x - y)]
    ring
  have hmono := hF x hx y hy
  have hlip := hLip x hx y hy
  have hlipsq : ‖F x - F y‖ ^ 2 ≤ L ^ 2 * ‖x - y‖ ^ 2 := by
    have := mul_self_le_mul_self (norm_nonneg _) hlip
    nlinarith [this]
  have hbound : ‖(x - y) - gamma • (F x - F y)‖ ^ 2
      ≤ (1 - 2 * gamma * mu + gamma ^ 2 * L ^ 2) * ‖x - y‖ ^ 2 := by
    rw [hexpand]
    nlinarith [sq_nonneg gamma, mul_nonneg hgamma hgamma]
  calc ‖u - v‖ ^ 2 ≤ ‖(x - gamma • F x) - (y - gamma • F y)‖ ^ 2 := by
        exact pow_le_pow_left₀ (norm_nonneg _) hstep 2
    _ = ‖(x - y) - gamma • (F x - F y)‖ ^ 2 := by rw [hrw]
    _ ≤ _ := hbound

/-! ### From the estimate to a solution -/

/-- A choice of projection onto a nonempty complete convex set. -/
noncomputable def projOnto (hne : s.Nonempty) (hc : IsComplete s) (hcv : Convex ℝ s)
    (u : E) : E :=
  Classical.choose (exists_isProj hne hc hcv u)

theorem projOnto_isProj (hne : s.Nonempty) (hc : IsComplete s) (hcv : Convex ℝ s) (u : E) :
    IsProj s u (projOnto hne hc hcv u) :=
  Classical.choose_spec (exists_isProj hne hc hcv u)

/-- The projected residual step whose fixed points are the equilibria. -/
noncomputable def projectedStep (hne : s.Nonempty) (hc : IsComplete s) (hcv : Convex ℝ s)
    (F : E → E) (gamma : ℝ) (x : E) : E :=
  projOnto hne hc hcv (x - gamma • F x)

/-- **Existence.** A strongly monotone, Lipschitz variational inequality on a nonempty
closed convex set has a solution, and the projected iteration converges to it.

The step length is the only free choice, and the hypothesis `hq` is exactly
`gamma * L^2 < 2 * mu` once `gamma` is positive. No fixed-point theorem for set-valued maps
is used: the projection is single-valued and nonexpansive, so Banach suffices. -/
theorem exists_solvesVI (hne : s.Nonempty) (hc : IsComplete s) (hcv : Convex ℝ s)
    (hF : StronglyMonotoneOn F s mu)
    (hLip : ∀ a ∈ s, ∀ b ∈ s, ‖F a - F b‖ ≤ L * ‖a - b‖)
    (hgamma : 0 < gamma)
    (hq : 1 - 2 * gamma * mu + gamma ^ 2 * L ^ 2 < 1)
    (hq0 : 0 ≤ 1 - 2 * gamma * mu + gamma ^ 2 * L ^ 2) :
    ∃ v, SolvesVI F s v := by
  classical
  set c : ℝ := 1 - 2 * gamma * mu + gamma ^ 2 * L ^ 2 with hc_def
  set T : E → E := projectedStep hne hc hcv F gamma with hT
  have hmaps : Set.MapsTo T s s := fun a _ => (projOnto_isProj hne hc hcv _).1
  have hstep : ∀ a ∈ s, ∀ b ∈ s, ‖T a - T b‖ ≤ √c * ‖a - b‖ := by
    intro a ha b hb
    have hsq := projected_step_dist_le hF hLip (le_of_lt hgamma) ha hb
      (projOnto_isProj hne hc hcv (a - gamma • F a))
      (projOnto_isProj hne hc hcv (b - gamma • F b))
    have hle : ‖T a - T b‖ ^ 2 ≤ (√c * ‖a - b‖) ^ 2 := by
      rw [mul_pow, Real.sq_sqrt hq0]
      exact hsq
    have hnn : 0 ≤ √c * ‖a - b‖ := by positivity
    have hroot := Real.sqrt_le_sqrt hle
    rwa [Real.sqrt_sq (norm_nonneg _), Real.sqrt_sq hnn] at hroot
  have hk1 : √c < 1 := by
    have : √c < √1 := Real.sqrt_lt_sqrt hq0 (by linarith)
    simpa using this
  have hcoe : ((Real.toNNReal (√c) : NNReal) : ℝ) = √c :=
    Real.coe_toNNReal _ (Real.sqrt_nonneg c)
  have hcontract : ContractingWith (Real.toNNReal (√c)) (hmaps.restrict T s s) := by
    constructor
    · rw [← NNReal.coe_lt_coe, hcoe, NNReal.coe_one]
      exact hk1
    · refine LipschitzWith.of_dist_le_mul fun a b => ?_
      rw [hcoe]
      simpa [Subtype.dist_eq, dist_eq_norm] using hstep a.1 a.2 b.1 b.2
  obtain ⟨z, hzs, hfix, -, -⟩ :=
    hcontract.exists_fixedPoint' hc hmaps (hne.choose_spec) (edist_ne_top _ _)
  refine ⟨z, ?_⟩
  have hproj : IsProj s (z - gamma • F z) z := by
    have := projOnto_isProj hne hc hcv (z - gamma • F z)
    rwa [show projOnto hne hc hcv (z - gamma • F z) = z from hfix] at this
  exact (isProj_sub_smul_iff_solvesVI hgamma hzs).mp hproj

/-- **Existence and uniqueness.** With `eq_of_solvesVI`, the solution is the only one. -/
theorem existsUnique_solvesVI (hne : s.Nonempty) (hc : IsComplete s) (hcv : Convex ℝ s)
    (hmu : 0 < mu) (hF : StronglyMonotoneOn F s mu)
    (hLip : ∀ a ∈ s, ∀ b ∈ s, ‖F a - F b‖ ≤ L * ‖a - b‖)
    (hgamma : 0 < gamma)
    (hq : 1 - 2 * gamma * mu + gamma ^ 2 * L ^ 2 < 1)
    (hq0 : 0 ≤ 1 - 2 * gamma * mu + gamma ^ 2 * L ^ 2) :
    ∃! v, SolvesVI F s v := by
  obtain ⟨v, hv⟩ := exists_solvesVI hne hc hcv hF hLip hgamma hq hq0
  exact ⟨v, hv, fun w hw => eq_of_solvesVI hmu hF hw hv⟩

end PricingPerspective.Operating.VariationalInequality
