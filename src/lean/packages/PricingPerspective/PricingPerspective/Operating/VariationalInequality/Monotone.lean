import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.Calculus.Deriv.MeanValue

set_option linter.style.longLine false

/-!
# Strong monotonicity and uniqueness of the operating equilibrium

The operating layer of Paper 6 makes two claims that are routinely conflated and are
logically independent: the capacity game has a **unique** Nash equilibrium, and the
simultaneous proximal map **converges**. This module proves the first and says nothing
about the second, which lives in `Operating/VariationalInequality/Proximal.lean`.

Uniqueness follows from strong monotonicity of the negative pseudo-gradient by a two-line
variational-inequality argument (Rosen 1965), with **no** bound on strategic
cross-sensitivity. The row-sum condition `L_game < mu` that the contraction route needs is
not required here and is not implied by what is proved here.

## Main results

* `eq_of_solvesVI` — a strongly monotone variational inequality has at most one solution.
* `StronglyMonotoneOn.add` — strong moduli add, so a monotone revenue part plus a
  strongly monotone cost part is strongly monotone.
* `stronglyMonotoneOn_smul_id` — an own-curvature quadratic contributes its own modulus.
* `stronglyMonotoneOn_of_segment_deriv` — the derivative criterion: a lower bound on the
  directional derivative along every segment integrates to strong monotonicity. This is the
  step that turns a pointwise Hessian bound into a global statement, and it is proved here
  rather than assumed.
* `node_block_nonneg` — the demand-node inequality that makes the revenue part monotone on
  the whole nonnegative orthant. It is the mathematical core of the uniqueness claim, and it
  needs only nonnegativity of the appeal vector, not a viability floor.

## What is deliberately absent

Existence. Uniqueness together with a computed equilibrium carries every claim the arm
makes, and the concave-game fixed-point argument that would supply existence is not needed
to support any of them.
-/

namespace PricingPerspective.Operating.VariationalInequality

open scoped RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable {F G : E → E} {s : Set E} {mu nu : ℝ} {x y : E}

/-- `F` is `mu`-strongly monotone on `s`. At `mu = 0` this is plain monotonicity. -/
def StronglyMonotoneOn (F : E → E) (s : Set E) (mu : ℝ) : Prop :=
  ∀ x ∈ s, ∀ y ∈ s, mu * ‖x - y‖ ^ 2 ≤ ⟪F x - F y, x - y⟫

/-- `x` solves the variational inequality for `F` on `s`.

For a game with payoffs concave in the own action and `F` the negative pseudo-gradient,
this is exactly the Nash condition. That equivalence is economics rather than mathematics, so
it is not proved at this level of generality. The rational-share specialization is
`PricingPerspective.Operating.RationalShare.isNash_iff_solvesVI` in
`Operating/RationalShare/NashVI.lean`, under that model's stated feasibility and curvature
hypotheses. -/
def SolvesVI (F : E → E) (s : Set E) (x : E) : Prop :=
  x ∈ s ∧ ∀ y ∈ s, 0 ≤ ⟪F x, y - x⟫

/-- **Uniqueness.** A strongly monotone variational inequality has at most one solution.

Testing each solution with the other and adding gives a nonpositive monotonicity pairing,
which strong monotonicity can only reconcile with coincident points. No condition on
cross-sensitivity appears. -/
theorem eq_of_solvesVI (hmu : 0 < mu) (hF : StronglyMonotoneOn F s mu)
    (hx : SolvesVI F s x) (hy : SolvesVI F s y) : x = y := by
  have h1 := hx.2 y hy.1
  have h2 := hy.2 x hx.1
  have e1 : ⟪F x, y - x⟫ = -⟪F x, x - y⟫ := by
    rw [show y - x = -(x - y) by abel, inner_neg_right]
  rw [e1] at h1
  have key : ⟪F x - F y, x - y⟫ ≤ 0 := by
    rw [inner_sub_left]; linarith
  have hlow := hF x hx.1 y hy.1
  have hsq : ‖x - y‖ ^ 2 = 0 := le_antisymm (by nlinarith) (sq_nonneg _)
  exact sub_eq_zero.mp (norm_eq_zero.mp (pow_eq_zero_iff two_ne_zero |>.mp hsq))

/-- Strong moduli add: a monotone part plus a strongly monotone part keeps the modulus.

This is how the economics composes. The revenue block is monotone on the whole nonnegative
orthant by `node_block_nonneg`, and the cost block supplies the own curvature. -/
theorem StronglyMonotoneOn.add (hF : StronglyMonotoneOn F s mu)
    (hG : StronglyMonotoneOn G s nu) :
    StronglyMonotoneOn (fun z => F z + G z) s (mu + nu) := by
  intro a ha b hb
  have hf := hF a ha b hb
  have hg := hG a ha b hb
  have : ⟪F a + G a - (F b + G b), a - b⟫ = ⟪F a - F b, a - b⟫ + ⟪G a - G b, a - b⟫ := by
    rw [← inner_add_left]; congr 1; abel
  rw [this]; nlinarith

/-- An own-curvature quadratic is strongly monotone with exactly its own modulus. -/
theorem stronglyMonotoneOn_smul_id (mu : ℝ) (s : Set E) :
    StronglyMonotoneOn (fun z => mu • z) s mu := by
  intro a _ b _
  have : (mu • a - mu • b) = mu • (a - b) := by rw [smul_sub]
  rw [this, real_inner_smul_left, real_inner_self_eq_norm_sq]

/-- **The derivative criterion.** A uniform lower bound on the directional derivative along
every segment of `s` integrates to strong monotonicity on `s`.

This is the step that carries a pointwise bound on the symmetric part of the Jacobian to a
global statement about the map. It is the reason a Hessian computation at sampled states is
evidence for, and not a substitute for, the uniqueness theorem. -/
theorem stronglyMonotoneOn_of_segment_deriv
    (hseg : ∀ x ∈ s, ∀ y ∈ s, ∃ q : ℝ → ℝ,
      (∀ t ∈ Set.Icc (0 : ℝ) 1,
        HasDerivAt (fun r : ℝ => ⟪F (y + r • (x - y)), x - y⟫) (q t) t) ∧
      (∀ t ∈ Set.Icc (0 : ℝ) 1, mu * ‖x - y‖ ^ 2 ≤ q t)) :
    StronglyMonotoneOn F s mu := by
  intro a ha b hb
  obtain ⟨q, hderiv, hlower⟩ := hseg a ha b hb
  set g : ℝ → ℝ := fun r => ⟪F (b + r • (a - b)), a - b⟫ with hg
  have hsub : interior (Set.Icc (0 : ℝ) 1) ⊆ Set.Icc (0 : ℝ) 1 := interior_subset
  have hcont : ContinuousOn g (Set.Icc (0 : ℝ) 1) := fun t ht =>
    ((hderiv t ht).differentiableAt.continuousAt).continuousWithinAt
  have hdiff : DifferentiableOn ℝ g (interior (Set.Icc (0 : ℝ) 1)) := fun t ht =>
    ((hderiv t (hsub ht)).differentiableAt).differentiableWithinAt
  have hval : ∀ t ∈ interior (Set.Icc (0 : ℝ) 1), mu * ‖a - b‖ ^ 2 ≤ deriv g t := by
    intro t ht
    rw [(hderiv t (hsub ht)).deriv]
    exact hlower t (hsub ht)
  have main :=
    (convex_Icc (0 : ℝ) 1).mul_sub_le_image_sub_of_le_deriv hcont hdiff hval 0
      (by simp) 1 (by simp) zero_le_one
  have h1 : g 1 = ⟪F a, a - b⟫ := by
    simp only [hg, one_smul]
    congr 2
    abel
  have h0 : g 0 = ⟪F b, a - b⟫ := by simp [hg]
  rw [h1, h0] at main
  rw [inner_sub_left]
  linarith [main]

/-- **The demand-node inequality.** At choice sharpness one the symmetric part of a demand
node's revenue block is positive semidefinite for every nonnegative appeal vector.

Written out: with `a` the nonnegative appeal vector, `a0 > 0` the outside option and
`D = a0 + sum a`, the node's quadratic form dominates `a0 * (|u|^2 + (sum u)^2)`, which is
nonnegative. The proof needs only `0 <= a i` -- no lower bound on aggregate attractiveness,
and so no viability floor and no invariant set. That is precisely why uniqueness holds on
the whole nonnegative orthant while the row-sum contraction certificate does not. -/
theorem node_block_nonneg {n : Type*} [Fintype n] (a u : n → ℝ) (a0 D : ℝ)
    (ha : ∀ i, 0 ≤ a i) (hD : D = a0 + ∑ i, a i) :
    a0 * ((∑ i, u i ^ 2) + (∑ i, u i) ^ 2)
      ≤ D * ((∑ i, u i ^ 2) + (∑ i, u i) ^ 2) - 2 * (∑ i, u i) * ∑ i, a i * u i := by
  have hS : ∀ i, u i ^ 2 ≤ ∑ j, u j ^ 2 := fun i =>
    Finset.single_le_sum (f := fun j => u j ^ 2) (fun j _ => sq_nonneg _) (Finset.mem_univ i)
  have key : 2 * (∑ i, u i) * (∑ i, a i * u i)
      ≤ (∑ i, a i) * ((∑ i, u i ^ 2) + (∑ i, u i) ^ 2) := by
    calc 2 * (∑ i, u i) * (∑ i, a i * u i)
        = ∑ i, a i * (2 * (∑ j, u j) * u i) := by
          rw [Finset.mul_sum]
          exact Finset.sum_congr rfl fun i _ => by ring
      _ ≤ ∑ i, a i * ((∑ j, u j ^ 2) + (∑ j, u j) ^ 2) := by
          refine Finset.sum_le_sum fun i _ => ?_
          have hsq : 2 * (∑ j, u j) * u i ≤ (∑ j, u j) ^ 2 + u i ^ 2 := by
            nlinarith [sq_nonneg ((∑ j, u j) - u i)]
          nlinarith [ha i, hS i]
      _ = (∑ i, a i) * ((∑ i, u i ^ 2) + (∑ i, u i) ^ 2) := by rw [← Finset.sum_mul]
  rw [hD]
  nlinarith [key]

end PricingPerspective.Operating.VariationalInequality
