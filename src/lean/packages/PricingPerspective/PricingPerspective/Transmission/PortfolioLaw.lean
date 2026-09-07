import PricingPerspective.Transmission.Frechet

set_option linter.style.longLine false

open scoped ENNReal NNReal MeasureTheory InnerProductSpace
open MeasureTheory WassersteinGeometry WassersteinGeometry.Hilbert Finset

/-!
# Portfolios need a joint law, not a family of pairwise couplings

`Transmission.Frechet` brackets the systematic covariance of *one pair* of assets. A
portfolio needs the whole matrix, and the two are not the same problem. This file develops
the matrix at the measure level; `Transmission.Portfolio` does it for finite clouds.

## The gap this closes

Each pairwise Fréchet class has a ceiling, attained by *its own* optimal coupling. Selecting
the pairwise optimum for every pair independently gives a matrix `[κ̄ᵢⱼ]` that need not be
positive semidefinite: those couplings can be mutually inconsistent, describing a joint
dependence structure that no single joint law realizes. The matrix would then be a valid
entrywise bound and a useless covariance matrix.

A single joint exposure law `J` on `A → H` fixes this by construction. Everything below
follows from having one law rather than `|A|²` couplings:

* `jointCovariance_posSemidef` — the induced matrix is PSD, because its quadratic form is
  the variance of an actual portfolio loading;
* `jointCovariance_eq_systCov` — each entry is nevertheless the `systCov` of the
  corresponding *pair marginal*, so every pairwise result still applies entrywise;
* `map_pair_mem_couplingSet` — that pair marginal really is a coupling of the two asset
  marginals, so each entry lies in the pairwise Fréchet class of `Transmission.Frechet`.

Read together: a globally coherent joint law gives a PSD matrix whose entries respect every
pairwise envelope. Pairwise optimization gives neither guarantee. That is the formal content
of the modelling constraint recorded in the session README under "Open boundaries".

## What is not proved

That a joint law *attaining* the pairwise ceilings simultaneously exists — it generally does
not, which is the point. Nothing here constructs `J`; it is the modeller's object, and its
choice is where the economics of cross-asset dependence lives.

## References

* `.context/chat/2026-08-16_random_functions/README.md`, §2 and "Open boundaries".
* `Transmission.Portfolio` — the finite counterpart.
-/

namespace PricingPerspective.Transmission

variable {A : Type*} [Fintype A]
variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]
  [MeasurableSpace H] [BorelSpace H] [SecondCountableTopology H] [Inhabited H]

/-! ### The portfolio loading and its variance -/

/-- The loading of a portfolio: the weighted sum of the assets' loadings.

    A portfolio aggregates *exposures*, so it depends on the joint draw `x` and not on the
    marginals separately. This is why the literal mixture `∑ wᵢ Pᵢ` — which randomly selects
    one asset — is the wrong object and carries no diversification benefit. -/
def portfolioLoading (w : A → ℝ) (x : A → H) : H := ∑ a, w a • x a

/-- The covariance matrix induced by a joint exposure law. -/
noncomputable def jointCovariance (J : Measure (A → H)) (a b : A) : ℝ :=
  ∫ x, ⟪x a, x b⟫_ℝ ∂J

/-- The systematic variance of a portfolio under a joint exposure law. -/
noncomputable def portfolioVariance (J : Measure (A → H)) (w : A → ℝ) : ℝ :=
  ∫ x, ‖portfolioLoading w x‖ ^ 2 ∂J

omit [MeasurableSpace H] [BorelSpace H] [SecondCountableTopology H] [Inhabited H] in
/-- Pointwise expansion of the squared portfolio loading. -/
lemma normSq_portfolioLoading (w : A → ℝ) (x : A → H) :
    ‖portfolioLoading w x‖ ^ 2 = ∑ a, ∑ b, w a * w b * ⟪x a, x b⟫_ℝ := by
  rw [← real_inner_self_eq_norm_sq]
  unfold portfolioLoading
  rw [sum_inner]
  refine Finset.sum_congr rfl fun a _ => ?_
  rw [real_inner_smul_left, inner_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun b _ => ?_
  rw [real_inner_smul_right]
  ring

omit [BorelSpace H] [SecondCountableTopology H] [Inhabited H] in
/-- **The portfolio variance is the quadratic form of the covariance matrix.**

    `Var(⟪B_w, F⟫) = ∑ₐ∑_b wₐ w_b Cₐ_b`, with `C` the matrix induced by the joint law. This
    is the identity that lets ordinary mean–variance machinery run on top of the random-
    exposure model. -/
theorem portfolioVariance_eq_quadraticForm
    (J : Measure (A → H)) (w : A → ℝ)
    (hint : ∀ a b : A, Integrable (fun x : A → H => ⟪x a, x b⟫_ℝ) J) :
    portfolioVariance J w = ∑ a, ∑ b, w a * w b * jointCovariance J a b := by
  unfold portfolioVariance jointCovariance
  simp_rw [normSq_portfolioLoading]
  rw [integral_finsetSum _ fun a _ =>
    integrable_finsetSum _ fun b _ => ((hint a b).const_mul (w a * w b))]
  refine Finset.sum_congr rfl fun a _ => ?_
  rw [integral_finsetSum _ fun b _ => ((hint a b).const_mul (w a * w b))]
  exact Finset.sum_congr rfl fun b _ => integral_const_mul _ _

omit [BorelSpace H] [SecondCountableTopology H] [Inhabited H] in
/-- **The induced covariance matrix is positive semidefinite.**

    Its quadratic form is the variance of a genuine portfolio loading, hence nonnegative.
    A family of independently chosen pairwise optimal couplings admits no such argument,
    because no single law underlies it. -/
theorem jointCovariance_posSemidef
    (J : Measure (A → H)) (w : A → ℝ)
    (hint : ∀ a b : A, Integrable (fun x : A → H => ⟪x a, x b⟫_ℝ) J) :
    0 ≤ ∑ a, ∑ b, w a * w b * jointCovariance J a b := by
  rw [← portfolioVariance_eq_quadraticForm J w hint]
  unfold portfolioVariance
  exact integral_nonneg fun x => by positivity

omit [Fintype A] [BorelSpace H] [SecondCountableTopology H] [Inhabited H] in
/-- The covariance matrix is symmetric. -/
lemma jointCovariance_symm (J : Measure (A → H)) (a b : A) :
    jointCovariance J a b = jointCovariance J b a := by
  unfold jointCovariance
  exact integral_congr_ae (Filter.Eventually.of_forall fun x => real_inner_comm _ _)

/-! ### Every entry is still a pairwise coupling -/

omit [Fintype A] [NormedAddCommGroup H] [InnerProductSpace ℝ H] [BorelSpace H] [SecondCountableTopology H] [Inhabited H] in
/-- The pair marginal of a joint law couples the two asset marginals.

    So each entry of the covariance matrix is constrained by the pairwise envelope of
    `Transmission.Frechet`, even though the matrix as a whole comes from one joint law. -/
lemma map_pair_mem_couplingSet (J : Measure (A → H)) (a b : A) :
    J.map (fun x : A → H => (x a, x b)) ∈
      couplingSet (J.map (fun x : A → H => x a)) (J.map (fun x : A → H => x b)) := by
  have hpair : Measurable fun x : A → H => (x a, x b) :=
    (measurable_pi_apply a).prodMk (measurable_pi_apply b)
  constructor
  · rw [Measure.map_map measurable_fst hpair]; rfl
  · rw [Measure.map_map measurable_snd hpair]; rfl

omit [Fintype A] [Inhabited H] in
/-- **Each matrix entry is the systematic covariance of a pair coupling.**

    `Cₐ_b = systCov (pair marginal)`. Combined with `map_pair_mem_couplingSet`, every entry
    lies in the Fréchet class of the two asset marginals, so `isGreatest_frechetClass` and
    `isLeast_frechetClass_of_reflection` bound it — while `jointCovariance_posSemidef`
    additionally makes the whole matrix usable. -/
theorem jointCovariance_eq_systCov (J : Measure (A → H)) (a b : A) :
    jointCovariance J a b = systCov (J.map (fun x : A → H => (x a, x b))) := by
  unfold jointCovariance systCov
  have hpair : Measurable fun x : A → H => (x a, x b) :=
    (measurable_pi_apply a).prodMk (measurable_pi_apply b)
  rw [integral_map hpair.aemeasurable
    (by exact (continuous_inner (𝕜 := ℝ)).aestronglyMeasurable)]

/-! ### Coherent matrices respect the pairwise identified sets -/

omit [Fintype A] in
/-- Package a pair marginal of one joint exposure law as an integrable coupling of its
    `P₂(H)` marginals. -/
noncomputable def jointPairCoupling
    (J : Measure (A → H)) (P : A → WassersteinMeasure H)
    (hP : ∀ a, (P a).measure = J.map fun x : A → H => x a)
    (a b : A) : IntegrableCoupling (P a) (P b) :=
  IntegrableCoupling.ofMeasure (P a) (P b)
    (J.map fun x : A → H => (x a, x b)) (by
      rw [hP a, hP b]
      exact map_pair_mem_couplingSet J a b)

omit [Fintype A] in
/-- Every entry induced by one joint exposure law belongs to the Fréchet class of its two
    marginals. -/
theorem jointCovariance_mem_frechetClass
    (J : Measure (A → H)) (P : A → WassersteinMeasure H)
    (hP : ∀ a, (P a).measure = J.map fun x : A → H => x a)
    (a b : A) : jointCovariance J a b ∈ frechetClass (P a) (P b) := by
  refine ⟨jointPairCoupling J P hP a b, ?_⟩
  exact (jointCovariance_eq_systCov J a b).symm

/-- **One joint law gives a PSD matrix inside every certified pairwise interval.**

    Pairwise endpoint certificates constrain each entry, while the shared joint law supplies
    the global PSD condition. The implication is one-way: entrywise interval membership and
    PSD do not by themselves construct a compatible joint law. -/
theorem jointCovariance_posSemidef_and_mem_pairwiseIntervals
    (J : Measure (A → H)) (P : A → WassersteinMeasure H)
    (hP : ∀ a, (P a).measure = J.map fun x : A → H => x a)
    (hint : ∀ a b : A, Integrable (fun x : A → H => ⟪x a, x b⟫_ℝ) J)
    (lo hi : A → A → ℝ)
    (hlo : ∀ a b, IsLeast (frechetClass (P a) (P b)) (lo a b))
    (hhi : ∀ a b, IsGreatest (frechetClass (P a) (P b)) (hi a b))
    (w : A → ℝ) :
    (∀ a b, jointCovariance J a b = jointCovariance J b a)
      ∧ 0 ≤ ∑ a, ∑ b, w a * w b * jointCovariance J a b
      ∧ ∀ a b, jointCovariance J a b ∈ Set.Icc (lo a b) (hi a b) := by
  constructor
  · exact jointCovariance_symm J
  · constructor
    · exact jointCovariance_posSemidef J w hint
    · intro a b
      have hmem := jointCovariance_mem_frechetClass J P hP a b
      exact ⟨(hlo a b).2 hmem, (hhi a b).2 hmem⟩

end PricingPerspective.Transmission
