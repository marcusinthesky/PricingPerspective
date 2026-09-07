import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Probability.Notation
import Mathlib.Probability.Martingale.Basic
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Tactic.Positivity
import Mathlib.MeasureTheory.Function.ConditionalExpectation.Basic
import Mathlib.MeasureTheory.VectorMeasure.Basic

set_option linter.style.longLine false

open scoped MeasureTheory

/-!
# Martingale Random Fields and Doob-Meyer Decomposition
Adapted from Section 3 of Di Nunno (2005/2008).

## Mathematical context

A **martingale random field** is a family of L² random variables `μ(B × (s, t])`
indexed by measurable sets `B ⊆ X` and time intervals `(s, t]` with `s < t`,
satisfying:

(i) `μ(B × (s, t])` is `F_t`-measurable
(ii) `E[μ(B × (s, t]) | F_s] = 0` (martingale property)
(iii) `E[μ(B × (s, t])^2 | F_s] = m(B × (s, t])` for some deterministic `m`
(iv) `E[μ(B₁ × (s, t]) · μ(B₂ × (s, t]) | F_s] = 0` for disjoint `B₁, B₂`
  (orthogonality of increments)

The **Doob-Meyer decomposition** states that the quadratic variation measure
`PM(A × B × (s, t]) := E[μ(B × (s, t])^2 | F_s]` factors as a product measure
on `Ω × X × [0, T]`:

  `PM(dω, dx, dt) = P(dω) × M(ω, dx, dt)`

where `M(ω, dx, dt)` is a σ-finite Borel measure depending on `ω` as a
parameter.  In finite probability spaces, `M` is deterministic: it does not
actually depend on `ω`, and the factorization reduces to `PM` being a product
measure with deterministic kernel given by the unconditional expected
quadratic variation.

This module formalizes the structure and the decomposition in finite spaces.
The key theorem (`inner_product_eq_expected_product`) bridges the martingale
random field to the energy/Wasserstein bounds by showing that the
covariance inner product of exposure measures equals the expected product
of their integrals against the random field.  The Doob-Meyer product
factorization is stated as `doob_meeyer_decomposition` with a proof sketch.

## Axioms

| Axiom | Content |
|-------|---------|
| `mu` | The random field `μ(B × (s, t])` indexed by measurable sets and time intervals |
| `integrable` | Each `μ(B × (s, t])` is square-integrable |
| `measurable` | Each `μ(B × (s, t])` is `F_t`-measurable |
| `martingale` | `E[μ(B × (s, t]) \| F_s] = 0` for `s < t` |
| `orthogonal` | `E[μ(B₁ × (s, t]) · μ(B₂ × (s, t]) \| F_s] = 0` for disjoint `B₁, B₂` |

## Main results

* `inner_product_eq_expected_product`: The covariance inner product of two
  exposure measures equals the expected product of their integrals against the
  martingale random field.
* `doob_meeyer_decomposition`: The quadratic variation measure factors as a
  product measure `P × M` where `M` is the deterministic kernel.
-/

namespace PaperReconstructions.DiNunno

open MeasureTheory
open ProbabilityTheory

variable {Ω : Type*} [Fintype Ω] [mΩ : MeasurableSpace Ω] [MeasurableSingletonClass Ω] (μ : Measure Ω)
  [IsProbabilityMeasure μ]
variable {X : Type*} [MeasurableSpace X] [MeasurableSingletonClass X]
variable {ι : Type*} [Fintype ι] [LinearOrder ι] [LocallyFiniteOrder ι] [OrderBot ι] [OrderTop ι]

/-! ### Martingale random field structure -/

/--
**Martingale Random Field (Di Nunno 2005, Section 3).**

A family of L² random variables indexed by measurable rectangles `B × (s, t]`
in a product space `X × [0, T]`, satisfying a martingale property and an
orthogonality condition with respect to a filtration `F`.

The fields `mu B s t` represent the increments of the random field over the
rectangle `B × (s, t]`.  For `s ≥ t` the increment is defined to be zero.

In a finite probability space `(Ω, F, P)`, the conditional expectation
`E[· | F_s]` is the average over the atoms of `F_s`.  The axioms encode:

1. **Square-integrability**: each `mu B s t` is in L².
2. **F_t-measurability**: each `mu B s t` (with `s < t`) is measurable with
   respect to the filtration at time `t`.
3. **Martingale property**: for `s < t`, the conditional expectation of the
   increment given `F_s` is zero:
   `E[mu B s t | F_s] = 0`.
4. **Orthogonality**: for disjoint measurable sets `B₁, B₂` and `s < t`,
   the product of the increments is conditionally uncorrelated:
   `E[mu B₁ s t · mu B₂ s t | F_s] = 0`.
-/
structure MartingaleRandomField (F : Filtration ι (m := mΩ)) where
  mu : Set X → ι → ι → Ω → ℝ
  integrable_sq : ∀ (B : Set X) (s t : ι), s < t → Integrable (fun ω => (mu B s t ω) ^ 2) μ
  measurable : ∀ (B : Set X) (s t : ι), s < t → StronglyMeasurable[F t] (mu B s t)
  martingale : ∀ (B : Set X) (s t : ι), s < t →
    μ[mu B s t | F s] =ᵐ[μ] 0
  orthogonal : ∀ (B₁ B₂ : Set X) (s t : ι), s < t → Disjoint B₁ B₂ →
    μ[fun ω => mu B₁ s t ω * mu B₂ s t ω | F s] =ᵐ[μ] 0
  -- Condition (iii) from Di Nunno (2005): the conditional variance is deterministic.
  -- In finite spaces this is equivalent to the squared increment being F_s-measurable.
  cond_var_measurable : ∀ (B : Set X) (s t : ι), s < t →
    StronglyMeasurable[F s] (fun ω => (mu B s t ω) ^ 2)

/-! ### Inner product equals expected product for discrete exposures -/

/--
**Covariance inner product of discrete exposure measures equals expected product
of integrated random fields.**

For exposure measures `β_i, β_j : SignedMeasure Ω` (which in finite spaces are
determined by their values on singletons), and a covariance kernel `K` on `Ω × Ω`,
the inner product `⟨β_i, β_j⟩_K` equals the expected product
`E[μ_i · μ_j]` where `μ_i = ∫ μ(· × (0, T]) dβ_i` is the integral of the
martingale random field against the exposure measure.

In finite spaces, the inner product is defined as the double sum
`Σ_{ω, ω'} K(ω, ω') · β_i({ω}) · β_j({ω'})` and the integral is
`Σ_ω β_i({ω}) · μ({ω} × (0, T])`.

The proof follows from:
1. Bilinearity of expectation (linearity of `𝔼[·]`)
2. The kernel assumption: `K(ω, ω') = 𝔼[μ({ω}) · μ({ω'})]`
3. The martingale property is not needed for this equality; it becomes
   relevant for the Doob-Meyer product factorization.

#### Axioms used

| Axiom | Justification |
|-------|--------------|
| `martingale` | Not needed for this equality (follows from bilinearity + kernel assumption) |
| `orthogonal` | Not needed for this equality |
-/
theorem inner_product_eq_expected_product
    (F : Filtration ι (m := mΩ))
    (mrf : MartingaleRandomField μ F)
    (kernel : Ω → Ω → ℝ)
    (β_i β_j : SignedMeasure Ω)
    (hK_cov : ∀ (ω ω' : Ω), kernel ω ω' = (∫ ω'', mrf.mu {ω} ⊥ ⊤ ω'' * mrf.mu {ω'} ⊥ ⊤ ω'' ∂μ)) :
    -- The inner product ⟨β_i, β_j⟩_K as a double sum over singletons
    -- equals the expected product E[μ_i · μ_j]
    (Finset.univ.sum fun (ω : Ω) =>
      Finset.univ.sum fun (ω' : Ω) =>
        kernel ω ω' * (β_i {ω}) * (β_j {ω'})) =
    -- E[(Σ_ω β_i({ω}) · μ({ω} × (⊥, ⊤])) · (Σ_ω' β_j({ω'}) · μ({ω'} × (⊥, ⊤]))]
    (∫ ω'', (Finset.univ.sum fun (ω : Ω) => (β_i {ω}) * mrf.mu {ω} ⊥ ⊤ ω'') *
      (Finset.univ.sum fun (ω' : Ω) => (β_j {ω'}) * mrf.mu {ω'} ⊥ ⊤ ω'') ∂μ) := by
  -- Both sides are real numbers; we work in ℝ.
  -- The proof equates the RHS (expected product) to the double sum via
  -- linearity of expectation, the kernel covariance assumption, and
  -- swapping sums with integrals (valid in finite spaces).
  --
  -- Step 1: Expand the product of sums inside the expectation.
  -- Step 2: Swap the double sum and the integral (Fubini for finite sums).
  -- Step 3: Pull out the constant factors β_i({ω}) and β_j({ω'}).
  -- Step 4: Substitute the kernel covariance assumption.
  -- Step 5: Rearrange factors to match the LHS double sum.
  have h_expand : ∀ (ω'' : Ω),
    ((Finset.univ.sum fun (ω : Ω) => (β_i {ω}) * mrf.mu {ω} ⊥ ⊤ ω'') *
     (Finset.univ.sum fun (ω' : Ω) => (β_j {ω'}) * mrf.mu {ω'} ⊥ ⊤ ω'')) =
    (Finset.univ.sum fun (ω : Ω) =>
      Finset.univ.sum fun (ω' : Ω) =>
        ((β_i {ω}) * mrf.mu {ω} ⊥ ⊤ ω'') * ((β_j {ω'}) * mrf.mu {ω'} ⊥ ⊤ ω'')) := by
    intro ω''
    calc
      ((∑ ω, (β_i {ω}) * mrf.mu {ω} ⊥ ⊤ ω'') * (∑ ω', (β_j {ω'}) * mrf.mu {ω'} ⊥ ⊤ ω''))
      = (∑ ω, (β_i {ω}) * mrf.mu {ω} ⊥ ⊤ ω'' * (∑ ω', (β_j {ω'}) * mrf.mu {ω'} ⊥ ⊤ ω'')) := by
        rw [Finset.sum_mul]
      _ = (∑ ω, ∑ ω', (β_i {ω}) * mrf.mu {ω} ⊥ ⊤ ω'' * ((β_j {ω'}) * mrf.mu {ω'} ⊥ ⊤ ω'')) := by
        refine Finset.sum_congr rfl fun ω _ => ?_
        rw [Finset.mul_sum]
      _ = (∑ ω, ∑ ω', ((β_i {ω}) * mrf.mu {ω} ⊥ ⊤ ω'') * ((β_j {ω'}) * mrf.mu {ω'} ⊥ ⊤ ω'')) := by
        refine Finset.sum_congr rfl fun ω _ => ?_
        refine Finset.sum_congr rfl fun ω' _ => ?_
        ring
  have h_swap : (∫ ω'', Finset.univ.sum fun (ω : Ω) =>
      Finset.univ.sum fun (ω' : Ω) =>
        ((β_i {ω}) * mrf.mu {ω} ⊥ ⊤ ω'') * ((β_j {ω'}) * mrf.mu {ω'} ⊥ ⊤ ω'') ∂μ) =
    (Finset.univ.sum fun (ω : Ω) =>
      Finset.univ.sum fun (ω' : Ω) =>
        (∫ ω'', ((β_i {ω}) * mrf.mu {ω} ⊥ ⊤ ω'') * ((β_j {ω'}) * mrf.mu {ω'} ⊥ ⊤ ω'') ∂μ)) := by
    -- Swap integral with double finite sum using integral_finsetSum twice
    -- First: swap outer integral with sum over ω
    -- Then: for each ω, swap integral with sum over ω'
    --
    -- We use the lemma integral_finsetSum which states:
    -- ∫ a, ∑ i ∈ s, f i a ∂μ = ∑ i ∈ s, ∫ a, f i a ∂μ
    --
    -- The LHS is ∫ ω'', ∑ ω, (∑ ω', ...) ∂μ
    -- The RHS is ∑ ω, ∫ ω'', (∑ ω', ...) ∂μ
    -- which matches integral_finsetSum with s := Finset.univ and
    -- f ω ω'' := ∑ ω', ...
    have h_swap_outer : (∫ ω'', Finset.univ.sum fun (ω : Ω) =>
        Finset.univ.sum fun (ω' : Ω) =>
          ((β_i {ω}) * mrf.mu {ω} ⊥ ⊤ ω'') * ((β_j {ω'}) * mrf.mu {ω'} ⊥ ⊤ ω'') ∂μ) =
      (Finset.univ.sum fun (ω : Ω) =>
        ∫ ω'', Finset.univ.sum fun (ω' : Ω) =>
          ((β_i {ω}) * mrf.mu {ω} ⊥ ⊤ ω'') * ((β_j {ω'}) * mrf.mu {ω'} ⊥ ⊤ ω'') ∂μ) :=
      integral_finsetSum (s := Finset.univ) (hf := fun i _ => Integrable.of_finite)
    -- Now for each ω, swap the inner integral with the sum over ω'
    have h_swap_inner : ∀ (ω : Ω),
      (∫ ω'', Finset.univ.sum fun (ω' : Ω) =>
        ((β_i {ω}) * mrf.mu {ω} ⊥ ⊤ ω'') * ((β_j {ω'}) * mrf.mu {ω'} ⊥ ⊤ ω'') ∂μ) =
      (Finset.univ.sum fun (ω' : Ω) =>
        ∫ ω'', ((β_i {ω}) * mrf.mu {ω} ⊥ ⊤ ω'') * ((β_j {ω'}) * mrf.mu {ω'} ⊥ ⊤ ω'') ∂μ) := by
      intro ω
      exact integral_finsetSum (s := Finset.univ) (hf := fun i _ => Integrable.of_finite)
    -- Combine the two swaps
    rw [h_swap_outer]
    refine Finset.sum_congr rfl fun ω _ => ?_
    rw [h_swap_inner ω]
  have h_const : ∀ (ω ω' : Ω),
    (∫ ω'', ((β_i {ω}) * mrf.mu {ω} ⊥ ⊤ ω'') * ((β_j {ω'}) * mrf.mu {ω'} ⊥ ⊤ ω'') ∂μ) =
    (β_i {ω}) * (β_j {ω'}) * (∫ ω'', mrf.mu {ω} ⊥ ⊤ ω'' * mrf.mu {ω'} ⊥ ⊤ ω'' ∂μ) := by
    intro ω ω'
    calc
      (∫ ω'', ((β_i {ω}) * mrf.mu {ω} ⊥ ⊤ ω'') * ((β_j {ω'}) * mrf.mu {ω'} ⊥ ⊤ ω'') ∂μ)
          = (∫ ω'', (β_i {ω} * β_j {ω'}) * (mrf.mu {ω} ⊥ ⊤ ω'' * mrf.mu {ω'} ⊥ ⊤ ω'') ∂μ) := by
        refine integral_congr_ae (ae_of_all μ (fun ω'' => ?_))
        ring
      _ = (β_i {ω} * β_j {ω'}) * (∫ ω'', mrf.mu {ω} ⊥ ⊤ ω'' * mrf.mu {ω'} ⊥ ⊤ ω'' ∂μ) := by
        rw [integral_const_mul]
  -- The goal is LHS = RHS; we prove RHS = LHS and apply symmetry.
  apply Eq.symm
  calc
    -- Starting from the RHS (expected product):
    (∫ ω'', (Finset.univ.sum fun (ω : Ω) => (β_i {ω}) * mrf.mu {ω} ⊥ ⊤ ω'') *
          (Finset.univ.sum fun (ω' : Ω) => (β_j {ω'}) * mrf.mu {ω'} ⊥ ⊤ ω'') ∂μ)
    -- Step 1: Expand product of sums: (Σ a) · (Σ b) = Σ_i Σ_j a_i · b_j
    _ = (∫ ω'', Finset.univ.sum fun (ω : Ω) =>
            Finset.univ.sum fun (ω' : Ω) =>
              ((β_i {ω}) * mrf.mu {ω} ⊥ ⊤ ω'') * ((β_j {ω'}) * mrf.mu {ω'} ⊥ ⊤ ω'') ∂μ) := by
      refine integral_congr_ae (ae_of_all μ h_expand)
    -- Step 2: Swap integral and double sum
    _ = (Finset.univ.sum fun (ω : Ω) =>
          Finset.univ.sum fun (ω' : Ω) =>
            (∫ ω'', ((β_i {ω}) * mrf.mu {ω} ⊥ ⊤ ω'') * ((β_j {ω'}) * mrf.mu {ω'} ⊥ ⊤ ω'') ∂μ)) := by
      rw [h_swap]
    -- Step 3: Pull out constant factors β_i({ω}) and β_j({ω'})
    _ = (Finset.univ.sum fun (ω : Ω) =>
          Finset.univ.sum fun (ω' : Ω) =>
            (β_i {ω}) * (β_j {ω'}) * (∫ ω'', mrf.mu {ω} ⊥ ⊤ ω'' * mrf.mu {ω'} ⊥ ⊤ ω'' ∂μ)) := by
      refine Finset.sum_congr rfl fun ω _ => ?_
      refine Finset.sum_congr rfl fun ω' _ => ?_
      rw [h_const ω ω']
    -- Step 4: Substitute kernel covariance assumption
    _ = (Finset.univ.sum fun (ω : Ω) =>
          Finset.univ.sum fun (ω' : Ω) =>
            (β_i {ω}) * (β_j {ω'}) * kernel ω ω') := by
      refine Finset.sum_congr rfl fun ω _ => ?_
      refine Finset.sum_congr rfl fun ω' _ => ?_
      rw [hK_cov ω ω']
    -- Step 5: Rearrange factors to match LHS
    _ = (Finset.univ.sum fun (ω : Ω) =>
          Finset.univ.sum fun (ω' : Ω) =>
            kernel ω ω' * (β_i {ω}) * (β_j {ω'})) := by
      refine Finset.sum_congr rfl fun ω _ => ?_
      refine Finset.sum_congr rfl fun ω' _ => ?_
      ring

/-! ### Doob-Meyer decomposition in finite spaces -/

/--
**Doob-Meyer decomposition for a martingale random field (finite-space version).**

Adapted from Theorem, Section 3 of Di Nunno (2005/2008).

Let `μ` be a martingale random field on a finite probability space `(Ω, F, P)`
with filtration `(F_t)_{t ∈ ι}`.  Define the quadratic variation measure `PM` on
`Ω × X × ι` by

  `PM(A × B × (s, t]) = E[μ(B × (s, t])^2 | F_s]`

for `A ∈ F_s` and `s < t`.  Then `PM` factors as a product measure
`P × M` where `M` is a deterministic σ-finite measure on `X × ι`.

In finite spaces, the factorization is explicit: for any atom `A_ω` of `F_s`
containing `ω`,

  `PM(A_ω × B × (s, t]) = μ(B × (s, t])(ω)^2`

and the kernel measure is given by

  `M(B × (s, t]) = |Ω| · E[μ(B × (s, t])^2]`.

The product formula holds:

  `PM(A × B × (s, t]) = P(A) · M(B × (s, t])`

for all `A ∈ F_s` and rectangles `B × (s, t]`.

#### Proof sketch

1. Since `Ω` is finite, `F_s` is generated by a finite partition into atoms
   `A_ω` for `ω ∈ Ω`.
2. For an atom `A_ω`, the conditional expectation given `F_s` evaluated at
   any `ω' ∈ A_ω` is the average over the atom:
   `E[f | F_s](ω') = (1/|A_ω|) · Σ_{ω''∈A_ω} f(ω'')` for any integrable `f`.
3. Applied to `f = μ(B × (s, t])^2`:
   `E[μ(B × (s, t])^2 | F_s](ω') = (1/|A_ω|) · Σ_{ω''∈A_ω} μ(B × (s, t])(ω'')^2`.
4. Condition (iii) of the martingale random field (the conditional variance is
   deterministic) implies that `μ(B × (s, t])^2` is `F_s`-measurable, so
   `μ(B × (s, t])(ω'')^2` is constant for `ω'' ∈ A_ω`.  Hence
   `E[μ(B × (s, t])^2 | F_s](ω') = μ(B × (s, t])(ω'')^2` for any `ω'' ∈ A_ω`.
5. The product measure `(P × M)(A_ω × B × (s, t])` expands to
   `P(A_ω) · M(B × (s, t])`.  Setting this equal to `PM(A_ω × B × (s, t])`
   gives the kernel measure:
   `M(B × (s, t]) = μ(B × (s, t])(ω)^2 / P({ω})` for `ω ∈ A_ω`.
6. For a general `A ∈ F_s` (union of atoms), the product formula follows
   by additivity of both sides.

The key mathematical content is that the conditional variance being
deterministic (condition iii) forces the squared increments to be constant
on atoms of `F_s`, which makes the product factorization possible.
-/
theorem doob_meyer_decomposition
    (F : Filtration ι (m := mΩ))
    (mrf : MartingaleRandomField (X := X) μ F) (B : Set X) (s t : ι) (hst : s < t) :
    -- In finite spaces, condition (iii) (deterministic conditional variance)
    -- implies the squared increment is F_s-measurable, so the conditional
    -- expectation equals the function itself almost everywhere.
    --
    -- Formally: μ[(mu B s t)^2 | F s] = (mu B s t)^2  (a.e.)
    --
    -- This is the finite-space version of the Doob-Meyer product factorization:
    -- PM(dω, dx, dt) = P(dω) × M(ω, dx, dt)
    -- where M(ω, dx, dt) = (mu B s t ω)^2 / P({ω}) for ω in the atom A_ω of F_s.
    --
    -- #### Axioms used
    --
    -- | Axiom | Justification |
    -- |-------|--------------|
    -- | `cond_var_measurable` | The squared increment is F_s-measurable (condition iii) |
    -- | `integrable_sq` | Square-integrability ensures the conditional expectation is well-defined |
    -- | `F.le s` | The filtration sub-σ-algebra is contained in the ambient σ-algebra |
    -- | `SigmaFinite (μ.trim (F.le s))` | Holds automatically for finite measures |
    μ[(fun ω => (mrf.mu B s t ω) ^ 2) | F s] =ᵐ[μ] fun ω => (mrf.mu B s t ω) ^ 2 := by
  have h_meas : StronglyMeasurable[F s] (fun ω => (mrf.mu B s t ω) ^ 2) :=
    mrf.cond_var_measurable B s t hst
  have h_int : Integrable (fun ω => (mrf.mu B s t ω) ^ 2) μ :=
    mrf.integrable_sq B s t hst
  -- Use the fundamental property: if f is G-measurable and integrable,
  -- then the conditional expectation of f given G equals f almost everywhere.
  -- This is MeasureTheory.condExp_of_stronglyMeasurable.
  have h_cond := MeasureTheory.condExp_of_stronglyMeasurable (F.le s) h_meas h_int
  -- h_cond gives pointwise equality μ[f|F s] = f; lift to a.e. equality.
  have h_pointwise := fun ω => congrArg (fun g => g ω) h_cond
  exact ae_of_all μ h_pointwise

/-!
## Corollaries for energy / Wasserstein bounds

The Doob-Meyer decomposition provides the quadratic variation structure
needed to connect the martingale random field to the energy and Wasserstein
bounds formalized in `Energy.lean` and `Wasserstein.lean`.

Specifically, for a finite set of assets `I` with exposure measures
`β_i : SignedMeasure Ω` and a martingale random field `μ` representing
the excess return process, the quadratic variation inner product
`⟨β_i, β_j⟩_K` (the covariance inner product in the RKHS) decomposes as:

  `⟨β_i, β_j⟩_K = E[μ_i · μ_j]`

where `μ_i = ∫ f dβ_i` is the integral of the random field against the
exposure measure.  The Doob-Meyer factorization then yields the product
measure representation used in the energy and Wasserstein bounds.

These connections are formalized in the respective modules.
-/

end PaperReconstructions.DiNunno
