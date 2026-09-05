import Mathlib.MeasureTheory.Measure.ProbabilityMeasure
import Mathlib.MeasureTheory.Integral.Prod
import Mathlib.Topology.MetricSpace.Basic

set_option linter.style.longLine false

/-!
# Types for Energy Statistics

Shared type definitions for the EnergyStatistics package.

## Main definitions

* `ProbabilityMeasure`: A probability measure on a measurable space
* `JointProbabilityMeasure`: A joint probability measure on a product space
* `JointProbabilityMeasure.marginalFst`: First marginal of a joint measure
* `JointProbabilityMeasure.marginalSnd`: Second marginal of a joint measure

## References

* Székely, G. J., & Rizzo, M. L. (2023). The Energy of Data and Distance Correlation.
  CRC Press.
-/

namespace EnergyStatistics

/-- A probability measure on a measurable space.

    This is a thin wrapper bundling a `MeasureTheory.Measure` with
    the proof that the total mass is 1. -/
structure ProbabilityMeasure (α : Type*) [MeasurableSpace α] where
  measure : MeasureTheory.Measure α
  is_probability : measure .univ = 1

/-- Joint probability measure on a product space.

    A probability measure on `α × β` whose marginals are well-defined
    via `Measure.map`. -/
structure JointProbabilityMeasure (α β : Type*)
    [MeasurableSpace α] [MeasurableSpace β] where
  measure : MeasureTheory.Measure (α × β)
  is_probability : measure .univ = 1

/-- The first marginal of a joint probability measure.

    Obtained by pushing forward through `Prod.fst`. -/
noncomputable def JointProbabilityMeasure.marginalFst
    {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    (π : JointProbabilityMeasure α β) : MeasureTheory.Measure α :=
  π.measure.map Prod.fst

/-- The second marginal of a joint probability measure.

    Obtained by pushing forward through `Prod.snd`. -/
noncomputable def JointProbabilityMeasure.marginalSnd
    {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    (π : JointProbabilityMeasure α β) : MeasureTheory.Measure β :=
  π.measure.map Prod.snd

/-- The wrapped measure of an `EnergyStatistics.ProbabilityMeasure` is a
    probability measure in the mathlib typeclass sense. Registering this as an
    instance makes `IsFiniteMeasure`, `SigmaFinite`, and `SFinite` on
    `μ.measure` available to instance search, which the Fubini lemmas
    (`MeasureTheory.integral_prod` and friends) require on both factors. -/
instance ProbabilityMeasure.isProbabilityMeasure
    {α : Type*} [MeasurableSpace α] (μ : ProbabilityMeasure α) :
    MeasureTheory.IsProbabilityMeasure μ.measure :=
  ⟨μ.is_probability⟩

/-- The wrapped measure of a `JointProbabilityMeasure` is a probability measure
    in the mathlib typeclass sense (see `ProbabilityMeasure.isProbabilityMeasure`). -/
instance JointProbabilityMeasure.isProbabilityMeasure
    {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    (π : JointProbabilityMeasure α β) :
    MeasureTheory.IsProbabilityMeasure π.measure :=
  ⟨π.is_probability⟩

/-- Finite joint first moment for a pair of probability measures: the distance
    function is Bochner-integrable with respect to `μ.measure.prod ν.measure`,
    i.e. `E‖X − Y‖ < ∞` for independent `X ~ μ`, `Y ~ ν`.

    Under Lean's junk-zero convention for non-integrable Bochner integrals,
    the iterated integrals in `energyDistanceSq` only agree with the intended
    expectations under this hypothesis (Fubini,
    `MeasureTheory.integral_integral`). Because `α` carries an arbitrary
    `MeasurableSpace` with no Borel link to the metric, this predicate also
    supplies the a.e.-strong measurability of `dist`, which is not otherwise
    derivable.

    Reference: Székely & Rizzo (2023), finite-first-moment condition of
    Theorem 3.1 / Proposition 3.1. -/
def FiniteDistMoment {α : Type*} [MeasurableSpace α] [PseudoMetricSpace α]
    (μ ν : ProbabilityMeasure α) : Prop :=
  MeasureTheory.Integrable (fun p : α × α => dist p.1 p.2)
    (μ.measure.prod ν.measure)

/-- Moment hypotheses for distance covariance and distance correlation over a
    joint measure `π` on `α × β`: Bochner integrability over `π ⊗ π` of the two
    coordinate distances, their squares, and their product.

    `fst`/`snd` make the S₂/S₃ integrals of `distanceCovarianceSq` honest;
    `distProd` makes S₁ honest; `fstSq`/`sndSq` are needed by
    `distanceVarianceSq`/`distanceVarianceSqSnd` (which contain squared
    distances) and by the Cauchy–Schwarz bound in `dcor_bounds`. `distProd` is
    derivable from `fst`, `snd`, `fstSq`, `sndSq` via `|ab| ≤ (a² + b²)/2`; it
    is kept as a field so statements need no side derivations.

    Reference: Székely & Rizzo (2023), Proposition 12.1 (finite-moment
    conditions); Lyons (2013), §2. -/
structure JointFiniteMoments {α β : Type*}
    [MeasurableSpace α] [MeasurableSpace β]
    [PseudoMetricSpace α] [PseudoMetricSpace β]
    (π : JointProbabilityMeasure α β) : Prop where
  fst : MeasureTheory.Integrable
    (fun z : (α × β) × (α × β) => dist z.1.1 z.2.1) (π.measure.prod π.measure)
  snd : MeasureTheory.Integrable
    (fun z : (α × β) × (α × β) => dist z.1.2 z.2.2) (π.measure.prod π.measure)
  fstSq : MeasureTheory.Integrable
    (fun z : (α × β) × (α × β) => dist z.1.1 z.2.1 ^ 2) (π.measure.prod π.measure)
  sndSq : MeasureTheory.Integrable
    (fun z : (α × β) × (α × β) => dist z.1.2 z.2.2 ^ 2) (π.measure.prod π.measure)
  distProd : MeasureTheory.Integrable
    (fun z : (α × β) × (α × β) => dist z.1.1 z.2.1 * dist z.1.2 z.2.2)
    (π.measure.prod π.measure)

/-- The metric `dist` is **conditionally negative definite** (the space has
    *negative type*): for every finite family of points `x : Fin k → α` and
    weights `w : Fin k → ℝ` summing to zero,

      `∑ i, ∑ j, w i * w j * dist (x i) (x j) ≤ 0`.

    Euclidean and Hilbert spaces have negative type (Schoenberg's theorem;
    Székely & Rizzo 2023 §3.2, §10; Lyons 2013 §3), while e.g. the circle with
    geodesic distance does not — so energy-statistics nonnegativity is *false*
    over a bare `PseudoMetricSpace`. This predicate makes the required
    condition an explicit hypothesis (HYPOTHESIS-CONDITIONAL in the roadmap's
    tagging) until a Euclidean/Hilbert instance is formalized.

    Reference: Székely & Rizzo (2023), §3.2; Lyons (2013), "Distance
    covariance in metric spaces", Definition 3.1. -/
def DistNegativeType (α : Type*) [PseudoMetricSpace α] : Prop :=
  ∀ (k : ℕ) (x : Fin k → α) (w : Fin k → ℝ), (∑ i, w i) = 0 →
    ∑ i, ∑ j, w i * w j * dist (x i) (x j) ≤ 0

/-- **Strict negative type.** A space has strict negative type when the negative-type inequality
    is strict for every nonzero zero-sum weighting of pairwise-distinct points. This is a
    finite-support property; it must not be confused with *strong* negative type, which requires
    the corresponding rigidity for arbitrary probability measures with finite first moments.

    Strict and strong negative type agree on finite spaces but not in general: Lyons (2013),
    Remark 3.3 gives a strict-negative-type space that is not of strong negative type.

    Reference: Lyons (2013), "Distance covariance in metric spaces", Definition 3.1 and
    Remark 3.3; Székely & Rizzo (2023), §3.1. -/
def StrictDistNegativeType (α : Type*) [PseudoMetricSpace α] : Prop :=
  DistNegativeType α ∧
    ∀ (k : ℕ) (x : Fin k → α) (w : Fin k → ℝ), Function.Injective x →
      (∑ i, w i) = 0 → (∃ i, w i ≠ 0) →
        ∑ i, ∑ j, w i * w j * dist (x i) (x j) < 0

/-- Symmetry of `FiniteDistMoment`: if the distance is integrable for `(μ, ν)`
    then it is also integrable for `(ν, μ)`.

    Reference: Székely & Rizzo (2023), Proposition 3.1. -/
lemma FiniteDistMoment.symm {α : Type*} [MeasurableSpace α] [PseudoMetricSpace α]
    {μ ν : ProbabilityMeasure α} (h : FiniteDistMoment μ ν) : FiniteDistMoment ν μ := by
  have hswap := MeasureTheory.Integrable.swap h
  refine hswap.congr ?_
  filter_upwards with p
  simp [dist_comm]

/-- Build `JointFiniteMoments` from the four coordinate distances and their squares.

    The only field that cannot be derived directly from the hypotheses is
    `distProd`; it follows from `|ab| ≤ (a² + b²)/2` (AM-GM inequality,
    `two_mul_le_add_sq`) and `Integrable.mono'`.

    Reference: Székely & Rizzo (2023), Proposition 12.1. -/
lemma JointFiniteMoments.mk_of_sq {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    [PseudoMetricSpace α] [PseudoMetricSpace β] {π : JointProbabilityMeasure α β}
    (h1 : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => dist z.1.1 z.2.1) (π.measure.prod π.measure))
    (h2 : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => dist z.1.2 z.2.2) (π.measure.prod π.measure))
    (h1sq : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => dist z.1.1 z.2.1 ^ 2) (π.measure.prod π.measure))
    (h2sq : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => dist z.1.2 z.2.2 ^ 2) (π.measure.prod π.measure)) :
    JointFiniteMoments π := by
  refine
    { fst := h1
      snd := h2
      fstSq := h1sq
      sndSq := h2sq
      distProd := ?_ }
  have h_bound : MeasureTheory.Integrable
    (fun z : (α × β) × (α × β) => (dist z.1.1 z.2.1 ^ 2 + dist z.1.2 z.2.2 ^ 2) / 2)
    (π.measure.prod π.measure) := by
    simpa using (h1sq.add h2sq).div_const (2 : ℝ)
  refine MeasureTheory.Integrable.mono' h_bound
    (h1.aestronglyMeasurable.mul h2.aestronglyMeasurable) ?_
  filter_upwards with z
  have ha : 0 ≤ dist z.1.1 z.2.1 := dist_nonneg
  have hb : 0 ≤ dist z.1.2 z.2.2 := dist_nonneg
  have hab_nonneg : 0 ≤ dist z.1.1 z.2.1 * dist z.1.2 z.2.2 := mul_nonneg ha hb
  calc
    ‖dist z.1.1 z.2.1 * dist z.1.2 z.2.2‖ = |dist z.1.1 z.2.1 * dist z.1.2 z.2.2| := by
      rw [Real.norm_eq_abs]
    _ = dist z.1.1 z.2.1 * dist z.1.2 z.2.2 := abs_of_nonneg hab_nonneg
    _ ≤ (dist z.1.1 z.2.1 ^ 2 + dist z.1.2 z.2.2 ^ 2) / 2 := by
      have h := two_mul_le_add_sq (dist z.1.1 z.2.1) (dist z.1.2 z.2.2)
      nlinarith

end EnergyStatistics
