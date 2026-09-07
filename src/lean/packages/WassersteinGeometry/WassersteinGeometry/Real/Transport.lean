import Mathlib.Probability.Kernel.Disintegration.StandardBorel
import Mathlib.MeasureTheory.Function.LpSeminorm.Basic
import WassersteinGeometry.Real.Defs

set_option linter.style.longLine false

open scoped ENNReal MeasureTheory
open MeasureTheory ProbabilityTheory

namespace WassersteinGeometry

namespace P2Real

/-- A conditional transport plan between two real `P₂` measures.

The joint measure carries both marginals.  The kernel field is a measurable Markov kernel and
`disintegration` records that it reconstructs the joint from the source marginal.  The cost field
keeps the quadratic conditional-cost identity explicit instead of hiding it in a later theorem.

Reference: Panaretos & Zemel (2020), Chapter 1 §1.2. -/
structure ConditionalTransport (μ ν : P2Real) where
  joint : Measure (ℝ × ℝ)
  marginal_fst : joint.map Prod.fst = μ.measure
  marginal_snd : joint.map Prod.snd = ν.measure
  kernel : Kernel ℝ ℝ
  kernel_markov : IsMarkovKernel kernel
  disintegration : joint.fst ⊗ₘ kernel = joint
  quadratic_cost :
    (∫⁻ p, (edist p.1 p.2) ^ 2 ∂joint) =
      ∫⁻ x, ∫⁻ y, (edist x y) ^ 2 ∂kernel x ∂μ.measure

/-- The source marginal of a conditional transport. -/
lemma ConditionalTransport.source_marginal {μ ν : P2Real}
    (π : ConditionalTransport μ ν) : π.joint.map Prod.fst = μ.measure :=
  π.marginal_fst

/-- The target marginal of a conditional transport. -/
lemma ConditionalTransport.target_marginal {μ ν : P2Real}
    (π : ConditionalTransport μ ν) : π.joint.map Prod.snd = ν.measure :=
  π.marginal_snd

/-- The conditional kernel is measurable in its source point. -/
lemma ConditionalTransport.kernel_measurable {μ ν : P2Real}
    (π : ConditionalTransport μ ν) : Measurable π.kernel :=
  π.kernel.measurable

/-- The conditional kernel has probability-measure values. -/
lemma ConditionalTransport.kernel_isProbabilityMeasure {μ ν : P2Real}
    (π : ConditionalTransport μ ν) (x : ℝ) : IsProbabilityMeasure (π.kernel x) := by
  exact π.kernel_markov.isProbabilityMeasure x

/-- A measurable, monotone, exactly optimal transport map witness on the real line.

The source-side `MemLp` field is the displacement integrability needed by tangent and exponential
constructions.  Existence is intentionally represented by this witness rather than asserted for
atomic sources.

Reference: Panaretos & Zemel (2020), Chapter 2 §2.2. -/
structure OptimalTransportMap (μ ν : P2Real) where
  toFun : ℝ → ℝ
  aemeasurable : AEMeasurable toFun μ.measure
  pushforward : Measure.map toFun μ.measure = ν.measure
  monotone : Monotone toFun
  displacement_memLp : MemLp (fun x => toFun x - x) 2 μ.measure
  optimal_cost :
    ∫⁻ x, (edist x (toFun x)) ^ 2 ∂μ.measure =
      WassersteinDistanceSq μ.toGeneric ν.toGeneric

instance : CoeFun (OptimalTransportMap μ ν) (fun _ => ℝ → ℝ) :=
  ⟨OptimalTransportMap.toFun⟩

/-- The source-side map witness is an explicit regularity contract. -/
def HasOptimalTransportMap (μ ν : P2Real) : Prop := Nonempty (OptimalTransportMap μ ν)

/-- Select a deterministic map only when its full witness is supplied. -/
noncomputable def chooseOptimalTransportMap {μ ν : P2Real}
    (h : HasOptimalTransportMap μ ν) : OptimalTransportMap μ ν :=
  Classical.choice h

/-- A map witness preserves the target measure by its stated pushforward equality. -/
lemma map_pushforward {μ ν : P2Real} (T : OptimalTransportMap μ ν) :
    Measure.map T μ.measure = ν.measure :=
  T.pushforward

/-- Almost-everywhere congruence gives equal pushforward measures. -/
lemma map_congr {μ : P2Real} {T S : ℝ → ℝ} (h : T =ᵐ[μ.measure] S) :
    Measure.map T μ.measure = Measure.map S μ.measure :=
  Measure.map_congr h

/-- Pushforwards compose for explicitly a.e.-measurable maps. -/
lemma map_pushforward_comp {μ : P2Real} {T : ℝ → ℝ} {S : ℝ → ℝ}
    (hT : AEMeasurable T μ.measure) (hS : AEMeasurable S (Measure.map T μ.measure)) :
    Measure.map S (Measure.map T μ.measure) = Measure.map (S ∘ T) μ.measure := by
  exact hS.map_map_of_aemeasurable hT

/-- A source measure is atomless when every singleton has zero mass. -/
def AtomlessSource (μ : P2Real) : Prop := ∀ x : ℝ, μ.measure {x} = 0

/-- Deterministic-map existence is scoped to an explicit witness or source regularity theorem.

This proposition is the interface consumed by logarithmic transport constructions.  It does not
claim that every probability measure has a deterministic optimal map.

Reference: Panaretos & Zemel (2020), Chapter 2 §2.2. -/
def RegularMapSource (μ : P2Real) : Prop :=
  AtomlessSource μ ∨ ∀ ν : P2Real, HasOptimalTransportMap μ ν

end P2Real

end WassersteinGeometry
