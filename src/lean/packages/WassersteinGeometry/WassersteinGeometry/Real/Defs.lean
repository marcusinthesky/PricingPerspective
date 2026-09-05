import Mathlib.MeasureTheory.Function.LpSeminorm.Basic
import Mathlib.MeasureTheory.Measure.ProbabilityMeasure
import WassersteinGeometry.Geodesics

set_option linter.style.longLine false

open scoped ENNReal MeasureTheory
open MeasureTheory ProbabilityTheory

namespace WassersteinGeometry

/-- A real-valued `P₂` measure with its identity in `L²` made explicit.

The generic carrier is retained so existing coupling and energy-bridge code can consume
`toWasserstein` without changing the generic API.  The `MemLp` witness is the real-line
specialization's square-integrability evidence.

Reference: Panaretos & Zemel (2020), Chapter 2 §2.1. -/
structure P2Real where
  toWasserstein : WassersteinMeasure ℝ
  memLp_id : MemLp id 2 toWasserstein.measure

namespace P2Real

/-- The underlying probability measure of a real `P₂` carrier. -/
def measure (μ : P2Real) : Measure ℝ := μ.toWasserstein.measure

instance (μ : P2Real) : IsProbabilityMeasure μ.measure where
  measure_univ := μ.toWasserstein.is_probability

/-- The explicit identity `L²` witness carried by a real `P₂` measure. -/
lemma id_memLp (μ : P2Real) : MemLp id 2 μ.measure :=
  μ.memLp_id

/-- Forget the real-line square-integrability witness. -/
def toGeneric (μ : P2Real) : WassersteinMeasure ℝ := μ.toWasserstein

@[simp] lemma toGeneric_measure (μ : P2Real) : μ.toGeneric.measure = μ.measure := rfl

/-- Build a real `P₂` carrier from a probability measure and an identity `L²` witness. -/
noncomputable def ofMeasure (μ : Measure ℝ) (hprob : μ Set.univ = 1)
    (hmem : MemLp id 2 μ) : P2Real :=
  ⟨⟨μ, hprob, by
    have h_integrable : Integrable (fun x : ℝ => x ^ 2) μ := hmem.integrable_sq
    have h_finite : (∫⁻ x, ENNReal.ofReal (x ^ 2) ∂μ) < ⊤ :=
      h_integrable.lintegral_lt_top
    have h_eq : (fun x : ℝ => (edist x (default : ℝ)) ^ 2) =
        (fun x : ℝ => ENNReal.ofReal (x ^ 2)) := by
      funext x
      rw [edist_dist, Real.dist_eq, show (default : ℝ) = 0 by rfl, sub_zero]
      rw [← ENNReal.ofReal_pow (abs_nonneg x)]
      congr 1
      rw [sq_abs]
    rw [h_eq]
    exact h_finite
  ⟩, hmem⟩

@[simp] lemma ofMeasure_measure (μ : Measure ℝ) (hprob : μ Set.univ = 1)
    (hmem : MemLp id 2 μ) : (ofMeasure μ hprob hmem).measure = μ := rfl

/-- A Dirac mass is a real `P₂` measure. -/
noncomputable def dirac (x : ℝ) : P2Real :=
  ofMeasure (Measure.dirac x) (by simp) <|
    (memLp_two_iff_integrable_sq measurable_id.aestronglyMeasurable).2 <|
      integrable_dirac (by simp)


/-- Equality of real carriers follows from equality of their underlying measures. -/
lemma ext_measure {μ ν : P2Real} (h : μ.measure = ν.measure) : μ = ν := by
  cases μ with
  | mk μ hμ =>
    cases ν with
    | mk ν hν =>
      cases μ with
      | mk m hprob hmoment =>
        cases ν with
        | mk n hprob' hmoment' =>
          simp only [P2Real.measure] at h
          cases h
          rfl
@[simp] lemma dirac_measure (x : ℝ) : (dirac x).measure = Measure.dirac x := rfl

/-- Equality of real carriers follows from equality of their generic carriers. -/
lemma ext {μ ν : P2Real} (h : μ.toGeneric = ν.toGeneric) : μ = ν := by
  cases μ
  cases ν
  cases h
  rfl

end P2Real

end WassersteinGeometry
