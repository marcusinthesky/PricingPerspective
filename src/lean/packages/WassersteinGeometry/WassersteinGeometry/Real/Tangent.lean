import Mathlib.MeasureTheory.Function.LpSeminorm.TriangleInequality
import WassersteinGeometry.Real.Transport

set_option linter.style.longLine false

open scoped ENNReal MeasureTheory
open MeasureTheory ProbabilityTheory

namespace WassersteinGeometry

namespace P2Real

/-- A one-dimensional tangent vector with actual measurability, `L²`, and monotone-gradient evidence.

On the real line, the convex-gradient condition is represented by monotonicity of
`x ↦ x + field x`.  This is the correct one-dimensional replacement for the former vacuous
`True` fields.

Reference: Panaretos & Zemel (2020), Chapter 4 §4.1. -/
structure TangentSpace (ν : P2Real) where
  field : ℝ → ℝ
  field_aemeasurable : AEMeasurable field ν.measure
  field_memLp : MemLp field 2 ν.measure
  displacement_aemeasurable : AEMeasurable (fun x => x + field x) ν.measure
  displacement_monotone : Monotone (fun x => x + field x)

/-- The displacement field of a tangent vector. -/
def TangentSpace.displacement {ν : P2Real} (v : TangentSpace ν) : ℝ → ℝ :=
  fun x => x + v.field x

lemma TangentSpace.displacement_eq {ν : P2Real} (v : TangentSpace ν) :
    v.displacement = fun x => x + v.field x := rfl

/-- The displacement map is square-integrable by the `L²` triangle inequality. -/
lemma TangentSpace.displacement_memLp {ν : P2Real} (v : TangentSpace ν) :
    MemLp v.displacement 2 ν.measure := by
  change MemLp (fun x : ℝ => x + v.field x) 2 ν.measure
  exact MemLp.add ν.id_memLp v.field_memLp

/-- Push forward a real `P₂` carrier by a tangent displacement. -/
noncomputable def exponentialMap (ν : P2Real) (v : TangentSpace ν) : P2Real :=
  P2Real.ofMeasure (Measure.map v.displacement ν.measure)
    (by
      letI : IsProbabilityMeasure ν.measure := inferInstance
      letI : IsProbabilityMeasure (Measure.map v.displacement ν.measure) :=
        Measure.isProbabilityMeasure_map v.displacement_aemeasurable
      exact (inferInstance : IsProbabilityMeasure (Measure.map v.displacement ν.measure)).measure_univ)
    (by
      apply (memLp_map_measure_iff measurable_id.aestronglyMeasurable
        v.displacement_aemeasurable).2
      change MemLp (fun x : ℝ => x + v.field x) 2 ν.measure
      exact v.displacement_memLp
    )

/-- The explicit map witness induces a tangent vector by subtracting the identity. -/
def logarithmicMap {ν μ : P2Real} (T : OptimalTransportMap ν μ) : TangentSpace ν where
  field := fun x => T x - x
  field_aemeasurable := T.aemeasurable.sub measurable_id.aemeasurable
  field_memLp := T.displacement_memLp
  displacement_aemeasurable := by
    simpa [Function.comp_def, add_sub_cancel_right] using T.aemeasurable
  displacement_monotone := by
    simpa [Function.comp_def, add_sub_cancel_right] using T.monotone

/-- The exponential and logarithmic maps reconstruct the target of an explicit optimal map.

The equality is of bundled real `P₂` values.  No deterministic map is selected for an arbitrary
atomic source; callers must provide `OptimalTransportMap` evidence.

Reference: Panaretos & Zemel (2020), Proposition 2.19. -/
theorem exp_log_inverse {ν μ : P2Real} (T : OptimalTransportMap ν μ) :
    exponentialMap ν (logarithmicMap T) = μ := by
  apply P2Real.ext_measure
  change Measure.map (fun x => x + (T x - x)) ν.measure = μ.measure
  rw [show (fun x => x + (T x - x)) = T by funext x; ring]
  exact T.pushforward

end P2Real

end WassersteinGeometry
