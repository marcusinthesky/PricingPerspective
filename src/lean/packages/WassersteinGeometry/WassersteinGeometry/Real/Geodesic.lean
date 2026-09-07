import Mathlib.MeasureTheory.Function.LpSeminorm.TriangleInequality
import WassersteinGeometry.Real.Quantile
import WassersteinGeometry.Geodesics

set_option linter.style.longLine false

open scoped ENNReal MeasureTheory
open MeasureTheory ProbabilityTheory

namespace WassersteinGeometry

namespace P2Real

/-- Pointwise interpolation of two increasing uniform representations. -/
noncomputable def interpolateQuantile {μ ν : P2Real} (qμ : QuantileRep μ) (qν : QuantileRep ν)
    (t : ℝ) : ℝ → ℝ :=
  fun u => (1 - t) * qμ.value u + t * qν.value u

lemma interpolateQuantile_measurable {μ ν : P2Real} (qμ : QuantileRep μ) (qν : QuantileRep ν)
    (t : ℝ) : Measurable (interpolateQuantile qμ qν t) := by
  unfold interpolateQuantile
  exact (measurable_const.mul qμ.measurable).add (measurable_const.mul qν.measurable)

lemma interpolateQuantile_memLp {μ ν : P2Real} (qμ : QuantileRep μ) (qν : QuantileRep ν)
    (t : ℝ) : MemLp (interpolateQuantile qμ qν t) 2 uniform01 := by
  unfold interpolateQuantile
  exact (qμ.memLp.const_mul (1 - t)).add (qν.memLp.const_mul t)

lemma interpolateQuantile_monotoneOn {μ ν : P2Real} (qμ : QuantileRep μ) (qν : QuantileRep ν)
    {t : ℝ} (ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    MonotoneOn (interpolateQuantile qμ qν t) (Set.Ioo (0 : ℝ) 1) := by
  intro x hx y hy hxy
  unfold interpolateQuantile
  exact add_le_add
    (mul_le_mul_of_nonneg_left (qμ.monotoneOn hx hy hxy) (sub_nonneg.mpr ht1))
    (mul_le_mul_of_nonneg_left (qν.monotoneOn hx hy hxy) ht0)

/-- The `P₂` measure induced by representation interpolation. -/
noncomputable def interpolate {μ ν : P2Real} (qμ : QuantileRep μ) (qν : QuantileRep ν)
    (t : ℝ) : P2Real :=
  P2Real.ofMeasure (Measure.map (interpolateQuantile qμ qν t) uniform01)
    (by
      letI : IsProbabilityMeasure
          (Measure.map (interpolateQuantile qμ qν t) uniform01) :=
        Measure.isProbabilityMeasure_map
          (interpolateQuantile_measurable qμ qν t).aemeasurable
      exact (inferInstance : IsProbabilityMeasure
        (Measure.map (interpolateQuantile qμ qν t) uniform01)).measure_univ)
    (by
      apply (memLp_map_measure_iff measurable_id.aestronglyMeasurable
        (interpolateQuantile_measurable qμ qν t).aemeasurable).2
      simpa [Function.comp_def] using interpolateQuantile_memLp qμ qν t)

/-- Interpolation itself carries an increasing uniform representation on `[0, 1]`. -/
noncomputable def interpolateRep {μ ν : P2Real} (qμ : QuantileRep μ) (qν : QuantileRep ν)
    {t : ℝ} (ht0 : 0 ≤ t) (ht1 : t ≤ 1) : QuantileRep (interpolate qμ qν t) :=
  { value := interpolateQuantile qμ qν t
    monotoneOn := interpolateQuantile_monotoneOn qμ qν ht0 ht1
    measurable := interpolateQuantile_measurable qμ qν t
    map_eq := by
      rfl
    memLp := interpolateQuantile_memLp qμ qν t }

lemma interpolate_zero {μ ν : P2Real} (qμ : QuantileRep μ) (qν : QuantileRep ν) :
    interpolate qμ qν 0 = μ := by
  apply P2Real.ext_measure
  calc
    Measure.map (interpolateQuantile qμ qν 0) uniform01 =
        Measure.map (fun u : ℝ => qμ.value u) uniform01 :=
      Measure.map_congr (Filter.Eventually.of_forall fun u => by
        simp [interpolateQuantile])
    _ = μ.measure := qμ.map_eq

lemma interpolate_one {μ ν : P2Real} (qμ : QuantileRep μ) (qν : QuantileRep ν) :
    interpolate qμ qν 1 = ν := by
  apply P2Real.ext_measure
  calc
    Measure.map (interpolateQuantile qμ qν 1) uniform01 =
        Measure.map (fun u : ℝ => qν.value u) uniform01 :=
      Measure.map_congr (Filter.Eventually.of_forall fun u => by
        simp [interpolateQuantile])
    _ = ν.measure := qν.map_eq

/-- A distance certificate for representation interpolation on the supported real-line domain. -/
structure GeodesicDistanceCertificate {μ ν : P2Real} (qμ : QuantileRep μ)
    (qν : QuantileRep ν) where
  distance_eq : ∀ s t : ℝ, 0 ≤ s → s ≤ t → t ≤ 1 →
    WassersteinDistance (interpolate qμ qν s).toGeneric (interpolate qμ qν t).toGeneric =
      (t - s) * WassersteinDistance μ.toGeneric ν.toGeneric

/-- Package an interpolation distance certificate as the generic `Geodesic` structure. -/
noncomputable def quantileGeodesic {μ ν : P2Real} (qμ : QuantileRep μ) (qν : QuantileRep ν)
    (h : GeodesicDistanceCertificate qμ qν) : Geodesic μ.toGeneric ν.toGeneric :=
  { path := fun t => (interpolate qμ qν t).toGeneric
    start := congrArg P2Real.toGeneric (interpolate_zero qμ qν)
    end_ := congrArg P2Real.toGeneric (interpolate_one qμ qν)
    constant_speed := fun s t hs hst ht => h.distance_eq s t hs hst ht }

/-- Real-line geodesic existence is exposed under an explicit interpolation-distance proof. -/
theorem geodesic_exists_of_quantile_certificate {μ ν : P2Real} (qμ : QuantileRep μ)
    (qν : QuantileRep ν) (h : GeodesicDistanceCertificate qμ qν) :
    Nonempty (Geodesic μ.toGeneric ν.toGeneric) :=
  ⟨quantileGeodesic qμ qν h⟩

/-- Honest separation contract for the real-line specialization.

The generic carrier remains a pseudometric until a CDF/representation separation proof is supplied.
-/
structure RealMetricSeparation where
  eq_of_dist_eq_zero : ∀ {μ ν : P2Real},
    WassersteinDistance μ.toGeneric ν.toGeneric = 0 → μ = ν

end P2Real

end WassersteinGeometry
