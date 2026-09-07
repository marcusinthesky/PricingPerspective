import WassersteinGeometry.Real.Geodesic
import WassersteinGeometry.Real.Barycenter
import WassersteinGeometry.Real.Tangent
import WassersteinGeometry.Real.Transport

set_option linter.style.longLine false

open scoped ENNReal MeasureTheory unitInterval
open MeasureTheory ProbabilityTheory

namespace WassersteinGeometry
namespace P2Real

/-- Dirac self-distance is zero through the generic pseudometric surface. -/
example (a : ℝ) : WassersteinDistance (dirac a).toGeneric (dirac a).toGeneric = 0 := by
  change dist (dirac a).toGeneric (dirac a).toGeneric = 0
  exact dist_self _

/-- Certified quantile couplings expose both marginals. -/
example {μ ν : P2Real} (qμ : QuantileRep μ) (qν : QuantileRep ν) :
    (quantileCoupling qμ qν).map Prod.fst = μ.measure ∧
      (quantileCoupling qμ qν).map Prod.snd = ν.measure :=
  ⟨quantileCoupling_fst qμ qν, quantileCoupling_snd qμ qν⟩

/-- Every real `P₂` law now has a canonical increasing uniform representation. -/
noncomputable example (μ : P2Real) : QuantileRep μ :=
  (canonicalQuantileSpec μ).toQuantileRep

/-- The canonical generalized inverse has the required uniform pushforward law. -/
example (μ : P2Real) : Measure.map (quantile μ) uniform01 = μ.measure :=
  quantile_pushforward μ
noncomputable example (ν : P2Real) (c : ℝ) (hc : MemLp (fun _ : ℝ => c) 2 ν.measure) :
    P2Real :=
  exponentialMap ν
    { field := fun _ => c
      field_aemeasurable := measurable_const.aemeasurable
      field_memLp := hc
      displacement_aemeasurable := (measurable_id.add measurable_const).aemeasurable
      displacement_monotone := by
        intro x y hxy
        linarith }

/-- A source-regularity map witness reconstructs the target through exp/log. -/
example {μ ν : P2Real} (T : OptimalTransportMap μ ν) :
    exponentialMap μ (logarithmicMap T) = ν :=
  exp_log_inverse T

/-- Atomic sources use a conditional plan interface rather than a forced map. -/
example {μ ν : P2Real} (π : ConditionalTransport μ ν) :
    π.joint.map Prod.snd = ν.measure :=
  π.target_marginal

/-- Under constant Dirac quantile certificates, the two-input barycenter is the midpoint Dirac. -/
example (a b : ℝ) (qa : QuantileRep (dirac a)) (qb : QuantileRep (dirac b))
    (ha : qa.value = fun _ => a) (hb : qb.value = fun _ => b) :
    (QuantileFamily.quantileBarycenter
      { items := [⟨dirac a, qa⟩, ⟨dirac b, qb⟩]
        nonempty := by simp }).measure = (dirac ((a + b) / 2)).measure := by
  let F : QuantileFamily :=
    { items := [⟨dirac a, qa⟩, ⟨dirac b, qb⟩]
      nonempty := by simp }
  have hmean : F.meanQuantile = fun _ => (a + b) / 2 := by
    funext u
    simp [F, QuantileFamily.meanQuantile, ha, hb]

  have htarget :
      (QuantileFamily.quantileBarycenter F).measure = (dirac ((a + b) / 2)).measure := by
    change Measure.map F.meanQuantile uniform01 = (dirac ((a + b) / 2)).measure
    rw [hmean]
    simp [P2Real.dirac_measure]
  simpa [F] using htarget

/-- The pointwise barycenter minimizes the completed quadratic quantile sum. -/
example {γ : P2Real} (F : QuantileFamily) (qγ : QuantileRep γ) (u : ℝ) :
    (F.items.map (fun q => (qγ.value u - q.2.value u) ^ 2)).sum =
      (F.items.length : ℝ) * (qγ.value u - QuantileFamily.meanQuantile F u) ^ 2 +
        (F.items.map (fun q =>
          (QuantileFamily.meanQuantile F u - q.2.value u) ^ 2)).sum :=
  QuantileFamily.frechet_square_completion F qγ u

/-- A constant-speed Dirac geodesic example is available once its distance certificate is proved. -/
example {μ ν : P2Real} (qμ : QuantileRep μ) (qν : QuantileRep ν)
    (h : GeodesicDistanceCertificate qμ qν) :
    Nonempty (Geodesic μ.toGeneric ν.toGeneric) :=
  geodesic_exists_of_quantile_certificate qμ qν h

end P2Real
end WassersteinGeometry
