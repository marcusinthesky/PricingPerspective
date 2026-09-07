import Mathlib.MeasureTheory.Function.LpSeminorm.TriangleInequality
import WassersteinGeometry.Real.Quantile

set_option linter.style.longLine false

open scoped ENNReal MeasureTheory
open MeasureTheory ProbabilityTheory

namespace WassersteinGeometry

namespace P2Real

/-- A finite nonempty family of real `P₂` measures carrying increasing uniform representations. -/
structure QuantileFamily where
  items : List (Σ μ : P2Real, QuantileRep μ)
  nonempty : items ≠ []

namespace QuantileFamily

/-- The pointwise average of the certified input representations. -/
noncomputable def meanQuantile (F : QuantileFamily) : ℝ → ℝ :=
  fun u => (F.items.map (fun q => q.2.value u)).sum / (F.items.length : ℝ)

private lemma measurable_sum_quantiles : ∀ xs : List (Σ μ : P2Real, QuantileRep μ),
    Measurable (fun u : ℝ => (xs.map (fun q => q.2.value u)).sum)
  | [] => measurable_const
  | q :: qs => by
      simpa [List.sum_cons, Pi.add_apply] using
        q.2.measurable.add (measurable_sum_quantiles qs)

private lemma monotone_sum_quantiles : ∀ xs : List (Σ μ : P2Real, QuantileRep μ),
    MonotoneOn (fun u : ℝ => (xs.map (fun q => q.2.value u)).sum) (Set.Ioo (0 : ℝ) 1)
  | [] => by
      intro x hx y hy hxy
      simp
  | q :: qs => by
      intro x hx y hy hxy
      simp only [List.map, List.sum_cons]
      exact add_le_add (q.2.monotoneOn hx hy hxy)
        (monotone_sum_quantiles qs hx hy hxy)

private lemma memLp_sum_quantiles : ∀ xs : List (Σ μ : P2Real, QuantileRep μ),
    MemLp (fun u : ℝ => (xs.map (fun q => q.2.value u)).sum) 2 uniform01
  | [] => by simpa using (MemLp.zero : MemLp (fun _ : ℝ => (0 : ℝ)) 2 uniform01)
  | q :: qs => by
      change MemLp (fun u : ℝ => q.2.value u + (qs.map (fun q => q.2.value u)).sum)
        2 uniform01
      exact q.2.memLp.add (memLp_sum_quantiles qs)

lemma meanQuantile_measurable (F : QuantileFamily) :
    Measurable F.meanQuantile := by
  unfold meanQuantile
  exact (measurable_sum_quantiles F.items).div_const _

lemma meanQuantile_monotoneOn (F : QuantileFamily) :
    MonotoneOn F.meanQuantile (Set.Ioo (0 : ℝ) 1) := by
  intro x hx y hy hxy
  unfold meanQuantile
  apply div_le_div_of_nonneg_right (monotone_sum_quantiles F.items hx hy hxy)
  exact Nat.cast_nonneg _

lemma meanQuantile_memLp (F : QuantileFamily) :
    MemLp F.meanQuantile 2 uniform01 := by
  unfold meanQuantile
  have hsum := memLp_sum_quantiles F.items
  have hscaled := hsum.const_mul ((F.items.length : ℝ)⁻¹)
  simpa [div_eq_mul_inv, mul_comm] using hscaled

/-- The quantile Fréchet barycenter as the pushforward of uniform measure by the average
representation.

This is a `P₂` construction before the measure-level minimization theorem is supplied.
-/
noncomputable def quantileBarycenter (F : QuantileFamily) : P2Real :=
  P2Real.ofMeasure (Measure.map F.meanQuantile uniform01)
    (by
      letI : IsProbabilityMeasure (Measure.map F.meanQuantile uniform01) :=
        Measure.isProbabilityMeasure_map (meanQuantile_measurable F).aemeasurable
      exact (inferInstance : IsProbabilityMeasure
        (Measure.map F.meanQuantile uniform01)).measure_univ)
    (by
      apply (memLp_map_measure_iff measurable_id.aestronglyMeasurable
        (meanQuantile_measurable F).aemeasurable).2
      simpa [Function.comp_def] using meanQuantile_memLp F)

/-- The averaged input representation is carried by the barycenter construction. -/
noncomputable def meanQuantileRep (F : QuantileFamily) :
    QuantileRep (quantileBarycenter F) :=
  { value := F.meanQuantile
    monotoneOn := meanQuantile_monotoneOn F
    measurable := meanQuantile_measurable F
    map_eq := by rfl
    memLp := meanQuantile_memLp F }

/-- Public Fréchet-mean name for the quantile barycenter construction. -/
noncomputable def FrechetMean (F : QuantileFamily) : P2Real :=
  quantileBarycenter F

/-- The finite-family Fréchet functional on real `P₂` measures. -/
noncomputable def frechetFunctional (F : QuantileFamily) (γ : P2Real) : ℝ≥0∞ :=
  (F.items.map (fun q => WassersteinDistanceSq γ.toGeneric q.1.toGeneric)).sum /
    (F.items.length : ℝ≥0∞)

lemma barycenter_quantile_balance (F : QuantileFamily) (u : ℝ) :
    (F.items.map (fun q => q.2.value u)).sum =
      (F.items.length : ℝ) * meanQuantile F u := by
  unfold meanQuantile
  have hlen : (F.items.length : ℝ) ≠ 0 := by
    have hlen_nat : F.items.length ≠ 0 :=
      Nat.ne_of_gt (List.length_pos_iff_ne_nil.mpr F.nonempty)
    exact_mod_cast hlen_nat
  field_simp

private lemma list_sum_sub {α : Type*} (xs : List α) (f : α → ℝ) (m : ℝ) :
    (xs.map (fun x => f x - m)).sum = (xs.map f).sum - (xs.length : ℝ) * m := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
      simp only [List.map, List.sum_cons, List.length_cons]
      rw [ih]
      push_cast
      ring


private lemma list_sum_sq_completion {α : Type*} (xs : List α) (f : α → ℝ) (z m : ℝ) :
    (xs.map (fun x => (z - f x) ^ 2)).sum =
      (xs.length : ℝ) * (z - m) ^ 2 +
        (xs.map (fun x => (m - f x) ^ 2)).sum +
          2 * (z - m) * (xs.map (fun x => m - f x)).sum := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
      simp only [List.map, List.sum_cons, List.length_cons]
      rw [ih]
      push_cast
      ring

/-- Pointwise square completion for the averaged quantile representation. -/
lemma frechet_square_completion {γ : P2Real} (F : QuantileFamily) (qγ : QuantileRep γ) (u : ℝ) :
    (F.items.map (fun q => (qγ.value u - q.2.value u) ^ 2)).sum =
      (F.items.length : ℝ) * (qγ.value u - meanQuantile F u) ^ 2 +
        (F.items.map (fun q => (meanQuantile F u - q.2.value u) ^ 2)).sum := by
  have hb :
      (F.items.map (fun q => q.2.value u - meanQuantile F u)).sum = 0 := by
    rw [list_sum_sub, barycenter_quantile_balance]
    ring
  have hbalance :
      (F.items.map (fun q => meanQuantile F u - q.2.value u)).sum = 0 := by
    have hneg_map :
        (F.items.map (fun q => -(q.2.value u - meanQuantile F u))).sum =
          -(F.items.map (fun q => q.2.value u - meanQuantile F u)).sum := by
      induction F.items with
      | nil => simp
      | cons q qs ih =>
          simp only [List.map, List.sum_cons]
          rw [ih]
          ring
    calc
      (F.items.map (fun q => meanQuantile F u - q.2.value u)).sum =
          (F.items.map (fun q => -(q.2.value u - meanQuantile F u))).sum := by
        induction F.items with
        | nil => simp
        | cons q qs ih =>
            simp only [List.map, List.sum_cons]
            rw [ih]
            congr 1
            ring
      _ = -(F.items.map (fun q => q.2.value u - meanQuantile F u)).sum := hneg_map
      _ = 0 := by rw [hb]; simp
  rw [list_sum_sq_completion]
  rw [hbalance]
  simp
/-- A candidate representation is a barycenter exactly when it equals the pointwise average
almost everywhere on the uniform parameter space.

The measure-level minimizer equivalence additionally requires the quantile W₂ formula and
monotone-rearrangement theorem.
-/
def IsQuantileBarycenter (F : QuantileFamily) (γ : P2Real) (qγ : QuantileRep γ) : Prop :=
  qγ.value =ᵐ[uniform01] meanQuantile F

lemma barycenter_tangent_balance (F : QuantileFamily) (u : ℝ) :
    (F.items.map (fun q => q.2.value u - meanQuantile F u)).sum = 0 := by
  rw [list_sum_sub, barycenter_quantile_balance]
  simp

end QuantileFamily

end P2Real

end WassersteinGeometry
