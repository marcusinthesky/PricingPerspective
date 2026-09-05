import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Positivity
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.MeasureTheory.Function.L2Space
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.Probability.Independence.Basic
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Analysis.Matrix.Order
import Mathlib.LinearAlgebra.Matrix.Hadamard
import Mathlib.Analysis.SpecificLimits.Basic
import EnergyStatistics.Defs
import EnergyStatistics.DistanceCovariance.Defs

set_option linter.style.longLine false

namespace EnergyStatistics

open Matrix in
/-- The negated empirically doubly-centered distance matrix `−Â` of a sample from a
    space of negative type is positive semidefinite: for any `v : Fin n → ℝ` the
    quadratic form `vᵀ(−Â)v` equals `−∑ᵢⱼ ṽᵢṽⱼ dᵢⱼ` with centered (zero-sum) weights
    `ṽ = v − (mean v)·1`, which is nonnegative by `DistNegativeType`.

    Reference: Székely & Rizzo (2023), Ch. 12; Lyons (2013), Thm 3.20. -/
private lemma neg_centered_dist_posSemidef {γ : Type*} [PseudoMetricSpace γ]
    (h_neg : DistNegativeType γ) {n : ℕ} (z : Fin n → γ) :
    Matrix.PosSemidef (Matrix.of fun i j : Fin n =>
      -(dist (z i) (z j) - (∑ k, dist (z i) (z k)) / n - (∑ k, dist (z k) (z j)) / n
        + (∑ k, ∑ l, dist (z k) (z l)) / (n ^ 2))) := by
  rcases Nat.eq_zero_or_pos n with hn0 | hnpos
  · subst hn0
    refine Matrix.PosSemidef.of_dotProduct_mulVec_nonneg ?_ fun v => ?_
    · ext i j; exact absurd i.2 (by simp)
    · simp [dotProduct]
  have hn : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hnpos.ne'
  refine Matrix.PosSemidef.of_dotProduct_mulVec_nonneg ?_ fun v => ?_
  · ext i j
    simp only [Matrix.conjTranspose_apply, Matrix.of_apply, star_trivial]
    have h1 : dist (z j) (z i) = dist (z i) (z j) := dist_comm _ _
    have h2 : (∑ k, dist (z j) (z k)) = ∑ k, dist (z k) (z j) :=
      Finset.sum_congr rfl fun k _ => dist_comm _ _
    have h3 : (∑ k, dist (z k) (z i)) = ∑ k, dist (z i) (z k) :=
      Finset.sum_congr rfl fun k _ => dist_comm _ _
    rw [h1, h2, h3]
    ring
  · -- quadratic form: reduces to the negative-type inequality with centered weights
    set s : ℝ := ∑ k, v k with hs
    set w : Fin n → ℝ := fun i => v i - s / n with hw
    have hw_sum : (∑ i, w i) = 0 := by
      simp only [hw, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ,
        Fintype.card_fin, nsmul_eq_mul, ← hs]
      field_simp
      ring
    have h_cnd := h_neg n z w hw_sum
    have key : star v ⬝ᵥ ((Matrix.of fun i j : Fin n =>
        -(dist (z i) (z j) - (∑ k, dist (z i) (z k)) / n - (∑ k, dist (z k) (z j)) / n
          + (∑ k, ∑ l, dist (z k) (z l)) / (n ^ 2))) *ᵥ v)
        = -(∑ i, ∑ j, w i * w j * dist (z i) (z j)) := by
      simp only [Matrix.mulVec, dotProduct, Matrix.of_apply, Pi.star_apply, star_trivial]
      set T := ∑ k, ∑ l, dist (z k) (z l) with hT
      set Q := ∑ i, ∑ j, v i * v j * dist (z i) (z j) with hQ
      set U := ∑ i, v i * ∑ k, dist (z i) (z k) with hU
      set W := ∑ j, v j * ∑ k, dist (z k) (z j) with hW
      -- evaluations of the four double sums arising from the centered matrix
      have hS2 : (∑ i, ∑ j, v i * v j * ((∑ k, dist (z i) (z k)) / n)) = s / n * U := by
        have h : ∀ i, (∑ j, v i * v j * ((∑ k, dist (z i) (z k)) / n))
            = s / n * (v i * ∑ k, dist (z i) (z k)) := by
          intro i
          calc (∑ j, v i * v j * ((∑ k, dist (z i) (z k)) / n))
              = ∑ j, v i * ((∑ k, dist (z i) (z k)) / n) * v j :=
                Finset.sum_congr rfl fun j _ => by ring
            _ = v i * ((∑ k, dist (z i) (z k)) / n) * ∑ j, v j := (Finset.mul_sum _ _ _).symm
            _ = s / n * (v i * ∑ k, dist (z i) (z k)) := by rw [← hs]; ring
        simp only [h]
        rw [← Finset.mul_sum, ← hU]
      have hS3 : (∑ i, ∑ j, v i * v j * ((∑ k, dist (z k) (z j)) / n)) = s / n * W := by
        rw [Finset.sum_comm]
        have h : ∀ j, (∑ i, v i * v j * ((∑ k, dist (z k) (z j)) / n))
            = s / n * (v j * ∑ k, dist (z k) (z j)) := by
          intro j
          calc (∑ i, v i * v j * ((∑ k, dist (z k) (z j)) / n))
              = ∑ i, v j * ((∑ k, dist (z k) (z j)) / n) * v i :=
                Finset.sum_congr rfl fun i _ => by ring
            _ = v j * ((∑ k, dist (z k) (z j)) / n) * ∑ i, v i := (Finset.mul_sum _ _ _).symm
            _ = s / n * (v j * ∑ k, dist (z k) (z j)) := by rw [← hs]; ring
        simp only [h]
        rw [← Finset.mul_sum, ← hW]
      have hS4 : (∑ i, ∑ j, v i * v j * (T / n ^ 2)) = T / n ^ 2 * (s * s) := by
        have h : ∀ i, (∑ j, v i * v j * (T / n ^ 2)) = T / n ^ 2 * s * v i := by
          intro i
          calc (∑ j, v i * v j * (T / n ^ 2))
              = ∑ j, v i * (T / n ^ 2) * v j := Finset.sum_congr rfl fun j _ => by ring
            _ = v i * (T / n ^ 2) * ∑ j, v j := (Finset.mul_sum _ _ _).symm
            _ = T / n ^ 2 * s * v i := by rw [← hs]; ring
        simp only [h]
        rw [← Finset.mul_sum, ← hs]
        ring
      -- evaluations of the double sums arising from the centered weights
      have hR2 : (∑ i, ∑ j, s / n * (v i * dist (z i) (z j))) = s / n * U := by
        have h : ∀ i, (∑ j, s / n * (v i * dist (z i) (z j)))
            = s / n * (v i * ∑ k, dist (z i) (z k)) := by
          intro i
          rw [← Finset.mul_sum, ← Finset.mul_sum]
        simp only [h]
        rw [← Finset.mul_sum, ← hU]
      have hR3 : (∑ i, ∑ j, s / n * (v j * dist (z i) (z j))) = s / n * W := by
        rw [Finset.sum_comm]
        have h : ∀ j, (∑ i, s / n * (v j * dist (z i) (z j)))
            = s / n * (v j * ∑ k, dist (z k) (z j)) := by
          intro j
          rw [← Finset.mul_sum, ← Finset.mul_sum]
        simp only [h]
        rw [← Finset.mul_sum, ← hW]
      have hR4 : (∑ i, ∑ j, (s / n) ^ 2 * dist (z i) (z j)) = (s / n) ^ 2 * T := by
        simp only [← Finset.mul_sum]
        rw [← hT]
      have hRHS : (∑ i, ∑ j, w i * w j * dist (z i) (z j))
          = Q - s / n * U - s / n * W + (s / n) ^ 2 * T := by
        have h2 : ∀ i j, w i * w j * dist (z i) (z j)
            = v i * v j * dist (z i) (z j) - s / n * (v i * dist (z i) (z j))
              - s / n * (v j * dist (z i) (z j)) + (s / n) ^ 2 * dist (z i) (z j) := by
          intro i j; simp only [hw]; ring
        simp only [h2, Finset.sum_add_distrib, Finset.sum_sub_distrib]
        rw [hR2, hR3, hR4, ← hQ]
      -- expand the left-hand side and assemble
      have h1 : ∀ i j : Fin n, v i * ((-(dist (z i) (z j) - (∑ k, dist (z i) (z k)) / n
            - (∑ k, dist (z k) (z j)) / n + T / n ^ 2)) * v j)
          = -(v i * v j * dist (z i) (z j))
            + v i * v j * ((∑ k, dist (z i) (z k)) / n)
            + v i * v j * ((∑ k, dist (z k) (z j)) / n)
            - v i * v j * (T / n ^ 2) := fun i j => by ring
      calc (∑ i, v i * ∑ j,
              (-(dist (z i) (z j) - (∑ k, dist (z i) (z k)) / n - (∑ k, dist (z k) (z j)) / n
                + T / n ^ 2)) * v j)
          = ∑ i, ∑ j, v i * ((-(dist (z i) (z j) - (∑ k, dist (z i) (z k)) / n
              - (∑ k, dist (z k) (z j)) / n + T / n ^ 2)) * v j) :=
            Finset.sum_congr rfl fun i _ => Finset.mul_sum _ _ _
        _ = ∑ i, ∑ j, (-(v i * v j * dist (z i) (z j))
              + v i * v j * ((∑ k, dist (z i) (z k)) / n)
              + v i * v j * ((∑ k, dist (z k) (z j)) / n)
              - v i * v j * (T / n ^ 2)) :=
            Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => h1 i j
        _ = -(∑ i, ∑ j, v i * v j * dist (z i) (z j))
              + (∑ i, ∑ j, v i * v j * ((∑ k, dist (z i) (z k)) / n))
              + (∑ i, ∑ j, v i * v j * ((∑ k, dist (z k) (z j)) / n))
              - (∑ i, ∑ j, v i * v j * (T / n ^ 2)) := by
            simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.sum_neg_distrib]
        _ = -Q + s / n * U + s / n * W - T / n ^ 2 * (s * s) := by
            rw [hS2, hS3, hS4, ← hQ]
        _ = -(Q - s / n * U - s / n * W + (s / n) ^ 2 * T) := by ring
        _ = -(∑ i, ∑ j, w i * w j * dist (z i) (z j)) := by rw [hRHS]
    rw [key]
    linarith

open Matrix in
/-- The Schur/Gram step of the discrete route to `dcov_nonneg`: the sum
    `∑ᵢⱼ Âᵢⱼ·B̂ᵢⱼ` of products of the empirically doubly-centered distance matrices of two
    samples from spaces of negative type is nonnegative.  Both `−Â` and `−B̂` are positive
    semidefinite (`neg_centered_dist_posSemidef`), so by the Schur product theorem
    (`Matrix.PosSemidef.hadamard`) their Hadamard product `Â ⊙ B̂ = (−Â) ⊙ (−B̂)` is
    positive semidefinite, and the claimed sum is its quadratic form at the all-ones vector.

    Reference: Székely & Rizzo (2023), Ch. 12; Lyons (2013), Thm 3.20
    (discrete Schur/Gram step). -/
lemma dcov_schur_sum_nonneg {α β : Type*}
    [PseudoMetricSpace α] [PseudoMetricSpace β]
    (h_negα : DistNegativeType α) (h_negβ : DistNegativeType β)
    {n : ℕ} (x : Fin n → α) (y : Fin n → β) :
    0 ≤ ∑ i, ∑ j,
      (dist (x i) (x j) - (∑ k, dist (x i) (x k)) / n - (∑ k, dist (x k) (x j)) / n
        + (∑ k, ∑ l, dist (x k) (x l)) / (n ^ 2))
      * (dist (y i) (y j) - (∑ k, dist (y i) (y k)) / n - (∑ k, dist (y k) (y j)) / n
        + (∑ k, ∑ l, dist (y k) (y l)) / (n ^ 2)) := by
  classical
  have hA := neg_centered_dist_posSemidef h_negα x
  have hB := neg_centered_dist_posSemidef h_negβ y
  have hAB := hA.hadamard hB
  have h1 := hAB.dotProduct_mulVec_nonneg fun _ => (1 : ℝ)
  have hsum : star (fun _ : Fin n => (1 : ℝ)) ⬝ᵥ
      (((Matrix.of fun i j : Fin n =>
          -(dist (x i) (x j) - (∑ k, dist (x i) (x k)) / n - (∑ k, dist (x k) (x j)) / n
            + (∑ k, ∑ l, dist (x k) (x l)) / (n ^ 2))) ⊙
        (Matrix.of fun i j : Fin n =>
          -(dist (y i) (y j) - (∑ k, dist (y i) (y k)) / n - (∑ k, dist (y k) (y j)) / n
            + (∑ k, ∑ l, dist (y k) (y l)) / (n ^ 2)))) *ᵥ fun _ => (1 : ℝ))
      = ∑ i, ∑ j,
        (dist (x i) (x j) - (∑ k, dist (x i) (x k)) / n - (∑ k, dist (x k) (x j)) / n
          + (∑ k, ∑ l, dist (x k) (x l)) / (n ^ 2))
        * (dist (y i) (y j) - (∑ k, dist (y i) (y k)) / n - (∑ k, dist (y k) (y j)) / n
          + (∑ k, ∑ l, dist (y k) (y l)) / (n ^ 2)) := by
    simp only [Matrix.mulVec, dotProduct, Pi.star_apply, star_trivial,
      Matrix.hadamard_apply, Matrix.of_apply, mul_one, one_mul]
    exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => neg_mul_neg _ _
  rw [hsum] at h1
  exact h1

/-- Cauchy–Schwarz inequality for real L² integrals over an arbitrary measure.

    (∫ f·g)² ≤ (∫ f²)·(∫ g²)

    Proof via the discriminant of the quadratic t ↦ ∫(t·f + g)² ≥ 0
    (nonnegativity of an integral of a nonnegative function).

    Reference: Székely & Rizzo (2023), Chapter 12 (the Cauchy–Schwarz
    machinery for the distance correlation bound). -/
lemma integral_mul_sq_le {α : Type*} [MeasurableSpace α] {μ : MeasureTheory.Measure α}
    (f g : α → ℝ)
    (hf2 : MeasureTheory.Integrable (fun x => f x ^ 2) μ)
    (hg2 : MeasureTheory.Integrable (fun x => g x ^ 2) μ)
    (hfg : MeasureTheory.Integrable (fun x => f x * g x) μ) :
    (∫ x, f x * g x ∂μ) ^ 2 ≤ (∫ x, f x ^ 2 ∂μ) * ∫ x, g x ^ 2 ∂μ := by
  set a := ∫ x, f x ^ 2 ∂μ
  set b := ∫ x, f x * g x ∂μ
  set c := ∫ x, g x ^ 2 ∂μ
  have h_nonneg : ∀ t : ℝ, 0 ≤ a * (t * t) + (2 * b) * t + c := by
    intro t
    have h_int_sq : MeasureTheory.Integrable (fun x => (t • f x + g x) ^ 2) μ := by
      have : (fun x => (t • f x + g x) ^ 2) = fun x =>
          t ^ 2 * (f x ^ 2) + (2 * t) * (f x * g x) + (g x ^ 2) := by
        ext x; ring
      rw [this]
      exact ((hf2.const_mul (t ^ 2)).add (hfg.const_mul (2 * t))).add hg2
    have h_int_nonneg : 0 ≤ ∫ x, (t • f x + g x) ^ 2 ∂μ :=
      MeasureTheory.integral_nonneg (fun x => sq_nonneg _)
    have h_expand : (∫ x, (t • f x + g x) ^ 2 ∂μ) = a * (t * t) + (2 * b) * t + c := by
      calc
        (∫ x, (t • f x + g x) ^ 2 ∂μ) =
            (∫ x, t ^ 2 * (f x ^ 2) + (2 * t) * (f x * g x) + (g x ^ 2) ∂μ) := by
          refine MeasureTheory.integral_congr_ae ?_
          filter_upwards with x
          ring
        _ = (∫ x, t ^ 2 * (f x ^ 2) + (2 * t) * (f x * g x) ∂μ) + (∫ x, g x ^ 2 ∂μ) :=
          MeasureTheory.integral_add ((hf2.const_mul (t ^ 2)).add (hfg.const_mul (2 * t))) hg2
        _ = ((∫ x, t ^ 2 * (f x ^ 2) ∂μ) + (∫ x, (2 * t) * (f x * g x) ∂μ)) + (∫ x, g x ^ 2 ∂μ) := by
          rw [MeasureTheory.integral_add (hf2.const_mul (t ^ 2)) (hfg.const_mul (2 * t))]
        _ = t ^ 2 * (∫ x, f x ^ 2 ∂μ) + (2 * t) * (∫ x, f x * g x ∂μ) + (∫ x, g x ^ 2 ∂μ) := by
          simp [MeasureTheory.integral_const_mul]
        _ = a * (t * t) + (2 * b) * t + c := by
          simp [a, b, c]
          ring
    linarith
  have h_discrim := discrim_le_zero h_nonneg
  unfold discrim at h_discrim
  nlinarith

/-- Jensen inequality for the square of the mean over a probability measure:
    (E[f])² ≤ E[f²].

    Follows from `integral_mul_sq_le` with g ≡ 1.

    Reference: Székely & Rizzo (2023), Chapter 12. -/
lemma sq_integral_le {α : Type*} [MeasurableSpace α] {μ : MeasureTheory.Measure α}
    [MeasureTheory.IsProbabilityMeasure μ] (f : α → ℝ)
    (hf : MeasureTheory.Integrable f μ)
    (hf2 : MeasureTheory.Integrable (fun x => f x ^ 2) μ) :
    (∫ x, f x ∂μ) ^ 2 ≤ ∫ x, f x ^ 2 ∂μ := by
  have h_one_sq : MeasureTheory.Integrable (fun _ : α => (1 : ℝ) ^ 2) μ :=
    MeasureTheory.integrable_const (c := (1 : ℝ) ^ 2)
  have h_f_mul_one : MeasureTheory.Integrable (fun x => f x * (1 : ℝ)) μ := by
    simpa using hf
  have h := integral_mul_sq_le f (fun _ => 1) hf2 h_one_sq h_f_mul_one
  -- h : (∫ x, f x * 1 ∂μ)^2 ≤ (∫ x, f x ^ 2 ∂μ) * (∫ x, 1^2 ∂μ)
  -- Simplify: (∫ f)^2 ≤ (∫ f^2) * 1 = (∫ f^2)
  have h_simp : (∫ x, f x * (1 : ℝ) ∂μ) = (∫ x, f x ∂μ) := by simp
  have h_one_int : (∫ x, (1 : ℝ) ^ 2 ∂μ) = (1 : ℝ) := by simp
  simpa [h_simp, h_one_int] using h

/-- Squared distance variance equals the L²(π⊗π) norm of the doubly-centered
    distance kernel (U-centered representation).

    `dVar²(X) = ∫ (centeredDistFst)² d(π⊗π)`.

    The proof expands `(d − a(Z) − a(Z') + D)²` over the product measure
    `π ⊗ π`, with `a p = ∫ q, dist p.1 q.1 ∂π` and
    `D = ∫∫ dist p.1 q.1`, grouping it as `(u − v)²` with `u = d − a(Z)` and
    `v = a(Z') − D`.  Each resulting expectation is evaluated via Fubini
    (`integral_prod`), the marginal identities (`integral_fun_fst`,
    `integral_fun_snd`), independence across the product coordinates
    (`integral_prod_mul`) and `dist_comm`.  All measurability/integrability is
    extracted from the `Integrable` fields of `hm` (there is no Borel link
    between the metric and the ambient measurable space).

    Reference: Székely & Rizzo (2023), Ch. 12 (U-centered representation). -/
lemma dvar_eq_integral_sq {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    [PseudoMetricSpace α] [PseudoMetricSpace β]
    (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π) :
    distanceVarianceSq π =
      ∫ z, centeredDistFst π z ^ 2 ∂(π.measure.prod π.measure) := by
  -- `g p = ∫ q, dist p.1 q.1 ∂π` is the marginal mean-distance function.
  set g : (α × β) → ℝ := fun p => ∫ q, dist p.1 q.1 ∂π.measure with hgdef
  -- Integrability of `g` and `g²` over `π`.
  have hg_int : MeasureTheory.Integrable g π.measure := hm.fst.integral_prod_left
  have hgsq_int : MeasureTheory.Integrable (fun p => g p ^ 2) π.measure := by
    have hbound : MeasureTheory.Integrable
        (fun p => ∫ q, dist p.1 q.1 ^ 2 ∂π.measure) π.measure := hm.fstSq.integral_prod_left
    have hmeas : MeasureTheory.AEStronglyMeasurable (fun p => g p ^ 2) π.measure := by
      refine (hg_int.aestronglyMeasurable.mul hg_int.aestronglyMeasurable).congr ?_
      filter_upwards with p
      simp [pow_two]
    refine hbound.mono' hmeas ?_
    filter_upwards [hm.fst.prod_right_ae, hm.fstSq.prod_right_ae] with p hp1 hp2
    have hj := sq_integral_le (fun q : α × β => dist p.1 q.1) hp1 hp2
    calc ‖g p ^ 2‖ = g p ^ 2 := by rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg (g p))]
      _ ≤ ∫ q, dist p.1 q.1 ^ 2 ∂π.measure := hj
  -- Lifts of `g`, `g²` and the raw distance to the product `π ⊗ π`.
  have hmd : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => dist z.1.1 z.2.1) (π.measure.prod π.measure) := hm.fst
  have hmdsq : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => dist z.1.1 z.2.1 ^ 2) (π.measure.prod π.measure) := hm.fstSq
  have hga : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => g z.1) (π.measure.prod π.measure) := hg_int.comp_fst π.measure
  have hgb : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => g z.2) (π.measure.prod π.measure) := hg_int.comp_snd π.measure
  have hgasq : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => g z.1 ^ 2) (π.measure.prod π.measure) :=
    hgsq_int.comp_fst π.measure
  have hgbsq : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => g z.2 ^ 2) (π.measure.prod π.measure) :=
    hgsq_int.comp_snd π.measure
  have hgagb : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => g z.1 * g z.2) (π.measure.prod π.measure) :=
    hg_int.mul_prod hg_int
  have hmdga : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => dist z.1.1 z.2.1 * g z.1) (π.measure.prod π.measure) := by
    refine MeasureTheory.Integrable.mono' ((hmdsq.add hgasq).div_const 2)
      (hmd.aestronglyMeasurable.mul hga.aestronglyMeasurable) ?_
    filter_upwards with z
    have ha : (0 : ℝ) ≤ dist z.1.1 z.2.1 := dist_nonneg
    have hb : (0 : ℝ) ≤ g z.1 := by
      show (0 : ℝ) ≤ ∫ q, dist z.1.1 q.1 ∂π.measure
      exact MeasureTheory.integral_nonneg fun q => dist_nonneg
    rw [Real.norm_eq_abs, abs_of_nonneg (mul_nonneg ha hb)]
    simp only [Pi.add_apply]
    nlinarith [two_mul_le_add_sq (dist z.1.1 z.2.1) (g z.1)]
  have hmdgb : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => dist z.1.1 z.2.1 * g z.2) (π.measure.prod π.measure) := by
    refine MeasureTheory.Integrable.mono' ((hmdsq.add hgbsq).div_const 2)
      (hmd.aestronglyMeasurable.mul hgb.aestronglyMeasurable) ?_
    filter_upwards with z
    have ha : (0 : ℝ) ≤ dist z.1.1 z.2.1 := dist_nonneg
    have hb : (0 : ℝ) ≤ g z.2 := by
      show (0 : ℝ) ≤ ∫ q, dist z.2.1 q.1 ∂π.measure
      exact MeasureTheory.integral_nonneg fun q => dist_nonneg
    rw [Real.norm_eq_abs, abs_of_nonneg (mul_nonneg ha hb)]
    simp only [Pi.add_apply]
    nlinarith [two_mul_le_add_sq (dist z.1.1 z.2.1) (g z.2)]
  -- Unfold the target of the equality and abbreviate the three aggregate terms.
  rw [distanceVarianceSq]
  set S₁ := ∫ p, ∫ q, dist p.1 q.1 ^ 2 ∂π.measure ∂π.measure with hS₁
  set Dm := ∫ p, ∫ q, dist p.1 q.1 ∂π.measure ∂π.measure with hDm
  set A₂ := ∫ p, (∫ q, dist p.1 q.1 ∂π.measure) ^ 2 ∂π.measure with hA₂
  -- Atomic expectations.
  have hV_md2 : (∫ z, dist z.1.1 z.2.1 ^ 2 ∂(π.measure.prod π.measure)) = S₁ := by
    rw [hS₁]; exact MeasureTheory.integral_prod _ hm.fstSq
  have hV_md : (∫ z, dist z.1.1 z.2.1 ∂(π.measure.prod π.measure)) = Dm := by
    rw [hDm]; exact MeasureTheory.integral_prod _ hm.fst
  have hV_ga : (∫ z, g z.1 ∂(π.measure.prod π.measure)) = Dm := by
    have h : (∫ z, g z.1 ∂(π.measure.prod π.measure))
        = (π.measure).real Set.univ • ∫ x, g x ∂π.measure :=
      MeasureTheory.integral_fun_fst g
    rw [h]; simp only [MeasureTheory.probReal_univ, one_smul, hgdef]; exact hDm.symm
  have hV_gb : (∫ z, g z.2 ∂(π.measure.prod π.measure)) = Dm := by
    have h : (∫ z, g z.2 ∂(π.measure.prod π.measure))
        = (π.measure).real Set.univ • ∫ y, g y ∂π.measure :=
      MeasureTheory.integral_fun_snd g
    rw [h]; simp only [MeasureTheory.probReal_univ, one_smul, hgdef]; exact hDm.symm
  have hV_gasq : (∫ z, g z.1 ^ 2 ∂(π.measure.prod π.measure)) = A₂ := by
    have h : (∫ z, g z.1 ^ 2 ∂(π.measure.prod π.measure))
        = (π.measure).real Set.univ • ∫ x, g x ^ 2 ∂π.measure :=
      MeasureTheory.integral_fun_fst (fun p => g p ^ 2)
    rw [h]; simp only [MeasureTheory.probReal_univ, one_smul, hgdef]; exact hA₂.symm
  have hV_gbsq : (∫ z, g z.2 ^ 2 ∂(π.measure.prod π.measure)) = A₂ := by
    have h : (∫ z, g z.2 ^ 2 ∂(π.measure.prod π.measure))
        = (π.measure).real Set.univ • ∫ y, g y ^ 2 ∂π.measure :=
      MeasureTheory.integral_fun_snd (fun p => g p ^ 2)
    rw [h]; simp only [MeasureTheory.probReal_univ, one_smul, hgdef]; exact hA₂.symm
  have hV_gagb : (∫ z, g z.1 * g z.2 ∂(π.measure.prod π.measure)) = Dm * Dm := by
    rw [MeasureTheory.integral_prod_mul]
  have hV_mdga : (∫ z, dist z.1.1 z.2.1 * g z.1 ∂(π.measure.prod π.measure)) = A₂ := by
    rw [MeasureTheory.integral_prod _ hmdga]
    have hinner : ∀ x, (∫ y, dist x.1 y.1 * g x ∂π.measure) = g x ^ 2 := by
      intro x
      rw [MeasureTheory.integral_mul_const,
        show (∫ y, dist x.1 y.1 ∂π.measure) = g x from rfl]
      ring
    simp_rw [hinner]
    rw [hA₂]
  have hV_mdgb : (∫ z, dist z.1.1 z.2.1 * g z.2 ∂(π.measure.prod π.measure)) = A₂ := by
    rw [MeasureTheory.integral_prod_symm _ hmdgb]
    have hinner : ∀ y, (∫ x, dist x.1 y.1 * g y ∂π.measure) = g y ^ 2 := by
      intro y
      rw [MeasureTheory.integral_mul_const]
      have hcomm : (∫ x, dist x.1 y.1 ∂π.measure) = g y := by
        show (∫ x, dist x.1 y.1 ∂π.measure) = ∫ q, dist y.1 q.1 ∂π.measure
        refine MeasureTheory.integral_congr_ae ?_
        filter_upwards with x
        rw [dist_comm]
      rw [hcomm]; ring
    simp_rw [hinner]
    rw [hA₂]
  -- Pointwise rewrite of the centered kernel and expansion of its square.
  have hc : ∀ z : (α × β) × (α × β),
      centeredDistFst π z = dist z.1.1 z.2.1 - g z.1 - g z.2 + Dm := by
    intro z; simp only [centeredDistFst, hgdef, hDm]
  have hExpand : ∀ z : (α × β) × (α × β),
      centeredDistFst π z ^ 2
        = (dist z.1.1 z.2.1 - g z.1) ^ 2
          - 2 * ((dist z.1.1 z.2.1 - g z.1) * (g z.2 - Dm))
          + (g z.2 - Dm) ^ 2 := by
    intro z; rw [hc z]; ring
  -- Integrability of the three grouped terms.
  have hu2i : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => (dist z.1.1 z.2.1 - g z.1) ^ 2) (π.measure.prod π.measure) := by
    have hEq : (fun z : (α × β) × (α × β) => (dist z.1.1 z.2.1 - g z.1) ^ 2)
        = fun z => dist z.1.1 z.2.1 ^ 2 - 2 * (dist z.1.1 z.2.1 * g z.1) + g z.1 ^ 2 := by
      funext z; ring
    rw [hEq]; exact (hmdsq.sub (hmdga.const_mul 2)).add hgasq
  have hv2i : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => (g z.2 - Dm) ^ 2) (π.measure.prod π.measure) := by
    have hEq : (fun z : (α × β) × (α × β) => (g z.2 - Dm) ^ 2)
        = fun z => g z.2 ^ 2 - 2 * Dm * g z.2 + Dm ^ 2 := by
      funext z; ring
    rw [hEq]
    exact (hgbsq.sub (hgb.const_mul (2 * Dm))).add (MeasureTheory.integrable_const (Dm ^ 2))
  have huvi : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => (dist z.1.1 z.2.1 - g z.1) * (g z.2 - Dm))
      (π.measure.prod π.measure) := by
    have hEq : (fun z : (α × β) × (α × β) => (dist z.1.1 z.2.1 - g z.1) * (g z.2 - Dm))
        = fun z => dist z.1.1 z.2.1 * g z.2 - Dm * dist z.1.1 z.2.1
                    - g z.1 * g z.2 + Dm * g z.1 := by
      funext z; ring
    rw [hEq]
    exact ((hmdgb.sub (hmd.const_mul Dm)).sub hgagb).add (hga.const_mul Dm)
  -- Values of the three grouped integrals.  Each linearity step is stated as a
  -- `have` with a flat (defeq-checked) statement so that `rw` can match it.
  have hu2v : (∫ z, (dist z.1.1 z.2.1 - g z.1) ^ 2 ∂(π.measure.prod π.measure)) = S₁ - A₂ := by
    have hcong : (∫ z, (dist z.1.1 z.2.1 - g z.1) ^ 2 ∂(π.measure.prod π.measure))
        = ∫ z, dist z.1.1 z.2.1 ^ 2 - 2 * (dist z.1.1 z.2.1 * g z.1) + g z.1 ^ 2
            ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_congr_ae (Filter.Eventually.of_forall fun z => by ring)
    have e1 : (∫ z, dist z.1.1 z.2.1 ^ 2 - 2 * (dist z.1.1 z.2.1 * g z.1) + g z.1 ^ 2
          ∂(π.measure.prod π.measure))
        = (∫ z, dist z.1.1 z.2.1 ^ 2 - 2 * (dist z.1.1 z.2.1 * g z.1) ∂(π.measure.prod π.measure))
          + ∫ z, g z.1 ^ 2 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_add (hmdsq.sub (hmdga.const_mul 2)) hgasq
    have e2 : (∫ z, dist z.1.1 z.2.1 ^ 2 - 2 * (dist z.1.1 z.2.1 * g z.1) ∂(π.measure.prod π.measure))
        = (∫ z, dist z.1.1 z.2.1 ^ 2 ∂(π.measure.prod π.measure))
          - ∫ z, 2 * (dist z.1.1 z.2.1 * g z.1) ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_sub hmdsq (hmdga.const_mul 2)
    have e3 : (∫ z, 2 * (dist z.1.1 z.2.1 * g z.1) ∂(π.measure.prod π.measure))
        = 2 * ∫ z, dist z.1.1 z.2.1 * g z.1 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_const_mul 2 _
    rw [hcong, e1, e2, e3, hV_md2, hV_mdga, hV_gasq]; ring
  have hv2v : (∫ z, (g z.2 - Dm) ^ 2 ∂(π.measure.prod π.measure)) = A₂ - Dm ^ 2 := by
    have hcong : (∫ z, (g z.2 - Dm) ^ 2 ∂(π.measure.prod π.measure))
        = ∫ z, g z.2 ^ 2 - 2 * Dm * g z.2 + Dm ^ 2 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_congr_ae (Filter.Eventually.of_forall fun z => by ring)
    have e1 : (∫ z, g z.2 ^ 2 - 2 * Dm * g z.2 + Dm ^ 2 ∂(π.measure.prod π.measure))
        = (∫ z, g z.2 ^ 2 - 2 * Dm * g z.2 ∂(π.measure.prod π.measure))
          + ∫ z, Dm ^ 2 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_add (hgbsq.sub (hgb.const_mul (2 * Dm)))
        (MeasureTheory.integrable_const (Dm ^ 2))
    have e2 : (∫ z, g z.2 ^ 2 - 2 * Dm * g z.2 ∂(π.measure.prod π.measure))
        = (∫ z, g z.2 ^ 2 ∂(π.measure.prod π.measure))
          - ∫ z, 2 * Dm * g z.2 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_sub hgbsq (hgb.const_mul (2 * Dm))
    have e3 : (∫ z, 2 * Dm * g z.2 ∂(π.measure.prod π.measure))
        = 2 * Dm * ∫ z, g z.2 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_const_mul (2 * Dm) _
    have e4 : (∫ z, Dm ^ 2 ∂(π.measure.prod π.measure)) = Dm ^ 2 := by
      rw [MeasureTheory.integral_const, MeasureTheory.probReal_univ, one_smul]
    rw [hcong, e1, e2, e3, e4, hV_gbsq, hV_gb]; ring
  have huvv : (∫ z, (dist z.1.1 z.2.1 - g z.1) * (g z.2 - Dm) ∂(π.measure.prod π.measure))
      = A₂ - Dm ^ 2 := by
    have hcong : (∫ z, (dist z.1.1 z.2.1 - g z.1) * (g z.2 - Dm) ∂(π.measure.prod π.measure))
        = ∫ z, dist z.1.1 z.2.1 * g z.2 - Dm * dist z.1.1 z.2.1
              - g z.1 * g z.2 + Dm * g z.1 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_congr_ae (Filter.Eventually.of_forall fun z => by ring)
    have e1 : (∫ z, dist z.1.1 z.2.1 * g z.2 - Dm * dist z.1.1 z.2.1
              - g z.1 * g z.2 + Dm * g z.1 ∂(π.measure.prod π.measure))
        = (∫ z, dist z.1.1 z.2.1 * g z.2 - Dm * dist z.1.1 z.2.1 - g z.1 * g z.2
              ∂(π.measure.prod π.measure))
          + ∫ z, Dm * g z.1 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_add ((hmdgb.sub (hmd.const_mul Dm)).sub hgagb) (hga.const_mul Dm)
    have e2 : (∫ z, dist z.1.1 z.2.1 * g z.2 - Dm * dist z.1.1 z.2.1 - g z.1 * g z.2
            ∂(π.measure.prod π.measure))
        = (∫ z, dist z.1.1 z.2.1 * g z.2 - Dm * dist z.1.1 z.2.1 ∂(π.measure.prod π.measure))
          - ∫ z, g z.1 * g z.2 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_sub (hmdgb.sub (hmd.const_mul Dm)) hgagb
    have e3 : (∫ z, dist z.1.1 z.2.1 * g z.2 - Dm * dist z.1.1 z.2.1 ∂(π.measure.prod π.measure))
        = (∫ z, dist z.1.1 z.2.1 * g z.2 ∂(π.measure.prod π.measure))
          - ∫ z, Dm * dist z.1.1 z.2.1 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_sub hmdgb (hmd.const_mul Dm)
    have e4 : (∫ z, Dm * dist z.1.1 z.2.1 ∂(π.measure.prod π.measure))
        = Dm * ∫ z, dist z.1.1 z.2.1 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_const_mul Dm _
    have e5 : (∫ z, Dm * g z.1 ∂(π.measure.prod π.measure))
        = Dm * ∫ z, g z.1 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_const_mul Dm _
    rw [hcong, e1, e2, e3, e4, e5, hV_mdgb, hV_md, hV_gagb, hV_ga]; ring
  -- Assemble.
  have hmain : (∫ z, centeredDistFst π z ^ 2 ∂(π.measure.prod π.measure))
      = S₁ + Dm ^ 2 - 2 * A₂ := by
    have hcong : (∫ z, centeredDistFst π z ^ 2 ∂(π.measure.prod π.measure))
        = ∫ z, ((dist z.1.1 z.2.1 - g z.1) ^ 2
                - 2 * ((dist z.1.1 z.2.1 - g z.1) * (g z.2 - Dm))
                + (g z.2 - Dm) ^ 2) ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_congr_ae (Filter.Eventually.of_forall hExpand)
    have e1 : (∫ z, ((dist z.1.1 z.2.1 - g z.1) ^ 2
                - 2 * ((dist z.1.1 z.2.1 - g z.1) * (g z.2 - Dm))
                + (g z.2 - Dm) ^ 2) ∂(π.measure.prod π.measure))
        = (∫ z, (dist z.1.1 z.2.1 - g z.1) ^ 2
                - 2 * ((dist z.1.1 z.2.1 - g z.1) * (g z.2 - Dm)) ∂(π.measure.prod π.measure))
          + ∫ z, (g z.2 - Dm) ^ 2 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_add (hu2i.sub (huvi.const_mul 2)) hv2i
    have e2 : (∫ z, (dist z.1.1 z.2.1 - g z.1) ^ 2
                - 2 * ((dist z.1.1 z.2.1 - g z.1) * (g z.2 - Dm)) ∂(π.measure.prod π.measure))
        = (∫ z, (dist z.1.1 z.2.1 - g z.1) ^ 2 ∂(π.measure.prod π.measure))
          - ∫ z, 2 * ((dist z.1.1 z.2.1 - g z.1) * (g z.2 - Dm)) ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_sub hu2i (huvi.const_mul 2)
    have e3 : (∫ z, 2 * ((dist z.1.1 z.2.1 - g z.1) * (g z.2 - Dm)) ∂(π.measure.prod π.measure))
        = 2 * ∫ z, (dist z.1.1 z.2.1 - g z.1) * (g z.2 - Dm) ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_const_mul 2 _
    rw [hcong, e1, e2, e3, hu2v, huvv, hv2v]; ring
  rw [hmain]

/-- Squared distance variance of the second component equals the L²(π⊗π) norm of the
    doubly-centered distance kernel (U-centered representation).

    `dVar²(Y) = ∫ (centeredDistSnd)² d(π⊗π)`.

    Mirror of `dvar_eq_integral_sq` for the second coordinate.

    Reference: Székely & Rizzo (2023), Ch. 12 (U-centered representation). -/
lemma dvarSnd_eq_integral_sq {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    [PseudoMetricSpace α] [PseudoMetricSpace β]
    (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π) :
    distanceVarianceSqSnd π =
      ∫ z, centeredDistSnd π z ^ 2 ∂(π.measure.prod π.measure) := by
  -- `g p = ∫ q, dist p.2 q.2 ∂π` is the marginal mean-distance function.
  set g : (α × β) → ℝ := fun p => ∫ q, dist p.2 q.2 ∂π.measure with hgdef
  -- Integrability of `g` and `g²` over `π`.
  have hg_int : MeasureTheory.Integrable g π.measure := hm.snd.integral_prod_left
  have hgsq_int : MeasureTheory.Integrable (fun p => g p ^ 2) π.measure := by
    have hbound : MeasureTheory.Integrable
        (fun p => ∫ q, dist p.2 q.2 ^ 2 ∂π.measure) π.measure := hm.sndSq.integral_prod_left
    have hmeas : MeasureTheory.AEStronglyMeasurable (fun p => g p ^ 2) π.measure := by
      refine (hg_int.aestronglyMeasurable.mul hg_int.aestronglyMeasurable).congr ?_
      filter_upwards with p
      simp [pow_two]
    refine hbound.mono' hmeas ?_
    filter_upwards [hm.snd.prod_right_ae, hm.sndSq.prod_right_ae] with p hp1 hp2
    have hj := sq_integral_le (fun q : α × β => dist p.2 q.2) hp1 hp2
    calc ‖g p ^ 2‖ = g p ^ 2 := by rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg (g p))]
      _ ≤ ∫ q, dist p.2 q.2 ^ 2 ∂π.measure := hj
  -- Lifts of `g`, `g²` and the raw distance to the product `π ⊗ π`.
  have hmd : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => dist z.1.2 z.2.2) (π.measure.prod π.measure) := hm.snd
  have hmdsq : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => dist z.1.2 z.2.2 ^ 2) (π.measure.prod π.measure) := hm.sndSq
  have hga : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => g z.1) (π.measure.prod π.measure) := hg_int.comp_fst π.measure
  have hgb : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => g z.2) (π.measure.prod π.measure) := hg_int.comp_snd π.measure
  have hgasq : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => g z.1 ^ 2) (π.measure.prod π.measure) :=
    hgsq_int.comp_fst π.measure
  have hgbsq : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => g z.2 ^ 2) (π.measure.prod π.measure) :=
    hgsq_int.comp_snd π.measure
  have hgagb : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => g z.1 * g z.2) (π.measure.prod π.measure) :=
    hg_int.mul_prod hg_int
  have hmdga : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => dist z.1.2 z.2.2 * g z.1) (π.measure.prod π.measure) := by
    refine MeasureTheory.Integrable.mono' ((hmdsq.add hgasq).div_const 2)
      (hmd.aestronglyMeasurable.mul hga.aestronglyMeasurable) ?_
    filter_upwards with z
    have ha : (0 : ℝ) ≤ dist z.1.2 z.2.2 := dist_nonneg
    have hb : (0 : ℝ) ≤ g z.1 := by
      show (0 : ℝ) ≤ ∫ q, dist z.1.2 q.2 ∂π.measure
      exact MeasureTheory.integral_nonneg fun q => dist_nonneg
    rw [Real.norm_eq_abs, abs_of_nonneg (mul_nonneg ha hb)]
    simp only [Pi.add_apply]
    nlinarith [two_mul_le_add_sq (dist z.1.2 z.2.2) (g z.1)]
  have hmdgb : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => dist z.1.2 z.2.2 * g z.2) (π.measure.prod π.measure) := by
    refine MeasureTheory.Integrable.mono' ((hmdsq.add hgbsq).div_const 2)
      (hmd.aestronglyMeasurable.mul hgb.aestronglyMeasurable) ?_
    filter_upwards with z
    have ha : (0 : ℝ) ≤ dist z.1.2 z.2.2 := dist_nonneg
    have hb : (0 : ℝ) ≤ g z.2 := by
      show (0 : ℝ) ≤ ∫ q, dist z.2.2 q.2 ∂π.measure
      exact MeasureTheory.integral_nonneg fun q => dist_nonneg
    rw [Real.norm_eq_abs, abs_of_nonneg (mul_nonneg ha hb)]
    simp only [Pi.add_apply]
    nlinarith [two_mul_le_add_sq (dist z.1.2 z.2.2) (g z.2)]
  -- Unfold the target of the equality and abbreviate the three aggregate terms.
  rw [distanceVarianceSqSnd]
  set S₁ := ∫ p, ∫ q, dist p.2 q.2 ^ 2 ∂π.measure ∂π.measure with hS₁
  set Dm := ∫ p, ∫ q, dist p.2 q.2 ∂π.measure ∂π.measure with hDm
  set A₂ := ∫ p, (∫ q, dist p.2 q.2 ∂π.measure) ^ 2 ∂π.measure with hA₂
  -- Atomic expectations.
  have hV_md2 : (∫ z, dist z.1.2 z.2.2 ^ 2 ∂(π.measure.prod π.measure)) = S₁ := by
    rw [hS₁]; exact MeasureTheory.integral_prod _ hm.sndSq
  have hV_md : (∫ z, dist z.1.2 z.2.2 ∂(π.measure.prod π.measure)) = Dm := by
    rw [hDm]; exact MeasureTheory.integral_prod _ hm.snd
  have hV_ga : (∫ z, g z.1 ∂(π.measure.prod π.measure)) = Dm := by
    have h : (∫ z, g z.1 ∂(π.measure.prod π.measure))
        = (π.measure).real Set.univ • ∫ x, g x ∂π.measure :=
      MeasureTheory.integral_fun_fst g
    rw [h]; simp only [MeasureTheory.probReal_univ, one_smul, hgdef]; exact hDm.symm
  have hV_gb : (∫ z, g z.2 ∂(π.measure.prod π.measure)) = Dm := by
    have h : (∫ z, g z.2 ∂(π.measure.prod π.measure))
        = (π.measure).real Set.univ • ∫ y, g y ∂π.measure :=
      MeasureTheory.integral_fun_snd g
    rw [h]; simp only [MeasureTheory.probReal_univ, one_smul, hgdef]; exact hDm.symm
  have hV_gasq : (∫ z, g z.1 ^ 2 ∂(π.measure.prod π.measure)) = A₂ := by
    have h : (∫ z, g z.1 ^ 2 ∂(π.measure.prod π.measure))
        = (π.measure).real Set.univ • ∫ x, g x ^ 2 ∂π.measure :=
      MeasureTheory.integral_fun_fst (fun p => g p ^ 2)
    rw [h]; simp only [MeasureTheory.probReal_univ, one_smul, hgdef]; exact hA₂.symm
  have hV_gbsq : (∫ z, g z.2 ^ 2 ∂(π.measure.prod π.measure)) = A₂ := by
    have h : (∫ z, g z.2 ^ 2 ∂(π.measure.prod π.measure))
        = (π.measure).real Set.univ • ∫ y, g y ^ 2 ∂π.measure :=
      MeasureTheory.integral_fun_snd (fun p => g p ^ 2)
    rw [h]; simp only [MeasureTheory.probReal_univ, one_smul, hgdef]; exact hA₂.symm
  have hV_gagb : (∫ z, g z.1 * g z.2 ∂(π.measure.prod π.measure)) = Dm * Dm := by
    rw [MeasureTheory.integral_prod_mul]
  have hV_mdga : (∫ z, dist z.1.2 z.2.2 * g z.1 ∂(π.measure.prod π.measure)) = A₂ := by
    rw [MeasureTheory.integral_prod _ hmdga]
    have hinner : ∀ x, (∫ y, dist x.2 y.2 * g x ∂π.measure) = g x ^ 2 := by
      intro x
      rw [MeasureTheory.integral_mul_const,
        show (∫ y, dist x.2 y.2 ∂π.measure) = g x from rfl]
      ring
    simp_rw [hinner]
    rw [hA₂]
  have hV_mdgb : (∫ z, dist z.1.2 z.2.2 * g z.2 ∂(π.measure.prod π.measure)) = A₂ := by
    rw [MeasureTheory.integral_prod_symm _ hmdgb]
    have hinner : ∀ y, (∫ x, dist x.2 y.2 * g y ∂π.measure) = g y ^ 2 := by
      intro y
      rw [MeasureTheory.integral_mul_const]
      have hcomm : (∫ x, dist x.2 y.2 ∂π.measure) = g y := by
        show (∫ x, dist x.2 y.2 ∂π.measure) = ∫ q, dist y.2 q.2 ∂π.measure
        refine MeasureTheory.integral_congr_ae ?_
        filter_upwards with x
        rw [dist_comm]
      rw [hcomm]; ring
    simp_rw [hinner]
    rw [hA₂]
  -- Pointwise rewrite of the centered kernel and expansion of its square.
  have hc : ∀ z : (α × β) × (α × β),
      centeredDistSnd π z = dist z.1.2 z.2.2 - g z.1 - g z.2 + Dm := by
    intro z; simp only [centeredDistSnd, hgdef, hDm]
  have hExpand : ∀ z : (α × β) × (α × β),
      centeredDistSnd π z ^ 2
        = (dist z.1.2 z.2.2 - g z.1) ^ 2
          - 2 * ((dist z.1.2 z.2.2 - g z.1) * (g z.2 - Dm))
          + (g z.2 - Dm) ^ 2 := by
    intro z; rw [hc z]; ring
  -- Integrability of the three grouped terms.
  have hu2i : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => (dist z.1.2 z.2.2 - g z.1) ^ 2) (π.measure.prod π.measure) := by
    have hEq : (fun z : (α × β) × (α × β) => (dist z.1.2 z.2.2 - g z.1) ^ 2)
        = fun z => dist z.1.2 z.2.2 ^ 2 - 2 * (dist z.1.2 z.2.2 * g z.1) + g z.1 ^ 2 := by
      funext z; ring
    rw [hEq]; exact (hmdsq.sub (hmdga.const_mul 2)).add hgasq
  have hv2i : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => (g z.2 - Dm) ^ 2) (π.measure.prod π.measure) := by
    have hEq : (fun z : (α × β) × (α × β) => (g z.2 - Dm) ^ 2)
        = fun z => g z.2 ^ 2 - 2 * Dm * g z.2 + Dm ^ 2 := by
      funext z; ring
    rw [hEq]
    exact (hgbsq.sub (hgb.const_mul (2 * Dm))).add (MeasureTheory.integrable_const (Dm ^ 2))
  have huvi : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => (dist z.1.2 z.2.2 - g z.1) * (g z.2 - Dm))
      (π.measure.prod π.measure) := by
    have hEq : (fun z : (α × β) × (α × β) => (dist z.1.2 z.2.2 - g z.1) * (g z.2 - Dm))
        = fun z => dist z.1.2 z.2.2 * g z.2 - Dm * dist z.1.2 z.2.2
                    - g z.1 * g z.2 + Dm * g z.1 := by
      funext z; ring
    rw [hEq]
    exact ((hmdgb.sub (hmd.const_mul Dm)).sub hgagb).add (hga.const_mul Dm)
  -- Values of the three grouped integrals.
  have hu2v : (∫ z, (dist z.1.2 z.2.2 - g z.1) ^ 2 ∂(π.measure.prod π.measure)) = S₁ - A₂ := by
    have hcong : (∫ z, (dist z.1.2 z.2.2 - g z.1) ^ 2 ∂(π.measure.prod π.measure))
        = ∫ z, dist z.1.2 z.2.2 ^ 2 - 2 * (dist z.1.2 z.2.2 * g z.1) + g z.1 ^ 2
            ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_congr_ae (Filter.Eventually.of_forall fun z => by ring)
    have e1 : (∫ z, dist z.1.2 z.2.2 ^ 2 - 2 * (dist z.1.2 z.2.2 * g z.1) + g z.1 ^ 2
          ∂(π.measure.prod π.measure))
        = (∫ z, dist z.1.2 z.2.2 ^ 2 - 2 * (dist z.1.2 z.2.2 * g z.1) ∂(π.measure.prod π.measure))
          + ∫ z, g z.1 ^ 2 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_add (hmdsq.sub (hmdga.const_mul 2)) hgasq
    have e2 : (∫ z, dist z.1.2 z.2.2 ^ 2 - 2 * (dist z.1.2 z.2.2 * g z.1) ∂(π.measure.prod π.measure))
        = (∫ z, dist z.1.2 z.2.2 ^ 2 ∂(π.measure.prod π.measure))
          - ∫ z, 2 * (dist z.1.2 z.2.2 * g z.1) ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_sub hmdsq (hmdga.const_mul 2)
    have e3 : (∫ z, 2 * (dist z.1.2 z.2.2 * g z.1) ∂(π.measure.prod π.measure))
        = 2 * ∫ z, dist z.1.2 z.2.2 * g z.1 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_const_mul 2 _
    rw [hcong, e1, e2, e3, hV_md2, hV_mdga, hV_gasq]; ring
  have hv2v : (∫ z, (g z.2 - Dm) ^ 2 ∂(π.measure.prod π.measure)) = A₂ - Dm ^ 2 := by
    have hcong : (∫ z, (g z.2 - Dm) ^ 2 ∂(π.measure.prod π.measure))
        = ∫ z, g z.2 ^ 2 - 2 * Dm * g z.2 + Dm ^ 2 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_congr_ae (Filter.Eventually.of_forall fun z => by ring)
    have e1 : (∫ z, g z.2 ^ 2 - 2 * Dm * g z.2 + Dm ^ 2 ∂(π.measure.prod π.measure))
        = (∫ z, g z.2 ^ 2 - 2 * Dm * g z.2 ∂(π.measure.prod π.measure))
          + ∫ z, Dm ^ 2 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_add (hgbsq.sub (hgb.const_mul (2 * Dm)))
        (MeasureTheory.integrable_const (Dm ^ 2))
    have e2 : (∫ z, g z.2 ^ 2 - 2 * Dm * g z.2 ∂(π.measure.prod π.measure))
        = (∫ z, g z.2 ^ 2 ∂(π.measure.prod π.measure))
          - ∫ z, 2 * Dm * g z.2 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_sub hgbsq (hgb.const_mul (2 * Dm))
    have e3 : (∫ z, 2 * Dm * g z.2 ∂(π.measure.prod π.measure))
        = 2 * Dm * ∫ z, g z.2 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_const_mul (2 * Dm) _
    have e4 : (∫ z, Dm ^ 2 ∂(π.measure.prod π.measure)) = Dm ^ 2 := by
      rw [MeasureTheory.integral_const, MeasureTheory.probReal_univ, one_smul]
    rw [hcong, e1, e2, e3, e4, hV_gbsq, hV_gb]; ring
  have huvv : (∫ z, (dist z.1.2 z.2.2 - g z.1) * (g z.2 - Dm) ∂(π.measure.prod π.measure))
      = A₂ - Dm ^ 2 := by
    have hcong : (∫ z, (dist z.1.2 z.2.2 - g z.1) * (g z.2 - Dm) ∂(π.measure.prod π.measure))
        = ∫ z, dist z.1.2 z.2.2 * g z.2 - Dm * dist z.1.2 z.2.2
              - g z.1 * g z.2 + Dm * g z.1 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_congr_ae (Filter.Eventually.of_forall fun z => by ring)
    have e1 : (∫ z, dist z.1.2 z.2.2 * g z.2 - Dm * dist z.1.2 z.2.2
              - g z.1 * g z.2 + Dm * g z.1 ∂(π.measure.prod π.measure))
        = (∫ z, dist z.1.2 z.2.2 * g z.2 - Dm * dist z.1.2 z.2.2 - g z.1 * g z.2
              ∂(π.measure.prod π.measure))
          + ∫ z, Dm * g z.1 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_add ((hmdgb.sub (hmd.const_mul Dm)).sub hgagb) (hga.const_mul Dm)
    have e2 : (∫ z, dist z.1.2 z.2.2 * g z.2 - Dm * dist z.1.2 z.2.2 - g z.1 * g z.2
            ∂(π.measure.prod π.measure))
        = (∫ z, dist z.1.2 z.2.2 * g z.2 - Dm * dist z.1.2 z.2.2 ∂(π.measure.prod π.measure))
          - ∫ z, g z.1 * g z.2 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_sub (hmdgb.sub (hmd.const_mul Dm)) hgagb
    have e3 : (∫ z, dist z.1.2 z.2.2 * g z.2 - Dm * dist z.1.2 z.2.2 ∂(π.measure.prod π.measure))
        = (∫ z, dist z.1.2 z.2.2 * g z.2 ∂(π.measure.prod π.measure))
          - ∫ z, Dm * dist z.1.2 z.2.2 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_sub hmdgb (hmd.const_mul Dm)
    have e4 : (∫ z, Dm * dist z.1.2 z.2.2 ∂(π.measure.prod π.measure))
        = Dm * ∫ z, dist z.1.2 z.2.2 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_const_mul Dm _
    have e5 : (∫ z, Dm * g z.1 ∂(π.measure.prod π.measure))
        = Dm * ∫ z, g z.1 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_const_mul Dm _
    rw [hcong, e1, e2, e3, e4, e5, hV_mdgb, hV_md, hV_gagb, hV_ga]; ring
  -- Assemble.
  have hmain : (∫ z, centeredDistSnd π z ^ 2 ∂(π.measure.prod π.measure))
      = S₁ + Dm ^ 2 - 2 * A₂ := by
    have hcong : (∫ z, centeredDistSnd π z ^ 2 ∂(π.measure.prod π.measure))
        = ∫ z, ((dist z.1.2 z.2.2 - g z.1) ^ 2
                - 2 * ((dist z.1.2 z.2.2 - g z.1) * (g z.2 - Dm))
                + (g z.2 - Dm) ^ 2) ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_congr_ae (Filter.Eventually.of_forall hExpand)
    have e1 : (∫ z, ((dist z.1.2 z.2.2 - g z.1) ^ 2
                - 2 * ((dist z.1.2 z.2.2 - g z.1) * (g z.2 - Dm))
                + (g z.2 - Dm) ^ 2) ∂(π.measure.prod π.measure))
        = (∫ z, (dist z.1.2 z.2.2 - g z.1) ^ 2
                - 2 * ((dist z.1.2 z.2.2 - g z.1) * (g z.2 - Dm)) ∂(π.measure.prod π.measure))
          + ∫ z, (g z.2 - Dm) ^ 2 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_add (hu2i.sub (huvi.const_mul 2)) hv2i
    have e2 : (∫ z, (dist z.1.2 z.2.2 - g z.1) ^ 2
                - 2 * ((dist z.1.2 z.2.2 - g z.1) * (g z.2 - Dm)) ∂(π.measure.prod π.measure))
        = (∫ z, (dist z.1.2 z.2.2 - g z.1) ^ 2 ∂(π.measure.prod π.measure))
          - ∫ z, 2 * ((dist z.1.2 z.2.2 - g z.1) * (g z.2 - Dm)) ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_sub hu2i (huvi.const_mul 2)
    have e3 : (∫ z, 2 * ((dist z.1.2 z.2.2 - g z.1) * (g z.2 - Dm)) ∂(π.measure.prod π.measure))
        = 2 * ∫ z, (dist z.1.2 z.2.2 - g z.1) * (g z.2 - Dm) ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_const_mul 2 _
    rw [hcong, e1, e2, e3, hu2v, huvv, hv2v]; ring
  rw [hmain]

/-- Squared distance variance is nonnegative: `dVar²(X) ≥ 0`.

    Immediate from the L²-representation `dvar_eq_integral_sq`, since the
    integrand is a square.

    Reference: Székely & Rizzo (2023), Ch. 12. -/
lemma distanceVarianceSq_nonneg {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    [PseudoMetricSpace α] [PseudoMetricSpace β]
    (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π) :
    0 ≤ distanceVarianceSq π := by
  rw [dvar_eq_integral_sq π hm]
  exact MeasureTheory.integral_nonneg fun z => sq_nonneg _

/-- Squared distance covariance equals the L²(π⊗π) pairing of the two doubly-centered
    distance kernels (U-centered representation).

    `dCov²(X,Y) = ∫ centeredDistFst · centeredDistSnd d(π⊗π)`.

    The proof mirrors `dvar_eq_integral_sq`: write each centered kernel as a difference
    `(d − g(Z)) − (g(Z') − D)`, expand the product into four binomial groups, and evaluate
    every resulting expectation by Fubini (`integral_prod`, `integral_prod_symm`), the
    marginal identities (`integral_fun_fst`, `integral_fun_snd`), independence across
    coordinates (`integral_prod_mul`) and `dist_comm`.

    Reference: Székely & Rizzo (2023), Ch. 12 (U-centered representation); Lyons (2013), §2. -/
lemma dcov_eq_integral_mul {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    [PseudoMetricSpace α] [PseudoMetricSpace β]
    (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π) :
    distanceCovarianceSq π =
      ∫ z, centeredDistFst π z * centeredDistSnd π z ∂(π.measure.prod π.measure) := by
  -- marginal mean-distance functions
  set gX : (α × β) → ℝ := fun p => ∫ q, dist p.1 q.1 ∂π.measure with hgXdef
  set gY : (α × β) → ℝ := fun p => ∫ q, dist p.2 q.2 ∂π.measure with hgYdef
  have hgX_nonneg : ∀ p, 0 ≤ gX p := fun p =>
    MeasureTheory.integral_nonneg fun q => dist_nonneg
  have hgY_nonneg : ∀ p, 0 ≤ gY p := fun p =>
    MeasureTheory.integral_nonneg fun q => dist_nonneg
  -- Integrability of the marginal functions and their squares over `π`.
  have hgX_int : MeasureTheory.Integrable gX π.measure := hm.fst.integral_prod_left
  have hgY_int : MeasureTheory.Integrable gY π.measure := hm.snd.integral_prod_left
  have hgXsq_int : MeasureTheory.Integrable (fun p => gX p ^ 2) π.measure := by
    have hbound : MeasureTheory.Integrable
        (fun p => ∫ q, dist p.1 q.1 ^ 2 ∂π.measure) π.measure := hm.fstSq.integral_prod_left
    have hmeas : MeasureTheory.AEStronglyMeasurable (fun p => gX p ^ 2) π.measure := by
      refine (hgX_int.aestronglyMeasurable.mul hgX_int.aestronglyMeasurable).congr ?_
      filter_upwards with p
      simp [pow_two]
    refine hbound.mono' hmeas ?_
    filter_upwards [hm.fst.prod_right_ae, hm.fstSq.prod_right_ae] with p hp1 hp2
    have hj := sq_integral_le (fun q : α × β => dist p.1 q.1) hp1 hp2
    calc ‖gX p ^ 2‖ = gX p ^ 2 := by rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg (gX p))]
      _ ≤ ∫ q, dist p.1 q.1 ^ 2 ∂π.measure := hj
  have hgYsq_int : MeasureTheory.Integrable (fun p => gY p ^ 2) π.measure := by
    have hbound : MeasureTheory.Integrable
        (fun p => ∫ q, dist p.2 q.2 ^ 2 ∂π.measure) π.measure := hm.sndSq.integral_prod_left
    have hmeas : MeasureTheory.AEStronglyMeasurable (fun p => gY p ^ 2) π.measure := by
      refine (hgY_int.aestronglyMeasurable.mul hgY_int.aestronglyMeasurable).congr ?_
      filter_upwards with p
      simp [pow_two]
    refine hbound.mono' hmeas ?_
    filter_upwards [hm.snd.prod_right_ae, hm.sndSq.prod_right_ae] with p hp1 hp2
    have hj := sq_integral_le (fun q : α × β => dist p.2 q.2) hp1 hp2
    calc ‖gY p ^ 2‖ = gY p ^ 2 := by rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg (gY p))]
      _ ≤ ∫ q, dist p.2 q.2 ^ 2 ∂π.measure := hj
  have hgXgY_int : MeasureTheory.Integrable (fun p => gX p * gY p) π.measure := by
    refine MeasureTheory.Integrable.mono' ((hgXsq_int.add hgYsq_int).div_const 2)
      (hgX_int.aestronglyMeasurable.mul hgY_int.aestronglyMeasurable) ?_
    filter_upwards with p
    rw [Real.norm_eq_abs, abs_of_nonneg (mul_nonneg (hgX_nonneg p) (hgY_nonneg p))]
    simp only [Pi.add_apply]
    nlinarith [two_mul_le_add_sq (gX p) (gY p)]
  -- Lifts to `π ⊗ π`.
  have hdX : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => dist z.1.1 z.2.1) (π.measure.prod π.measure) := hm.fst
  have hdY : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => dist z.1.2 z.2.2) (π.measure.prod π.measure) := hm.snd
  have hgX1 : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => gX z.1) (π.measure.prod π.measure) := hgX_int.comp_fst π.measure
  have hgX2 : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => gX z.2) (π.measure.prod π.measure) := hgX_int.comp_snd π.measure
  have hgY1' : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => gY z.1) (π.measure.prod π.measure) := hgY_int.comp_fst π.measure
  have hgY2 : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => gY z.2) (π.measure.prod π.measure) := hgY_int.comp_snd π.measure
  have hgYsq1 : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => gY z.1 ^ 2) (π.measure.prod π.measure) :=
    hgYsq_int.comp_fst π.measure
  have hgYsq2 : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => gY z.2 ^ 2) (π.measure.prod π.measure) :=
    hgYsq_int.comp_snd π.measure
  have hgXsq1 : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => gX z.1 ^ 2) (π.measure.prod π.measure) :=
    hgXsq_int.comp_fst π.measure
  have hgXsq2 : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => gX z.2 ^ 2) (π.measure.prod π.measure) :=
    hgXsq_int.comp_snd π.measure
  have hdXdY : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => dist z.1.1 z.2.1 * dist z.1.2 z.2.2)
      (π.measure.prod π.measure) := hm.distProd
  have hgXgY1 : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => gX z.1 * gY z.1) (π.measure.prod π.measure) :=
    hgXgY_int.comp_fst π.measure
  have hgXgY2 : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => gX z.2 * gY z.2) (π.measure.prod π.measure) :=
    hgXgY_int.comp_snd π.measure
  have hgX1gY2 : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => gX z.1 * gY z.2) (π.measure.prod π.measure) :=
    hgX_int.mul_prod hgY_int
  have hgX2gY1 : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => gX z.2 * gY z.1) (π.measure.prod π.measure) := by
    have h := hgY_int.mul_prod hgX_int
    exact h.congr (Filter.Eventually.of_forall fun z => by ring)
  have hdXgY1 : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => dist z.1.1 z.2.1 * gY z.1) (π.measure.prod π.measure) := by
    refine MeasureTheory.Integrable.mono' ((hm.fstSq.add hgYsq1).div_const 2)
      (hdX.aestronglyMeasurable.mul hgY1'.aestronglyMeasurable) ?_
    filter_upwards with z
    rw [Real.norm_eq_abs, abs_of_nonneg (mul_nonneg dist_nonneg (hgY_nonneg z.1))]
    simp only [Pi.add_apply]
    nlinarith [two_mul_le_add_sq (dist z.1.1 z.2.1) (gY z.1)]
  have hdXgY2 : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => dist z.1.1 z.2.1 * gY z.2) (π.measure.prod π.measure) := by
    refine MeasureTheory.Integrable.mono' ((hm.fstSq.add hgYsq2).div_const 2)
      (hdX.aestronglyMeasurable.mul hgY2.aestronglyMeasurable) ?_
    filter_upwards with z
    rw [Real.norm_eq_abs, abs_of_nonneg (mul_nonneg dist_nonneg (hgY_nonneg z.2))]
    simp only [Pi.add_apply]
    nlinarith [two_mul_le_add_sq (dist z.1.1 z.2.1) (gY z.2)]
  have hdYgX1 : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => dist z.1.2 z.2.2 * gX z.1) (π.measure.prod π.measure) := by
    refine MeasureTheory.Integrable.mono' ((hm.sndSq.add hgXsq1).div_const 2)
      (hdY.aestronglyMeasurable.mul hgX1.aestronglyMeasurable) ?_
    filter_upwards with z
    rw [Real.norm_eq_abs, abs_of_nonneg (mul_nonneg dist_nonneg (hgX_nonneg z.1))]
    simp only [Pi.add_apply]
    nlinarith [two_mul_le_add_sq (dist z.1.2 z.2.2) (gX z.1)]
  have hdYgX2 : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => dist z.1.2 z.2.2 * gX z.2) (π.measure.prod π.measure) := by
    refine MeasureTheory.Integrable.mono' ((hm.sndSq.add hgXsq2).div_const 2)
      (hdY.aestronglyMeasurable.mul hgX2.aestronglyMeasurable) ?_
    filter_upwards with z
    rw [Real.norm_eq_abs, abs_of_nonneg (mul_nonneg dist_nonneg (hgX_nonneg z.2))]
    simp only [Pi.add_apply]
    nlinarith [two_mul_le_add_sq (dist z.1.2 z.2.2) (gX z.2)]
  -- Unfold the target and abbreviate the aggregate quantities.
  rw [distanceCovarianceSq]
  set S₁ := ∫ p, ∫ q, dist p.1 q.1 * dist p.2 q.2 ∂π.measure ∂π.measure with hS₁
  set DX := ∫ p, ∫ q, dist p.1 q.1 ∂π.measure ∂π.measure with hDX
  set DY := ∫ p, ∫ q, dist p.2 q.2 ∂π.measure ∂π.measure with hDY
  set G := ∫ p, (∫ q, dist p.1 q.1 ∂π.measure) * (∫ r, dist p.2 r.2 ∂π.measure) ∂π.measure
    with hG
  -- Atomic expectations over `π ⊗ π`.
  have hV_dXdY : (∫ z, dist z.1.1 z.2.1 * dist z.1.2 z.2.2 ∂(π.measure.prod π.measure)) = S₁ := by
    rw [hS₁]; exact MeasureTheory.integral_prod _ hm.distProd
  have hV_dX : (∫ z, dist z.1.1 z.2.1 ∂(π.measure.prod π.measure)) = DX := by
    rw [hDX]; exact MeasureTheory.integral_prod _ hm.fst
  have hV_dY : (∫ z, dist z.1.2 z.2.2 ∂(π.measure.prod π.measure)) = DY := by
    rw [hDY]; exact MeasureTheory.integral_prod _ hm.snd
  have hV_gX1 : (∫ z, gX z.1 ∂(π.measure.prod π.measure)) = DX := by
    have h : (∫ z, gX z.1 ∂(π.measure.prod π.measure))
        = (π.measure).real Set.univ • ∫ x, gX x ∂π.measure :=
      MeasureTheory.integral_fun_fst gX
    rw [h]; simp only [MeasureTheory.probReal_univ, one_smul, hgXdef]; exact hDX.symm
  have hV_gX2 : (∫ z, gX z.2 ∂(π.measure.prod π.measure)) = DX := by
    have h : (∫ z, gX z.2 ∂(π.measure.prod π.measure))
        = (π.measure).real Set.univ • ∫ y, gX y ∂π.measure :=
      MeasureTheory.integral_fun_snd gX
    rw [h]; simp only [MeasureTheory.probReal_univ, one_smul, hgXdef]; exact hDX.symm
  have hV_gY1 : (∫ z, gY z.1 ∂(π.measure.prod π.measure)) = DY := by
    have h : (∫ z, gY z.1 ∂(π.measure.prod π.measure))
        = (π.measure).real Set.univ • ∫ x, gY x ∂π.measure :=
      MeasureTheory.integral_fun_fst gY
    rw [h]; simp only [MeasureTheory.probReal_univ, one_smul, hgYdef]; exact hDY.symm
  have hV_gY2 : (∫ z, gY z.2 ∂(π.measure.prod π.measure)) = DY := by
    have h : (∫ z, gY z.2 ∂(π.measure.prod π.measure))
        = (π.measure).real Set.univ • ∫ y, gY y ∂π.measure :=
      MeasureTheory.integral_fun_snd gY
    rw [h]; simp only [MeasureTheory.probReal_univ, one_smul, hgYdef]; exact hDY.symm
  have hV_gXgY1 : (∫ z, gX z.1 * gY z.1 ∂(π.measure.prod π.measure)) = G := by
    have h : (∫ z, gX z.1 * gY z.1 ∂(π.measure.prod π.measure))
        = (π.measure).real Set.univ • ∫ x, gX x * gY x ∂π.measure :=
      MeasureTheory.integral_fun_fst (fun p => gX p * gY p)
    rw [h]; simp only [MeasureTheory.probReal_univ, one_smul, hgXdef, hgYdef]; exact hG.symm
  have hV_gXgY2 : (∫ z, gX z.2 * gY z.2 ∂(π.measure.prod π.measure)) = G := by
    have h : (∫ z, gX z.2 * gY z.2 ∂(π.measure.prod π.measure))
        = (π.measure).real Set.univ • ∫ y, gX y * gY y ∂π.measure :=
      MeasureTheory.integral_fun_snd (fun p => gX p * gY p)
    rw [h]; simp only [MeasureTheory.probReal_univ, one_smul, hgXdef, hgYdef]; exact hG.symm
  have hV_gX1gY2 : (∫ z, gX z.1 * gY z.2 ∂(π.measure.prod π.measure)) = DX * DY := by
    rw [MeasureTheory.integral_prod_mul, hgXdef, hgYdef]
  have hV_gX2gY1 : (∫ z, gX z.2 * gY z.1 ∂(π.measure.prod π.measure)) = DX * DY := by
    have hcong : (∫ z, gX z.2 * gY z.1 ∂(π.measure.prod π.measure))
        = ∫ z, gY z.1 * gX z.2 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_congr_ae (Filter.Eventually.of_forall fun z => by ring)
    rw [hcong, MeasureTheory.integral_prod_mul, hgXdef, hgYdef, hDX, hDY]
    ring
  have hV_dXgY1 : (∫ z, dist z.1.1 z.2.1 * gY z.1 ∂(π.measure.prod π.measure)) = G := by
    rw [MeasureTheory.integral_prod _ hdXgY1]
    have hinner : ∀ x, (∫ y, dist x.1 y.1 * gY x ∂π.measure) = gX x * gY x := by
      intro x
      rw [MeasureTheory.integral_mul_const,
        show (∫ y, dist x.1 y.1 ∂π.measure) = gX x from rfl]
    simp_rw [hinner]
    simp only [hgXdef, hgYdef]
    exact hG.symm
  have hV_dXgY2 : (∫ z, dist z.1.1 z.2.1 * gY z.2 ∂(π.measure.prod π.measure)) = G := by
    rw [MeasureTheory.integral_prod_symm _ hdXgY2]
    have hinner : ∀ y, (∫ x, dist x.1 y.1 * gY y ∂π.measure) = gX y * gY y := by
      intro y
      rw [MeasureTheory.integral_mul_const]
      have hcomm : (∫ x, dist x.1 y.1 ∂π.measure) = gX y := by
        show (∫ x, dist x.1 y.1 ∂π.measure) = ∫ q, dist y.1 q.1 ∂π.measure
        refine MeasureTheory.integral_congr_ae ?_
        filter_upwards with x
        rw [dist_comm]
      rw [hcomm]
    simp_rw [hinner]
    simp only [hgXdef, hgYdef]
    exact hG.symm
  have hV_dYgX1 : (∫ z, dist z.1.2 z.2.2 * gX z.1 ∂(π.measure.prod π.measure)) = G := by
    rw [MeasureTheory.integral_prod _ hdYgX1]
    have hinner : ∀ x, (∫ y, dist x.2 y.2 * gX x ∂π.measure) = gX x * gY x := by
      intro x
      rw [MeasureTheory.integral_mul_const,
        show (∫ y, dist x.2 y.2 ∂π.measure) = gY x from rfl]
      ring
    simp_rw [hinner]
    simp only [hgXdef, hgYdef]
    exact hG.symm
  have hV_dYgX2 : (∫ z, dist z.1.2 z.2.2 * gX z.2 ∂(π.measure.prod π.measure)) = G := by
    rw [MeasureTheory.integral_prod_symm _ hdYgX2]
    have hinner : ∀ y, (∫ x, dist x.2 y.2 * gX y ∂π.measure) = gX y * gY y := by
      intro y
      rw [MeasureTheory.integral_mul_const]
      have hcomm : (∫ x, dist x.2 y.2 ∂π.measure) = gY y := by
        show (∫ x, dist x.2 y.2 ∂π.measure) = ∫ q, dist y.2 q.2 ∂π.measure
        refine MeasureTheory.integral_congr_ae ?_
        filter_upwards with x
        rw [dist_comm]
      rw [hcomm]
      ring
    simp_rw [hinner]
    simp only [hgXdef, hgYdef]
    exact hG.symm
  -- Pointwise form of the centered kernels.
  have hcX : ∀ z : (α × β) × (α × β),
      centeredDistFst π z = dist z.1.1 z.2.1 - gX z.1 - gX z.2 + DX := by
    intro z; simp only [centeredDistFst, hgXdef, hDX]
  have hcY : ∀ z : (α × β) × (α × β),
      centeredDistSnd π z = dist z.1.2 z.2.2 - gY z.1 - gY z.2 + DY := by
    intro z; simp only [centeredDistSnd, hgYdef, hDY]
  have hExpand : ∀ z : (α × β) × (α × β),
      centeredDistFst π z * centeredDistSnd π z
        = (dist z.1.1 z.2.1 - gX z.1) * (dist z.1.2 z.2.2 - gY z.1)
          - (dist z.1.1 z.2.1 - gX z.1) * (gY z.2 - DY)
          - (gX z.2 - DX) * (dist z.1.2 z.2.2 - gY z.1)
          + (gX z.2 - DX) * (gY z.2 - DY) := by
    intro z; rw [hcX z, hcY z]; ring
  -- Integrability of the four binomial groups.
  have hI_uu : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) =>
        (dist z.1.1 z.2.1 - gX z.1) * (dist z.1.2 z.2.2 - gY z.1))
      (π.measure.prod π.measure) := by
    have hEq : (fun z : (α × β) × (α × β) =>
        (dist z.1.1 z.2.1 - gX z.1) * (dist z.1.2 z.2.2 - gY z.1))
        = fun z => dist z.1.1 z.2.1 * dist z.1.2 z.2.2 - dist z.1.1 z.2.1 * gY z.1
            - dist z.1.2 z.2.2 * gX z.1 + gX z.1 * gY z.1 := by
      funext z; ring
    rw [hEq]
    exact ((hdXdY.sub hdXgY1).sub hdYgX1).add hgXgY1
  have hI_uv : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => (dist z.1.1 z.2.1 - gX z.1) * (gY z.2 - DY))
      (π.measure.prod π.measure) := by
    have hEq : (fun z : (α × β) × (α × β) => (dist z.1.1 z.2.1 - gX z.1) * (gY z.2 - DY))
        = fun z => dist z.1.1 z.2.1 * gY z.2 - DY * dist z.1.1 z.2.1
            - gX z.1 * gY z.2 + DY * gX z.1 := by
      funext z; ring
    rw [hEq]
    exact ((hdXgY2.sub (hdX.const_mul DY)).sub hgX1gY2).add (hgX1.const_mul DY)
  have hI_vu : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => (gX z.2 - DX) * (dist z.1.2 z.2.2 - gY z.1))
      (π.measure.prod π.measure) := by
    have hEq : (fun z : (α × β) × (α × β) => (gX z.2 - DX) * (dist z.1.2 z.2.2 - gY z.1))
        = fun z => dist z.1.2 z.2.2 * gX z.2 - gX z.2 * gY z.1
            - DX * dist z.1.2 z.2.2 + DX * gY z.1 := by
      funext z; ring
    rw [hEq]
    exact ((hdYgX2.sub hgX2gY1).sub (hdY.const_mul DX)).add (hgY1'.const_mul DX)
  have hI_vv : MeasureTheory.Integrable
      (fun z : (α × β) × (α × β) => (gX z.2 - DX) * (gY z.2 - DY))
      (π.measure.prod π.measure) := by
    have hEq : (fun z : (α × β) × (α × β) => (gX z.2 - DX) * (gY z.2 - DY))
        = fun z => gX z.2 * gY z.2 - DY * gX z.2 - DX * gY z.2 + DX * DY := by
      funext z; ring
    rw [hEq]
    exact ((hgXgY2.sub (hgX2.const_mul DY)).sub (hgY2.const_mul DX)).add
      (MeasureTheory.integrable_const (DX * DY))
  -- Values of the four binomial groups.
  have hV_uu : (∫ z, (dist z.1.1 z.2.1 - gX z.1) * (dist z.1.2 z.2.2 - gY z.1)
      ∂(π.measure.prod π.measure)) = S₁ - G - G + G := by
    have hcong : (∫ z, (dist z.1.1 z.2.1 - gX z.1) * (dist z.1.2 z.2.2 - gY z.1)
        ∂(π.measure.prod π.measure))
        = ∫ z, dist z.1.1 z.2.1 * dist z.1.2 z.2.2 - dist z.1.1 z.2.1 * gY z.1
            - dist z.1.2 z.2.2 * gX z.1 + gX z.1 * gY z.1 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_congr_ae (Filter.Eventually.of_forall fun z => by ring)
    have e1 : (∫ z, dist z.1.1 z.2.1 * dist z.1.2 z.2.2 - dist z.1.1 z.2.1 * gY z.1
            - dist z.1.2 z.2.2 * gX z.1 + gX z.1 * gY z.1 ∂(π.measure.prod π.measure))
        = (∫ z, dist z.1.1 z.2.1 * dist z.1.2 z.2.2 - dist z.1.1 z.2.1 * gY z.1
            - dist z.1.2 z.2.2 * gX z.1 ∂(π.measure.prod π.measure))
          + ∫ z, gX z.1 * gY z.1 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_add ((hdXdY.sub hdXgY1).sub hdYgX1) hgXgY1
    have e2 : (∫ z, dist z.1.1 z.2.1 * dist z.1.2 z.2.2 - dist z.1.1 z.2.1 * gY z.1
            - dist z.1.2 z.2.2 * gX z.1 ∂(π.measure.prod π.measure))
        = (∫ z, dist z.1.1 z.2.1 * dist z.1.2 z.2.2 - dist z.1.1 z.2.1 * gY z.1
            ∂(π.measure.prod π.measure))
          - ∫ z, dist z.1.2 z.2.2 * gX z.1 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_sub (hdXdY.sub hdXgY1) hdYgX1
    have e3 : (∫ z, dist z.1.1 z.2.1 * dist z.1.2 z.2.2 - dist z.1.1 z.2.1 * gY z.1
            ∂(π.measure.prod π.measure))
        = (∫ z, dist z.1.1 z.2.1 * dist z.1.2 z.2.2 ∂(π.measure.prod π.measure))
          - ∫ z, dist z.1.1 z.2.1 * gY z.1 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_sub hdXdY hdXgY1
    rw [hcong, e1, e2, e3, hV_dXdY, hV_dXgY1, hV_dYgX1, hV_gXgY1]
  have hV_uv : (∫ z, (dist z.1.1 z.2.1 - gX z.1) * (gY z.2 - DY)
      ∂(π.measure.prod π.measure)) = G - DX * DY := by
    have hcong : (∫ z, (dist z.1.1 z.2.1 - gX z.1) * (gY z.2 - DY)
        ∂(π.measure.prod π.measure))
        = ∫ z, dist z.1.1 z.2.1 * gY z.2 - DY * dist z.1.1 z.2.1
            - gX z.1 * gY z.2 + DY * gX z.1 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_congr_ae (Filter.Eventually.of_forall fun z => by ring)
    have e1 : (∫ z, dist z.1.1 z.2.1 * gY z.2 - DY * dist z.1.1 z.2.1
            - gX z.1 * gY z.2 + DY * gX z.1 ∂(π.measure.prod π.measure))
        = (∫ z, dist z.1.1 z.2.1 * gY z.2 - DY * dist z.1.1 z.2.1
            - gX z.1 * gY z.2 ∂(π.measure.prod π.measure))
          + ∫ z, DY * gX z.1 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_add ((hdXgY2.sub (hdX.const_mul DY)).sub hgX1gY2)
        (hgX1.const_mul DY)
    have e2 : (∫ z, dist z.1.1 z.2.1 * gY z.2 - DY * dist z.1.1 z.2.1
            - gX z.1 * gY z.2 ∂(π.measure.prod π.measure))
        = (∫ z, dist z.1.1 z.2.1 * gY z.2 - DY * dist z.1.1 z.2.1
            ∂(π.measure.prod π.measure))
          - ∫ z, gX z.1 * gY z.2 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_sub (hdXgY2.sub (hdX.const_mul DY)) hgX1gY2
    have e3 : (∫ z, dist z.1.1 z.2.1 * gY z.2 - DY * dist z.1.1 z.2.1
            ∂(π.measure.prod π.measure))
        = (∫ z, dist z.1.1 z.2.1 * gY z.2 ∂(π.measure.prod π.measure))
          - ∫ z, DY * dist z.1.1 z.2.1 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_sub hdXgY2 (hdX.const_mul DY)
    have e4 : (∫ z, DY * dist z.1.1 z.2.1 ∂(π.measure.prod π.measure))
        = DY * ∫ z, dist z.1.1 z.2.1 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_const_mul DY _
    have e5 : (∫ z, DY * gX z.1 ∂(π.measure.prod π.measure))
        = DY * ∫ z, gX z.1 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_const_mul DY _
    rw [hcong, e1, e2, e3, e4, e5, hV_dXgY2, hV_dX, hV_gX1gY2, hV_gX1]
    ring
  have hV_vu : (∫ z, (gX z.2 - DX) * (dist z.1.2 z.2.2 - gY z.1)
      ∂(π.measure.prod π.measure)) = G - DX * DY := by
    have hcong : (∫ z, (gX z.2 - DX) * (dist z.1.2 z.2.2 - gY z.1)
        ∂(π.measure.prod π.measure))
        = ∫ z, dist z.1.2 z.2.2 * gX z.2 - gX z.2 * gY z.1
            - DX * dist z.1.2 z.2.2 + DX * gY z.1 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_congr_ae (Filter.Eventually.of_forall fun z => by ring)
    have e1 : (∫ z, dist z.1.2 z.2.2 * gX z.2 - gX z.2 * gY z.1
            - DX * dist z.1.2 z.2.2 + DX * gY z.1 ∂(π.measure.prod π.measure))
        = (∫ z, dist z.1.2 z.2.2 * gX z.2 - gX z.2 * gY z.1
            - DX * dist z.1.2 z.2.2 ∂(π.measure.prod π.measure))
          + ∫ z, DX * gY z.1 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_add ((hdYgX2.sub hgX2gY1).sub (hdY.const_mul DX))
        (hgY1'.const_mul DX)
    have e2 : (∫ z, dist z.1.2 z.2.2 * gX z.2 - gX z.2 * gY z.1
            - DX * dist z.1.2 z.2.2 ∂(π.measure.prod π.measure))
        = (∫ z, dist z.1.2 z.2.2 * gX z.2 - gX z.2 * gY z.1 ∂(π.measure.prod π.measure))
          - ∫ z, DX * dist z.1.2 z.2.2 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_sub (hdYgX2.sub hgX2gY1) (hdY.const_mul DX)
    have e3 : (∫ z, dist z.1.2 z.2.2 * gX z.2 - gX z.2 * gY z.1 ∂(π.measure.prod π.measure))
        = (∫ z, dist z.1.2 z.2.2 * gX z.2 ∂(π.measure.prod π.measure))
          - ∫ z, gX z.2 * gY z.1 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_sub hdYgX2 hgX2gY1
    have e4 : (∫ z, DX * dist z.1.2 z.2.2 ∂(π.measure.prod π.measure))
        = DX * ∫ z, dist z.1.2 z.2.2 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_const_mul DX _
    have e5 : (∫ z, DX * gY z.1 ∂(π.measure.prod π.measure))
        = DX * ∫ z, gY z.1 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_const_mul DX _
    rw [hcong, e1, e2, e3, e4, e5, hV_dYgX2, hV_gX2gY1, hV_dY, hV_gY1]
    ring
  have hV_vv : (∫ z, (gX z.2 - DX) * (gY z.2 - DY)
      ∂(π.measure.prod π.measure)) = G - DX * DY := by
    have hcong : (∫ z, (gX z.2 - DX) * (gY z.2 - DY) ∂(π.measure.prod π.measure))
        = ∫ z, gX z.2 * gY z.2 - DY * gX z.2 - DX * gY z.2 + DX * DY
            ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_congr_ae (Filter.Eventually.of_forall fun z => by ring)
    have e1 : (∫ z, gX z.2 * gY z.2 - DY * gX z.2 - DX * gY z.2 + DX * DY
            ∂(π.measure.prod π.measure))
        = (∫ z, gX z.2 * gY z.2 - DY * gX z.2 - DX * gY z.2 ∂(π.measure.prod π.measure))
          + ∫ z, (DX * DY : ℝ) ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_add ((hgXgY2.sub (hgX2.const_mul DY)).sub (hgY2.const_mul DX))
        (MeasureTheory.integrable_const (DX * DY))
    have e2 : (∫ z, gX z.2 * gY z.2 - DY * gX z.2 - DX * gY z.2 ∂(π.measure.prod π.measure))
        = (∫ z, gX z.2 * gY z.2 - DY * gX z.2 ∂(π.measure.prod π.measure))
          - ∫ z, DX * gY z.2 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_sub (hgXgY2.sub (hgX2.const_mul DY)) (hgY2.const_mul DX)
    have e3 : (∫ z, gX z.2 * gY z.2 - DY * gX z.2 ∂(π.measure.prod π.measure))
        = (∫ z, gX z.2 * gY z.2 ∂(π.measure.prod π.measure))
          - ∫ z, DY * gX z.2 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_sub hgXgY2 (hgX2.const_mul DY)
    have e4 : (∫ z, DY * gX z.2 ∂(π.measure.prod π.measure))
        = DY * ∫ z, gX z.2 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_const_mul DY _
    have e5 : (∫ z, DX * gY z.2 ∂(π.measure.prod π.measure))
        = DX * ∫ z, gY z.2 ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_const_mul DX _
    have e6 : (∫ z, (DX * DY : ℝ) ∂(π.measure.prod π.measure)) = DX * DY := by
      rw [MeasureTheory.integral_const, MeasureTheory.probReal_univ, one_smul]
    rw [hcong, e1, e2, e3, e4, e5, e6, hV_gXgY2, hV_gX2, hV_gY2]
    ring
  -- Assemble the four groups.
  have hmain : (∫ z, centeredDistFst π z * centeredDistSnd π z ∂(π.measure.prod π.measure))
      = S₁ - 2 * G + DX * DY := by
    have hcong : (∫ z, centeredDistFst π z * centeredDistSnd π z ∂(π.measure.prod π.measure))
        = ∫ z, ((dist z.1.1 z.2.1 - gX z.1) * (dist z.1.2 z.2.2 - gY z.1)
            - (dist z.1.1 z.2.1 - gX z.1) * (gY z.2 - DY)
            - (gX z.2 - DX) * (dist z.1.2 z.2.2 - gY z.1)
            + (gX z.2 - DX) * (gY z.2 - DY)) ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_congr_ae (Filter.Eventually.of_forall hExpand)
    have e1 : (∫ z, ((dist z.1.1 z.2.1 - gX z.1) * (dist z.1.2 z.2.2 - gY z.1)
            - (dist z.1.1 z.2.1 - gX z.1) * (gY z.2 - DY)
            - (gX z.2 - DX) * (dist z.1.2 z.2.2 - gY z.1)
            + (gX z.2 - DX) * (gY z.2 - DY)) ∂(π.measure.prod π.measure))
        = (∫ z, (dist z.1.1 z.2.1 - gX z.1) * (dist z.1.2 z.2.2 - gY z.1)
            - (dist z.1.1 z.2.1 - gX z.1) * (gY z.2 - DY)
            - (gX z.2 - DX) * (dist z.1.2 z.2.2 - gY z.1) ∂(π.measure.prod π.measure))
          + ∫ z, (gX z.2 - DX) * (gY z.2 - DY) ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_add ((hI_uu.sub hI_uv).sub hI_vu) hI_vv
    have e2 : (∫ z, (dist z.1.1 z.2.1 - gX z.1) * (dist z.1.2 z.2.2 - gY z.1)
            - (dist z.1.1 z.2.1 - gX z.1) * (gY z.2 - DY)
            - (gX z.2 - DX) * (dist z.1.2 z.2.2 - gY z.1) ∂(π.measure.prod π.measure))
        = (∫ z, (dist z.1.1 z.2.1 - gX z.1) * (dist z.1.2 z.2.2 - gY z.1)
            - (dist z.1.1 z.2.1 - gX z.1) * (gY z.2 - DY) ∂(π.measure.prod π.measure))
          - ∫ z, (gX z.2 - DX) * (dist z.1.2 z.2.2 - gY z.1) ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_sub (hI_uu.sub hI_uv) hI_vu
    have e3 : (∫ z, (dist z.1.1 z.2.1 - gX z.1) * (dist z.1.2 z.2.2 - gY z.1)
            - (dist z.1.1 z.2.1 - gX z.1) * (gY z.2 - DY) ∂(π.measure.prod π.measure))
        = (∫ z, (dist z.1.1 z.2.1 - gX z.1) * (dist z.1.2 z.2.2 - gY z.1)
            ∂(π.measure.prod π.measure))
          - ∫ z, (dist z.1.1 z.2.1 - gX z.1) * (gY z.2 - DY) ∂(π.measure.prod π.measure) :=
      MeasureTheory.integral_sub hI_uu hI_uv
    rw [hcong, e1, e2, e3, hV_uu, hV_uv, hV_vu, hV_vv]
    ring
  rw [hmain]
  ring

/-- The doubly-centered first-coordinate distance kernel is square-integrable over `π ⊗ π`.

    Each of its four summands is in L²: the raw distance by `hm.fstSq`, the two marginal
    mean-distance terms by Jensen (`sq_integral_le`) against `hm.fstSq`, and the constant
    trivially.

    Reference: Székely & Rizzo (2023), Ch. 12. -/
lemma memLp_two_centeredDistFst {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    [PseudoMetricSpace α] [PseudoMetricSpace β]
    (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π) :
    MeasureTheory.MemLp (centeredDistFst π) 2 (π.measure.prod π.measure) := by
  set gX : (α × β) → ℝ := fun p => ∫ q, dist p.1 q.1 ∂π.measure with hgXdef
  have hgX_int : MeasureTheory.Integrable gX π.measure := hm.fst.integral_prod_left
  have hgXsq_int : MeasureTheory.Integrable (fun p => gX p ^ 2) π.measure := by
    have hbound : MeasureTheory.Integrable
        (fun p => ∫ q, dist p.1 q.1 ^ 2 ∂π.measure) π.measure := hm.fstSq.integral_prod_left
    have hmeas : MeasureTheory.AEStronglyMeasurable (fun p => gX p ^ 2) π.measure := by
      refine (hgX_int.aestronglyMeasurable.mul hgX_int.aestronglyMeasurable).congr ?_
      filter_upwards with p
      simp [pow_two]
    refine hbound.mono' hmeas ?_
    filter_upwards [hm.fst.prod_right_ae, hm.fstSq.prod_right_ae] with p hp1 hp2
    have hj := sq_integral_le (fun q : α × β => dist p.1 q.1) hp1 hp2
    calc ‖gX p ^ 2‖ = gX p ^ 2 := by rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg (gX p))]
      _ ≤ ∫ q, dist p.1 q.1 ^ 2 ∂π.measure := hj
  have h1 : MeasureTheory.MemLp (fun z : (α × β) × (α × β) => dist z.1.1 z.2.1) 2
      (π.measure.prod π.measure) :=
    (MeasureTheory.memLp_two_iff_integrable_sq hm.fst.aestronglyMeasurable).mpr hm.fstSq
  have h2 : MeasureTheory.MemLp (fun z : (α × β) × (α × β) => gX z.1) 2
      (π.measure.prod π.measure) :=
    (MeasureTheory.memLp_two_iff_integrable_sq
      (hgX_int.comp_fst π.measure).aestronglyMeasurable).mpr (hgXsq_int.comp_fst π.measure)
  have h3 : MeasureTheory.MemLp (fun z : (α × β) × (α × β) => gX z.2) 2
      (π.measure.prod π.measure) :=
    (MeasureTheory.memLp_two_iff_integrable_sq
      (hgX_int.comp_snd π.measure).aestronglyMeasurable).mpr (hgXsq_int.comp_snd π.measure)
  have h4 : MeasureTheory.MemLp
      (fun _ : (α × β) × (α × β) => ∫ p, ∫ q, dist p.1 q.1 ∂π.measure ∂π.measure) 2
      (π.measure.prod π.measure) := MeasureTheory.memLp_const _
  have hEq : centeredDistFst π = fun z : (α × β) × (α × β) =>
      dist z.1.1 z.2.1 - gX z.1 - gX z.2
        + ∫ p, ∫ q, dist p.1 q.1 ∂π.measure ∂π.measure := by
    funext z; simp only [centeredDistFst, hgXdef]
  rw [hEq]
  exact ((h1.sub h2).sub h3).add h4

/-- The doubly-centered second-coordinate distance kernel is square-integrable over `π ⊗ π`.

    Mirror of `memLp_two_centeredDistFst`. -/
lemma memLp_two_centeredDistSnd {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    [PseudoMetricSpace α] [PseudoMetricSpace β]
    (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π) :
    MeasureTheory.MemLp (centeredDistSnd π) 2 (π.measure.prod π.measure) := by
  set gY : (α × β) → ℝ := fun p => ∫ q, dist p.2 q.2 ∂π.measure with hgYdef
  have hgY_int : MeasureTheory.Integrable gY π.measure := hm.snd.integral_prod_left
  have hgYsq_int : MeasureTheory.Integrable (fun p => gY p ^ 2) π.measure := by
    have hbound : MeasureTheory.Integrable
        (fun p => ∫ q, dist p.2 q.2 ^ 2 ∂π.measure) π.measure := hm.sndSq.integral_prod_left
    have hmeas : MeasureTheory.AEStronglyMeasurable (fun p => gY p ^ 2) π.measure := by
      refine (hgY_int.aestronglyMeasurable.mul hgY_int.aestronglyMeasurable).congr ?_
      filter_upwards with p
      simp [pow_two]
    refine hbound.mono' hmeas ?_
    filter_upwards [hm.snd.prod_right_ae, hm.sndSq.prod_right_ae] with p hp1 hp2
    have hj := sq_integral_le (fun q : α × β => dist p.2 q.2) hp1 hp2
    calc ‖gY p ^ 2‖ = gY p ^ 2 := by rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg (gY p))]
      _ ≤ ∫ q, dist p.2 q.2 ^ 2 ∂π.measure := hj
  have h1 : MeasureTheory.MemLp (fun z : (α × β) × (α × β) => dist z.1.2 z.2.2) 2
      (π.measure.prod π.measure) :=
    (MeasureTheory.memLp_two_iff_integrable_sq hm.snd.aestronglyMeasurable).mpr hm.sndSq
  have h2 : MeasureTheory.MemLp (fun z : (α × β) × (α × β) => gY z.1) 2
      (π.measure.prod π.measure) :=
    (MeasureTheory.memLp_two_iff_integrable_sq
      (hgY_int.comp_fst π.measure).aestronglyMeasurable).mpr (hgYsq_int.comp_fst π.measure)
  have h3 : MeasureTheory.MemLp (fun z : (α × β) × (α × β) => gY z.2) 2
      (π.measure.prod π.measure) :=
    (MeasureTheory.memLp_two_iff_integrable_sq
      (hgY_int.comp_snd π.measure).aestronglyMeasurable).mpr (hgYsq_int.comp_snd π.measure)
  have h4 : MeasureTheory.MemLp
      (fun _ : (α × β) × (α × β) => ∫ p, ∫ q, dist p.2 q.2 ∂π.measure ∂π.measure) 2
      (π.measure.prod π.measure) := MeasureTheory.memLp_const _
  have hEq : centeredDistSnd π = fun z : (α × β) × (α × β) =>
      dist z.1.2 z.2.2 - gY z.1 - gY z.2
        + ∫ p, ∫ q, dist p.2 q.2 ∂π.measure ∂π.measure := by
    funext z; simp only [centeredDistSnd, hgYdef]
  rw [hEq]
  exact ((h1.sub h2).sub h3).add h4

end EnergyStatistics
