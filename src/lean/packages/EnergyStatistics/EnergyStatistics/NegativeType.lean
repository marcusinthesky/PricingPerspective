import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Probability.Distributions.Gaussian.Multivariate
import EnergyStatistics.Defs

set_option linter.style.longLine false

/-!
# DistNegativeType for Inner Product Spaces

This module proves that every real inner product space (hence every Euclidean
space) has **negative type** — i.e. `DistNegativeType E` holds — which is the
key hypothesis required by the energy-statistics nonnegativity theorems
(`v_statistic_nonneg`, `dcov_nonneg`, `energy_distance_nonneg`).

## Main results

* `distNegativeType_of_innerProductSpace`: the `DistNegativeType` proof
  for `[NormedAddCommGroup E] [InnerProductSpace ℝ E]`.
* `sq_dist_cnd_of_inner'`: the squared-distance CND identity in a real inner
  product space (standalone quantitative companion, reproved here to keep
  `EnergyStatistics` self-contained; not needed by the main proof).

## Mathematical content

The proof is a Gaussian-projection argument (classically equivalent to the
Schoenberg/Bernstein-function route, but built entirely from what is in
mathlib today):

1. **One-dimensional CND** (`cnd_abs_sub_real`): for zero-sum weights,
   `Σᵢⱼ wᵢwⱼ|sᵢ−sⱼ| ≤ 0` on `ℝ`, via the min-kernel identity
   `|a−b| = a + b − 2·min a b` and the integral representation
   `Σᵢⱼ wᵢwⱼ (min(sᵢ,sⱼ) − A) = ∫ (Σᵢ wᵢ·𝟙_{(A,sᵢ]})² ≥ 0`.

2. **Gaussian projection** (`integral_abs_inner_stdGaussian`): on a
   finite-dimensional real inner product space,
   `∫ |⟪v,g⟫| dγ(g) = c·‖v‖` with `γ = stdGaussian E` and
   `c = ∫|t| dN(0,1) > 0`, so the distance double sum is a Gaussian average
   of one-dimensional double sums, each ≤ 0 by step 1
   (`cnd_dist_of_finiteDimensional`).

3. **Span reduction**: in a general inner product space the finitely many
   points embed isometrically in `Submodule.span ℝ (Set.range x)`, which is
   finite dimensional, so step 2 applies.

   Reference: Schoenberg (1938), "Metric spaces and positive definite
   functions"; Lyons (2013), "Distance covariance in metric spaces", §3;
   Székely & Rizzo (2023), §3.2, §10.

## References

* Székely, G. J., & Rizzo, M. L. (2023). The Energy of Data and Distance
  Correlation. CRC Press.
* Lyons, R. (2013). Distance covariance in metric spaces. Ann. Probab.
* Schoenberg, I. J. (1938). Metric spaces and positive definite functions.
  Trans. Amer. Math. Soc.
-/

namespace EnergyStatistics

/-- **Portfolio norm expansion as a double inner-product sum.**

For weights `wᵢ` and points `βᵢ` in a real inner product space:

`‖Σᵢ wᵢ • βᵢ‖² = Σᵢⱼ wᵢ wⱼ ⟪βᵢ, βⱼ⟫`

Reproduced from `PricingPerspective.Continuous.Energy` to keep
`EnergyStatistics` self-contained.

Reference: Székely & Rizzo (2023), §3.2. -/
lemma portfolio_norm_sq_eq_double_sum' {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] {n : ℕ} (w : Fin n → ℝ) (β : Fin n → E) :
    ‖∑ i, w i • β i‖ ^ 2 =
      ∑ i, ∑ j, w i * w j * @Inner.inner ℝ E _ (β i) (β j) := by
  calc
    ‖∑ i, w i • β i‖ ^ 2 =
        @Inner.inner ℝ E _ (∑ i, w i • β i) (∑ i, w i • β i) := by
      rw [real_inner_self_eq_norm_sq]
    _ = ∑ i, @Inner.inner ℝ E _ (w i • β i) (∑ j, w j • β j) := by
      rw [sum_inner]
    _ = ∑ i, ∑ j, @Inner.inner ℝ E _ (w i • β i) (w j • β j) := by
      simp_rw [inner_sum]
    _ = ∑ i, ∑ j, (w i * @Inner.inner ℝ E _ (β i) (w j • β j)) := by
      simp_rw [real_inner_smul_left]
    _ = ∑ i, ∑ j, (w i * (w j * @Inner.inner ℝ E _ (β i) (β j))) := by
      simp_rw [real_inner_smul_right]
    _ = ∑ i, ∑ j, w i * w j * @Inner.inner ℝ E _ (β i) (β j) := by
      refine Finset.sum_congr rfl fun i _ => ?_
      refine Finset.sum_congr rfl fun j _ => ?_
      ring

/-- **Squared-distance CND identity** (Hilbert-space algebra).

For zero-sum weights `Σᵢ wᵢ = 0` and points `xᵢ` in a real inner product
space, the weighted double sum of squared distances equals `−2‖Σᵢ wᵢxᵢ‖²`.

This is the quantitative Hilbert form of `DistNegativeType` for *squared*
distance.

Reference: Székely & Rizzo (2023), §3.2. -/
lemma sq_dist_cnd_of_inner' {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] {n : ℕ} (x : Fin n → E) (w : Fin n → ℝ)
    (hsum : (∑ i, w i) = 0) :
    ∑ i, ∑ j, w i * w j * ‖x i - x j‖ ^ 2 = -2 * ‖∑ i, w i • x i‖ ^ 2 := by
  have lhs_eq : ∑ i : Fin n, ∑ j : Fin n, w i * w j * ‖x i - x j‖ ^ 2 =
      -2 * ∑ i : Fin n, ∑ j : Fin n,
        w i * w j * @Inner.inner ℝ E _ (x i) (x j) := by
    have step1 : ∑ i : Fin n, ∑ j : Fin n, w i * w j * ‖x i - x j‖ ^ 2 =
        ∑ i : Fin n, ∑ j : Fin n,
          (w i * w j * ‖x i‖ ^ 2 -
           2 * (w i * w j * @Inner.inner ℝ E _ (x i) (x j)) +
           w i * w j * ‖x j‖ ^ 2) := by
      congr 1; ext i; congr 1; ext j
      rw [norm_sub_sq_real]; ring
    rw [step1]
    have h1 : ∑ i : Fin n, ∑ j : Fin n, w i * w j * ‖x i‖ ^ 2 = 0 := by
      simp_rw [show ∀ i j : Fin n, w i * w j * ‖x i‖ ^ 2 =
        (w i * ‖x i‖ ^ 2) * w j from fun i j => by ring,
        ← Finset.mul_sum, hsum, mul_zero, Finset.sum_const_zero]
    have h3 : ∑ i : Fin n, ∑ j : Fin n, w i * w j * ‖x j‖ ^ 2 = 0 := by
      simp_rw [show ∀ i j : Fin n, w i * w j * ‖x j‖ ^ 2 =
        w i * (w j * ‖x j‖ ^ 2) from fun i j => by ring,
        ← Finset.mul_sum, ← Finset.sum_mul, hsum, zero_mul]
    simp_rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, h1, h3,
      ← Finset.mul_sum]
    ring
  rw [lhs_eq, ← portfolio_norm_sq_eq_double_sum']

section OneDimensionalCND

open MeasureTheory Set

/-- The overlap of two intervals `Ioc A a`, `Ioc A b` has Lebesgue measure `min a b - A`.
This is the integral form of the *min kernel* `(a, b) ↦ min a b`, the positive-definiteness
engine behind conditional negative definiteness of `|a - b|` on `ℝ`. -/
private lemma integral_indicator_Ioc_mul {A a b : ℝ} (ha : A ≤ a) (hb : A ≤ b) :
    ∫ u, (Ioc A a).indicator (1 : ℝ → ℝ) u * (Ioc A b).indicator 1 u = min a b - A := by
  have hset : Ioc A a ∩ Ioc A b = Ioc A (min a b) := by
    ext u
    simp only [mem_inter_iff, mem_Ioc, le_min_iff]
    tauto
  have hfun : (fun u => (Ioc A a).indicator (1 : ℝ → ℝ) u * (Ioc A b).indicator 1 u) =
      (Ioc A (min a b)).indicator 1 := by
    funext u
    rw [← Set.inter_indicator_mul, hset]
    simp [Pi.one_def]
  rw [hfun, integral_indicator_one measurableSet_Ioc]
  simp [Real.volume_real_Ioc, ha, hb]

/-- **Conditional negative definiteness of `|a − b|` on `ℝ`.**

For zero-sum weights, `∑ᵢⱼ wᵢwⱼ|sᵢ−sⱼ| ≤ 0`.  Proof: `|a−b| = a + b − 2·min a b`; the linear
terms vanish by `∑w = 0`, and the min-kernel double sum is the integral of the square
`(∑ᵢ wᵢ·𝟙_{Ioc A sᵢ})²`, hence nonnegative. -/
private lemma cnd_abs_sub_real {k : ℕ} (s w : Fin k → ℝ) (hw : (∑ i, w i) = 0) :
    ∑ i, ∑ j, w i * w j * |s i - s j| ≤ 0 := by
  rcases isEmpty_or_nonempty (Fin k) with hk | hk
  · simp
  -- a uniform lower bound for the sample points
  set A : ℝ := Finset.univ.inf' Finset.univ_nonempty s with hA
  have hAle : ∀ i, A ≤ s i := fun i => Finset.inf'_le _ (Finset.mem_univ i)
  -- the indicator functions of (A, sᵢ]
  set f : Fin k → ℝ → ℝ := fun i => (Ioc A (s i)).indicator 1 with hf
  have hf_int : ∀ i, Integrable (f i) := fun i =>
    (integrable_indicator_iff measurableSet_Ioc).2 (integrableOn_const measure_Ioc_lt_top.ne)
  have hff_int : ∀ i j, Integrable (fun u => w i * w j * (f i u * f j u)) := by
    intro i j
    have heq : (fun u => f i u * f j u) = (Ioc A (s i) ∩ Ioc A (s j)).indicator 1 := by
      funext u
      rw [hf, ← Set.inter_indicator_mul]
      simp [Pi.one_def]
    refine Integrable.const_mul ?_ _
    rw [heq]
    exact (integrable_indicator_iff (measurableSet_Ioc.inter measurableSet_Ioc)).2
      (integrableOn_const ((measure_mono Set.inter_subset_left).trans_lt measure_Ioc_lt_top).ne)
  -- |sᵢ − sⱼ| = sᵢ + sⱼ − 2 min(sᵢ, sⱼ)
  have habs : ∀ i j, |s i - s j| = s i + s j - 2 * min (s i) (s j) := by
    intro i j
    rcases le_total (s i) (s j) with h | h
    · rw [abs_sub_comm, abs_of_nonneg (by linarith), min_eq_left h]; ring
    · rw [abs_of_nonneg (by linarith), min_eq_right h]; ring
  -- the linear terms vanish by zero-sum weights
  have hlin₁ : ∑ i, ∑ j, w i * w j * s i = 0 := by
    simp_rw [show ∀ i j : Fin k, w i * w j * s i = (w i * s i) * w j from fun i j => by ring,
      ← Finset.mul_sum, hw, mul_zero, Finset.sum_const_zero]
  have hlin₂ : ∑ i, ∑ j, w i * w j * s j = 0 := by
    simp_rw [show ∀ i j : Fin k, w i * w j * s j = w i * (w j * s j) from fun i j => by ring,
      ← Finset.mul_sum, ← Finset.sum_mul, hw, zero_mul]
  have hsplit : ∑ i, ∑ j, w i * w j * |s i - s j| =
      -2 * ∑ i, ∑ j, w i * w j * min (s i) (s j) := by
    simp_rw [habs]
    have expand : ∀ i j : Fin k, w i * w j * (s i + s j - 2 * min (s i) (s j)) =
        w i * w j * s i + w i * w j * s j - 2 * (w i * w j * min (s i) (s j)) :=
      fun i j => by ring
    simp_rw [expand, Finset.sum_sub_distrib, Finset.sum_add_distrib, hlin₁, hlin₂,
      ← Finset.mul_sum]
    ring
  rw [hsplit]
  -- the min-kernel double sum is ≥ 0: it equals ∫ (∑ᵢ wᵢ fᵢ)²
  have hshift : ∑ i, ∑ j, w i * w j * min (s i) (s j) =
      ∑ i, ∑ j, w i * w j * (min (s i) (s j) - A) := by
    have hAterm : ∑ i, ∑ j, w i * w j * A = 0 := by
      simp_rw [show ∀ i j : Fin k, w i * w j * A = (w i * A) * w j from fun i j => by ring,
        ← Finset.mul_sum, hw, mul_zero, Finset.sum_const_zero]
    have expand : ∀ i j : Fin k, w i * w j * (min (s i) (s j) - A) =
        w i * w j * min (s i) (s j) - w i * w j * A := fun i j => by ring
    simp_rw [expand, Finset.sum_sub_distrib, hAterm, sub_zero]
  have hkernel : ∑ i, ∑ j, w i * w j * (min (s i) (s j) - A) =
      ∫ u, (∑ i, w i * f i u) ^ 2 := by
    have hsq : ∀ u, (∑ i, w i * f i u) ^ 2 = ∑ i, ∑ j, w i * w j * (f i u * f j u) := by
      intro u
      rw [sq, Finset.sum_mul_sum]
      exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring
    simp_rw [hsq]
    rw [integral_finsetSum _ fun i _ => integrable_finsetSum _ fun j _ => hff_int i j]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [integral_finsetSum _ fun j _ => hff_int i j]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [integral_const_mul, integral_indicator_Ioc_mul (hAle i) (hAle j)]
  have hnonneg : 0 ≤ ∑ i, ∑ j, w i * w j * min (s i) (s j) := by
    rw [hshift, hkernel]
    exact integral_nonneg fun u => sq_nonneg _
  linarith

end OneDimensionalCND

section GaussianProjection

open MeasureTheory ProbabilityTheory
open scoped RealInnerProductSpace NNReal

/-- The absolute first moment of a centred real Gaussian scales as `√V`. -/
private lemma integral_abs_gaussianReal (V : ℝ≥0) :
    ∫ t, |t| ∂(gaussianReal 0 V) = Real.sqrt V * ∫ t, |t| ∂(gaussianReal 0 1) := by
  have hmap : (gaussianReal 0 1).map (Real.sqrt V * ·) = gaussianReal 0 V := by
    rw [gaussianReal_map_const_mul, mul_zero]
    congr 1
    exact NNReal.coe_injective (by simp [Real.sq_sqrt V.coe_nonneg])
  rw [← hmap, integral_map (by fun_prop) (by fun_prop)]
  simp_rw [abs_mul, abs_of_nonneg (Real.sqrt_nonneg _)]
  rw [integral_const_mul]

/-- The standard Gaussian absolute first moment is positive (we never need its value
`√(2/π)`, only positivity). -/
private lemma integral_abs_gaussianReal_one_pos : 0 < ∫ t, |t| ∂(gaussianReal (0 : ℝ) 1) := by
  have hint : Integrable (fun t : ℝ => |t|) (gaussianReal 0 1) :=
    (memLp_one_iff_integrable.mp (memLp_id_gaussianReal 1)).abs
  rw [integral_pos_iff_support_of_nonneg (fun t => abs_nonneg t) hint]
  have hsupp : Function.support (fun t : ℝ => |t|) = {(0 : ℝ)}ᶜ := by
    ext t
    simp [Function.mem_support]
  haveI : NoAtoms (gaussianReal (0 : ℝ) 1) := noAtoms_gaussianReal one_ne_zero
  rw [hsupp, measure_compl (measurableSet_singleton 0) (measure_ne_top _ _), measure_singleton]
  simp

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
  [MeasurableSpace E] [BorelSpace E]

/-- Pushing the standard Gaussian forward along `⟪v, ·⟫` gives a centred real Gaussian with
variance `‖v‖²`. -/
private lemma map_stdGaussian_innerSL (v : E) :
    (stdGaussian E).map (innerSL ℝ v) = gaussianReal 0 (‖v‖₊ ^ 2) := by
  rw [IsGaussian.map_eq_gaussianReal (innerSL ℝ v)]
  congr 1
  · exact integral_strongDual_stdGaussian (innerSL ℝ v)
  · rw [variance_dual_stdGaussian, innerSL_apply_norm]
    refine NNReal.coe_injective ?_
    rw [Real.coe_toNNReal _ (sq_nonneg _)]
    simp

lemma integrable_abs_inner (v : E) :
    Integrable (fun g : E => |⟪v, g⟫|) (stdGaussian E) :=
  ((innerSL ℝ v).integrable_comp IsGaussian.integrable_id).abs

/-- **Gaussian projection representation of the norm**: `∫ |⟪v, g⟫| dγ(g) = ‖v‖ · c` with
`c = ∫|t| dN(0,1) > 0`.  This is what reduces Hilbert-space CND to the one-dimensional case. -/
lemma integral_abs_inner_stdGaussian (v : E) :
    ∫ g, |⟪v, g⟫| ∂(stdGaussian E) = ‖v‖ * ∫ t, |t| ∂(gaussianReal (0 : ℝ) 1) := by
  have h1 : ∫ g, |⟪v, g⟫| ∂(stdGaussian E) =
      ∫ t, |t| ∂((stdGaussian E).map (innerSL ℝ v)) := by
    rw [integral_map (by fun_prop) (by fun_prop)]
    simp
  rw [h1, map_stdGaussian_innerSL, integral_abs_gaussianReal]
  congr 1
  push_cast
  exact Real.sqrt_sq (norm_nonneg v)

/-- CND of the distance on a finite-dimensional real inner product space, by Gaussian
projection onto the one-dimensional case. -/
private lemma cnd_dist_of_finiteDimensional {k : ℕ} (x : Fin k → E) (w : Fin k → ℝ)
    (hw : (∑ i, w i) = 0) :
    ∑ i, ∑ j, w i * w j * ‖x i - x j‖ ≤ 0 := by
  set c := ∫ t, |t| ∂(gaussianReal (0 : ℝ) 1) with hc
  have hcpos : 0 < c := integral_abs_gaussianReal_one_pos
  have hterm_int : ∀ i j : Fin k,
      Integrable (fun g : E => w i * w j * |⟪x i, g⟫ - ⟪x j, g⟫|) (stdGaussian E) := by
    intro i j
    have h := (integrable_abs_inner (x i - x j)).const_mul (w i * w j)
    simpa [inner_sub_left] using h
  have key : ∫ g, ∑ i, ∑ j, w i * w j * |⟪x i, g⟫ - ⟪x j, g⟫| ∂(stdGaussian E) =
      (∑ i, ∑ j, w i * w j * ‖x i - x j‖) * c := by
    rw [integral_finsetSum _ fun i _ => integrable_finsetSum _ fun j _ => hterm_int i j,
      Finset.sum_mul]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [integral_finsetSum _ fun j _ => hterm_int i j, Finset.sum_mul]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [integral_const_mul]
    simp_rw [← inner_sub_left]
    rw [integral_abs_inner_stdGaussian (x i - x j), ← hc]
    ring
  have hle : ∫ g, ∑ i, ∑ j, w i * w j * |⟪x i, g⟫ - ⟪x j, g⟫| ∂(stdGaussian E) ≤ 0 :=
    integral_nonpos fun g => cnd_abs_sub_real (fun i => ⟪x i, g⟫) w hw
  rw [key] at hle
  nlinarith [hle, hcpos]

end GaussianProjection

/-- **Euclidean/Hilbert spaces have negative type** (Schoenberg's theorem).

For any real inner-product space `E`, the metric `dist` is conditionally
negative definite: for every finite family of points and zero-sum weights,
the weighted double sum of pairwise distances is ≤ 0.

This is the key lemma that discharges the `DistNegativeType` hypothesis from
`v_statistic_nonneg`, `energy_distance_nonneg`, and `dcov_nonneg` when
applied to Euclidean or Hilbert spaces.

**Proof outline** (Gaussian projection, classically equivalent to Schoenberg's
Bernstein-function argument):
1. The points lie in the finite-dimensional subspace `F = span ℝ (range x)`, so it
   suffices to prove the inequality there.
2. On a finite-dimensional space, `‖v‖ · c = ∫ |⟪v, g⟫| dγ(g)` where `γ` is the standard
   Gaussian and `c = ∫|t| dN(0,1) > 0` (`integral_abs_inner_stdGaussian`).
3. This reduces the claim pointwise in `g` to the one-dimensional inequality
   `Σᵢⱼ wᵢwⱼ|sᵢ−sⱼ| ≤ 0` for `sᵢ = ⟪xᵢ, g⟫`, proven via the min-kernel identity
   (`cnd_abs_sub_real`).

Reference: Schoenberg (1938), Lyons (2013) §3, Székely & Rizzo (2023) §3.2. -/
theorem distNegativeType_of_innerProductSpace
    (E : Type*) [NormedAddCommGroup E] [InnerProductSpace ℝ E] :
    DistNegativeType E := by
  intro k x w hw
  simp_rw [dist_eq_norm]
  -- reduce to the span of the points, which is finite dimensional
  set F : Submodule ℝ E := Submodule.span ℝ (Set.range x) with hF
  haveI : FiniteDimensional ℝ F := FiniteDimensional.span_of_finite ℝ (Set.finite_range x)
  letI : MeasurableSpace F := borel F
  haveI : BorelSpace F := ⟨rfl⟩
  set y : Fin k → F := fun i => ⟨x i, Submodule.subset_span (Set.mem_range_self i)⟩ with hy
  have hxy : ∀ i j : Fin k, ‖x i - x j‖ = ‖y i - y j‖ := by
    intro i j
    rw [Submodule.coe_norm]
    norm_cast
  simp_rw [hxy]
  exact cnd_dist_of_finiteDimensional y w hw

end EnergyStatistics
