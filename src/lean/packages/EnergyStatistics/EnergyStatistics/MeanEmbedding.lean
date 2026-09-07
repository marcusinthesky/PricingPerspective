import EnergyStatistics.EnergyDistance
import EnergyStatistics.NegativeType

set_option linter.style.longLine false

/-!
# Mean embedding of probability measures into an energy L² space (§4.6 keystone)

This module builds the explicit RKHS-free realization needed to discharge the
`EnergyKernelEmbedding.identity` axiom used by Paper 3 (`Continuous/Energy.lean`). Stages s1--s3
are implemented here; the remaining s4 connection-layer wiring will expose the ROADMAP §4.6 keystone
`energy_sq_eq_energyDistanceSq` (Sejdinovic et al. 2013, Thm 22):

`‖energyEmbed P − energyEmbed Q‖²_H = energyDistanceSq P Q`.

Mathlib has no RKHS, so the target Hilbert space is built explicitly as
`Lp ℝ 2 ((stdGaussian E).prod volume)`, reusing the Gaussian-projection machinery already
proven for `distNegativeType_of_innerProductSpace` (`NegativeType.lean`).

## Construction (staged; this file lands the stages bottom-up)

* **s1 (this file, PROVEN): the slice lemma.** For the *signed interval indicator*
  `signedIoc a := 𝟙_{(0,a]} − 𝟙_{(a,0]}` on `ℝ`,
  `∫ u, signedIoc a u · signedIoc b u = (|a| + |b| − |a − b|) / 2`.
  This is the one-dimensional kernel `k(a,b) = ½(|a|+|b|−|a−b|)` written as an honest L²
  inner product `⟪signedIoc a, signedIoc b⟫_{L²(volume)}`, so `k` is positive semidefinite by
  construction and `signedIoc` is the point embedding on `ℝ`.

* **s2 (PROVEN).** `energyPointEmbed` packages the point embedding
  `Φ : E → Lp ℝ 2 ((stdGaussian E).prod volume)`,
  `Φ x := fun (g, u) ↦ √(2/c) · signedIoc ⟪x, g⟫ u` (`c = gaussAbsMoment = ∫|t| dN(0,1) > 0`), as a
  genuine `MemLp.toLp` element (reusing `integral_gaussianEnergyKernel`'s Gaussian-average kernel
  identity and `integrable_abs_inner`/`integral_abs_inner_stdGaussian`, `NegativeType.lean`).
  `inner_energyPointEmbed` proves `⟪Φ x, Φ y⟫ = ‖x‖ + ‖y‖ − dist x y` and
  `norm_sub_energyPointEmbed_sq` the companion norm form `‖Φ x − Φ y‖² = 2·dist x y`.

* **s3 (PROVEN).** `energyEmbed P := ∫ energyPointEmbed dP` is the Bochner mean in `Lp`, and
  `norm_sub_energyEmbed_sq` proves
  `‖energyEmbed P − energyEmbed Q‖² = energyDistanceSq P Q` under `Integrable (‖·‖)`. The
  `‖x‖`/`‖y‖` terms cancel because `P − Q` carries total mass 0, leaving exactly
  `2∫∫dist dP dQ − ∫∫dist dP dP − ∫∫dist dQ dQ`.

* **s4 (only pending stage).** Canonical `EnergyKernelEmbedding E` instance (`sys_var := ‖energyEmbed·‖²`,
  `sys_cov := ⟪energyEmbed·, energyEmbed·⟫`, `energy_sq := energyDistanceSq`, `identity` by
  polarization `mean_embedding_polarization`) + the named keystone
  `energy_sq_eq_energyDistanceSq`, wired into `Connections/EnergyStatistics.lean`.

Constants (`c`, the `½`, the `√(2/c)` rescale) are absorbed into `Φ` by construction, which also
settles the ½κ² bookkeeping recorded in ROADMAP Appendix C.

## References

* Sejdinovic, Sriperumbudur, Gretton, Fukumizu (2013). Equivalence of distance-based and
  RKHS-based statistics. Ann. Statist. 41(5), Thm 22.
* Székely & Rizzo (2023), §3.2; Schoenberg (1938); Lyons (2013), §3.
-/

namespace EnergyStatistics

open MeasureTheory Set

/-- **Signed interval indicator** — the point embedding of `ℝ` into `L²(volume)`.

`signedIoc a u = 𝟙_{(0,a]}(u) − 𝟙_{(a,0]}(u)`, i.e. `+1` on `(0, a]` when `a > 0`, `−1` on
`(a, 0]` when `a < 0`, and `0` when `a = 0`. Its self–inner-product is `‖signedIoc a‖²_{L²} = |a|`
and `⟪signedIoc a, signedIoc b⟫ = ½(|a|+|b|−|a−b|)` (`integral_signedIoc_mul`). -/
noncomputable def signedIoc (a : ℝ) : ℝ → ℝ :=
  fun u => (Ioc 0 a).indicator 1 u - (Ioc a 0).indicator 1 u

/-- **Slice lemma (s1).** The L² inner product of two signed interval indicators realizes the
one-dimensional energy kernel `k(a,b) = ½(|a| + |b| − |a − b|)`:

`∫ u, signedIoc a u · signedIoc b u = (|a| + |b| − |a − b|) / 2`.

Because the left side is `⟪signedIoc a, signedIoc b⟫_{L²(volume)}`, this exhibits `k` as a genuine
Gram kernel — the positive-semidefinite engine behind the mean-embedding keystone (Sejdinovic
2013, Thm 22). Reference: Székely & Rizzo (2023), §3.2. -/
lemma integral_signedIoc_mul (a b : ℝ) :
    ∫ u, signedIoc a u * signedIoc b u = (|a| + |b| - |a - b|) / 2 := by
  -- Overlap of two half-open intervals sharing the lower endpoint `0`: `∫ = min x y`.
  have hlow : ∀ x y : ℝ, 0 ≤ x → 0 ≤ y →
      ∫ u, (Ioc 0 x).indicator (1 : ℝ → ℝ) u * (Ioc 0 y).indicator 1 u = min x y := by
    intro x y hx hy
    have hset : Ioc (0 : ℝ) x ∩ Ioc 0 y = Ioc 0 (min x y) := by
      ext u; simp only [mem_inter_iff, mem_Ioc, le_min_iff]; tauto
    have hfun : (fun u => (Ioc (0 : ℝ) x).indicator (1 : ℝ → ℝ) u * (Ioc 0 y).indicator 1 u) =
        (Ioc 0 (min x y)).indicator 1 := by
      funext u; rw [← Set.inter_indicator_mul, hset]; simp [Pi.one_def]
    rw [hfun, integral_indicator_one measurableSet_Ioc,
      Real.volume_real_Ioc_of_le (le_min hx hy), sub_zero]
  -- Overlap of two half-open intervals sharing the upper endpoint `0`: `∫ = -max x y`.
  have hupp : ∀ x y : ℝ, x ≤ 0 → y ≤ 0 →
      ∫ u, (Ioc x 0).indicator (1 : ℝ → ℝ) u * (Ioc y 0).indicator 1 u = -max x y := by
    intro x y hx hy
    have hset : Ioc x (0 : ℝ) ∩ Ioc y 0 = Ioc (max x y) 0 := by
      ext u; simp only [mem_inter_iff, mem_Ioc, max_lt_iff]; tauto
    have hfun : (fun u => (Ioc x (0 : ℝ)).indicator (1 : ℝ → ℝ) u * (Ioc y 0).indicator 1 u) =
        (Ioc (max x y) 0).indicator 1 := by
      funext u; rw [← Set.inter_indicator_mul, hset]; simp [Pi.one_def]
    rw [hfun, integral_indicator_one measurableSet_Ioc,
      Real.volume_real_Ioc_of_le (max_le hx hy), zero_sub]
  -- Opposite-sign intervals `(0, x]` and `(y, 0]` are disjoint.
  have hempty : ∀ x y : ℝ, 0 ≤ x → y ≤ 0 → Ioc (0 : ℝ) x ∩ Ioc y 0 = ∅ := by
    intro x y _ _
    rw [Set.eq_empty_iff_forall_notMem]
    intro v hv
    simp only [mem_inter_iff, mem_Ioc] at hv
    linarith [hv.1.1, hv.2.2]
  rcases le_total 0 a with ha | ha <;> rcases le_total 0 b with hb | hb
  · -- `0 ≤ a`, `0 ≤ b`: both indicators positive, integral `= min a b`.
    have hca : signedIoc a = (Ioc 0 a).indicator 1 := by
      funext u
      simp only [signedIoc, Set.Ioc_eq_empty (not_lt.mpr ha), Set.indicator_empty, sub_zero]
    have hcb : signedIoc b = (Ioc 0 b).indicator 1 := by
      funext u
      simp only [signedIoc, Set.Ioc_eq_empty (not_lt.mpr hb), Set.indicator_empty, sub_zero]
    rw [hca, hcb, hlow a b ha hb, abs_of_nonneg ha, abs_of_nonneg hb]
    rcases le_total a b with hab | hab
    · rw [min_eq_left hab, abs_of_nonpos (by linarith : a - b ≤ 0)]; ring
    · rw [min_eq_right hab, abs_of_nonneg (by linarith : (0 : ℝ) ≤ a - b)]; ring
  · -- `0 ≤ a`, `b ≤ 0`: disjoint supports, integral `= 0`.
    have hz : (fun u => signedIoc a u * signedIoc b u) = fun _ => (0 : ℝ) := by
      funext u
      have h1 : signedIoc a u = (Ioc 0 a).indicator 1 u := by
        simp only [signedIoc, Set.Ioc_eq_empty (not_lt.mpr ha), Set.indicator_empty, sub_zero]
      have h2 : signedIoc b u = -(Ioc b 0).indicator 1 u := by
        simp only [signedIoc, Set.Ioc_eq_empty (not_lt.mpr hb), Set.indicator_empty, zero_sub]
      have h3 : (Ioc (0 : ℝ) a).indicator (1 : ℝ → ℝ) u * (Ioc b 0).indicator 1 u = 0 := by
        rw [← Set.inter_indicator_mul, hempty a b ha hb]; simp
      rw [h1, h2, mul_neg, h3, neg_zero]
    rw [hz, integral_zero, abs_of_nonneg ha, abs_of_nonpos hb,
      abs_of_nonneg (by linarith : (0 : ℝ) ≤ a - b)]
    ring
  · -- `a ≤ 0`, `0 ≤ b`: disjoint supports, integral `= 0`.
    have hz : (fun u => signedIoc a u * signedIoc b u) = fun _ => (0 : ℝ) := by
      funext u
      have h1 : signedIoc a u = -(Ioc a 0).indicator 1 u := by
        simp only [signedIoc, Set.Ioc_eq_empty (not_lt.mpr ha), Set.indicator_empty, zero_sub]
      have h2 : signedIoc b u = (Ioc 0 b).indicator 1 u := by
        simp only [signedIoc, Set.Ioc_eq_empty (not_lt.mpr hb), Set.indicator_empty, sub_zero]
      have h3 : (Ioc a (0 : ℝ)).indicator (1 : ℝ → ℝ) u * (Ioc 0 b).indicator 1 u = 0 := by
        rw [mul_comm, ← Set.inter_indicator_mul, hempty b a hb ha]; simp
      rw [h1, h2, neg_mul, h3, neg_zero]
    rw [hz, integral_zero, abs_of_nonpos ha, abs_of_nonneg hb,
      abs_of_nonpos (by linarith : a - b ≤ 0)]
    ring
  · -- `a ≤ 0`, `b ≤ 0`: both indicators negated, integral `= -max a b`.
    have hca : signedIoc a = -(Ioc a 0).indicator 1 := by
      funext u
      simp only [signedIoc, Set.Ioc_eq_empty (not_lt.mpr ha), Set.indicator_empty,
        Pi.neg_apply, zero_sub]
    have hcb : signedIoc b = -(Ioc b 0).indicator 1 := by
      funext u
      simp only [signedIoc, Set.Ioc_eq_empty (not_lt.mpr hb), Set.indicator_empty,
        Pi.neg_apply, zero_sub]
    rw [hca, hcb]
    simp only [Pi.neg_apply, neg_mul_neg]
    rw [hupp a b ha hb, abs_of_nonpos ha, abs_of_nonpos hb]
    rcases le_total a b with hab | hab
    · rw [max_eq_right hab, abs_of_nonpos (by linarith : a - b ≤ 0)]; ring
    · rw [max_eq_left hab, abs_of_nonneg (by linarith : (0 : ℝ) ≤ a - b)]; ring

section GaussianKernel

open MeasureTheory ProbabilityTheory
open scoped RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [FiniteDimensional ℝ E] [MeasurableSpace E] [BorelSpace E]

/-- **Gaussian energy kernel (s2 core).**

Averaging the one-dimensional energy kernel `k(a,b) = ½(|a|+|b|−|a−b|)` (`integral_signedIoc_mul`)
along the Gaussian projections `a = ⟪x,g⟫`, `b = ⟪y,g⟫` recovers the Hilbert distance kernel up to
the constant `c = ∫|t| dN(0,1)`:

`∫ g, ½(|⟪x,g⟫|+|⟪y,g⟫|−|⟪x,g⟫−⟪y,g⟫|) ∂γ = (c/2)·(‖x‖+‖y‖−dist x y)`.

This is the mathematical core of the point embedding (s2): it turns the abstract requirement
`⟪Φ x, Φ y⟫ = ‖x‖+‖y‖−dist x y` into a computed Gaussian integral, applying
`integral_abs_inner_stdGaussian` to each of `x`, `y`, and `x − y` (using
`⟪x,g⟫ − ⟪y,g⟫ = ⟪x−y,g⟫`, `inner_sub_left`). The Lp packaging that makes `Φ x` a genuine element
of `Lp ℝ 2 (γ.prod volume)` is completed by `energyPointEmbed` in s2. -/
lemma integral_gaussianEnergyKernel (x y : E) :
    ∫ g, (|⟪x, g⟫| + |⟪y, g⟫| - |⟪x, g⟫ - ⟪y, g⟫|) / 2 ∂(stdGaussian E)
      = (∫ t, |t| ∂(gaussianReal (0 : ℝ) 1)) / 2 * (‖x‖ + ‖y‖ - dist x y) := by
  have hx := integrable_abs_inner x
  have hy := integrable_abs_inner y
  have hxy := integrable_abs_inner (x - y)
  have hadd : Integrable (fun g => |⟪x, g⟫| + |⟪y, g⟫|) (stdGaussian E) := hx.add hy
  simp_rw [← inner_sub_left]
  rw [integral_div, integral_sub hadd hxy, integral_add hx hy,
    integral_abs_inner_stdGaussian x, integral_abs_inner_stdGaussian y,
    integral_abs_inner_stdGaussian (x - y), dist_eq_norm]
  ring

/-- The absolute first moment of the standard real Gaussian, `c = ∫|t| dN(0,1) = √(2/π)` — only
positivity is ever used (never the closed form), so this is kept as an internal constant. -/
private noncomputable def gaussAbsMoment : ℝ := ∫ t, |t| ∂(gaussianReal (0 : ℝ) 1)

private lemma gaussAbsMoment_pos : 0 < gaussAbsMoment := by
  unfold gaussAbsMoment
  have hint : Integrable (fun t : ℝ => |t|) (gaussianReal 0 1) :=
    (memLp_one_iff_integrable.mp (memLp_id_gaussianReal 1)).abs
  rw [integral_pos_iff_support_of_nonneg (fun t => abs_nonneg t) hint]
  have hsupp : Function.support (fun t : ℝ => |t|) = {(0 : ℝ)}ᶜ := by
    ext t; simp [Function.mem_support]
  haveI : NoAtoms (gaussianReal (0 : ℝ) 1) := noAtoms_gaussianReal one_ne_zero
  rw [hsupp, measure_compl (measurableSet_singleton 0) (measure_ne_top _ _), measure_singleton]
  simp

/-- `signedIoc a` squared self-integrates to `|a|` (s1 at `b := a`, with the `|a-a|=0` term
dropped). Feeds the L² norm computation for the point embedding. -/
private lemma integral_signedIoc_sq (a : ℝ) :
    ∫ u, signedIoc a u * signedIoc a u = |a| := by
  have h := integral_signedIoc_mul a a
  rw [sub_self, abs_zero] at h
  linarith

/-- `signedIoc a` takes values in `[-1, 1]` pointwise. -/
private lemma abs_signedIoc_le_one (a u : ℝ) : |signedIoc a u| ≤ 1 := by
  unfold signedIoc
  rcases le_total 0 a with ha | ha
  · simp only [Set.Ioc_eq_empty (show ¬ a < 0 by linarith), Set.indicator_empty, sub_zero]
    rw [Set.indicator_apply]
    split_ifs <;> simp
  · simp only [Set.Ioc_eq_empty (show ¬ 0 < a by linarith), Set.indicator_empty, zero_sub, abs_neg]
    rw [Set.indicator_apply]
    split_ifs <;> simp

/-- `signedIoc a` is the difference of two finite-measure interval indicators, hence integrable. -/
private lemma integrable_signedIoc (a : ℝ) : Integrable (signedIoc a) := by
  unfold signedIoc
  exact ((integrable_indicator_iff measurableSet_Ioc).2
      (integrableOn_const measure_Ioc_lt_top.ne)).sub
    ((integrable_indicator_iff measurableSet_Ioc).2
      (integrableOn_const measure_Ioc_lt_top.ne))

/-- `signedIoc a` squared is integrable, dominated by `|signedIoc a| ≤ 1`. -/
private lemma integrable_signedIoc_sq (a : ℝ) :
    Integrable (fun u => signedIoc a u ^ 2) := by
  have hf := integrable_signedIoc a
  refine Integrable.mono' hf.abs (hf.aestronglyMeasurable.pow 2) ?_
  filter_upwards with u
  rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _), ← sq_abs]
  exact sq_le (abs_nonneg _) (abs_signedIoc_le_one a u)

/-- **Joint measurability of `signedIoc`.** Written as a difference of indicators of the
measurable sets `{0 < u ≤ a}` and `{a < u ≤ 0}` (both jointly measurable in `(a, u)`). -/
private lemma measurable_signedIoc_uncurry :
    Measurable (fun q : ℝ × ℝ => signedIoc q.1 q.2) := by
  have hS1 : MeasurableSet {q : ℝ × ℝ | (0 : ℝ) < q.2 ∧ q.2 ≤ q.1} :=
    (measurableSet_lt measurable_const measurable_snd).inter
      (measurableSet_le measurable_snd measurable_fst)
  have hS2 : MeasurableSet {q : ℝ × ℝ | q.1 < q.2 ∧ q.2 ≤ (0 : ℝ)} :=
    (measurableSet_lt measurable_fst measurable_snd).inter
      (measurableSet_le measurable_snd measurable_const)
  have heq : (fun q : ℝ × ℝ => signedIoc q.1 q.2) =
      Set.indicator {q : ℝ × ℝ | (0 : ℝ) < q.2 ∧ q.2 ≤ q.1} (1 : ℝ × ℝ → ℝ) -
        Set.indicator {q : ℝ × ℝ | q.1 < q.2 ∧ q.2 ≤ (0 : ℝ)} (1 : ℝ × ℝ → ℝ) := by
    funext q
    simp [signedIoc, Set.indicator_apply, Set.mem_Ioc, Set.mem_setOf_eq]
  rw [heq]
  exact (Measurable.indicator measurable_const hS1).sub (Measurable.indicator measurable_const hS2)

omit [FiniteDimensional ℝ E] in
/-- Fixing `x`, `p ↦ signedIoc ⟪x, p.1⟫ p.2` is measurable on `E × ℝ` (composition of
`measurable_signedIoc_uncurry` with the continuous projection `p ↦ (⟪x, p.1⟫, p.2)`). -/
private lemma measurable_signedIoc_inner (x : E) :
    Measurable (fun p : E × ℝ => signedIoc ⟪x, p.1⟫ p.2) := by
  have hcomp : Measurable (fun p : E × ℝ => (⟪x, p.1⟫, p.2)) :=
    ((innerSL ℝ x).continuous.measurable.comp measurable_fst).prodMk measurable_snd
  exact measurable_signedIoc_uncurry.comp hcomp

/-- **Helper C.** `p ↦ signedIoc ⟪x, p.1⟫ p.2` lies in `L²(γ.prod volume)`: the slice integral
over `u` is `|⟪x, g⟫|` (`integral_signedIoc_sq`) and `g ↦ |⟪x, g⟫|` is `γ`-integrable
(`integrable_abs_inner`), so Fubini (`integrable_prod_iff`) gives the square integrable. -/
private lemma memLp_signedIoc_inner (x : E) :
    MemLp (fun p : E × ℝ => signedIoc ⟪x, p.1⟫ p.2) 2
      ((stdGaussian E).prod (volume : Measure ℝ)) := by
  have hmeas : AEStronglyMeasurable (fun p : E × ℝ => signedIoc ⟪x, p.1⟫ p.2)
      ((stdGaussian E).prod (volume : Measure ℝ)) :=
    (measurable_signedIoc_inner x).aestronglyMeasurable
  rw [memLp_two_iff_integrable_sq hmeas]
  have hmeasSq : AEStronglyMeasurable (fun p : E × ℝ => signedIoc ⟪x, p.1⟫ p.2 ^ 2)
      ((stdGaussian E).prod (volume : Measure ℝ)) :=
    ((measurable_signedIoc_inner x).pow_const 2).aestronglyMeasurable
  rw [integrable_prod_iff hmeasSq]
  constructor
  · filter_upwards with g
    simpa using integrable_signedIoc_sq ⟪x, g⟫
  · have heq : (fun g : E => ∫ u, ‖signedIoc ⟪x, g⟫ u ^ 2‖ ∂(volume : Measure ℝ)) =
        fun g : E => |⟪x, g⟫| := by
      funext g
      have h1 : ∀ u, ‖signedIoc ⟪x, g⟫ u ^ 2‖ = signedIoc ⟪x, g⟫ u * signedIoc ⟪x, g⟫ u := by
        intro u
        rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _), sq]
      simp_rw [h1]
      exact integral_signedIoc_sq ⟪x, g⟫
    rw [heq]
    exact integrable_abs_inner x

/-- **Point embedding (s2, fully packaged).** `Φ x := √(2/c) · signedIoc ⟪x, ·⟫` as a genuine
element of `Lp ℝ 2 (γ.prod volume)`, `γ = stdGaussian E`, `c = gaussAbsMoment > 0`. This is the
`MemLp.toLp` packaging promised in the module docstring; `inner_energyPointEmbed` below recovers
the Hilbert kernel `‖x‖ + ‖y‖ − dist x y`. -/
noncomputable def energyPointEmbed (x : E) :
    Lp ℝ 2 ((stdGaussian E).prod (volume : Measure ℝ)) :=
  ((memLp_signedIoc_inner x).const_smul (Real.sqrt (2 / gaussAbsMoment))).toLp
    (fun p : E × ℝ => Real.sqrt (2 / gaussAbsMoment) * signedIoc ⟪x, p.1⟫ p.2)

/-- **s2, discharged.** The point embedding realizes the Hilbert distance kernel:
`⟪Φ x, Φ y⟫ = ‖x‖ + ‖y‖ − dist x y`. Unfolds `⟪·,·⟫` on `Lp ℝ 2 _` to the product-measure integral
(`L2.inner_def`), rescales by `c² = 2/gaussAbsMoment`, applies Fubini (`integral_prod`,
integrability via Cauchy–Schwarz `MemLp.integrable_mul`) to the slice kernel
`integral_signedIoc_mul` (s1), and closes with the Gaussian average `integral_gaussianEnergyKernel`
(s2 core). -/
lemma inner_energyPointEmbed (x y : E) :
    ⟪energyPointEmbed x, energyPointEmbed y⟫ = ‖x‖ + ‖y‖ - dist x y := by
  set c := Real.sqrt (2 / gaussAbsMoment) with hc
  have step1 : ⟪energyPointEmbed x, energyPointEmbed y⟫ =
      ∫ p : E × ℝ, (c * signedIoc ⟪x, p.1⟫ p.2) * (c * signedIoc ⟪y, p.1⟫ p.2)
        ∂((stdGaussian E).prod (volume : Measure ℝ)) := by
    rw [L2.inner_def]
    refine integral_congr_ae ?_
    filter_upwards [MemLp.coeFn_toLp ((memLp_signedIoc_inner x).const_smul c),
      MemLp.coeFn_toLp ((memLp_signedIoc_inner y).const_smul c)] with p hxp hyp
    rw [RCLike.inner_apply, conj_trivial]
    show (energyPointEmbed y) p * (energyPointEmbed x) p = _
    rw [(show (energyPointEmbed x) p = c * signedIoc ⟪x, p.1⟫ p.2 from hxp),
      (show (energyPointEmbed y) p = c * signedIoc ⟪y, p.1⟫ p.2 from hyp)]
    ring
  have step2 : (∫ p : E × ℝ, (c * signedIoc ⟪x, p.1⟫ p.2) * (c * signedIoc ⟪y, p.1⟫ p.2)
        ∂((stdGaussian E).prod (volume : Measure ℝ))) =
      c * c * ∫ p : E × ℝ, signedIoc ⟪x, p.1⟫ p.2 * signedIoc ⟪y, p.1⟫ p.2
        ∂((stdGaussian E).prod (volume : Measure ℝ)) := by
    rw [← integral_const_mul]
    congr 1
    funext p
    ring
  have hIntProd : Integrable (fun p : E × ℝ => signedIoc ⟪x, p.1⟫ p.2 * signedIoc ⟪y, p.1⟫ p.2)
      ((stdGaussian E).prod (volume : Measure ℝ)) :=
    (memLp_signedIoc_inner x).integrable_mul (memLp_signedIoc_inner y)
  have step3 : (∫ p : E × ℝ, signedIoc ⟪x, p.1⟫ p.2 * signedIoc ⟪y, p.1⟫ p.2
        ∂((stdGaussian E).prod (volume : Measure ℝ))) =
      ∫ g, ∫ u, signedIoc ⟪x, g⟫ u * signedIoc ⟪y, g⟫ u ∂(volume : Measure ℝ) ∂(stdGaussian E) :=
    integral_prod _ hIntProd
  have step4 : (∫ g, ∫ u, signedIoc ⟪x, g⟫ u * signedIoc ⟪y, g⟫ u ∂(volume : Measure ℝ)
        ∂(stdGaussian E)) =
      ∫ g, (|⟪x, g⟫| + |⟪y, g⟫| - |⟪x, g⟫ - ⟪y, g⟫|) / 2 ∂(stdGaussian E) := by
    refine integral_congr_ae (Filter.Eventually.of_forall fun g => ?_)
    exact integral_signedIoc_mul ⟪x, g⟫ ⟪y, g⟫
  have step5 : (∫ g, (|⟪x, g⟫| + |⟪y, g⟫| - |⟪x, g⟫ - ⟪y, g⟫|) / 2 ∂(stdGaussian E)) =
      gaussAbsMoment / 2 * (‖x‖ + ‖y‖ - dist x y) :=
    integral_gaussianEnergyKernel x y
  have hcc : c * c = 2 / gaussAbsMoment :=
    Real.mul_self_sqrt (div_nonneg (by norm_num) gaussAbsMoment_pos.le)
  rw [step1, step2, step3, step4, step5, hcc]
  field_simp [gaussAbsMoment_pos.ne']

/-- **s2, norm form.** The companion identity `‖Φ x − Φ y‖² = 2·dist x y`, obtained from
`inner_energyPointEmbed` via the real Hilbert-space expansion `norm_sub_sq_real` (the `‖x‖`, `‖y‖`
terms cancel; `⟪Φ x, Φ x⟫ = 2‖x‖` since `dist x x = 0`). -/
lemma norm_sub_energyPointEmbed_sq (x y : E) :
    ‖energyPointEmbed x - energyPointEmbed y‖ ^ 2 = 2 * dist x y := by
  have hxx : ⟪energyPointEmbed x, energyPointEmbed x⟫ = 2 * ‖x‖ := by
    rw [inner_energyPointEmbed x x, dist_self]; ring
  have hyy : ⟪energyPointEmbed y, energyPointEmbed y⟫ = 2 * ‖y‖ := by
    rw [inner_energyPointEmbed y y, dist_self]; ring
  have hxx2 : ‖energyPointEmbed x‖ ^ 2 = 2 * ‖x‖ :=
    (real_inner_self_eq_norm_sq (energyPointEmbed x)).symm.trans hxx
  have hyy2 : ‖energyPointEmbed y‖ ^ 2 = 2 * ‖y‖ :=
    (real_inner_self_eq_norm_sq (energyPointEmbed y)).symm.trans hyy
  rw [norm_sub_sq_real, hxx2, hyy2, inner_energyPointEmbed x y]
  ring

/-! ### s3: Bochner mean embedding of probability measures -/

/-- The squared norm of the point embedding is twice the norm of the source point. This is the
diagonal case of `inner_energyPointEmbed` and supplies the moment bound for Bochner integrability.
-/
lemma norm_energyPointEmbed_sq (x : E) :
    ‖energyPointEmbed x‖ ^ 2 = 2 * ‖x‖ := by
  calc
    ‖energyPointEmbed x‖ ^ 2 = ⟪energyPointEmbed x, energyPointEmbed x⟫ :=
      (real_inner_self_eq_norm_sq (energyPointEmbed x)).symm
    _ = 2 * ‖x‖ := by rw [inner_energyPointEmbed x x, dist_self]; ring

/-- The point embedding is a square-root isometry:
`dist (Φ x) (Φ y) = √(2 * dist x y)`. -/
lemma dist_energyPointEmbed (x y : E) :
    dist (energyPointEmbed x) (energyPointEmbed y) = Real.sqrt (2 * dist x y) := by
  rw [dist_eq_norm, eq_comm,
    Real.sqrt_eq_iff_eq_sq (mul_nonneg (by norm_num) dist_nonneg) (norm_nonneg _)]
  exact (norm_sub_energyPointEmbed_sq x y).symm

/-- The point embedding is continuous. Its target distance is the continuous transform
`√(2 * dist x y)` of the source distance. -/
lemma continuous_energyPointEmbed : Continuous (energyPointEmbed (E := E)) := by
  rw [continuous_iff_continuous_dist]
  convert Real.continuous_sqrt.comp (continuous_const.mul continuous_dist) using 1
  ext p
  exact dist_energyPointEmbed p.1 p.2

/-- A finite first moment of the source measure makes the point embedding Bochner-integrable.
The squared target norm is exactly `2 * ‖x‖`; finite measure then lowers the integrable exponent
from two to one. -/
lemma integrable_energyPointEmbed (μ : ProbabilityMeasure E)
    (hμ : Integrable (fun x : E => ‖x‖) μ.measure) :
    Integrable (energyPointEmbed (E := E)) μ.measure := by
  have hmeas : AEStronglyMeasurable (energyPointEmbed (E := E)) μ.measure :=
    continuous_energyPointEmbed.aestronglyMeasurable
  have hsq : Integrable (fun x : E => ‖energyPointEmbed x‖ ^ 2) μ.measure := by
    refine (hμ.const_mul 2).congr ?_
    filter_upwards with x
    exact (norm_energyPointEmbed_sq x).symm
  have hone : Integrable (fun x : E => ‖energyPointEmbed x‖ ^ 1) μ.measure :=
    integrable_norm_pow_of_le hmeas (by norm_num) hsq
  exact (integrable_norm_iff hmeas).mp (by simpa using hone)

/-- Finite first moments of two normed-space-valued probability measures imply a finite joint
distance moment. The proof dominates `dist x y` by `‖x‖ + ‖y‖` on the product measure. -/
lemma finiteDistMoment_of_integrable_norm (μ ν : ProbabilityMeasure E)
    (hμ : Integrable (fun x : E => ‖x‖) μ.measure)
    (hν : Integrable (fun y : E => ‖y‖) ν.measure) :
    FiniteDistMoment μ ν := by
  refine Integrable.mono' ((hμ.comp_fst ν.measure).add (hν.comp_snd μ.measure))
    ((continuous_fst.dist continuous_snd).aestronglyMeasurable) ?_
  filter_upwards with p
  rw [Real.norm_of_nonneg dist_nonneg]
  exact dist_le_norm_add_norm p.1 p.2

omit [InnerProductSpace ℝ E] [FiniteDimensional ℝ E] in
/-- A finite first moment makes every distance slice `y ↦ dist x y` integrable. -/
private lemma integrable_dist_right (ν : ProbabilityMeasure E)
    (hν : Integrable (fun y : E => ‖y‖) ν.measure) (x : E) :
    Integrable (fun y => dist x y) ν.measure := by
  refine Integrable.mono' ((integrable_const ‖x‖).add hν)
    ((continuous_const.dist continuous_id).aestronglyMeasurable) ?_
  filter_upwards with y
  rw [Real.norm_of_nonneg dist_nonneg]
  exact dist_le_norm_add_norm x y

/-- **Bochner mean embedding (s3).** The probability measure `μ` is represented by the Bochner
integral of the point embedding `Φ` in `L²((stdGaussian E) × volume)`. Mathlib's integral is total;
the theorems below require the explicit finite-first-moment hypothesis that makes this integral
the intended Bochner mean. -/
noncomputable def energyEmbed (μ : ProbabilityMeasure E) :
    Lp ℝ 2 ((stdGaussian E).prod (volume : Measure ℝ)) :=
  ∫ x, energyPointEmbed x ∂μ.measure

/-- The inner product of two mean embeddings is the distance-induced kernel expectation:

`⟪m μ, m ν⟫ = ∫‖x‖dμ + ∫‖y‖dν - ∫∫dist x y dν dμ`.

This is the integral form of `inner_energyPointEmbed`; the finite-first-moment hypotheses justify
all Bochner integrals and the Fubini slice used by the final term. -/
lemma inner_energyEmbed (μ ν : ProbabilityMeasure E)
    (hμ : Integrable (fun x : E => ‖x‖) μ.measure)
    (hν : Integrable (fun y : E => ‖y‖) ν.measure) :
    ⟪energyEmbed μ, energyEmbed ν⟫ =
      (∫ x, ‖x‖ ∂μ.measure) + (∫ y, ‖y‖ ∂ν.measure) -
        ∫ x, ∫ y, dist x y ∂ν.measure ∂μ.measure := by
  have hΦμ := integrable_energyPointEmbed μ hμ
  have hΦν := integrable_energyPointEmbed ν hν
  have hdist : FiniteDistMoment μ ν := finiteDistMoment_of_integrable_norm μ ν hμ hν
  have hpotential : Integrable (fun x => ∫ y, dist x y ∂ν.measure) μ.measure := by
    have h : Integrable (fun x : E => ∫ y : E, ‖dist x y‖ ∂ν.measure) μ.measure :=
      hdist.integral_norm_prod_left
    simpa only [Real.norm_of_nonneg dist_nonneg] using h
  have hinner : ∀ x : E,
      (∫ y, ⟪energyPointEmbed x, energyPointEmbed y⟫ ∂ν.measure) =
        ‖x‖ + (∫ y, ‖y‖ ∂ν.measure) - ∫ y, dist x y ∂ν.measure := by
    intro x
    simp_rw [inner_energyPointEmbed]
    calc
      (∫ y, ‖x‖ + ‖y‖ - dist x y ∂ν.measure) =
          (∫ y, ‖x‖ + ‖y‖ ∂ν.measure) - ∫ y, dist x y ∂ν.measure :=
        integral_sub ((integrable_const ‖x‖).add hν) (integrable_dist_right ν hν x)
      _ = ((∫ _y, ‖x‖ ∂ν.measure) + ∫ y, ‖y‖ ∂ν.measure) -
          ∫ y, dist x y ∂ν.measure := by
        rw [integral_add (integrable_const ‖x‖) hν]
      _ = ‖x‖ + (∫ y, ‖y‖ ∂ν.measure) - ∫ y, dist x y ∂ν.measure := by
        simp [integral_const]
  calc
    ⟪energyEmbed μ, energyEmbed ν⟫ = ⟪energyEmbed ν, energyEmbed μ⟫ :=
      real_inner_comm _ _
    _ = ∫ x, ⟪energyEmbed ν, energyPointEmbed x⟫ ∂μ.measure := by
      exact (integral_inner hΦμ (energyEmbed ν)).symm
    _ = ∫ x, ∫ y, ⟪energyPointEmbed x, energyPointEmbed y⟫ ∂ν.measure ∂μ.measure := by
      apply integral_congr_ae
      filter_upwards with x
      rw [real_inner_comm]
      exact (integral_inner hΦν (energyPointEmbed x)).symm
    _ = ∫ x, (‖x‖ + (∫ y, ‖y‖ ∂ν.measure) -
        ∫ y, dist x y ∂ν.measure) ∂μ.measure := by
      apply integral_congr_ae
      filter_upwards with x
      exact hinner x
    _ = (∫ x, ‖x‖ ∂μ.measure) + (∫ y, ‖y‖ ∂ν.measure) -
        ∫ x, ∫ y, dist x y ∂ν.measure ∂μ.measure := by
      calc
        (∫ x, ‖x‖ + (∫ y, ‖y‖ ∂ν.measure) -
            ∫ y, dist x y ∂ν.measure ∂μ.measure) =
            (∫ x, ‖x‖ + (∫ y, ‖y‖ ∂ν.measure) ∂μ.measure) -
              ∫ x, ∫ y, dist x y ∂ν.measure ∂μ.measure :=
          integral_sub (hμ.add (integrable_const _)) hpotential
        _ = ((∫ x, ‖x‖ ∂μ.measure) +
              ∫ _x, (∫ y, ‖y‖ ∂ν.measure) ∂μ.measure) -
              ∫ x, ∫ y, dist x y ∂ν.measure ∂μ.measure := by
          rw [integral_add hμ (integrable_const _)]
        _ = (∫ x, ‖x‖ ∂μ.measure) + (∫ y, ‖y‖ ∂ν.measure) -
              ∫ x, ∫ y, dist x y ∂ν.measure ∂μ.measure := by
          simp [integral_const]

/-- **Energy--MMD identity (s3).** For finite-first-moment probability measures on a
finite-dimensional real inner-product space, squared distance between their explicit Bochner mean
embeddings equals population squared energy distance:

`‖energyEmbed μ - energyEmbed ν‖² = energyDistanceSq μ ν`.

This is the RKHS-free realization of Sejdinovic et al. (2013), Theorem 22 for the distance-induced
kernel constructed in this file. It proves equality of the two functionals; injectivity of the
measure embedding is the separate strong-negative-type question. -/
theorem norm_sub_energyEmbed_sq (μ ν : ProbabilityMeasure E)
    (hμ : Integrable (fun x : E => ‖x‖) μ.measure)
    (hν : Integrable (fun y : E => ‖y‖) ν.measure) :
    ‖energyEmbed μ - energyEmbed ν‖ ^ 2 = energyDistanceSq μ ν := by
  rw [norm_sub_sq_real, ← real_inner_self_eq_norm_sq (energyEmbed μ),
    ← real_inner_self_eq_norm_sq (energyEmbed ν),
    inner_energyEmbed μ μ hμ hμ, inner_energyEmbed ν ν hν hν,
    inner_energyEmbed μ ν hμ hν]
  unfold energyDistanceSq
  ring

end GaussianKernel

end EnergyStatistics
