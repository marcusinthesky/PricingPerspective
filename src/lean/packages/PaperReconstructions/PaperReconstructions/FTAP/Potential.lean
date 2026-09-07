import PaperReconstructions.FTAP.Defs
import Mathlib.MeasureTheory.Function.SpecialFunctions.Inner
import Mathlib.Analysis.Calculus.ParametricIntegral
import PaperReconstructions.FTAP.Softplus

/-!
# The softplus potential and its global minimiser

The backward direction of the one-period FTAP minimises the **softplus potential**

`f(θ) = ∫ softplus ⟪θ, Y⟫ ∂P`,

whose first-order condition hands back the equivalent martingale measure. This module
builds the minimiser.

The potential is constant along the gains kernel `N = {θ : ⟪θ, Y⟫ = 0 a.e.}` — the
redundant portfolio directions — and, under no arbitrage, *coercive* on `Nᗮ`. So a
minimiser exists on `Nᗮ` and, by translation-invariance along `N`, is a **global**
minimiser. Working on `Nᗮ` rather than all of `F` is exactly what lets the argument drop
the usual non-redundancy hypothesis.

## Provenance

Ported from [formal-mathfin](https://github.com/raphaelrrcoelho/formal-mathfin)
(`MathFin/Foundations/FTAPOnePeriodVector.lean`, Apache-2.0, © Raphael Coelho).
-/

namespace PaperReconstructions.FTAP

open MeasureTheory

variable {Ω : Type*} [MeasureSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
  {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F] [FiniteDimensional ℝ F]
  [MeasurableSpace F] [BorelSpace F] (Y : Ω → F)

/-- The **softplus potential** `f(θ) = ∫ softplus ⟪θ, Y⟫ ∂P`, minimised in the backward
direction of the FTAP; its first-order condition produces the equivalent martingale
measure. -/
noncomputable def potential (θ : F) : ℝ :=
  ∫ ω, softplus (inner ℝ θ (Y ω)) ∂P

omit [FiniteDimensional ℝ F] [MeasurableSpace F] [BorelSpace F] in
/-- `softplus ⟪θ, Y⟫` is `P`-integrable, dominated by `‖θ‖‖Y‖ + log 2`. -/
lemma integrable_softplus_inner (hYint : Integrable Y P) (θ : F) :
    Integrable (fun ω => softplus (inner ℝ θ (Y ω))) P := by
  have hmeas : AEStronglyMeasurable (fun ω => softplus (inner ℝ θ (Y ω))) P :=
    continuous_softplus.comp_aestronglyMeasurable
      ((continuous_const.inner continuous_id).comp_aestronglyMeasurable
        hYint.aestronglyMeasurable)
  refine Integrable.mono' (g := fun ω => ‖θ‖ * ‖Y ω‖ + Real.log 2) ?_ hmeas ?_
  · exact (hYint.norm.const_mul ‖θ‖).add (integrable_const _)
  · filter_upwards with ω
    rw [Real.norm_eq_abs, abs_of_nonneg (softplus_nonneg _)]
    exact (softplus_le _).trans (by gcongr; exact abs_real_inner_le_norm _ _)

/-- **Directional derivative of the potential**. For `Y ∈ L¹`, `t ↦ f(θ + t • u)` is
differentiable at `0` with derivative `∫ σ⟪θ, Y⟫ · ⟪u, Y⟫ ∂P` — differentiation under the
integral, dominated by `‖u‖‖Y‖` since `σ ∈ (0, 1)`. -/
lemma hasDerivAt_potential_dir (hY : Measurable Y) (hYint : Integrable Y P) (θ u : F) :
    HasDerivAt (fun t : ℝ => potential P Y (θ + t • u))
      (∫ ω, logistic (inner ℝ θ (Y ω)) * inner ℝ u (Y ω) ∂P) 0 := by
  have hbmeas : Measurable (fun ω => inner ℝ u (Y ω)) := measurable_const.inner hY
  have hexp : ∀ (t : ℝ) (ω : Ω),
      inner ℝ (θ + t • u) (Y ω) = inner ℝ θ (Y ω) + t * inner ℝ u (Y ω) := fun t ω => by
    rw [inner_add_left, real_inner_smul_left]
  set Φ : ℝ → Ω → ℝ :=
    fun t ω => softplus (inner ℝ θ (Y ω) + t * inner ℝ u (Y ω)) with hΦ
  set Φ' : ℝ → Ω → ℝ :=
    fun t ω => logistic (inner ℝ θ (Y ω) + t * inner ℝ u (Y ω)) * inner ℝ u (Y ω) with hΦ'
  have hmeas_arg : ∀ t : ℝ, AEStronglyMeasurable
      (fun ω => inner ℝ θ (Y ω) + t * inner ℝ u (Y ω)) P :=
    fun t => ((measurable_const.inner hY).add (hbmeas.const_mul t)).aestronglyMeasurable
  have hΦ_meas : ∀ᶠ t in nhds (0 : ℝ), AEStronglyMeasurable (Φ t) P :=
    Filter.Eventually.of_forall fun t =>
      continuous_softplus.comp_aestronglyMeasurable (hmeas_arg t)
  have hΦ_int : Integrable (Φ 0) P := by
    simp only [hΦ, zero_mul, add_zero]; exact integrable_softplus_inner P Y hYint θ
  have hΦ'_meas : AEStronglyMeasurable (Φ' 0) P :=
    (continuous_logistic.comp_aestronglyMeasurable (hmeas_arg 0)).mul
      hbmeas.aestronglyMeasurable
  have h_bound : ∀ᵐ ω ∂P, ∀ x ∈ (Set.univ : Set ℝ), ‖Φ' x ω‖ ≤ ‖u‖ * ‖Y ω‖ := by
    filter_upwards with ω x _
    rw [hΦ', Real.norm_eq_abs, abs_mul]
    have h1 : |logistic (inner ℝ θ (Y ω) + x * inner ℝ u (Y ω))| ≤ 1 := by
      rw [abs_of_pos (logistic_pos _)]; exact (logistic_lt_one _).le
    have h2 : |inner ℝ u (Y ω)| ≤ ‖u‖ * ‖Y ω‖ := abs_real_inner_le_norm u (Y ω)
    calc |logistic (inner ℝ θ (Y ω) + x * inner ℝ u (Y ω))| * |inner ℝ u (Y ω)|
        ≤ 1 * (‖u‖ * ‖Y ω‖) := mul_le_mul h1 h2 (abs_nonneg _) zero_le_one
      _ = ‖u‖ * ‖Y ω‖ := one_mul _
  have h_diff : ∀ᵐ ω ∂P, ∀ x ∈ (Set.univ : Set ℝ),
      HasDerivAt (fun t => Φ t ω) (Φ' x ω) x := by
    filter_upwards with ω x _
    have haff : HasDerivAt (fun t : ℝ => inner ℝ θ (Y ω) + t * inner ℝ u (Y ω))
        (inner ℝ u (Y ω)) x := by
      simpa using ((hasDerivAt_id x).mul_const (inner ℝ u (Y ω))).const_add (inner ℝ θ (Y ω))
    have hc := (hasDerivAt_softplus _).comp x haff
    simpa only [hΦ, hΦ', Function.comp_def] using hc
  obtain ⟨-, hderiv⟩ := hasDerivAt_integral_of_dominated_loc_of_deriv_le (μ := P)
    (bound := fun ω => ‖u‖ * ‖Y ω‖) Filter.univ_mem hΦ_meas hΦ_int hΦ'_meas h_bound
    (hYint.norm.const_mul ‖u‖) h_diff
  have hpot : (fun t : ℝ => potential P Y (θ + t • u)) = fun t => ∫ ω, Φ t ω ∂P := by
    funext t; simp only [potential, hΦ]
    exact integral_congr_ae (by filter_upwards with ω; rw [hexp t ω])
  have hval : (∫ ω, Φ' 0 ω ∂P) = ∫ ω, logistic (inner ℝ θ (Y ω)) * inner ℝ u (Y ω) ∂P := by
    refine integral_congr_ae ?_; filter_upwards with ω; simp only [hΦ', zero_mul, add_zero]
  rw [hpot, ← hval]; exact hderiv

omit [IsProbabilityMeasure P] [FiniteDimensional ℝ F] [MeasurableSpace F] [BorelSpace F] in
/-- For a `1`-Lipschitz `φ`, the averaged map `θ ↦ ∫ φ⟪θ, Y⟫ ∂P` is `(∫‖Y‖)`-Lipschitz
(`φ` is `1`-Lipschitz and `θ ↦ ⟪θ, Y ω⟫` is `‖Y ω‖`-Lipschitz, by Cauchy–Schwarz). -/
lemma lipschitzWith_integral_inner {φ : ℝ → ℝ} (hφ : LipschitzWith 1 φ)
    (hint : ∀ θ : F, Integrable (fun ω => φ (inner ℝ θ (Y ω))) P)
    (hYint : Integrable Y P) :
    LipschitzWith (∫ ω, ‖Y ω‖ ∂P).toNNReal (fun θ => ∫ ω, φ (inner ℝ θ (Y ω)) ∂P) := by
  have hnn : 0 ≤ ∫ ω, ‖Y ω‖ ∂P := integral_nonneg fun ω => norm_nonneg _
  refine LipschitzWith.of_dist_le_mul fun θ θ' => ?_
  rw [Real.dist_eq, Real.coe_toNNReal _ hnn, ← integral_sub (hint θ) (hint θ'), dist_eq_norm]
  have hbound : ∀ ω, ‖φ (inner ℝ θ (Y ω)) - φ (inner ℝ θ' (Y ω))‖ ≤ ‖Y ω‖ * ‖θ - θ'‖ := by
    intro ω
    have h1 := hφ.dist_le_mul (inner ℝ θ (Y ω)) (inner ℝ θ' (Y ω))
    rw [Real.dist_eq, Real.dist_eq, NNReal.coe_one, one_mul] at h1
    calc ‖φ (inner ℝ θ (Y ω)) - φ (inner ℝ θ' (Y ω))‖
        = |φ (inner ℝ θ (Y ω)) - φ (inner ℝ θ' (Y ω))| := Real.norm_eq_abs _
      _ ≤ |inner ℝ θ (Y ω) - inner ℝ θ' (Y ω)| := h1
      _ = |inner ℝ (θ - θ') (Y ω)| := by rw [inner_sub_left]
      _ ≤ ‖Y ω‖ * ‖θ - θ'‖ := by rw [mul_comm]; exact abs_real_inner_le_norm (θ - θ') (Y ω)
  calc |∫ ω, (φ (inner ℝ θ (Y ω)) - φ (inner ℝ θ' (Y ω))) ∂P|
      ≤ ∫ ω, ‖φ (inner ℝ θ (Y ω)) - φ (inner ℝ θ' (Y ω))‖ ∂P := abs_integral_le_integral_abs ..
    _ ≤ ∫ ω, ‖Y ω‖ * ‖θ - θ'‖ ∂P :=
        integral_mono_ae ((hint θ).sub (hint θ')).norm (hYint.norm.mul_const _)
          (Filter.Eventually.of_forall hbound)
    _ = (∫ ω, ‖Y ω‖ ∂P) * ‖θ - θ'‖ := integral_mul_const _ _

omit [FiniteDimensional ℝ F] [MeasurableSpace F] [BorelSpace F] in
/-- The potential is continuous. -/
lemma continuous_potential (hYint : Integrable Y P) : Continuous (potential P Y) :=
  (lipschitzWith_integral_inner P Y lipschitzWith_softplus
    (integrable_softplus_inner P Y hYint) hYint).continuous

omit [IsProbabilityMeasure P] [FiniteDimensional ℝ F] [MeasurableSpace F] [BorelSpace F] in
/-- `max ⟪θ, Y⟫ 0` is `P`-integrable (dominated by `‖θ‖‖Y‖`). -/
lemma integrable_posPart_inner (hYint : Integrable Y P) (θ : F) :
    Integrable (fun ω => max (inner ℝ θ (Y ω)) 0) P := by
  have hmeas : AEStronglyMeasurable (fun ω => max (inner ℝ θ (Y ω)) 0) P :=
    (continuous_id.max continuous_const).comp_aestronglyMeasurable
      ((continuous_const.inner continuous_id).comp_aestronglyMeasurable
        hYint.aestronglyMeasurable)
  refine Integrable.mono' (hYint.norm.const_mul ‖θ‖) hmeas
    (Filter.Eventually.of_forall fun ω => ?_)
  rw [Real.norm_eq_abs, abs_of_nonneg (le_max_right _ _), max_le_iff]
  exact ⟨(le_abs_self _).trans (abs_real_inner_le_norm θ (Y ω)), by positivity⟩

omit [IsProbabilityMeasure P] [FiniteDimensional ℝ F] [MeasurableSpace F] [BorelSpace F] in
/-- The positive-gain average `g(θ) = ∫⟪θ, Y⟫⁺ ∂P` is continuous. It lower-bounds the
potential (`softplus s ≥ s⁺`) and drives the coercivity argument. -/
lemma continuous_gainsPos (hYint : Integrable Y P) :
    Continuous (fun θ => ∫ ω, max (inner ℝ θ (Y ω)) 0 ∂P) :=
  (lipschitzWith_integral_inner P Y lipschitzWith_posPart
    (integrable_posPart_inner P Y hYint) hYint).continuous

omit [MeasurableSpace F] [BorelSpace F] in
/-- **Coercivity** of the potential on `Nᗮ` (no arbitrage).

The positive gain average `g(θ) = ∫⟪θ, Y⟫⁺` is positive on `Nᗮ \ {0}` (no arbitrage, plus
`N ⊓ Nᗮ = ⊥`), continuous and positively homogeneous; its minimum `c` over the unit sphere
of `Nᗮ` is positive, and `softplus s ≥ s⁺` gives `c‖θ‖ ≤ f(θ)` for `θ ∈ Nᗮ`. -/
lemma exists_pos_lower_bound (hYint : Integrable Y P) (hNA : NoArbitrageFTAP P Y)
    (hNbot : (gainsKernel P Y)ᗮ ≠ ⊥) :
    ∃ c > 0, ∀ θ ∈ (gainsKernel P Y)ᗮ, c * ‖θ‖ ≤ potential P Y θ := by
  set N := gainsKernel P Y
  set g : F → ℝ := fun θ => ∫ ω, max (inner ℝ θ (Y ω)) 0 ∂P with hg
  have hg_nonneg : ∀ θ, 0 ≤ g θ := fun θ => integral_nonneg fun ω => le_max_right _ _
  -- `g` is positive on `Nᗮ \ {0}`
  have hg_pos : ∀ θ ∈ Nᗮ, θ ≠ 0 → 0 < g θ := by
    intro θ hθK hθ
    refine (hg_nonneg θ).lt_of_ne fun h => hθ ?_
    have hmax : (fun ω => max (inner ℝ θ (Y ω)) 0) =ᵐ[P] 0 :=
      (integral_eq_zero_iff_of_nonneg_ae
        (Filter.Eventually.of_forall fun ω => le_max_right _ _)
        (integrable_posPart_inner P Y hYint θ)).mp h.symm
    have hnonpos : (fun ω => inner ℝ θ (Y ω)) ≤ᵐ[P] 0 := by
      filter_upwards [hmax] with ω hm
      have hle : inner ℝ θ (Y ω) ≤ max (inner ℝ θ (Y ω)) 0 := le_max_left _ _
      simp only [Pi.zero_apply] at hm ⊢; rwa [hm] at hle
    have hneg := hNA (-θ) (by
      filter_upwards [hnonpos] with ω h
      simp only [Pi.zero_apply] at h ⊢; rw [inner_neg_left]; linarith)
    have hθN : θ ∈ N := by
      show (fun ω => inner ℝ θ (Y ω)) =ᵐ[P] 0
      filter_upwards [hneg] with ω hh
      simp only [Pi.zero_apply, inner_neg_left] at hh ⊢; linarith
    exact inner_self_eq_zero.mp (N.inner_right_of_mem_orthogonal hθN hθK)
  -- `g` is positively homogeneous
  have hg_hom : ∀ (r : ℝ), 0 ≤ r → ∀ θ, g (r • θ) = r * g θ := by
    intro r hr θ
    simp only [hg]
    rw [← integral_const_mul]
    refine integral_congr_ae (Filter.Eventually.of_forall fun ω => ?_)
    show max (inner ℝ (r • θ) (Y ω)) 0 = r * max (inner ℝ θ (Y ω)) 0
    rw [real_inner_smul_left]
    rcases le_total 0 (inner ℝ θ (Y ω)) with hs | hs
    · rw [max_eq_left hs, max_eq_left (mul_nonneg hr hs)]
    · rw [max_eq_right hs, max_eq_right (mul_nonpos_of_nonneg_of_nonpos hr hs), mul_zero]
  -- the minimum of `g` over the unit sphere of `Nᗮ` is positive
  have hScompact : IsCompact ((Nᗮ : Set F) ∩ Metric.sphere 0 1) :=
    (isCompact_sphere 0 1).inter_left Nᗮ.closed_of_finiteDimensional
  have hSne : ((Nᗮ : Set F) ∩ Metric.sphere 0 1).Nonempty := by
    obtain ⟨v, hvK, hv0⟩ := (Submodule.ne_bot_iff _).mp hNbot
    have hvnorm : 0 < ‖v‖ := norm_pos_iff.mpr hv0
    refine ⟨(‖v‖⁻¹ : ℝ) • v, Nᗮ.smul_mem _ hvK, ?_⟩
    rw [Metric.mem_sphere, dist_zero_right, norm_smul, norm_inv, Real.norm_eq_abs,
      abs_of_pos hvnorm, inv_mul_cancel₀ hvnorm.ne']
  obtain ⟨u₀, hu₀mem, hu₀min⟩ :=
    hScompact.exists_isMinOn hSne (continuous_gainsPos P Y hYint).continuousOn
  obtain ⟨hu₀K, hu₀S⟩ := hu₀mem
  have hu₀ne : u₀ ≠ 0 := fun h => by
    rw [Metric.mem_sphere, h, dist_self] at hu₀S; exact one_ne_zero hu₀S.symm
  refine ⟨g u₀, hg_pos u₀ hu₀K hu₀ne, fun θ hθK => ?_⟩
  have hpg : g θ ≤ potential P Y θ := by
    rw [hg, potential]
    exact integral_mono_ae (integrable_posPart_inner P Y hYint θ)
      (integrable_softplus_inner P Y hYint θ)
      (Filter.Eventually.of_forall fun ω => posPart_le_softplus _)
  refine le_trans ?_ hpg
  rcases eq_or_ne θ 0 with rfl | hθ
  · simpa using hg_nonneg 0
  · have hθ0 : (0 : ℝ) < ‖θ‖ := norm_pos_iff.mpr hθ
    have hunit : (‖θ‖⁻¹ : ℝ) • θ ∈ (Nᗮ : Set F) ∩ Metric.sphere 0 1 := by
      refine ⟨Nᗮ.smul_mem _ hθK, ?_⟩
      rw [Metric.mem_sphere, dist_zero_right, norm_smul, norm_inv, Real.norm_eq_abs,
        abs_of_pos hθ0, inv_mul_cancel₀ hθ0.ne']
    calc g u₀ * ‖θ‖ = ‖θ‖ * g u₀ := mul_comm _ _
      _ ≤ ‖θ‖ * g ((‖θ‖⁻¹ : ℝ) • θ) :=
          mul_le_mul_of_nonneg_left (isMinOn_iff.mp hu₀min _ hunit) hθ0.le
      _ = g θ := by
          rw [hg_hom ‖θ‖⁻¹ (inv_nonneg.mpr hθ0.le) θ, ← mul_assoc, mul_inv_cancel₀ hθ0.ne',
            one_mul]

omit [MeasurableSpace F] [BorelSpace F] in
/-- **The potential attains a global minimum** (no arbitrage).

On `Nᗮ`, coercivity makes the minimum over a large closed ball global; and the potential
is constant along `N` (`⟪n, Y⟫ = 0` a.e.), so a minimiser over `Nᗮ` minimises over all of
`F` after the decomposition `θ = n + z`, `n ∈ N`, `z ∈ Nᗮ`. -/
lemma exists_global_min_potential (hYint : Integrable Y P) (hNA : NoArbitrageFTAP P Y)
    (hNbot : (gainsKernel P Y)ᗮ ≠ ⊥) :
    ∃ θ₀, ∀ θ, potential P Y θ₀ ≤ potential P Y θ := by
  set N := gainsKernel P Y
  obtain ⟨c, hc, hlb⟩ := exists_pos_lower_bound P Y hYint hNA hNbot
  -- the potential is constant along `N`
  have hinv : ∀ ψ : F, ∀ n ∈ N, potential P Y (ψ + n) = potential P Y ψ := by
    intro ψ n hn
    have hn' : (fun ω => inner ℝ n (Y ω)) =ᵐ[P] 0 := hn
    refine integral_congr_ae ?_
    filter_upwards [hn'] with ω he
    simp only [Pi.zero_apply] at he
    show softplus (inner ℝ (ψ + n) (Y ω)) = softplus (inner ℝ ψ (Y ω))
    rw [inner_add_left, he, add_zero]
  -- minimise over the compact set `Nᗮ ∩ closedBall 0 R`
  have hp0 : 0 ≤ potential P Y 0 := integral_nonneg fun ω => softplus_nonneg _
  set R : ℝ := potential P Y 0 / c with hRdef
  have hR0 : 0 ≤ R := div_nonneg hp0 hc.le
  have hKcompact : IsCompact ((Nᗮ : Set F) ∩ Metric.closedBall 0 R) :=
    (isCompact_closedBall 0 R).inter_left Nᗮ.closed_of_finiteDimensional
  have hKne : ((Nᗮ : Set F) ∩ Metric.closedBall 0 R).Nonempty :=
    ⟨0, Nᗮ.zero_mem, Metric.mem_closedBall_self hR0⟩
  obtain ⟨θ₀, hθ₀mem, hθ₀min⟩ :=
    hKcompact.exists_isMinOn hKne (continuous_potential P Y hYint).continuousOn
  obtain ⟨hθ₀K, _⟩ := hθ₀mem
  have hcR : c * R = potential P Y 0 := by rw [hRdef]; field_simp
  -- `θ₀` minimises over all of `Nᗮ` (coercivity escapes the ball)
  have hθ₀minK : ∀ θ ∈ Nᗮ, potential P Y θ₀ ≤ potential P Y θ := by
    intro θ hθK
    rcases le_or_gt ‖θ‖ R with hle | hlt
    · exact isMinOn_iff.mp hθ₀min θ
        ⟨hθK, by rw [Metric.mem_closedBall, dist_zero_right]; exact hle⟩
    · calc potential P Y θ₀
          ≤ potential P Y 0 :=
            isMinOn_iff.mp hθ₀min 0 ⟨Nᗮ.zero_mem, Metric.mem_closedBall_self hR0⟩
        _ = c * R := hcR.symm
        _ ≤ c * ‖θ‖ := mul_le_mul_of_nonneg_left hlt.le hc.le
        _ ≤ potential P Y θ := hlb θ hθK
  -- lift to all of `F`: `f(θ) = f(z) ≥ f(θ₀)` for the `Nᗮ`-component `z` of `θ`
  refine ⟨θ₀, fun θ => ?_⟩
  obtain ⟨n, hn, z, hz, hnz⟩ : ∃ n ∈ N, ∃ z ∈ Nᗮ, n + z = θ := by
    have hmem : θ ∈ N ⊔ Nᗮ := by
      rw [Submodule.sup_orthogonal_of_hasOrthogonalProjection]; trivial
    exact Submodule.mem_sup.mp hmem
  have hfθ : potential P Y θ = potential P Y z := by
    rw [← hnz, add_comm n z, hinv z n hn]
  rw [hfθ]; exact hθ₀minK z hz

end PaperReconstructions.FTAP
