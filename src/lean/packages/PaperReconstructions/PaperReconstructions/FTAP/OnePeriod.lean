import PaperReconstructions.FTAP.Potential

/-!
# One-period FTAP — the backward direction and the equivalence

The first-order condition at the potential's global minimiser `θ₀` says the gradient
`∫ σ⟪θ₀, Y⟫ • Y ∂P` vanishes, so the strictly-positive bounded weight
`z = σ⟪θ₀, Y⟫ ∈ (0, 1)` already makes `Y` fair. Normalising it to `z / E[z]` gives the
density of an equivalent martingale measure: a bounded logistic tilt, the softplus
analogue of an Esscher measure.

Integrability of `Y` is then dropped by first passing to the equivalent probability
measure with density proportional to `(1 + ‖Y‖)⁻¹`, under which `Y` *is* integrable; no
arbitrage is an a.e. notion and survives the change, and `≪` composes.

No Hahn–Banach, no `L⁰`-cone closedness, no measurable selection, and no non-redundancy
hypothesis — those are needed only for the general-`Ω` *multi-period* Dalang–Morton–
Willinger theorem, which stays out of scope.

## Provenance

Ported from [formal-mathfin](https://github.com/raphaelrrcoelho/formal-mathfin)
(Apache-2.0, © Raphael Coelho): `MathFin/Foundations/FTAPOnePeriodVector.lean` and the
change-of-measure helper from `MathFin/Foundations/EquivMeasure.lean`.
-/

namespace PaperReconstructions.FTAP

open MeasureTheory

/-! ### Equivalent probability measure from a positive normalised density -/

/-- A measurable, strictly-positive, `P`-integrable density `g` with `∫ g ∂P = 1` makes
`Q = P.withDensity (fun ω => ENNReal.ofReal (g ω))` a probability measure equivalent to
`P`: `IsProbabilityMeasure Q`, `Q ≪ P`, and `P ≪ Q`.

This is the change-of-measure ritual shared by both equivalent-martingale-measure
constructions below — the logistic Esscher weight, and the `(1 + ‖Y‖)⁻¹` tempering used
for the `L¹` reduction. -/
theorem isEquivProbMeasure_withDensity {Ω : Type*} {mΩ : MeasurableSpace Ω} (P : Measure Ω)
    {g : Ω → ℝ} (hmeas : Measurable g) (hpos : ∀ ω, 0 < g ω) (hint : Integrable g P)
    (hsum : ∫ ω, g ω ∂P = 1) :
    IsProbabilityMeasure (P.withDensity (fun ω => ENNReal.ofReal (g ω))) ∧
      P.withDensity (fun ω => ENNReal.ofReal (g ω)) ≪ P ∧
      P ≪ P.withDensity (fun ω => ENNReal.ofReal (g ω)) := by
  have hofReal_meas : Measurable (fun ω => ENNReal.ofReal (g ω)) :=
    ENNReal.measurable_ofReal.comp hmeas
  refine ⟨⟨?_⟩, withDensity_absolutelyContinuous _ _,
    withDensity_absolutelyContinuous' hofReal_meas.aemeasurable
      (Filter.Eventually.of_forall fun ω => ?_)⟩
  · rw [withDensity_apply _ MeasurableSet.univ, Measure.restrict_univ,
      ← ofReal_integral_eq_lintegral_ofReal hint
        (Filter.Eventually.of_forall fun ω => (hpos ω).le), hsum, ENNReal.ofReal_one]
  · simp only [ne_eq, ENNReal.ofReal_eq_zero, not_le]; exact hpos ω

variable {Ω : Type*} [MeasureSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
  {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F] [FiniteDimensional ℝ F]
  [MeasurableSpace F] [BorelSpace F] (Y : Ω → F)

/-! ### First-order condition -/

/-- **First-order condition**. At a global minimiser `θ₀` of the potential, every
directional derivative vanishes (`IsLocalMin.hasDerivAt_eq_zero`), so the gradient
`∫ σ⟪θ₀, Y⟫ • Y` is the zero vector — the candidate density `z = σ⟪θ₀, Y⟫` makes `Y`
fair. -/
lemma integral_logistic_smul_eq_zero (hY : Measurable Y) (hYint : Integrable Y P)
    {θ₀ : F} (hmin : ∀ θ, potential P Y θ₀ ≤ potential P Y θ) :
    ∫ ω, logistic (inner ℝ θ₀ (Y ω)) • Y ω ∂P = 0 := by
  -- every directional derivative at `θ₀` is `0`
  have hdir : ∀ u, ∫ ω, logistic (inner ℝ θ₀ (Y ω)) * inner ℝ u (Y ω) ∂P = 0 := by
    intro u
    have hmin0 : IsLocalMin (fun t : ℝ => potential P Y (θ₀ + t • u)) 0 :=
      Filter.Eventually.of_forall fun t => by simp only [zero_smul, add_zero]; exact hmin _
    exact hmin0.hasDerivAt_eq_zero (hasDerivAt_potential_dir P Y hY hYint θ₀ u)
  -- the gradient vector is `0`: it is `inner`-orthogonal to everything
  have hGint : Integrable (fun ω => logistic (inner ℝ θ₀ (Y ω)) • Y ω) P := by
    refine Integrable.mono' hYint.norm
      ((continuous_logistic.comp_aestronglyMeasurable
        ((continuous_const.inner continuous_id).comp_aestronglyMeasurable
          hYint.aestronglyMeasurable)).smul hYint.aestronglyMeasurable)
      (Filter.Eventually.of_forall fun ω => ?_)
    rw [norm_smul, Real.norm_eq_abs, abs_of_pos (logistic_pos _)]
    exact mul_le_of_le_one_left (norm_nonneg _) (logistic_lt_one _).le
  have hGu : ∀ u, inner ℝ (∫ ω, logistic (inner ℝ θ₀ (Y ω)) • Y ω ∂P) u = (0 : ℝ) := by
    intro u
    rw [real_inner_comm, ← integral_inner hGint]
    simp_rw [real_inner_smul_right]
    exact hdir u
  exact inner_self_eq_zero.mp (hGu _)

/-! ### Backward direction -/

/-- **Integrable backward direction** (finite-dim market). For an integrable `Y`, no
arbitrage gives an equivalent martingale measure.

If every direction is redundant (`Nᗮ = ⊥`) then `E[Y] = 0` already and `Q = P`;
otherwise the logistic density `z = σ⟪θ₀, Y⟫` at the potential's global minimiser is
strictly positive, bounded, and fair, so `Q = P.withDensity (z / ∫ z)` is the EMM. -/
theorem exists_isEMMFTAP_of_noArbitrage_integrable (hY : Measurable Y)
    (hYint : Integrable Y P) (hNA : NoArbitrageFTAP P Y) :
    ∃ Q, IsEMMFTAP P Y Q := by
  by_cases hY0 : (gainsKernel P Y)ᗮ = ⊥
  · -- `Nᗮ = ⊥ ⟹ N = ⊤`: `Y` is a.e. orthogonal to every `θ`, so `E[Y] = 0` and `Q = P`
    refine ⟨P, inferInstance, Measure.AbsolutelyContinuous.refl P,
      Measure.AbsolutelyContinuous.refl P, hYint, ?_⟩
    have hNtop : gainsKernel P Y = ⊤ := by
      have hoo := Submodule.orthogonal_orthogonal (gainsKernel P Y)
      rw [hY0, Submodule.bot_orthogonal_eq_top] at hoo
      exact hoo.symm
    have hall : ∀ θ : F, inner ℝ θ (∫ ω, Y ω ∂P) = (0 : ℝ) := by
      intro θ
      have hmem : θ ∈ gainsKernel P Y := by rw [hNtop]; exact Submodule.mem_top
      have hθN : (fun ω => inner ℝ θ (Y ω)) =ᵐ[P] 0 := (mem_gainsKernel P Y).mp hmem
      calc inner ℝ θ (∫ ω, Y ω ∂P)
          = ∫ ω, inner ℝ θ (Y ω) ∂P := (integral_inner hYint θ).symm
        _ = 0 := by rw [integral_congr_ae hθN]; simp
    exact inner_self_eq_zero.mp (hall _)
  · obtain ⟨θ₀, hmin⟩ := exists_global_min_potential P Y hYint hNA hY0
    have hfair := integral_logistic_smul_eq_zero P Y hY hYint hmin
    set z : Ω → ℝ := fun ω => logistic (inner ℝ θ₀ (Y ω))
    have hzpos : ∀ ω, 0 < z ω := fun ω => logistic_pos _
    have hzlt : ∀ ω, z ω < 1 := fun ω => logistic_lt_one _
    have hzmeas : Measurable z := continuous_logistic.measurable.comp (measurable_const.inner hY)
    have hzint : Integrable z P :=
      ⟨hzmeas.aestronglyMeasurable, HasFiniteIntegral.of_bounded
        (Filter.Eventually.of_forall fun ω => by
          rw [Real.norm_eq_abs, abs_of_pos (hzpos ω)]; exact (hzlt ω).le)⟩
    set ζ : ℝ := ∫ ω, z ω ∂P with hζ
    have hζpos : 0 < ζ := by
      rw [hζ, integral_pos_iff_support_of_nonneg_ae
          (Filter.Eventually.of_forall fun ω => (hzpos ω).le) hzint,
        show Function.support z = Set.univ from Set.eq_univ_of_forall fun ω => (hzpos ω).ne']
      rw [measure_univ]; exact one_pos
    set dens : Ω → ℝ := fun ω => z ω / ζ with hdens
    have hdpos : ∀ ω, 0 < dens ω := fun ω => div_pos (hzpos ω) hζpos
    have hdmeas : Measurable dens := hzmeas.div_const ζ
    have hdint : Integrable dens P := hzint.div_const ζ
    have hdsum : ∫ ω, dens ω ∂P = 1 := by
      simp only [hdens, div_eq_inv_mul]
      rw [integral_const_mul, ← hζ, inv_mul_cancel₀ hζpos.ne']
    have hdbound : ∀ ω, dens ω ≤ ζ⁻¹ := fun ω => by
      rw [hdens, div_le_iff₀ hζpos, inv_mul_cancel₀ hζpos.ne']; exact (hzlt ω).le
    set Q : Measure Ω := P.withDensity (fun ω => ENNReal.ofReal (dens ω)) with hQ
    have hofReal_meas : Measurable (fun ω => ENNReal.ofReal (dens ω)) :=
      ENNReal.measurable_ofReal.comp hdmeas
    obtain ⟨hQprob, hQP, hPQ⟩ := isEquivProbMeasure_withDensity P hdmeas hdpos hdint hdsum
    rw [← hQ] at hQprob hQP hPQ
    haveI := hQprob
    have hdY_int : Integrable (fun ω => dens ω • Y ω) P := by
      refine Integrable.mono' (hYint.norm.const_mul ζ⁻¹)
        (hdmeas.aestronglyMeasurable.smul hYint.aestronglyMeasurable)
        (Filter.Eventually.of_forall fun ω => ?_)
      rw [norm_smul, Real.norm_eq_abs, abs_of_pos (hdpos ω)]
      exact mul_le_mul_of_nonneg_right (hdbound ω) (norm_nonneg _)
    have hYintQ : Integrable Y Q := by
      rw [hQ, integrable_withDensity_iff_integrable_smul' hofReal_meas
        (Filter.Eventually.of_forall fun ω => ENNReal.ofReal_lt_top)]
      refine hdY_int.congr (Filter.Eventually.of_forall fun ω => ?_)
      show dens ω • Y ω = (ENNReal.ofReal (dens ω)).toReal • Y ω
      rw [ENNReal.toReal_ofReal (hdpos ω).le]
    have hQfair : ∫ ω, Y ω ∂Q = 0 := by
      rw [hQ, integral_withDensity_eq_integral_toReal_smul hofReal_meas
        (Filter.Eventually.of_forall fun ω => ENNReal.ofReal_lt_top)]
      have heq : (fun ω => (ENNReal.ofReal (dens ω)).toReal • Y ω)
          = fun ω => ζ⁻¹ • (z ω • Y ω) := by
        funext ω
        show (ENNReal.ofReal (dens ω)).toReal • Y ω = ζ⁻¹ • (z ω • Y ω)
        rw [ENNReal.toReal_ofReal (hdpos ω).le]
        show (z ω / ζ) • Y ω = ζ⁻¹ • (z ω • Y ω)
        rw [div_eq_inv_mul, mul_smul]
      rw [heq, integral_smul, hfair, smul_zero]
    exact ⟨Q, hQprob, hQP, hPQ, hYintQ, hQfair⟩

/-- **General backward direction** (finite-dim market, integrability dropped).

For a measurable `Y`, no arbitrage gives an EMM. Pass to the equivalent probability
measure `P̃ = P.withDensity (w / κ)`, `w = (1 + ‖Y‖)⁻¹`, under which `Y` is integrable; no
arbitrage is an a.e. notion preserved by `P̃ ~ P`, so the integrable backward direction
applies, and `Q ~ P̃ ~ P` by transitivity. -/
theorem exists_isEMMFTAP_of_noArbitrage (hY : Measurable Y) (hNA : NoArbitrageFTAP P Y) :
    ∃ Q, IsEMMFTAP P Y Q := by
  set w : Ω → ℝ := fun ω => (1 + ‖Y ω‖)⁻¹ with hwdef
  have hw_meas : Measurable w := (measurable_const.add hY.norm).inv
  have hden_pos : ∀ ω, (0 : ℝ) < 1 + ‖Y ω‖ := fun ω => by positivity
  have hw_pos : ∀ ω, 0 < w ω := fun ω => by simp only [hwdef]; exact inv_pos.mpr (hden_pos ω)
  have hw_le_one : ∀ ω, w ω ≤ 1 := fun ω => by
    simp only [hwdef]; exact inv_le_one_of_one_le₀ (by linarith [norm_nonneg (Y ω)])
  have hw_int : Integrable w P :=
    ⟨hw_meas.aestronglyMeasurable, HasFiniteIntegral.of_bounded
      (Filter.Eventually.of_forall fun ω => by
        rw [Real.norm_eq_abs, abs_of_pos (hw_pos ω)]; exact hw_le_one ω)⟩
  set κ : ℝ := ∫ ω, w ω ∂P with hκdef
  have hκ_pos : 0 < κ := by
    rw [hκdef, integral_pos_iff_support_of_nonneg_ae
        (Filter.Eventually.of_forall fun ω => (hw_pos ω).le) hw_int,
      show Function.support w = Set.univ from Set.eq_univ_of_forall fun ω => (hw_pos ω).ne']
    rw [measure_univ]; exact one_pos
  set dens : Ω → ℝ := fun ω => w ω / κ with hddef
  have hd_meas : Measurable dens := hw_meas.div_const κ
  have hd_pos : ∀ ω, 0 < dens ω := fun ω => div_pos (hw_pos ω) hκ_pos
  have hd_int : Integrable dens P := hw_int.div_const κ
  have hd_sum : ∫ ω, dens ω ∂P = 1 := by
    simp only [hddef, div_eq_inv_mul]
    rw [integral_const_mul, ← hκdef, inv_mul_cancel₀ hκ_pos.ne']
  set Pt : Measure Ω := P.withDensity (fun ω => ENNReal.ofReal (dens ω)) with hPtdef
  have hd_ofReal_meas : Measurable (fun ω => ENNReal.ofReal (dens ω)) :=
    ENNReal.measurable_ofReal.comp hd_meas
  obtain ⟨hPt_prob, hPt_ll_P, hP_ll_Pt⟩ :=
    isEquivProbMeasure_withDensity P hd_meas hd_pos hd_int hd_sum
  rw [← hPtdef] at hPt_prob hPt_ll_P hP_ll_Pt
  haveI := hPt_prob
  have hdY_int : Integrable (fun ω => dens ω • Y ω) P := by
    refine ⟨(hd_meas.aestronglyMeasurable.smul hY.aestronglyMeasurable),
      HasFiniteIntegral.of_bounded (C := κ⁻¹) (Filter.Eventually.of_forall fun ω => ?_)⟩
    rw [norm_smul, Real.norm_eq_abs, abs_of_pos (hd_pos ω)]
    have h1 : w ω * ‖Y ω‖ ≤ 1 := by
      simp only [hwdef, inv_mul_eq_div, div_le_one (hden_pos ω)]
      linarith [norm_nonneg (Y ω)]
    simp only [hddef, div_mul_eq_mul_div]
    rw [div_le_iff₀ hκ_pos, inv_mul_cancel₀ hκ_pos.ne']; exact h1
  have hYintPt : Integrable Y Pt := by
    rw [hPtdef, integrable_withDensity_iff_integrable_smul' hd_ofReal_meas
      (Filter.Eventually.of_forall fun ω => ENNReal.ofReal_lt_top)]
    refine hdY_int.congr (Filter.Eventually.of_forall fun ω => ?_)
    show dens ω • Y ω = (ENNReal.ofReal (dens ω)).toReal • Y ω
    rw [ENNReal.toReal_ofReal (hd_pos ω).le]
  have hNAt : NoArbitrageFTAP Pt Y := fun θ h =>
    hPt_ll_P.ae_eq (hNA θ (hP_ll_Pt.ae_le h))
  obtain ⟨Q, hQ⟩ := exists_isEMMFTAP_of_noArbitrage_integrable Pt Y hY hYintPt hNAt
  exact ⟨Q, hQ.prob, hQ.absP.trans hPt_ll_P, hP_ll_Pt.trans hQ.Pabs, hQ.int, hQ.fair⟩

end PaperReconstructions.FTAP
