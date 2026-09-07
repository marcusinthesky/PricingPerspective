import WassersteinGeometry.Geodesics

set_option linter.style.longLine false

open scoped ENNReal NNReal MeasureTheory
open MeasureTheory

/-!
# Wasserstein contraction under a Lipschitz pushforward

If `f` is `L`-Lipschitz then pushing both marginals forward under `f` contracts quadratic
transport by at most `L`:

  `W₂(f_#μ, f_#ν) ≤ L · W₂(μ, ν)`.

## Why this matters downstream

This is the measure-level statement of the transmission channel. Reading `f` as the
characteristic-to-exposure map `t` and `μ, ν` as two firms' characteristic laws, it says the
induced exposure laws are no further apart than the transmitted characteristic laws — the
channel can erase distinctions but never manufacture them.

Two consequences worth stating explicitly, because they bound what a transmission
assumption may claim:

* *A lower bi-Lipschitz bound is a genuinely separate assumption.* The upper result does not
  imply it.  When an `AntilipschitzWith` bound and measurable embedding are supplied, the
  reverse results below construct the measurable inverse coupling explicitly.
* *Leakage is contraction too.* Additive slack `B = t(X) + U` with a law `Q` common to both
  firms gives `W₂(t_#μ ∗ Q, t_#ν ∗ Q) ≤ W₂(t_#μ, t_#ν)`. That is **not** an instance of the
  theorems below — it needs a coupling built on `π ⊗ Q` rather than a pushforward of the
  marginals under one shared map — and it needs the group structure of the target, so it is
  proved separately in `PricingPerspective.Transmission.Kernel`
  (`wassersteinDistanceSq_conv_le`).

## Main results

* `map_prodMap_mem_couplingSet` — coordinatewise pushforward maps `Π(μ,ν)` into
  `Π(f_#μ, f_#ν)`.
* `lintegral_edist_sq_map_le` — the cost of a pushed-forward coupling is at most `L²` times
  the original cost.
* `wassersteinDistanceSq_map_le` / `wassersteinDistance_map_le` — the contraction itself.
* `wassersteinDistanceSq_le_antilipschitz_map` /
  `wassersteinDistance_le_antilipschitz_map` — the reverse bound under a measurable
  antilipschitz embedding.
* Their `_of_measurable` corollaries — on a standard Borel metric source,
  Lusin–Souslin derives the embedding from measurability and the injectivity
  supplied by `AntilipschitzWith`.

The pushforward `WassersteinMeasure` is supplied by the caller rather than constructed here,
matching the certificate discipline `OptimalCoupling` already uses: the statement is about
transport cost, not about re-deriving finiteness of a second moment.

## References

* Panaretos & Zemel (2020), *An Invitation to Statistics in Wasserstein Space*, Ch. 1–2.
* Villani (2009), *Optimal Transport: Old and New*, Ch. 6.
-/

namespace WassersteinGeometry

variable {Ω Ω' : Type*}
  [MeasurableSpace Ω] [PseudoMetricSpace Ω] [Inhabited Ω]
  [MeasurableSpace Ω'] [PseudoMetricSpace Ω'] [Inhabited Ω']

omit [PseudoMetricSpace Ω] [Inhabited Ω] [PseudoMetricSpace Ω'] [Inhabited Ω'] in
/-- The coordinatewise pushforward of a coupling is a coupling of the pushforwards.

    If `π ∈ Π(μ, ν)` then `(f × f)_#π ∈ Π(f_#μ, f_#ν)`: composing the pushforward with a
    projection is the same as projecting first and then pushing forward. -/
lemma map_prodMap_mem_couplingSet
    {f : Ω → Ω'} (hf : Measurable f)
    {μ ν : Measure Ω} {π : Measure (Ω × Ω)} (hπ : π ∈ couplingSet μ ν) :
    π.map (fun p => (f p.1, f p.2)) ∈ couplingSet (μ.map f) (ν.map f) := by
  have hff : Measurable fun p : Ω × Ω => (f p.1, f p.2) :=
    (hf.comp measurable_fst).prodMk (hf.comp measurable_snd)
  constructor
  · rw [Measure.map_map measurable_fst hff]
    have : (Prod.fst ∘ fun p : Ω × Ω => (f p.1, f p.2)) = f ∘ Prod.fst := rfl
    rw [this, ← Measure.map_map hf measurable_fst, hπ.1]
  · rw [Measure.map_map measurable_snd hff]
    have : (Prod.snd ∘ fun p : Ω × Ω => (f p.1, f p.2)) = f ∘ Prod.snd := rfl
    rw [this, ← Measure.map_map hf measurable_snd, hπ.2]

omit [Inhabited Ω] [Inhabited Ω'] in
/-- Pushing a coupling forward under an `L`-Lipschitz map scales its cost by at most `L²`. -/
lemma lintegral_edist_sq_map_le
    [OpensMeasurableSpace Ω'] [SecondCountableTopology Ω']
    {L : ℝ≥0} {f : Ω → Ω'} (hf : LipschitzWith L f) (hfm : Measurable f)
    (π : Measure (Ω × Ω)) :
    ∫⁻ p, (edist p.1 p.2) ^ 2 ∂(π.map (fun p => (f p.1, f p.2))) ≤
      (L : ℝ≥0∞) ^ 2 * ∫⁻ p, (edist p.1 p.2) ^ 2 ∂π := by
  have hff : Measurable fun p : Ω × Ω => (f p.1, f p.2) :=
    (hfm.comp measurable_fst).prodMk (hfm.comp measurable_snd)
  have hcost : Measurable fun p : Ω' × Ω' => (edist p.1 p.2) ^ 2 :=
    measurable_edist.pow_const 2
  rw [lintegral_map hcost hff]
  calc ∫⁻ p : Ω × Ω, (edist (f p.1) (f p.2)) ^ 2 ∂π
      ≤ ∫⁻ p : Ω × Ω, ((L : ℝ≥0∞) * edist p.1 p.2) ^ 2 ∂π := by
        refine lintegral_mono fun p => ?_
        exact pow_le_pow_left' (hf.edist_le_mul p.1 p.2) 2
    _ = ∫⁻ p : Ω × Ω, (L : ℝ≥0∞) ^ 2 * (edist p.1 p.2) ^ 2 ∂π := by
        simp_rw [mul_pow]
    _ = (L : ℝ≥0∞) ^ 2 * ∫⁻ p : Ω × Ω, (edist p.1 p.2) ^ 2 ∂π :=
        lintegral_const_mul' _ _ (by simp)

/-- **Quadratic transport contracts under a Lipschitz pushforward.**

    `W₂²(f_#μ, f_#ν) ≤ L² · W₂²(μ, ν)` for `f` `L`-Lipschitz and measurable.

    The pushforward measures are supplied as `WassersteinMeasure`s with the pushforward
    identity as a hypothesis, so the caller discharges finiteness of the pushed-forward
    second moment once, where the ambient space is concrete. -/
theorem wassersteinDistanceSq_map_le
    [OpensMeasurableSpace Ω'] [SecondCountableTopology Ω']
    {L : ℝ≥0} {f : Ω → Ω'} (hf : LipschitzWith L f) (hfm : Measurable f)
    (μ ν : WassersteinMeasure Ω) (μ' ν' : WassersteinMeasure Ω')
    (hμ' : μ'.measure = μ.measure.map f) (hν' : ν'.measure = ν.measure.map f) :
    WassersteinDistanceSq μ' ν' ≤ (L : ℝ≥0∞) ^ 2 * WassersteinDistanceSq μ ν := by
  unfold WassersteinDistanceSq
  set S : Set ℝ≥0∞ :=
    {c | ∃ π ∈ couplingSet μ.measure ν.measure, c = ∫⁻ p, (edist p.1 p.2) ^ 2 ∂π} with hS
  set S' : Set ℝ≥0∞ :=
    {c | ∃ π ∈ couplingSet μ'.measure ν'.measure, c = ∫⁻ p, (edist p.1 p.2) ^ 2 ∂π} with hS'
  -- Every admissible original cost, scaled by `L²`, dominates some admissible pushed cost.
  have key : ∀ c ∈ S, sInf S' ≤ (L : ℝ≥0∞) ^ 2 * c := by
    rintro c ⟨π, hπ, rfl⟩
    refine le_trans (sInf_le ?_) (lintegral_edist_sq_map_le hf hfm π)
    refine ⟨π.map (fun p => (f p.1, f p.2)), ?_, rfl⟩
    rw [hμ', hν']
    exact map_prodMap_mem_couplingSet hfm hπ
  -- `S` is never empty: the independent coupling is always admissible.
  have hprod : μ.measure.prod ν.measure ∈ couplingSet μ.measure ν.measure := by
    haveI : IsProbabilityMeasure μ.measure := ⟨μ.is_probability⟩
    haveI : IsProbabilityMeasure ν.measure := ⟨ν.is_probability⟩
    exact ⟨by rw [Measure.map_fst_prod]; simp [ν.is_probability],
           by rw [Measure.map_snd_prod]; simp [μ.is_probability]⟩
  have hmemS : (∫⁻ p, (edist p.1 p.2) ^ 2 ∂(μ.measure.prod ν.measure)) ∈ S :=
    ⟨_, hprod, rfl⟩
  rcases eq_or_ne L 0 with hL | hL
  · -- `L = 0`: the scaled bound collapses to `0`, and `key` already delivers it.
    subst hL
    simpa using key _ hmemS
  · have h0 : ((L : ℝ≥0∞) ^ 2) ≠ 0 :=
      pow_ne_zero 2 (by exact_mod_cast hL)
    have htop : ((L : ℝ≥0∞) ^ 2) ≠ ⊤ := by
      simp [ENNReal.pow_eq_top_iff]
    calc sInf S' ≤ ⨅ c : S, (L : ℝ≥0∞) ^ 2 * (c : ℝ≥0∞) :=
          le_iInf fun c => key c c.2
      _ = (L : ℝ≥0∞) ^ 2 * ⨅ c : S, (c : ℝ≥0∞) := (ENNReal.mul_iInf_of_ne h0 htop).symm
      _ = (L : ℝ≥0∞) ^ 2 * sInf S := by rw [← sInf_eq_iInf']

/-- **Wasserstein contraction.** `W₂(f_#μ, f_#ν) ≤ L · W₂(μ, ν)` for `f` `L`-Lipschitz.

    The distance-level form of `wassersteinDistanceSq_map_le`, obtained by taking real
    parts and square roots. Finiteness of the source distance is what licenses
    `ENNReal.toReal_mul` here, so the source space carries the measurability hypotheses of
    `wassersteinDistanceSq_lt_top`. -/
theorem wassersteinDistance_map_le
    [OpensMeasurableSpace Ω] [SecondCountableTopology Ω]
    [OpensMeasurableSpace Ω'] [SecondCountableTopology Ω']
    {L : ℝ≥0} {f : Ω → Ω'} (hf : LipschitzWith L f) (hfm : Measurable f)
    (μ ν : WassersteinMeasure Ω) (μ' ν' : WassersteinMeasure Ω')
    (hμ' : μ'.measure = μ.measure.map f) (hν' : ν'.measure = ν.measure.map f) :
    WassersteinDistance μ' ν' ≤ (L : ℝ) * WassersteinDistance μ ν := by
  have hsq := wassersteinDistanceSq_map_le hf hfm μ ν μ' ν' hμ' hν'
  have hfin : WassersteinDistanceSq μ ν ≠ ⊤ := (wassersteinDistanceSq_lt_top μ ν).ne
  have hprod : ((L : ℝ≥0∞) ^ 2 * WassersteinDistanceSq μ ν) ≠ ⊤ := by
    simp [ENNReal.mul_eq_top, hfin, ENNReal.pow_eq_top_iff]
  -- Compare real parts, then square roots.
  have hreal : (WassersteinDistanceSq μ' ν').toReal ≤
      (L : ℝ) ^ 2 * (WassersteinDistanceSq μ ν).toReal := by
    have := ENNReal.toReal_mono hprod hsq
    rwa [ENNReal.toReal_mul, ENNReal.toReal_pow, ENNReal.coe_toReal] at this
  have hLnn : (0 : ℝ) ≤ (L : ℝ) := L.coe_nonneg
  have hrhs : (0 : ℝ) ≤ (L : ℝ) ^ 2 * (WassersteinDistanceSq μ ν).toReal :=
    mul_nonneg (by positivity) ENNReal.toReal_nonneg
  unfold WassersteinDistance
  calc (WassersteinDistanceSq μ' ν').toReal ^ (1 / 2 : ℝ)
      ≤ ((L : ℝ) ^ 2 * (WassersteinDistanceSq μ ν).toReal) ^ (1 / 2 : ℝ) :=
        Real.rpow_le_rpow ENNReal.toReal_nonneg hreal (by norm_num)
    _ = ((L : ℝ) ^ 2) ^ (1 / 2 : ℝ) * (WassersteinDistanceSq μ ν).toReal ^ (1 / 2 : ℝ) :=
        Real.mul_rpow (by positivity) ENNReal.toReal_nonneg
    _ = (L : ℝ) * (WassersteinDistanceSq μ ν).toReal ^ (1 / 2 : ℝ) := by
        congr 1
        rw [← Real.rpow_natCast (L : ℝ) 2, ← Real.rpow_mul hLnn]
        norm_num

/-! ### Reverse transport under a measurable antilipschitz embedding -/

omit [Inhabited Ω'] in
/-- Pulling a coupling back through a measurable embedding with an
`K`-antilipschitz forward map scales its quadratic cost by at most `K²`.

The inverse is only controlled on `Set.range f`.  Both marginals of the target
coupling are pushforwards through `f`, so the coupling lies in
`Set.range f ×ˢ Set.range f` almost everywhere; this is exactly what licenses
the pointwise antilipschitz estimate under the integral.
-/
lemma lintegral_edist_sq_invFun_map_le
    [OpensMeasurableSpace Ω] [SecondCountableTopology Ω]
    {K : ℝ≥0} {f : Ω → Ω'} (hf : AntilipschitzWith K f)
    (hme : MeasurableEmbedding f) (π : Measure (Ω' × Ω'))
    (μ ν : Measure Ω)
    (hπ : π ∈ couplingSet (μ.map f) (ν.map f)) :
    ∫⁻ p, (edist p.1 p.2) ^ 2
        ∂(π.map (fun p => (hme.invFun p.1, hme.invFun p.2))) ≤
      (K : ℝ≥0∞) ^ 2 * ∫⁻ p, (edist p.1 p.2) ^ 2 ∂π := by
  let g : Ω' → Ω := hme.invFun
  have hg : Measurable g := hme.measurable_invFun
  have hgg : Measurable fun p : Ω' × Ω' => (g p.1, g p.2) :=
    (hg.comp measurable_fst).prodMk (hg.comp measurable_snd)
  have hcost : Measurable fun p : Ω × Ω => (edist p.1 p.2) ^ 2 :=
    measurable_edist.pow_const 2
  have hfst : ∀ᵐ p ∂π, p.1 ∈ Set.range f := by
    apply ae_of_ae_map measurable_fst.aemeasurable
    rw [hπ.1]
    exact ae_map_mem_range f hme.measurableSet_range μ
  have hsnd : ∀ᵐ p ∂π, p.2 ∈ Set.range f := by
    apply ae_of_ae_map measurable_snd.aemeasurable
    rw [hπ.2]
    exact ae_map_mem_range f hme.measurableSet_range ν
  rw [lintegral_map hcost hgg]
  calc
    ∫⁻ p, (edist (g p.1) (g p.2)) ^ 2 ∂π ≤
        ∫⁻ p, ((K : ℝ≥0∞) * edist p.1 p.2) ^ 2 ∂π := by
      refine lintegral_mono_ae ?_
      filter_upwards [hfst, hsnd] with p hp hq
      rcases hp with ⟨x, hx⟩
      rcases hq with ⟨y, hy⟩
      rw [← hx, ← hy]
      have hgx : g (f x) = x := hme.leftInverse_invFun x
      have hgy : g (f y) = y := hme.leftInverse_invFun y
      rw [hgx, hgy]
      exact pow_le_pow_left' (hf x y) 2
    _ = ∫⁻ p, (K : ℝ≥0∞) ^ 2 * (edist p.1 p.2) ^ 2 ∂π := by
      simp_rw [mul_pow]
    _ = (K : ℝ≥0∞) ^ 2 * ∫⁻ p, (edist p.1 p.2) ^ 2 ∂π :=
      lintegral_const_mul' _ _ (by simp)

omit [PseudoMetricSpace Ω] [PseudoMetricSpace Ω'] [Inhabited Ω'] in
/-- Pulling a coupling of two pushforwards back through the measurable inverse
produces a coupling of the original measures. -/
lemma map_invFun_prodMap_mem_couplingSet
    {f : Ω → Ω'} (hme : MeasurableEmbedding f)
    {μ ν : Measure Ω} {π : Measure (Ω' × Ω')}
    (hπ : π ∈ couplingSet (μ.map f) (ν.map f)) :
    π.map (fun p => (hme.invFun p.1, hme.invFun p.2)) ∈ couplingSet μ ν := by
  let g : Ω' → Ω := hme.invFun
  have hg : Measurable g := hme.measurable_invFun
  have hgg : Measurable fun p : Ω' × Ω' => (g p.1, g p.2) :=
    (hg.comp measurable_fst).prodMk (hg.comp measurable_snd)
  constructor
  · rw [Measure.map_map measurable_fst hgg]
    change π.map (g ∘ Prod.fst) = μ
    rw [← Measure.map_map hg measurable_fst, hπ.1, Measure.map_map hg hme.measurable]
    simp [g, Function.LeftInverse.id hme.leftInverse_invFun]
  · rw [Measure.map_map measurable_snd hgg]
    change π.map (g ∘ Prod.snd) = ν
    rw [← Measure.map_map hg measurable_snd, hπ.2, Measure.map_map hg hme.measurable]
    simp [g, Function.LeftInverse.id hme.leftInverse_invFun]

/-- **Reverse quadratic transport under an antilipschitz measurable embedding.**

If `f` is `K`-antilipschitz and measurably invertible on its image, then
`W₂²(μ,ν) ≤ K² W₂²(f#μ,f#ν)`.  Unlike an axiom-level lower pushforward premise,
this theorem constructs the inverse coupling explicitly.
-/
theorem wassersteinDistanceSq_le_antilipschitz_map
    [OpensMeasurableSpace Ω] [SecondCountableTopology Ω]
    {K : ℝ≥0} {f : Ω → Ω'} (hf : AntilipschitzWith K f)
    (hme : MeasurableEmbedding f)
    (μ ν : WassersteinMeasure Ω) (μ' ν' : WassersteinMeasure Ω')
    (hμ' : μ'.measure = μ.measure.map f) (hν' : ν'.measure = ν.measure.map f) :
    WassersteinDistanceSq μ ν ≤
      (K : ℝ≥0∞) ^ 2 * WassersteinDistanceSq μ' ν' := by
  unfold WassersteinDistanceSq
  set S : Set ℝ≥0∞ :=
    {c | ∃ π ∈ couplingSet μ.measure ν.measure,
      c = ∫⁻ p, (edist p.1 p.2) ^ 2 ∂π} with hS
  set S' : Set ℝ≥0∞ :=
    {c | ∃ π ∈ couplingSet μ'.measure ν'.measure,
      c = ∫⁻ p, (edist p.1 p.2) ^ 2 ∂π} with hS'
  have key : ∀ c ∈ S', sInf S ≤ (K : ℝ≥0∞) ^ 2 * c := by
    rintro c ⟨π, hπ, rfl⟩
    have hpushed : π ∈ couplingSet (μ.measure.map f) (ν.measure.map f) := by
      rwa [← hμ', ← hν']
    refine le_trans (sInf_le ?_) (lintegral_edist_sq_invFun_map_le hf hme π
      μ.measure ν.measure hpushed)
    exact ⟨π.map (fun p => (hme.invFun p.1, hme.invFun p.2)),
      map_invFun_prodMap_mem_couplingSet hme hpushed, rfl⟩
  have hprod : μ'.measure.prod ν'.measure ∈
      couplingSet μ'.measure ν'.measure := by
    haveI : IsProbabilityMeasure μ'.measure := ⟨μ'.is_probability⟩
    haveI : IsProbabilityMeasure ν'.measure := ⟨ν'.is_probability⟩
    exact ⟨by rw [Measure.map_fst_prod]; simp [ν'.is_probability],
      by rw [Measure.map_snd_prod]; simp [μ'.is_probability]⟩
  have hmemS' :
      (∫⁻ p, (edist p.1 p.2) ^ 2 ∂(μ'.measure.prod ν'.measure)) ∈ S' :=
    ⟨_, hprod, rfl⟩
  rcases eq_or_ne K 0 with hK | hK
  · subst hK
    simpa using key _ hmemS'
  · have h0 : ((K : ℝ≥0∞) ^ 2) ≠ 0 :=
      pow_ne_zero 2 (by exact_mod_cast hK)
    have htop : ((K : ℝ≥0∞) ^ 2) ≠ ⊤ := by
      simp [ENNReal.pow_eq_top_iff]
    calc
      sInf S ≤ ⨅ c : S', (K : ℝ≥0∞) ^ 2 * (c : ℝ≥0∞) :=
        le_iInf fun c => key c c.2
      _ = (K : ℝ≥0∞) ^ 2 * ⨅ c : S', (c : ℝ≥0∞) :=
        (ENNReal.mul_iInf_of_ne h0 htop).symm
      _ = (K : ℝ≥0∞) ^ 2 * sInf S' := by rw [← sInf_eq_iInf']

/-- **Reverse Wasserstein pushforward bound.**

`W₂(μ,ν) ≤ K W₂(f#μ,f#ν)` for a `K`-antilipschitz measurable embedding.
-/
theorem wassersteinDistance_le_antilipschitz_map
    [OpensMeasurableSpace Ω] [SecondCountableTopology Ω]
    [OpensMeasurableSpace Ω'] [SecondCountableTopology Ω']
    {K : ℝ≥0} {f : Ω → Ω'} (hf : AntilipschitzWith K f)
    (hme : MeasurableEmbedding f)
    (μ ν : WassersteinMeasure Ω) (μ' ν' : WassersteinMeasure Ω')
    (hμ' : μ'.measure = μ.measure.map f) (hν' : ν'.measure = ν.measure.map f) :
    WassersteinDistance μ ν ≤ (K : ℝ) * WassersteinDistance μ' ν' := by
  have hsq := wassersteinDistanceSq_le_antilipschitz_map hf hme μ ν μ' ν' hμ' hν'
  have hfin : WassersteinDistanceSq μ' ν' ≠ ⊤ :=
    (wassersteinDistanceSq_lt_top μ' ν').ne
  have hprod : ((K : ℝ≥0∞) ^ 2 * WassersteinDistanceSq μ' ν') ≠ ⊤ := by
    simp [ENNReal.mul_eq_top, hfin, ENNReal.pow_eq_top_iff]
  have hreal : (WassersteinDistanceSq μ ν).toReal ≤
      (K : ℝ) ^ 2 * (WassersteinDistanceSq μ' ν').toReal := by
    have := ENNReal.toReal_mono hprod hsq
    rwa [ENNReal.toReal_mul, ENNReal.toReal_pow, ENNReal.coe_toReal] at this
  have hKnn : (0 : ℝ) ≤ (K : ℝ) := K.coe_nonneg
  unfold WassersteinDistance
  calc
    (WassersteinDistanceSq μ ν).toReal ^ (1 / 2 : ℝ) ≤
        ((K : ℝ) ^ 2 * (WassersteinDistanceSq μ' ν').toReal) ^ (1 / 2 : ℝ) :=
      Real.rpow_le_rpow ENNReal.toReal_nonneg hreal (by norm_num)
    _ = ((K : ℝ) ^ 2) ^ (1 / 2 : ℝ) *
        (WassersteinDistanceSq μ' ν').toReal ^ (1 / 2 : ℝ) :=
      Real.mul_rpow (by positivity) ENNReal.toReal_nonneg
    _ = (K : ℝ) * (WassersteinDistanceSq μ' ν').toReal ^ (1 / 2 : ℝ) := by
      congr 1
      rw [← Real.rpow_natCast (K : ℝ) 2, ← Real.rpow_mul hKnn]
      norm_num

section StandardBorelSource

variable {S T : Type*}
  [MeasurableSpace S] [MetricSpace S] [Inhabited S]
  [OpensMeasurableSpace S] [SecondCountableTopology S] [StandardBorelSpace S]
  [MeasurableSpace T] [PseudoMetricSpace T] [Inhabited T]
  [OpensMeasurableSpace T] [SecondCountableTopology T]
  [MeasurableSpace.CountablySeparated T]

omit [OpensMeasurableSpace T] [SecondCountableTopology T] in
/-- On a standard Borel metric source, measurability and `AntilipschitzWith`
automatically provide the measurable embedding needed by the reverse squared
Wasserstein theorem. -/
theorem wassersteinDistanceSq_le_antilipschitz_map_of_measurable
    {K : ℝ≥0} {f : S → T} (hf : AntilipschitzWith K f) (hm : Measurable f)
    (μ ν : WassersteinMeasure S) (μ' ν' : WassersteinMeasure T)
    (hμ' : μ'.measure = μ.measure.map f) (hν' : ν'.measure = ν.measure.map f) :
    WassersteinDistanceSq μ ν ≤
      (K : ℝ≥0∞) ^ 2 * WassersteinDistanceSq μ' ν' :=
  wassersteinDistanceSq_le_antilipschitz_map hf
    (hm.measurableEmbedding hf.injective) μ ν μ' ν' hμ' hν'

/-- On a standard Borel metric source, measurability and `AntilipschitzWith`
automatically provide the measurable embedding needed by the reverse
Wasserstein theorem. -/
theorem wassersteinDistance_le_antilipschitz_map_of_measurable
    {K : ℝ≥0} {f : S → T} (hf : AntilipschitzWith K f) (hm : Measurable f)
    (μ ν : WassersteinMeasure S) (μ' ν' : WassersteinMeasure T)
    (hμ' : μ'.measure = μ.measure.map f) (hν' : ν'.measure = ν.measure.map f) :
    WassersteinDistance μ ν ≤ (K : ℝ) * WassersteinDistance μ' ν' :=
  wassersteinDistance_le_antilipschitz_map hf
    (hm.measurableEmbedding hf.injective) μ ν μ' ν' hμ' hν'

end StandardBorelSource

end WassersteinGeometry
