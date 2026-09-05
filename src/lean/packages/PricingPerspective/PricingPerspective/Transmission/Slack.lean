import PricingPerspective.Transmission.Kernel

set_option linter.style.longLine false

open scoped ENNReal NNReal MeasureTheory
open MeasureTheory WassersteinGeometry

/-!
# Heterogeneous slack: why two firms' exposure laws differ

The transfer result the random-functional model needs. For leaky transmission
`Bᵢ = t(Xᵢ) + Uᵢ` with `Xᵢ ∼ Cᵢ` and firm-specific slack `Uᵢ ∼ Qᵢ`,

  `W₂(Pᵢ, Pⱼ) ≤ L · W₂(Cᵢ, Cⱼ) + W₂(Qᵢ, Qⱼ)`,

that is

  **beta-law difference ≤ transmitted characteristic difference + difference in slack.**

Two firms' loading laws can differ for exactly two reasons, and the bound separates them: a
genuine difference in what the firms are, and a difference in how leaky their transmission
is. Neither term is estimated here; the point is that nothing else can contribute.

## Why this is the right form of the transfer bound

The finite development in `Transmission.Transfer` carries slack as a *norm ball*
`‖δᵢ‖ ≤ τᵢ`, which inflates the bound by `2(τᵢ+τⱼ)D + (τᵢ+τⱼ)²` — a term with no
interpretation beyond "the slack is small". Treating slack as a *law* replaces that with
`W₂(Qᵢ, Qⱼ)`, which is zero exactly when the two firms leak the same way, however much they
leak. Common leakage, however large, costs nothing.

That is not a sharpening for its own sake. It is what makes the bound economically
readable: `τ` bounds the magnitude of what transmission drops, whereas `W₂(Qᵢ, Qⱼ)` measures
the part of it that is *idiosyncratic to the firm*, which is the only part that can separate
two exposure laws.

## Proof

Three steps, each already available:

1. triangle inequality through the waypoint `t_#Cⱼ ∗ Qᵢ` (`wasserstein_triangle`);
2. common slack is non-expansive, twice — once on each side
   (`wassersteinDistance_conv_le`, `wassersteinDistance_conv_le_left`);
3. the transmitted laws contract by `L` (`wassersteinDistance_map_le`).

## Caller-supplied laws

Every intermediate law — the transmitted laws, the two leaky laws, and the triangle waypoint
— is supplied as a `WassersteinMeasure` with its defining identity as a hypothesis. Nothing
here re-derives finiteness of a second moment; a caller over a concrete space discharges each
once. This is the same discipline as `WassersteinGeometry.Contraction` and
`Transmission.Frechet`.

## References

* `.context/chat/2026-08-16_random_functions/figures/hilbert_model.md`, final turn,
  "Heterogeneous slack gives another useful bound".
* `.context/chat/2026-08-16_random_functions/README.md`, "Transmission is contractive".
-/

namespace PricingPerspective.Transmission

variable {X H : Type*}
  [MeasurableSpace X] [PseudoMetricSpace X] [Inhabited X]
  [OpensMeasurableSpace X] [SecondCountableTopology X]
  [NormedAddCommGroup H] [MeasurableSpace H] [BorelSpace H]
  [SecondCountableTopology H] [CompleteSpace H] [Inhabited H] [MeasurableAdd₂ H]

/-- **The heterogeneous-slack transfer bound.**

    `W₂(Pᵢ, Pⱼ) ≤ L · W₂(Cᵢ, Cⱼ) + W₂(Qᵢ, Qⱼ)` for leaky transmission
    `Pᵢ = t_#Cᵢ ∗ Qᵢ` with `t` `L`-Lipschitz.

    A firm's exposure law is displaced from another's by at most the transmitted difference
    in characteristics plus the difference in transmission slack. Setting `Qᵢ = Qⱼ` recovers
    pure contraction: identical leakage cannot separate two firms at all, no matter how large
    it is. Setting both to a point mass recovers the deterministic map. -/
theorem wassersteinDistance_leaky_le
    {L : ℝ≥0} {t : X → H} (hL : LipschitzWith L t) (ht : Measurable t)
    (Ci Cj : WassersteinMeasure X) (Qi Qj : WassersteinMeasure H)
    -- the transmitted characteristic laws
    (PA PB : WassersteinMeasure H)
    (hPA : PA.measure = Ci.measure.map t) (hPB : PB.measure = Cj.measure.map t)
    -- the two leaky exposure laws, and the triangle waypoint
    (Pi Pj Mid : WassersteinMeasure H)
    (hPi : Pi.measure = PA.measure ∗ Qi.measure)
    (hPj : Pj.measure = PB.measure ∗ Qj.measure)
    (hMid : Mid.measure = PB.measure ∗ Qi.measure) :
    WassersteinDistance Pi Pj ≤
      (L : ℝ) * WassersteinDistance Ci Cj + WassersteinDistance Qi Qj := by
  haveI : IsProbabilityMeasure Qi.measure := ⟨Qi.is_probability⟩
  haveI : IsProbabilityMeasure Qj.measure := ⟨Qj.is_probability⟩
  haveI : IsProbabilityMeasure PB.measure := ⟨PB.is_probability⟩
  -- 1. Route through the waypoint `t_#Cⱼ ∗ Qᵢ`.
  have htri : WassersteinDistance Pi Pj ≤
      WassersteinDistance Pi Mid + WassersteinDistance Mid Pj :=
    wasserstein_triangle Pi Pj Mid
  -- 2a. Left leg: the slack law `Qᵢ` is common, so it is non-expansive.
  have hleft : WassersteinDistance Pi Mid ≤ WassersteinDistance PA PB :=
    wassersteinDistance_conv_le (Q := Qi.measure) PA PB Pi Mid hPi hMid
  -- 2b. Right leg: now the transmitted law `PB` is common, on the other side.
  have hright : WassersteinDistance Mid Pj ≤ WassersteinDistance Qi Qj :=
    wassersteinDistance_conv_le_left (R := PB.measure) Qi Qj Mid Pj
      (by rw [hMid, Measure.conv_comm]) (by rw [hPj, Measure.conv_comm])
  -- 3. The transmitted laws contract by the Lipschitz constant of `t`.
  have hmap : WassersteinDistance PA PB ≤ (L : ℝ) * WassersteinDistance Ci Cj :=
    wassersteinDistance_map_le hL ht Ci Cj PA PB hPA hPB
  linarith

omit [CompleteSpace H] in
/-- **Common slack cannot separate two firms.**

    The `Qᵢ = Qⱼ` corner of `wassersteinDistance_leaky_le`: with identical leakage the bound
    collapses to pure contraction, `W₂(Pᵢ, Pⱼ) ≤ L · W₂(Cᵢ, Cⱼ)`.

    This is the formal reason a *lower* bi-Lipschitz assumption on transmission is
    substantive rather than technical. Leakage can only compress; assuming it never does is
    an economic claim about how much firm-specific information survives transmission, and
    the covariance floor does not need that claim. -/
theorem wassersteinDistance_leaky_le_of_common_slack
    {L : ℝ≥0} {t : X → H} (hL : LipschitzWith L t) (ht : Measurable t)
    (Ci Cj : WassersteinMeasure X) (Q : WassersteinMeasure H)
    (PA PB : WassersteinMeasure H)
    (hPA : PA.measure = Ci.measure.map t) (hPB : PB.measure = Cj.measure.map t)
    (Pi Pj : WassersteinMeasure H)
    (hPi : Pi.measure = PA.measure ∗ Q.measure)
    (hPj : Pj.measure = PB.measure ∗ Q.measure) :
    WassersteinDistance Pi Pj ≤ (L : ℝ) * WassersteinDistance Ci Cj := by
  haveI : IsProbabilityMeasure Q.measure := ⟨Q.is_probability⟩
  have hconv : WassersteinDistance Pi Pj ≤ WassersteinDistance PA PB :=
    wassersteinDistance_conv_le (Q := Q.measure) PA PB Pi Pj hPi hPj
  exact hconv.trans (wassersteinDistance_map_le hL ht Ci Cj PA PB hPA hPB)

/-! ### Bounded slack riding on the characteristic draw

The convolution form above treats slack as an *independent* law. The manuscript's maintained
assumption is different and, for its purposes, more natural: slack is a measurable function
of the draw itself,

  `Bᵢ = T(ξᵢ) + εᵢ(ξᵢ)`,  `‖Γ^{1/2}εᵢ(x)‖ ≤ τᵢ` pointwise,

so it is *synchronous* with the characteristic rather than independent of it. Neither model
implies the other: independent slack can be unbounded, and synchronous slack can be perfectly
correlated with the characteristic. This section proves the transfer bracket for the second,
which is what `cor:random-w2-metric` states.

Working throughout in risk coordinates, so `t` below is `T_Γ = Γ^{1/2}∘T` and the maps `uᵢ`
are `T_Γ + Γ^{1/2}εᵢ`; `Transmission.RiskCoordinates` supplies that translation. -/

section BoundedSlack

omit [PseudoMetricSpace X] [Inhabited X] [OpensMeasurableSpace X] [SecondCountableTopology X]
  [NormedAddCommGroup H] [BorelSpace H] [SecondCountableTopology H] [CompleteSpace H]
  [Inhabited H] [MeasurableAdd₂ H] in
/-- The synchronous coupling: push one law forward under both maps at once. -/
lemma map_pair_mem_couplingSet_of_map
    {μ : Measure X} {f g : X → H} (hf : Measurable f) (hg : Measurable g) :
    μ.map (fun x => (f x, g x)) ∈ couplingSet (μ.map f) (μ.map g) := by
  have hpair : Measurable fun x : X => (f x, g x) := hf.prodMk hg
  constructor
  · rw [Measure.map_map measurable_fst hpair]; rfl
  · rw [Measure.map_map measurable_snd hpair]; rfl

omit [OpensMeasurableSpace X] [SecondCountableTopology X] [CompleteSpace H]
  [MeasurableAdd₂ H] in
/-- **An RMS perturbation controls squared Wasserstein distance.**

The synchronous coupling only needs an integrated quadratic slack bound; the
manuscript's pointwise radius is a stronger sufficient condition.
-/
theorem wassersteinDistanceSq_le_of_lintegral_edist_sq_le
    (C : WassersteinMeasure X) {f g : X → H} (hf : Measurable f) (hg : Measurable g)
    {s : ℝ≥0}
    (hs : ∫⁻ x, (edist (f x) (g x)) ^ 2 ∂C.measure ≤ (s : ℝ≥0∞) ^ 2)
    (P P' : WassersteinMeasure H)
    (hP : P.measure = C.measure.map f) (hP' : P'.measure = C.measure.map g) :
    WassersteinDistanceSq P P' ≤ (s : ℝ≥0∞) ^ 2 := by
  have hpair : Measurable fun x : X ↦ (f x, g x) := hf.prodMk hg
  have hcost : Measurable fun z : H × H ↦ (edist z.1 z.2) ^ 2 :=
    measurable_edist.pow_const 2
  refine le_trans (sInf_le ⟨C.measure.map (fun x ↦ (f x, g x)), ?_, rfl⟩) ?_
  · rw [hP, hP']
    exact map_pair_mem_couplingSet_of_map hf hg
  · rw [lintegral_map hcost hpair]
    exact hs

omit [OpensMeasurableSpace X] [SecondCountableTopology X] [CompleteSpace H]
  [MeasurableAdd₂ H] in
/-- Distance form of the synchronous RMS perturbation bound. -/
theorem wassersteinDistance_le_of_lintegral_edist_sq_le
    (C : WassersteinMeasure X) {f g : X → H} (hf : Measurable f) (hg : Measurable g)
    {s : ℝ≥0}
    (hs : ∫⁻ x, (edist (f x) (g x)) ^ 2 ∂C.measure ≤ (s : ℝ≥0∞) ^ 2)
    (P P' : WassersteinMeasure H)
    (hP : P.measure = C.measure.map f) (hP' : P'.measure = C.measure.map g) :
    WassersteinDistance P P' ≤ (s : ℝ) := by
  have hsq := wassersteinDistanceSq_le_of_lintegral_edist_sq_le
    C hf hg hs P P' hP hP'
  have hle : (WassersteinDistanceSq P P').toReal ≤ (s : ℝ) ^ 2 := by
    have := ENNReal.toReal_mono (by simp : ((s : ℝ≥0∞) ^ 2) ≠ ⊤) hsq
    rwa [ENNReal.toReal_pow, ENNReal.coe_toReal] at this
  unfold WassersteinDistance
  calc
    (WassersteinDistanceSq P P').toReal ^ (1 / 2 : ℝ) ≤
        ((s : ℝ) ^ 2) ^ (1 / 2 : ℝ) :=
      Real.rpow_le_rpow ENNReal.toReal_nonneg hle (by norm_num)
    _ = (s : ℝ) := by
      rw [← Real.rpow_natCast (s : ℝ) 2, ← Real.rpow_mul s.coe_nonneg]
      norm_num

omit [OpensMeasurableSpace X] [SecondCountableTopology X] [CompleteSpace H] [MeasurableAdd₂ H] in
/-- **A pointwise-bounded perturbation moves a law by at most that bound.**

    Coupling `B` with `T(ξ)` through the shared draw `ξ` gives a coupling whose cost is the
    mean squared slack, which the pointwise bound caps at `τ²`. This is the manuscript's
    first and third transfer legs. -/
theorem wassersteinDistanceSq_le_of_edist_le
    (C : WassersteinMeasure X) {f g : X → H} (hf : Measurable f) (hg : Measurable g)
    {τ : ℝ≥0} (hτ : ∀ x, edist (f x) (g x) ≤ (τ : ℝ≥0∞))
    (P P' : WassersteinMeasure H)
    (hP : P.measure = C.measure.map f) (hP' : P'.measure = C.measure.map g) :
    WassersteinDistanceSq P P' ≤ (τ : ℝ≥0∞) ^ 2 := by
  have hpair : Measurable fun x : X => (f x, g x) := hf.prodMk hg
  have hcost : Measurable fun z : H × H => (edist z.1 z.2) ^ 2 :=
    measurable_edist.pow_const 2
  refine le_trans (sInf_le ⟨C.measure.map (fun x => (f x, g x)), ?_, rfl⟩) ?_
  · rw [hP, hP']; exact map_pair_mem_couplingSet_of_map hf hg
  · rw [lintegral_map hcost hpair]
    calc ∫⁻ x, (edist (f x) (g x)) ^ 2 ∂C.measure
        ≤ ∫⁻ _, (τ : ℝ≥0∞) ^ 2 ∂C.measure :=
          lintegral_mono fun x => pow_le_pow_left' (hτ x) 2
      _ = (τ : ℝ≥0∞) ^ 2 := by
          rw [lintegral_const, C.is_probability, mul_one]

omit [OpensMeasurableSpace X] [SecondCountableTopology X] [CompleteSpace H] [MeasurableAdd₂ H] in
/-- The distance form of `wassersteinDistanceSq_le_of_edist_le`. -/
theorem wassersteinDistance_le_of_edist_le
    (C : WassersteinMeasure X) {f g : X → H} (hf : Measurable f) (hg : Measurable g)
    {τ : ℝ≥0} (hτ : ∀ x, edist (f x) (g x) ≤ (τ : ℝ≥0∞))
    (P P' : WassersteinMeasure H)
    (hP : P.measure = C.measure.map f) (hP' : P'.measure = C.measure.map g) :
    WassersteinDistance P P' ≤ (τ : ℝ) := by
  have hsq := wassersteinDistanceSq_le_of_edist_le C hf hg hτ P P' hP hP'
  have hle : (WassersteinDistanceSq P P').toReal ≤ ((τ : ℝ)) ^ 2 := by
    have := ENNReal.toReal_mono (by simp : ((τ : ℝ≥0∞) ^ 2) ≠ ⊤) hsq
    rwa [ENNReal.toReal_pow, ENNReal.coe_toReal] at this
  unfold WassersteinDistance
  calc (WassersteinDistanceSq P P').toReal ^ (1 / 2 : ℝ)
      ≤ ((τ : ℝ) ^ 2) ^ (1 / 2 : ℝ) :=
        Real.rpow_le_rpow ENNReal.toReal_nonneg hle (by norm_num)
    _ = (τ : ℝ) := by
        rw [← Real.rpow_natCast (τ : ℝ) 2, ← Real.rpow_mul τ.coe_nonneg]
        norm_num

omit [MeasurableAdd₂ H] in
/-- **The metric transfer bracket, upper half.**

    `W₂(Pᵢ, Pⱼ) ≤ τᵢ + L·W₂(Cᵢ, Cⱼ) + τⱼ` for exposure maps that differ from a common
    `L`-Lipschitz map by a pointwise-bounded, draw-synchronous slack.

    This is `cor:random-w2-metric`, equation `eq:random-w2-upper-metric`, and the proof is the
    manuscript's own: two synchronous-coupling legs bounding the slack, one contraction leg
    for the transmitted laws, chained by the `W₂` triangle inequality on `P₂(H)`.

    Slack enters **additively**, with no support diameter — unlike the finite development in
    `Transmission.Transfer`, whose bound carries `2(τᵢ+τⱼ)D + (τᵢ+τⱼ)²`. -/
theorem wassersteinDistance_boundedSlack_le
    {L τi τj : ℝ≥0} {t ui uj : X → H}
    (hL : LipschitzWith L t) (ht : Measurable t)
    (hui : Measurable ui) (huj : Measurable uj)
    (hτi : ∀ x, edist (ui x) (t x) ≤ (τi : ℝ≥0∞))
    (hτj : ∀ x, edist (uj x) (t x) ≤ (τj : ℝ≥0∞))
    (Ci Cj : WassersteinMeasure X)
    (Pi Pj TA TB : WassersteinMeasure H)
    (hPi : Pi.measure = Ci.measure.map ui) (hPj : Pj.measure = Cj.measure.map uj)
    (hTA : TA.measure = Ci.measure.map t) (hTB : TB.measure = Cj.measure.map t) :
    WassersteinDistance Pi Pj
      ≤ (τi : ℝ) + (L : ℝ) * WassersteinDistance Ci Cj + (τj : ℝ) := by
  -- Leg 1: the slack on firm `i`.
  have h1 : WassersteinDistance Pi TA ≤ (τi : ℝ) :=
    wassersteinDistance_le_of_edist_le Ci hui ht hτi Pi TA hPi hTA
  -- Leg 3: the slack on firm `j`, read in the reverse direction.
  have h3 : WassersteinDistance TB Pj ≤ (τj : ℝ) := by
    rw [wasserstein_symm]
    exact wassersteinDistance_le_of_edist_le Cj huj ht hτj Pj TB hPj hTB
  -- Leg 2: the transmitted laws contract.
  have h2 : WassersteinDistance TA TB ≤ (L : ℝ) * WassersteinDistance Ci Cj :=
    wassersteinDistance_map_le hL ht Ci Cj TA TB hTA hTB
  -- Chain the three legs.
  have t1 : WassersteinDistance Pi Pj
      ≤ WassersteinDistance Pi TA + WassersteinDistance TA Pj :=
    wasserstein_triangle Pi Pj TA
  have t2 : WassersteinDistance TA Pj
      ≤ WassersteinDistance TA TB + WassersteinDistance TB Pj :=
    wasserstein_triangle TA Pj TB
  linarith

end BoundedSlack

end PricingPerspective.Transmission
