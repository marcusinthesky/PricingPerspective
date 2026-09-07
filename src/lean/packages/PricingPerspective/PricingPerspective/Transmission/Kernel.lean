import WassersteinGeometry.Contraction
import Mathlib.MeasureTheory.Measure.GiryMonad
import Mathlib.MeasureTheory.Group.Convolution

set_option linter.style.longLine false

open scoped ENNReal NNReal MeasureTheory
open MeasureTheory WassersteinGeometry

/-!
# The transmission kernel: characteristics to a law of exposures

The channel `C ↦ P` that turns a firm's distribution-valued characteristic into a *law* of
factor loadings, rather than into a single loading. A draw from the characteristic law
generates a draw from the exposure law,

  `X ∼ C`,  `B = T(X, U)`,

so the induced object is a stochastic kernel `K(x, db) = ℙ(B ∈ db ∣ X = x)` and

  `P = C.bind K`.

Two special cases carry the economics:

* **Deterministic transmission** `B = t(X)` gives an ordinary pushforward `P = t_#C`
  (`transmit_deterministic`);
* **Leaky transmission** `B = t(X) + U` with `U ∼ Q` independent gives a convolution
  `P = t_#C ∗ Q` (`leakyTransmit_eq_conv`).

## The main result: leakage contracts

`wassersteinDistanceSq_conv_le` proves that adding a *common* slack law `Q` to both firms
cannot separate their exposure laws:

  `W₂(μ ∗ Q, ν ∗ Q) ≤ W₂(μ, ν)`.

This is the formal content of information leakage, and it is the item
`WassersteinGeometry.Contraction` flags as open: it is genuinely not an instance of the
Lipschitz-pushforward contraction, because the shifted coupling is built on `π ⊗ Q` rather
than obtained by pushing both marginals through one shared map. In fact the shifted coupling
has *exactly* the original cost, so the admissible cost set only grows and the infimum can
only fall.

Two readings, both worth stating because they constrain what a transmission assumption may
claim:

* Common unobserved variation can make two exposure laws strictly closer than the
  transmitted characteristic laws. Distinctions are destroyed on the way into exposure
  space, never created.
* Therefore a **lower** bi-Lipschitz bound on transmission is not a harmless regularity
  condition — it asserts leakage never erases anything, and the covariance floor does not
  need it.

## References

* `.context/chat/2026-08-16_random_functions/figures/hilbert_model.md`, final turn.
* `.context/chat/2026-08-16_random_functions/README.md`, "The transmission kernel".
-/

namespace PricingPerspective.Transmission

variable {X H : Type*}
  [MeasurableSpace X]
  [NormedAddCommGroup H] [MeasurableSpace H] [BorelSpace H]
  [SecondCountableTopology H] [Inhabited H]

/-! ### The channel -/

/-- The transmission channel `P = C K`: a characteristic law composed with a kernel.

    `K x` is the conditional law of the loading given the characteristic draw `x`. -/
noncomputable def transmit (C : Measure X) (K : X → Measure H) : Measure H :=
  C.bind K

omit [NormedAddCommGroup H] [BorelSpace H] [SecondCountableTopology H] [Inhabited H] in
/-- **Deterministic transmission is a pushforward.**

    When the kernel is a point mass, `B = t(X)`, the channel collapses to `P = t_#C` — the
    mapping model, recovered as the zero-slack corner of the kernel model. -/
theorem transmit_deterministic (C : Measure X) {t : X → H} (ht : Measurable t) :
    transmit C (fun x => Measure.dirac (t x)) = C.map t :=
  Measure.bind_dirac_eq_map C ht

/-- **Leaky transmission**: deterministic transmission followed by independent slack. -/
noncomputable def leakyTransmit (C : Measure X) (t : X → H) (Q : Measure H) : Measure H :=
  (C.map t) ∗ Q

omit [BorelSpace H] [SecondCountableTopology H] [Inhabited H] in
/-- Leaky transmission is by definition the convolution of the transmitted law with the
    slack law; recorded as a lemma so downstream files never unfold the definition. -/
theorem leakyTransmit_eq_conv (C : Measure X) (t : X → H) (Q : Measure H) :
    leakyTransmit C t Q = (C.map t) ∗ Q := rfl

/-! ### Common slack cannot separate two exposure laws -/

section Convolution

variable [MeasurableAdd₂ H]

omit [BorelSpace H] [SecondCountableTopology H] [Inhabited H] in
/-- Translating both coordinates of a coupling by a common slack draw yields a coupling of
    the convolved marginals.

    The shift `(x, y, u) ↦ (x + u, y + u)` moves `Π(μ, ν)` into `Π(μ ∗ Q, ν ∗ Q)`. -/
lemma map_shift_mem_couplingSet
    {μ ν : Measure H} {Q : Measure H} [SFinite Q] [IsProbabilityMeasure Q]
    {π : Measure (H × H)} [SFinite π] (hπ : π ∈ couplingSet μ ν) :
    (π.prod Q).map (fun z : (H × H) × H => (z.1.1 + z.2, z.1.2 + z.2)) ∈
      couplingSet (μ ∗ Q) (ν ∗ Q) := by
  have hshift : Measurable fun z : (H × H) × H => (z.1.1 + z.2, z.1.2 + z.2) :=
    (((measurable_fst.comp measurable_fst).add measurable_snd)).prodMk
      (((measurable_snd.comp measurable_fst).add measurable_snd))
  have hpf : Measurable (Prod.map (Prod.fst : H × H → H) (id : H → H)) :=
    measurable_fst.prodMap measurable_id
  have hps : Measurable (Prod.map (Prod.snd : H × H → H) (id : H → H)) :=
    measurable_snd.prodMap measurable_id
  constructor
  · -- First marginal: project the coupling, then convolve.
    have hmarg : Measure.map (Prod.map (Prod.fst : H × H → H) (id : H → H)) (π.prod Q)
        = μ.prod Q := by
      rw [← Measure.map_prod_map π Q measurable_fst measurable_id, hπ.1, Measure.map_id]
    have hcomp : (Prod.fst ∘ fun z : (H × H) × H => (z.1.1 + z.2, z.1.2 + z.2)) =
        (fun p : H × H => p.1 + p.2) ∘ (Prod.map Prod.fst (id : H → H)) := rfl
    rw [Measure.map_map measurable_fst hshift, hcomp,
      ← Measure.map_map measurable_add hpf, hmarg]
    rfl
  · -- Second marginal: identical, through the other projection.
    have hmarg : Measure.map (Prod.map (Prod.snd : H × H → H) (id : H → H)) (π.prod Q)
        = ν.prod Q := by
      rw [← Measure.map_prod_map π Q measurable_snd measurable_id, hπ.2, Measure.map_id]
    have hcomp : (Prod.snd ∘ fun z : (H × H) × H => (z.1.1 + z.2, z.1.2 + z.2)) =
        (fun p : H × H => p.1 + p.2) ∘ (Prod.map Prod.snd (id : H → H)) := rfl
    rw [Measure.map_map measurable_snd hshift, hcomp,
      ← Measure.map_map measurable_add hps, hmarg]
    rfl

omit [Inhabited H] in
/-- A common translation leaves transport cost unchanged.

    `edist (x + u) (y + u) = edist x y`, and the slack coordinate then integrates out
    against a probability measure. -/
lemma lintegral_edist_sq_map_shift
    {Q : Measure H} [SFinite Q] [IsProbabilityMeasure Q]
    (π : Measure (H × H)) [SFinite π] :
    ∫⁻ p, (edist p.1 p.2) ^ 2 ∂((π.prod Q).map
        (fun z : (H × H) × H => (z.1.1 + z.2, z.1.2 + z.2))) =
      ∫⁻ p, (edist p.1 p.2) ^ 2 ∂π := by
  have hshift : Measurable fun z : (H × H) × H => (z.1.1 + z.2, z.1.2 + z.2) :=
    (((measurable_fst.comp measurable_fst).add measurable_snd)).prodMk
      (((measurable_snd.comp measurable_fst).add measurable_snd))
  have hcost : Measurable fun p : H × H => (edist p.1 p.2) ^ 2 :=
    measurable_edist.pow_const 2
  rw [lintegral_map hcost hshift]
  have hshift_cost : ∀ z : (H × H) × H,
      (edist (z.1.1 + z.2) (z.1.2 + z.2)) ^ 2 = (edist z.1.1 z.1.2) ^ 2 := by
    intro z
    rw [edist_dist, edist_dist, dist_add_right]
  simp_rw [hshift_cost]
  -- The integrand no longer depends on the slack coordinate, so it integrates out.
  rw [← lintegral_map hcost measurable_fst, Measure.map_fst_prod]
  simp

/-- **Common slack contracts quadratic transport.**

    `W₂²(μ ∗ Q, ν ∗ Q) ≤ W₂²(μ, ν)`: convolving both laws with one slack law `Q` cannot
    push them apart.

    The proof is set inclusion rather than an inequality between costs. Every admissible
    cost for `(μ, ν)` is *also* admissible for the convolved pair, realized by the shifted
    coupling at exactly the same value, so the infimum on the convolved side is taken over a
    larger set. -/
theorem wassersteinDistanceSq_conv_le
    {Q : Measure H} [SFinite Q] [IsProbabilityMeasure Q]
    (μ ν : WassersteinMeasure H) (μ' ν' : WassersteinMeasure H)
    (hμ' : μ'.measure = μ.measure ∗ Q) (hν' : ν'.measure = ν.measure ∗ Q) :
    WassersteinDistanceSq μ' ν' ≤ WassersteinDistanceSq μ ν := by
  unfold WassersteinDistanceSq
  refine sInf_le_sInf ?_
  rintro c ⟨π, hπ, rfl⟩
  haveI : IsProbabilityMeasure π := coupling_isProbabilityMeasure μ.is_probability hπ
  refine ⟨(π.prod Q).map (fun z : (H × H) × H => (z.1.1 + z.2, z.1.2 + z.2)), ?_,
    (lintegral_edist_sq_map_shift π).symm⟩
  rw [hμ', hν']
  exact map_shift_mem_couplingSet hπ

/-- The distance form of `wassersteinDistanceSq_conv_le`. -/
theorem wassersteinDistance_conv_le
    {Q : Measure H} [SFinite Q] [IsProbabilityMeasure Q]
    (μ ν : WassersteinMeasure H) (μ' ν' : WassersteinMeasure H)
    (hμ' : μ'.measure = μ.measure ∗ Q) (hν' : ν'.measure = ν.measure ∗ Q) :
    WassersteinDistance μ' ν' ≤ WassersteinDistance μ ν := by
  have hsq := wassersteinDistanceSq_conv_le μ ν μ' ν' hμ' hν'
  have hfin : WassersteinDistanceSq μ ν ≠ ⊤ := (wassersteinDistanceSq_lt_top μ ν).ne
  unfold WassersteinDistance
  exact Real.rpow_le_rpow ENNReal.toReal_nonneg (ENNReal.toReal_mono hfin hsq) (by norm_num)

/-- Convolution is non-expansive in its *left* argument too, by commutativity. -/
theorem wassersteinDistance_conv_le_left
    {R : Measure H} [SFinite R] [IsProbabilityMeasure R]
    (μ ν : WassersteinMeasure H) [SFinite μ.measure] [SFinite ν.measure]
    (μ' ν' : WassersteinMeasure H)
    (hμ' : μ'.measure = R ∗ μ.measure) (hν' : ν'.measure = R ∗ ν.measure) :
    WassersteinDistance μ' ν' ≤ WassersteinDistance μ ν :=
  wassersteinDistance_conv_le (Q := R) μ ν μ' ν'
    (by rw [hμ', Measure.conv_comm]) (by rw [hν', Measure.conv_comm])

end Convolution

end PricingPerspective.Transmission
