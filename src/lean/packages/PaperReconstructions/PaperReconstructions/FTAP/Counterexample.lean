import PaperReconstructions.FTAP.Defs

/-!
# `Measurable Y` is necessary in the one-period FTAP

`PaperReconstructions.ftap_one_period_vector_iff` carries a `Measurable Y` hypothesis.
This module shows that hypothesis is **load-bearing**: dropping it makes the statement
false, so its presence is a soundness requirement rather than a convenience for the proof.

## The witness

`CoarseBool` is the two-point space under the **trivial** σ-algebra `⊥` (only `∅` and
`univ` are measurable), carrying the Dirac probability measure. The market is `F = ℝ` and
the excess return is `Y b = if b then 1 else -1`.

* **No arbitrage holds.** On `⊥` a probability measure has no nonempty null set
  (`eq_empty_of_measure_zero_bot`), so "a.e." collapses to "everywhere". A gain `θ · Y`
  that is everywhere `≥ 0` forces `θ = 0`, because `Y` takes both signs; and at `θ = 0`
  the conclusion is immediate. So every nonzero `θ` fails the hypothesis and the
  implication is vacuous.

* **No equivalent martingale measure exists.** `IsEMMFTAP` demands `Integrable Y Q`,
  hence `AEStronglyMeasurable Y Q`. As `Q` and the base measure are mutually absolutely
  continuous they share null sets, so `Y` would have to agree *everywhere* with a
  `⊥`-measurable `g`. But then `g ⁻¹' (Ioi 0)` contains `true` and misses `false`, so it
  is neither `∅` nor `univ` — contradicting `⊥`-measurability.

Non-measurability is the only available failure mode: with `Measurable Y` the theorem is
true, and the `(1 + ‖Y‖)⁻¹` change of measure inside
`FTAP.exists_isEMMFTAP_of_noArbitrage` removes any *integrability* obstruction.

The refuted statement is quantified over `Type` rather than `Type*`; that is a genuine
instance of the universe-polymorphic form, so refuting it refutes the general statement.
-/

namespace PaperReconstructions.FTAP

open MeasureTheory Set

/-- The two-point space carrying the **trivial** σ-algebra.

A type synonym is needed because `Bool` already has a `MeasurableSpace` instance (the
discrete one); wrapping it keeps `⊥` unambiguous for instance synthesis. -/
def CoarseBool : Type := Bool

namespace CoarseBool

/-- The trivial σ-algebra: only `∅` and `univ` are measurable. -/
instance instMeasurableSpace : MeasurableSpace CoarseBool := ⊥

/-- The inclusion `Bool → CoarseBool` (the identity on the underlying type). -/
def of (b : Bool) : CoarseBool := b

/-- The projection `CoarseBool → Bool` (the identity on the underlying type). -/
def toBool (b : CoarseBool) : Bool := b

/-- Every point is `of true` or `of false`. -/
lemma eq_of_true_or_of_false (b : CoarseBool) : b = of true ∨ b = of false := by
  rcases (b : Bool) with _ | _
  · exact Or.inr rfl
  · exact Or.inl rfl

noncomputable instance instMeasureSpace : MeasureSpace CoarseBool :=
  ⟨Measure.dirac (of true)⟩

instance instIsProbabilityMeasure : IsProbabilityMeasure (volume : Measure CoarseBool) :=
  Measure.dirac.isProbabilityMeasure

end CoarseBool

/-- Under the trivial σ-algebra a probability measure has no nonempty null set: the
measurable hull of a null set is measurable with measure `0`, hence `∅` rather than
`univ`. Consequently "almost everywhere" collapses to "everywhere". -/
lemma eq_empty_of_measure_zero_bot {s : Set CoarseBool}
    (hs : (volume : Measure CoarseBool) s = 0) : s = ∅ := by
  have hsub : s ⊆ toMeasurable (volume : Measure CoarseBool) s := subset_toMeasurable _ s
  have hmeas : MeasurableSet (toMeasurable (volume : Measure CoarseBool) s) :=
    measurableSet_toMeasurable _ s
  have hzero : (volume : Measure CoarseBool) (toMeasurable (volume : Measure CoarseBool) s)
      = 0 := by rw [measure_toMeasurable]; exact hs
  rcases MeasurableSpace.measurableSet_bot_iff.mp hmeas with h | h
  · exact Set.subset_eq_empty hsub h
  · rw [h, measure_univ] at hzero; exact absurd hzero one_ne_zero

/-- On `CoarseBool`, an almost-everywhere property holds everywhere. -/
lemma forall_of_ae {p : CoarseBool → Prop}
    (hp : ∀ᵐ b ∂(volume : Measure CoarseBool), p b) : ∀ b, p b := by
  intro b
  have hnull : (volume : Measure CoarseBool) {b | ¬ p b} = 0 := hp
  have hempty : {b | ¬ p b} = ∅ := eq_empty_of_measure_zero_bot hnull
  by_contra hb
  exact (Set.eq_empty_iff_forall_notMem.mp hempty) b hb

/-- The witness excess return: `+1` at `true`, `-1` at `false`. -/
def witnessY : CoarseBool → ℝ := fun b => if CoarseBool.toBool b then (1 : ℝ) else -1

@[simp] lemma witnessY_true : witnessY (CoarseBool.of true) = 1 := by
  simp [witnessY, CoarseBool.toBool, CoarseBool.of]

@[simp] lemma witnessY_false : witnessY (CoarseBool.of false) = -1 := by
  simp [witnessY, CoarseBool.toBool, CoarseBool.of]

/-- **The unconditional one-period FTAP is false.**

Dropping `Measurable Y` from `PaperReconstructions.ftap_one_period_vector_iff` leaves a
statement refuted by the two-point space under the trivial σ-algebra: no arbitrage holds
there, yet no equivalent martingale measure can exist, because `IsEMMFTAP` requires `Y`
to be integrable and `Y` is not measurable. -/
theorem ftap_one_period_vector_iff_not_unconditional :
    ¬ ∀ (F : Type) [NormedAddCommGroup F] [InnerProductSpace ℝ F] [FiniteDimensional ℝ F]
        (Ω : Type) [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
        (Y : Ω → F),
        NoArbitrageFTAP (volume : Measure Ω) Y ↔
          ∃ Q, IsEMMFTAP (volume : Measure Ω) Y Q := by
  intro h
  -- no arbitrage: every nonzero `θ` fails the hypothesis, since `Y` takes both signs
  have hNA : NoArbitrageFTAP (volume : Measure CoarseBool) witnessY := by
    intro θ hpos
    have hall : ∀ b, 0 ≤ inner ℝ θ (witnessY b) := forall_of_ae hpos
    have h1 : (0 : ℝ) ≤ θ * 1 := by
      simpa [RCLike.inner_apply] using hall (CoarseBool.of true)
    have h2 : (0 : ℝ) ≤ θ * (-1) := by
      simpa [RCLike.inner_apply] using hall (CoarseBool.of false)
    have hθ : θ = 0 := le_antisymm (by linarith) (by linarith)
    filter_upwards with b
    simp [hθ, RCLike.inner_apply]
  -- but no EMM can exist
  obtain ⟨Q, hQ⟩ := (h ℝ CoarseBool witnessY).mp hNA
  obtain ⟨g, hgmeas, hgae⟩ := hQ.int.aestronglyMeasurable
  -- `Q ~ volume`, so `witnessY = g` a.e.-`volume`, hence everywhere
  have hYg : ∀ b, witnessY b = g b := forall_of_ae (hQ.Pabs.ae_eq hgae)
  -- `g ⁻¹' (Ioi 0)` separates `true` from `false` — impossible for a `⊥`-measurable `g`
  have hpre : MeasurableSet (g ⁻¹' Set.Ioi 0) := hgmeas.measurable measurableSet_Ioi
  have htrue : CoarseBool.of true ∈ g ⁻¹' Set.Ioi 0 := by
    have hg1 : g (CoarseBool.of true) = 1 := by rw [← hYg]; simp
    simp [Set.mem_preimage, hg1]
  have hfalse : CoarseBool.of false ∉ g ⁻¹' Set.Ioi 0 := by
    have hg2 : g (CoarseBool.of false) = -1 := by rw [← hYg]; simp
    simp [Set.mem_preimage, hg2]
  rcases MeasurableSpace.measurableSet_bot_iff.mp hpre with hb | hb
  · rw [hb] at htrue; exact absurd htrue (Set.notMem_empty _)
  · rw [hb] at hfalse; exact hfalse (Set.mem_univ _)

end PaperReconstructions.FTAP
