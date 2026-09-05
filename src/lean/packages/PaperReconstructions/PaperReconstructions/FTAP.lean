import PaperReconstructions.FTAP.Defs
import PaperReconstructions.FTAP.Softplus
import PaperReconstructions.FTAP.Potential
import PaperReconstructions.FTAP.OnePeriod
import PaperReconstructions.FTAP.Counterexample

/-!
# One-period Fundamental Theorem of Asset Pricing

Roll-up for the one-period FTAP on a finite-dimensional market. The headline result is
`PaperReconstructions.ftap_one_period_vector_iff`.

| Module | Content |
|---|---|
| `FTAP.Defs` | `NoArbitrageFTAP`, `IsEMMFTAP`, the forward direction, the gains kernel |
| `FTAP.Softplus` | `softplus`, `logistic`, and their bounds |
| `FTAP.Potential` | the softplus potential, coercivity, the global minimiser |
| `FTAP.OnePeriod` | the first-order condition and the EMM construction |
| `FTAP.Counterexample` | why `Measurable Y` cannot be dropped |

## Provenance

The construction is ported from
[formal-mathfin](https://github.com/raphaelrrcoelho/formal-mathfin)
(`MathFin/Foundations/FTAPOnePeriodVector.lean` and `MathFin/Foundations/EquivMeasure.lean`,
Apache-2.0, © Raphael Coelho), adapted to this workspace's `NoArbitrageFTAP` /
`IsEMMFTAP` spelling and to the pinned Lean `v4.31.0` mathlib.

## Statement repair (2026-08-07)

The previous statement of `ftap_one_period_vector_iff` carried **no measurability
hypothesis** on `Y` and had the backward direction left as a gap. That statement is
**false**, not merely unproven — see
`FTAP.ftap_one_period_vector_iff_not_unconditional`. The hypothesis `Measurable Y` (and the ambient
`[MeasurableSpace F] [BorelSpace F]` needed to express it) is therefore part of the
theorem, exactly as upstream has it.
-/

namespace PaperReconstructions

open MeasureTheory

/-- **One-period Fundamental Theorem of Asset Pricing** (finite-dimensional market,
general `Ω`).

For a measurable `F`-valued discounted excess return `Y` (the `d`-asset case is
`F = EuclideanSpace ℝ (Fin d)`), no arbitrage holds iff there is an equivalent martingale
measure `Q ~ P` with `Y` integrable and `E_Q[Y] = 0`.

The backward direction is explicit: minimise the softplus potential
`θ ↦ ∫ log(1 + exp⟪θ, Y⟫)` over the gains kernel's orthogonal complement; its logistic
weight `σ⟪θ₀, Y⟫ ∈ (0, 1)`, normalised, is the EMM density. No Hahn–Banach, no
`L⁰`-cone closedness, no measurable selection, and **no non-redundancy hypothesis**.

`Measurable Y` is necessary, not merely convenient: without it the statement is refuted by
`ftap_one_period_vector_iff_not_unconditional`. -/
theorem ftap_one_period_vector_iff {F : Type*} [NormedAddCommGroup F]
    [InnerProductSpace ℝ F] [FiniteDimensional ℝ F] [MeasurableSpace F] [BorelSpace F]
    {Ω : Type*} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
    (Y : Ω → F) (hY : Measurable Y) :
    NoArbitrageFTAP (volume : Measure Ω) Y ↔
      ∃ Q, IsEMMFTAP (volume : Measure Ω) Y Q :=
  ⟨fun hNA => FTAP.exists_isEMMFTAP_of_noArbitrage (volume : Measure Ω) Y hY hNA,
   fun ⟨_, hQ⟩ => noArbitrage_of_isEMMFTAP (volume : Measure Ω) Y hQ⟩

end PaperReconstructions
