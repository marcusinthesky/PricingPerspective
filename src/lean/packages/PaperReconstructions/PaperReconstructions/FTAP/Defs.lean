import Mathlib.MeasureTheory.Function.L2Space
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.MeasureTheory.Measure.Typeclasses.Probability
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.LinearAlgebra.FiniteDimensional.Defs
import Mathlib.Algebra.Module.Submodule.Defs

/-!
# One-period FTAP — definitions and the forward direction

No-arbitrage and equivalent-martingale-measure predicates for a one-period,
finite-dimensional market, together with the easy direction `EMM ⟹ no arbitrage`
and the *gains kernel* of redundant portfolio directions.

The market is a finite-dimensional real inner-product space `F` (the `d`-asset case is
`F = EuclideanSpace ℝ (Fin d)`), the discounted excess return is `Y : Ω → F`, and
portfolios are **constant** vectors `θ : F` (trivial initial information).

## Provenance

`NoArbitrageFTAP` and `IsEMMFTAP` were previously stated in
`PaperReconstructions.Ross1976.Model`; they are relocated here unchanged, so
`import PaperReconstructions.Ross1976.Model` still provides them transitively.

The formulation follows `MathFin.OnePeriodVector` in
[formal-mathfin](https://github.com/raphaelrrcoelho/formal-mathfin)
(`MathFin/Foundations/FTAPOnePeriodVector.lean`, Apache-2.0, © Raphael Coelho).
-/

namespace PaperReconstructions

open MeasureTheory

/-! ### No-arbitrage -/

/-- **No-arbitrage** (finite-dim market, one period).

No constant portfolio `θ ∈ F` turns zero cost into a sure non-negative discounted gain
`⟪θ, Y⟫` with a chance of profit: any `θ` whose gain is `≥ 0` a.e. already has
`⟪θ, Y⟫ = 0` a.e. Uses the inner-product formulation appropriate for
finite-dimensional normed vector spaces. -/
def NoArbitrageFTAP {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]
    [FiniteDimensional ℝ F] {Ω : Type*} [MeasureTheory.MeasureSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (Y : Ω → F) : Prop :=
  ∀ θ : F, 0 ≤ᵐ[P] (fun ω => inner ℝ θ (Y ω)) →
    (fun ω => inner ℝ θ (Y ω)) =ᵐ[P] 0

/-! ### Equivalent martingale measure -/

/-- **Equivalent martingale measure** (one period, finite-dim market).

`Q ~ P`, the excess return `Y` is `Q`-integrable, and `E_Q[Y] = 0 ∈ F`. Equivalent to
the stochastic-discount-factor formulation via the Radon–Nikodym derivative
`m = dQ/dP`. -/
structure IsEMMFTAP {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]
    [FiniteDimensional ℝ F] {Ω : Type*} [MeasureTheory.MeasureSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (Y : Ω → F) (Q : Measure Ω) : Prop where
  prob : IsProbabilityMeasure Q
  absP : Q ≪ P
  Pabs : P ≪ Q
  int  : Integrable Y Q
  fair : ∫ ω, Y ω ∂Q = 0

variable {Ω : Type*} [MeasureSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
  {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F] [FiniteDimensional ℝ F]
  (Y : Ω → F)

/-! ### Forward direction -/

/-- **Forward direction**: an equivalent martingale measure precludes arbitrage.

Under `Q`, `∫ ⟪θ, Y⟫ ∂Q = ⟪θ, E_Q[Y]⟫ = 0`, so a non-negative `⟪θ, Y⟫` is `0` a.e.;
equivalence transports this back to `P`. This direction needs no measurability
hypothesis beyond the `Integrable Y Q` carried by `IsEMMFTAP`. -/
theorem noArbitrage_of_isEMMFTAP {Q : Measure Ω} (hQ : IsEMMFTAP P Y Q) :
    NoArbitrageFTAP P Y := by
  haveI := hQ.prob
  intro θ hpos
  have hposQ : 0 ≤ᵐ[Q] (fun ω => inner ℝ θ (Y ω)) := hQ.absP.ae_le hpos
  have hint : ∫ ω, inner ℝ θ (Y ω) ∂Q = 0 := by
    rw [integral_inner hQ.int, hQ.fair, inner_zero_right]
  have hzeroQ : (fun ω => inner ℝ θ (Y ω)) =ᵐ[Q] 0 :=
    (integral_eq_zero_iff_of_nonneg_ae hposQ (hQ.int.const_inner θ)).mp hint
  exact hQ.Pabs.ae_eq hzeroQ

/-! ### The gains kernel -/

/-- The **gains kernel** `N = {θ : ⟪θ, Y⟫ = 0 a.e.}`: the portfolio directions whose
discounted gain vanishes almost surely. A linear subspace of `F`; the market is
non-redundant exactly when `N = ⊥`.

The backward direction of the FTAP minimises the softplus potential over `Nᗮ`, which is
what lets it dispense with a non-redundancy hypothesis. -/
def gainsKernel : Submodule ℝ F where
  carrier := {θ | (fun ω => inner ℝ θ (Y ω)) =ᵐ[P] 0}
  zero_mem' := by
    show (fun ω => inner ℝ (0 : F) (Y ω)) =ᵐ[P] 0
    filter_upwards with ω; simp
  add_mem' := by
    intro a b ha hb
    show (fun ω => inner ℝ (a + b) (Y ω)) =ᵐ[P] 0
    have ha' : (fun ω => inner ℝ a (Y ω)) =ᵐ[P] 0 := ha
    have hb' : (fun ω => inner ℝ b (Y ω)) =ᵐ[P] 0 := hb
    filter_upwards [ha', hb'] with ω ea eb
    simp only [Pi.zero_apply] at ea eb ⊢
    rw [inner_add_left, ea, eb, add_zero]
  smul_mem' := by
    intro c b hb
    show (fun ω => inner ℝ (c • b) (Y ω)) =ᵐ[P] 0
    have hb' : (fun ω => inner ℝ b (Y ω)) =ᵐ[P] 0 := hb
    filter_upwards [hb'] with ω eb
    simp only [Pi.zero_apply] at eb ⊢
    rw [real_inner_smul_left, eb, mul_zero]

omit [IsProbabilityMeasure P] [FiniteDimensional ℝ F] in
@[simp] lemma mem_gainsKernel {θ : F} :
    θ ∈ gainsKernel P Y ↔ (fun ω => inner ℝ θ (Y ω)) =ᵐ[P] 0 := Iff.rfl

end PaperReconstructions
