import EnergyStatistics.Defs
import EnergyStatistics.EnergyDistance
import WassersteinGeometry.Geodesics
import Mathlib.MeasureTheory.Integral.Prod
import Mathlib.MeasureTheory.Integral.MeanInequalities
import Mathlib.MeasureTheory.Constructions.BorelSpace.Basic
import Mathlib.MeasureTheory.Function.L1Space.Integrable

set_option linter.style.longLine false
set_option linter.unusedSectionVars false
set_option maxHeartbeats 400000

open scoped ENNReal MeasureTheory
open MeasureTheory

/-!
# Energy–Wasserstein Bridge

Connects the squared energy distance D_E²(μ, ν) (Székely & Rizzo 2023) to the
2-Wasserstein distance W₂(μ, ν) (Panaretos & Zemel 2020) via the coupling
route: for any transference plan π ∈ Π(μ, ν), the centred distance kernel
g x = ∫ dist(x,y) dν − ∫ dist(x,y) dμ satisfies a 2-Lipschitz bound, which
combined with Cauchy–Schwarz yields D_E²(μ,ν) ≤ 2·W₂(μ,ν).

## Main definitions

* `toEnergyProb`: coerce a `WassersteinMeasure` to an `EnergyStatistics.ProbabilityMeasure`
* `distPotential`: the averaged distance function κ ↦ (x ↦ ∫ dist(x,y) dκ(y))

## Landed helpers

* `integrability_slice`: product integrability implies a.e. slice integrability,
  then all-slice integrability via the triangle inequality
* `distPotential_abs_sub_le`: |distPotential κ x − distPotential κ x'| ≤ dist x x'
* `distPotential_lipschitzWith`: distPotential κ is 1-Lipschitz

Note: `coupling_isProbabilityMeasure` and `wassersteinDistanceSq_lt_top` are now
imported from `WassersteinGeometry.Geodesics` (canonical source, no duplication).

## Contents: helpers H1–H5, H7 and the master theorem `energy_le_two_wasserstein`, all proven (2026-07-08),
plus the Paper 5 energy-robustness T6 ball-inclusion corollaries `energyDistanceSq_le_of_wasserstein_le` and
`wasserstein_ball_subset_energy_ball`, proven (2026-07-12)

## References

* Székely, G. J., & Rizzo, M. L. (2023). The Energy of Data and Distance Correlation.
  CRC Press. §3.
* Panaretos, V. M., & Zemel, Y. (2020). An Invitation to Statistics in Wasserstein Space.
  Springer. Ch. 2.
-/

namespace PricingPerspective.Connections

variable {Ω : Type*} [MeasurableSpace Ω] [PseudoMetricSpace Ω] [Inhabited Ω]

/-- View a Wasserstein measure as an EnergyStatistics probability measure.

    Reference: Panaretos & Zemel (2020), Ch. 2 §2.1 (P₂(Ω) ⊂ P(Ω)). -/
def toEnergyProb (μ : WassersteinGeometry.WassersteinMeasure Ω) :
    EnergyStatistics.ProbabilityMeasure Ω :=
  ⟨μ.measure, μ.is_probability⟩

/-- The distance potential: x ↦ ∫ dist(x, y) dκ(y).

    This is the "average distance from x" under the measure κ. The centred
    difference g x = distPotential ν x − distPotential μ x is 2-Lipschitz
    (see `distPotential_abs_sub_le`), which is the key kernel in the
    coupling route to the energy–Wasserstein inequality.

    Reference: Székely & Rizzo (2023), §3. -/
noncomputable def distPotential (κ : MeasureTheory.Measure Ω) (x : Ω) : ℝ :=
  ∫ y, dist x y ∂κ

/-! ### H1: product integrability implies all-slice integrability -/

/-- If dist is integrable over the product κ' ⊗ κ, then for every x, the function
    y ↦ dist(x, y) is integrable under κ.

    Proof sketch:
    (1) `Integrable.prod_right_ae` gives a.e.-x integrability;
    (2) since κ' is a probability measure, the ae filter is NeBot, so we extract a
        witness x₀ via `Filter.Eventually.exists`;
    (3) for arbitrary x, `dist x y ≤ dist x x₀ + dist x₀ y` plus
        `Integrable.mono'` with the integrable bound `const + hx₀`.

    Reference: Mathlib `Integrable.prod_right_ae` (Integral/Prod.lean:297);
    Székely & Rizzo (2023), finite-first-moment condition. -/
lemma integrability_slice
    [OpensMeasurableSpace Ω] [SecondCountableTopology Ω]
    {κ' κ : MeasureTheory.Measure Ω}
    [MeasureTheory.IsProbabilityMeasure κ']
    [MeasureTheory.IsProbabilityMeasure κ]
    (h_int : MeasureTheory.Integrable (fun p : Ω × Ω => dist p.1 p.2) (κ'.prod κ)) :
    ∀ x, MeasureTheory.Integrable (fun y => dist x y) κ := by
  -- Step 1: a.e.-x slice integrability
  have hae : ∀ᵐ x ∂κ', MeasureTheory.Integrable (fun y => dist x y) κ :=
    h_int.prod_right_ae
  -- Step 2: extract a concrete witness x₀
  have ⟨x₀, hx₀⟩ : ∃ x₀, MeasureTheory.Integrable (fun y => dist x₀ y) κ :=
    hae.exists
  -- Step 3: extend to all x via the triangle inequality
  intro x
  apply MeasureTheory.Integrable.mono'
      (hg := (MeasureTheory.integrable_const (dist x x₀)).add hx₀)
  · -- AEStronglyMeasurable via continuity + OpensMeasurableSpace
    exact ((continuous_const.dist continuous_id).measurable).aestronglyMeasurable
  · -- pointwise bound ‖dist x y‖ ≤ dist x x₀ + dist x₀ y
    filter_upwards [] with y
    simp only [Pi.add_apply]
    rw [Real.norm_of_nonneg dist_nonneg]
    exact dist_triangle x x₀ y

/-! ### H2: distPotential is 1-Lipschitz -/

/-- |distPotential κ x − distPotential κ x'| ≤ dist x x'.

    Proof:
    `distPotential κ x − distPotential κ x' = ∫ y, (dist x y − dist x' y) ∂κ`
    (by linearity); then `|∫| ≤ ∫|·|` and `|dist x y − dist x' y| ≤ dist x x'`
    (reverse triangle inequality); finally `∫ dist x x' ∂κ = dist x x'` since κ
    is a probability measure.

    Reference: Székely & Rizzo (2023), §3; Mathlib `abs_dist_sub_le`. -/
lemma distPotential_abs_sub_le
    {κ : MeasureTheory.Measure Ω}
    [MeasureTheory.IsProbabilityMeasure κ]
    (h_int : ∀ z, MeasureTheory.Integrable (fun y => dist z y) κ)
    (x x' : Ω) :
    |distPotential κ x - distPotential κ x'| ≤ dist x x' := by
  unfold distPotential
  rw [← MeasureTheory.integral_sub (h_int x) (h_int x')]
  calc |∫ y, (dist x y - dist x' y) ∂κ|
      ≤ ∫ y, |dist x y - dist x' y| ∂κ :=
        MeasureTheory.abs_integral_le_integral_abs
    _ ≤ ∫ _, dist x x' ∂κ := by
        apply MeasureTheory.integral_mono
        · exact ((h_int x).sub (h_int x')).norm
        · exact MeasureTheory.integrable_const _
        · intro y
          exact abs_dist_sub_le x x' y
    _ = dist x x' := by
        simp [MeasureTheory.integral_const]

/-- distPotential κ is 1-Lipschitz (with respect to the metric on Ω).

    This is an immediate consequence of `distPotential_abs_sub_le` together with
    the definition of `LipschitzWith`.

    Reference: Panaretos & Zemel (2020), Ch. 2; Székely & Rizzo (2023), §3. -/
lemma distPotential_lipschitzWith
    {κ : MeasureTheory.Measure Ω}
    [MeasureTheory.IsProbabilityMeasure κ]
    (h_int : ∀ z, MeasureTheory.Integrable (fun y => dist z y) κ) :
    LipschitzWith 1 (distPotential κ) := by
  apply LipschitzWith.of_dist_le_mul
  intro x x'
  simp only [NNReal.coe_one, one_mul]
  rw [Real.dist_eq]
  exact distPotential_abs_sub_le h_int x x'

/-! ### H3: Fubini rewrite of the energy distance -/

/-- H3: Fubini rewrite of the energy distance.
    2∫∫ d∂ν∂μ − ∫∫ d∂μ∂μ − ∫∫ d∂ν∂ν = (∫ a_ν dμ − ∫ a_μ dμ) − (∫ a_ν dν − ∫ a_μ dν)
    where a_κ(x) = ∫ dist(x,y) dκ(y).

    Proof: unfold `energyDistanceSq` and `distPotential`; the four real numbers
    satisfy A = D via `integral_integral_swap` (Fubini); then `linarith`.

    Reference: Székely & Rizzo (2023), §3; Panaretos & Zemel (2020), Ch. 2. -/
lemma energyDistanceSq_eq_integral_sub
    (μ ν : WassersteinGeometry.WassersteinMeasure Ω)
    (h_μν : EnergyStatistics.FiniteDistMoment (toEnergyProb μ) (toEnergyProb ν))
    (h_μμ : EnergyStatistics.FiniteDistMoment (toEnergyProb μ) (toEnergyProb μ))
    (h_νν : EnergyStatistics.FiniteDistMoment (toEnergyProb ν) (toEnergyProb ν)) :
    EnergyStatistics.energyDistanceSq (toEnergyProb μ) (toEnergyProb ν) =
      (∫ x, (distPotential ν.measure x - distPotential μ.measure x) ∂μ.measure)
        - ∫ x, (distPotential ν.measure x - distPotential μ.measure x) ∂ν.measure := by
  haveI : MeasureTheory.IsProbabilityMeasure μ.measure := ⟨μ.is_probability⟩
  haveI : MeasureTheory.IsProbabilityMeasure ν.measure := ⟨ν.is_probability⟩
  -- integrability of the four average-distance functions
  have h_int_ν_over_μ : MeasureTheory.Integrable (fun x => ∫ y, dist x y ∂ν.measure) μ.measure :=
    h_μν.integral_prod_left
  have h_int_μ_over_μ : MeasureTheory.Integrable (fun x => ∫ y, dist x y ∂μ.measure) μ.measure :=
    h_μμ.integral_prod_left
  have h_int_ν_over_ν : MeasureTheory.Integrable (fun x => ∫ y, dist x y ∂ν.measure) ν.measure :=
    h_νν.integral_prod_left
  have h_int_μ_over_ν : MeasureTheory.Integrable (fun x => ∫ y, dist x y ∂μ.measure) ν.measure := by
    simpa [toEnergyProb, dist_comm] using h_μν.swap.integral_prod_left
  -- key swap identity: ∫∫ dist(x,y) dν dμ = ∫∫ dist(x,y) dμ dν
  have h_swap : (∫ x, ∫ y, dist x y ∂ν.measure ∂μ.measure) =
      (∫ x, ∫ y, dist x y ∂μ.measure ∂ν.measure) := by
    have h := MeasureTheory.integral_integral_swap (f := fun (x y : Ω) => dist x y) h_μν
    simpa [toEnergyProb, dist_comm] using h
  -- unfold definitions and apply the algebraic identity
  unfold EnergyStatistics.energyDistanceSq distPotential
  simp [toEnergyProb]
  rw [MeasureTheory.integral_sub h_int_ν_over_μ h_int_μ_over_μ,
    MeasureTheory.integral_sub h_int_ν_over_ν h_int_μ_over_ν]
  -- RHS expands to A - B - C + D where A = ∫∫ dist dν dμ, B = ∫∫ dist dμ dμ,
  -- C = ∫∫ dist dν dν, D = ∫∫ dist dμ dν; LHS is 2*A - B - C; equality reduces to A = D
  linarith

/-! ### H4: coupling integral rewrite -/

/-- H4: For any coupling π ∈ Π(μ, ν), the difference of centred distance potentials
    integrates to the same value under π as under the marginal difference formula.

    Proof: let g(x) = distPotential ν x - distPotential μ x.
    Integrability of g under each marginal follows from the H3 side-goals.
    Using `hπ.1` and `hπ.2`, rewrite the two μ- and ν-marginal integrals
    via `integral_map` to integrals over π; then `integral_sub` on π.

    Reference: Panaretos & Zemel (2020), Ch. 1 §1.2; Székely & Rizzo (2023), §3. -/
lemma integral_sub_eq_integral_coupling
    (μ ν : WassersteinGeometry.WassersteinMeasure Ω)
    (h_μν : EnergyStatistics.FiniteDistMoment (toEnergyProb μ) (toEnergyProb ν))
    (h_μμ : EnergyStatistics.FiniteDistMoment (toEnergyProb μ) (toEnergyProb μ))
    (h_νν : EnergyStatistics.FiniteDistMoment (toEnergyProb ν) (toEnergyProb ν))
    {π : MeasureTheory.Measure (Ω × Ω)}
    (hπ : π ∈ WassersteinGeometry.couplingSet μ.measure ν.measure) :
    (∫ x, (distPotential ν.measure x - distPotential μ.measure x) ∂μ.measure)
        - ∫ x, (distPotential ν.measure x - distPotential μ.measure x) ∂ν.measure =
      ∫ p, ((distPotential ν.measure p.1 - distPotential μ.measure p.1)
        - (distPotential ν.measure p.2 - distPotential μ.measure p.2)) ∂π := by
  haveI : MeasureTheory.IsProbabilityMeasure μ.measure := ⟨μ.is_probability⟩
  haveI : MeasureTheory.IsProbabilityMeasure ν.measure := ⟨ν.is_probability⟩
  -- integrability of g := distPotential ν - distPotential μ under each marginal
  have h_int_g_μ : MeasureTheory.Integrable
      (distPotential ν.measure - distPotential μ.measure) μ.measure :=
    h_μν.integral_prod_left.sub h_μμ.integral_prod_left
  have h_int_g_ν : MeasureTheory.Integrable
      (distPotential ν.measure - distPotential μ.measure) ν.measure := by
    have h2 : MeasureTheory.Integrable (fun x => ∫ y, dist x y ∂μ.measure) ν.measure := by
      simpa [toEnergyProb, dist_comm] using h_μν.swap.integral_prod_left
    exact h_νν.integral_prod_left.sub h2
  -- integrability of g ∘ fst and g ∘ snd under π via Integrable.comp_aemeasurable
  have h_int_g_fst : MeasureTheory.Integrable
      (fun p : Ω × Ω => (distPotential ν.measure - distPotential μ.measure) p.1) π := by
    have h_int_g_map : MeasureTheory.Integrable
        (distPotential ν.measure - distPotential μ.measure) (π.map Prod.fst) := by
      rw [hπ.1]
      exact h_int_g_μ
    exact h_int_g_map.comp_aemeasurable measurable_fst.aemeasurable
  have h_int_g_snd : MeasureTheory.Integrable
      (fun p : Ω × Ω => (distPotential ν.measure - distPotential μ.measure) p.2) π := by
    have h_int_g_map : MeasureTheory.Integrable
        (distPotential ν.measure - distPotential μ.measure) (π.map Prod.snd) := by
      rw [hπ.2]
      exact h_int_g_ν
    exact h_int_g_map.comp_aemeasurable measurable_snd.aemeasurable
  -- rewrite marginal integrals to integrals over π via integral_map
  have h_map_fst : (∫ x, (distPotential ν.measure - distPotential μ.measure) x ∂μ.measure) =
      (∫ p, (distPotential ν.measure - distPotential μ.measure) p.1 ∂π) := by
    have h_int_g_map : MeasureTheory.Integrable
        (distPotential ν.measure - distPotential μ.measure) (π.map Prod.fst) := by
      rw [hπ.1]
      exact h_int_g_μ
    calc
      (∫ x, (distPotential ν.measure - distPotential μ.measure) x ∂μ.measure) =
          (∫ x, (distPotential ν.measure - distPotential μ.measure) x ∂(π.map Prod.fst)) := by rw [hπ.1]
      _ = (∫ p, (distPotential ν.measure - distPotential μ.measure) (Prod.fst p) ∂π) :=
        MeasureTheory.integral_map measurable_fst.aemeasurable
          (h_int_g_map.aestronglyMeasurable)
      _ = (∫ p, (distPotential ν.measure - distPotential μ.measure) p.1 ∂π) := rfl
  have h_map_snd : (∫ x, (distPotential ν.measure - distPotential μ.measure) x ∂ν.measure) =
      (∫ p, (distPotential ν.measure - distPotential μ.measure) p.2 ∂π) := by
    have h_int_g_map : MeasureTheory.Integrable
        (distPotential ν.measure - distPotential μ.measure) (π.map Prod.snd) := by
      rw [hπ.2]
      exact h_int_g_ν
    calc
      (∫ x, (distPotential ν.measure - distPotential μ.measure) x ∂ν.measure) =
          (∫ x, (distPotential ν.measure - distPotential μ.measure) x ∂(π.map Prod.snd)) := by rw [hπ.2]
      _ = (∫ p, (distPotential ν.measure - distPotential μ.measure) (Prod.snd p) ∂π) :=
        MeasureTheory.integral_map measurable_snd.aemeasurable
          (h_int_g_map.aestronglyMeasurable)
      _ = (∫ p, (distPotential ν.measure - distPotential μ.measure) p.2 ∂π) := rfl
  calc
    (∫ x, (distPotential ν.measure x - distPotential μ.measure x) ∂μ.measure)
        - ∫ x, (distPotential ν.measure x - distPotential μ.measure x) ∂ν.measure
        = (∫ x, (distPotential ν.measure - distPotential μ.measure) x ∂μ.measure) -
          (∫ x, (distPotential ν.measure - distPotential μ.measure) x ∂ν.measure) := by
      simp [Pi.sub_apply]
    _ = (∫ p, (distPotential ν.measure - distPotential μ.measure) p.1 ∂π) -
        (∫ p, (distPotential ν.measure - distPotential μ.measure) p.2 ∂π) := by
      rw [h_map_fst, h_map_snd]
    _ = ∫ p, ((distPotential ν.measure - distPotential μ.measure) p.1 -
        (distPotential ν.measure - distPotential μ.measure) p.2) ∂π := by
      rw [MeasureTheory.integral_sub h_int_g_fst h_int_g_snd]
    _ = ∫ p, ((distPotential ν.measure p.1 - distPotential μ.measure p.1)
        - (distPotential ν.measure p.2 - distPotential μ.measure p.2)) ∂π := by
      simp [Pi.sub_apply]

/-! ### H5: per-coupling squared bound -/

/-- H5: For any coupling π ∈ Π(μ, ν), the squared energy distance is bounded by
    four times the squared Wasserstein cost of π.

    Proof: Let D := energyDistanceSq … and g(x) := distPotential ν x − distPotential μ x.
    Case D ≤ 0: trivial.  Case 0 < D: D = ∫ F ∂π (H3 + H4) where F(p) = g p.1 − g p.2,
    so `ENNReal.ofReal D ≤ ∫⁻ ENNReal.ofReal |F|` (via `abs_integral_le_integral_abs` +
    `ofReal_integral_eq_lintegral_ofReal`).  The pointwise bound `|F p| ≤ 2·dist p.1 p.2`
    (from `distPotential_abs_sub_le` twice) gives `ENNReal.ofReal |F p| ≤ 2·edist p.1 p.2`.
    Hölder with p = q = 2 yields `∫⁻ edist ≤ (∫⁻ edist²)^(1/2)·(π univ)^(1/2) =
    (∫⁻ edist²)^(1/2)`.  Chain inequalities and square.

    Reference: Székely & Rizzo (2023), §3; Panaretos & Zemel (2020), Ch. 2. -/
lemma per_coupling_sq_bound
    [OpensMeasurableSpace Ω] [SecondCountableTopology Ω]
    (μ ν : WassersteinGeometry.WassersteinMeasure Ω)
    (h_μν : EnergyStatistics.FiniteDistMoment (toEnergyProb μ) (toEnergyProb ν))
    (h_μμ : EnergyStatistics.FiniteDistMoment (toEnergyProb μ) (toEnergyProb μ))
    (h_νν : EnergyStatistics.FiniteDistMoment (toEnergyProb ν) (toEnergyProb ν))
    {π : MeasureTheory.Measure (Ω × Ω)}
    (hπ : π ∈ WassersteinGeometry.couplingSet μ.measure ν.measure) :
    (ENNReal.ofReal (EnergyStatistics.energyDistanceSq (toEnergyProb μ) (toEnergyProb ν)))^2
      ≤ 4 * ∫⁻ p, (edist p.1 p.2)^2 ∂π := by
  haveI : MeasureTheory.IsProbabilityMeasure μ.measure := ⟨μ.is_probability⟩
  haveI : MeasureTheory.IsProbabilityMeasure ν.measure := ⟨ν.is_probability⟩
  haveI : MeasureTheory.IsProbabilityMeasure π := WassersteinGeometry.coupling_isProbabilityMeasure μ.is_probability hπ
  set D := EnergyStatistics.energyDistanceSq (toEnergyProb μ) (toEnergyProb ν) with hD
  set g := (distPotential ν.measure - distPotential μ.measure) with hg
  -- integrability of g under each marginal (from the FiniteDistMoment hypotheses)
  have h_int_g_μ : MeasureTheory.Integrable g μ.measure :=
    h_μν.integral_prod_left.sub h_μμ.integral_prod_left
  have h_int_g_ν : MeasureTheory.Integrable g ν.measure := by
    have h2 : MeasureTheory.Integrable (fun x => ∫ y, dist x y ∂μ.measure) ν.measure := by
      simpa [toEnergyProb, dist_comm] using h_μν.swap.integral_prod_left
    exact h_νν.integral_prod_left.sub h2
  -- integrability of F(p) := g p.1 - g p.2 under π
  have h_int_F : MeasureTheory.Integrable
    (fun p : Ω × Ω => g p.1 - g p.2) π := by
    have h_fst : MeasureTheory.Integrable (fun p : Ω × Ω => g p.1) π := by
      have h_int_g_map : MeasureTheory.Integrable g (π.map Prod.fst) := by
        rw [hπ.1]
        exact h_int_g_μ
      exact h_int_g_map.comp_aemeasurable measurable_fst.aemeasurable
    have h_snd : MeasureTheory.Integrable (fun p : Ω × Ω => g p.2) π := by
      have h_int_g_map : MeasureTheory.Integrable g (π.map Prod.snd) := by
        rw [hπ.2]
        exact h_int_g_ν
      exact h_int_g_map.comp_aemeasurable measurable_snd.aemeasurable
    exact h_fst.sub h_snd
  have h_int_absF : MeasureTheory.Integrable (fun p : Ω × Ω => |g p.1 - g p.2|) π :=
    h_int_F.abs
  -- D = ∫ F ∂π (H3 + H4)
  have hD_eq : D = ∫ p, (g p.1 - g p.2) ∂π := by
    rw [hD, energyDistanceSq_eq_integral_sub μ ν h_μν h_μμ h_νν,
      integral_sub_eq_integral_coupling μ ν h_μν h_μμ h_νν hπ]
    simp [hg, Pi.sub_apply]
  -- case split on D ≤ 0
  by_cases hD_le : D ≤ 0
  · -- LHS = 0, RHS ≥ 0
    have h_ofReal : ENNReal.ofReal D = 0 := ENNReal.ofReal_of_nonpos hD_le
    simp [h_ofReal]
  · -- 0 < D
    have hD_pos : 0 < D := lt_of_not_ge hD_le
    -- ENNReal.ofReal D ≤ ENNReal.ofReal (∫ |F|) = ∫⁻ ENNReal.ofReal |F|
    have h_abs_int : D ≤ ∫ p, |g p.1 - g p.2| ∂π := by
      calc
        D = |D| := by rw [abs_of_pos hD_pos]
        _ = |∫ p, (g p.1 - g p.2) ∂π| := by rw [hD_eq]
        _ ≤ ∫ p, |g p.1 - g p.2| ∂π := MeasureTheory.abs_integral_le_integral_abs
    have h_ofReal_le : ENNReal.ofReal D ≤ ENNReal.ofReal (∫ p, |g p.1 - g p.2| ∂π) :=
      ENNReal.ofReal_le_ofReal h_abs_int
    have h_lintegral_eq : ENNReal.ofReal (∫ p, |g p.1 - g p.2| ∂π) =
        ∫⁻ p, ENNReal.ofReal (|g p.1 - g p.2|) ∂π :=
      MeasureTheory.ofReal_integral_eq_lintegral_ofReal h_int_absF
        (MeasureTheory.ae_of_all π (fun _ => abs_nonneg _))
    -- pointwise bound: |g p.1 - g p.2| ≤ 2 * dist p.1 p.2
    have h_pointwise_bound : ∀ p : Ω × Ω, |g p.1 - g p.2| ≤ 2 * dist p.1 p.2 := by
      intro p
      have h_ν : |distPotential ν.measure p.1 - distPotential ν.measure p.2| ≤ dist p.1 p.2 := by
        have h_int_ν : ∀ z, MeasureTheory.Integrable (fun y => dist z y) ν.measure :=
          integrability_slice (κ' := ν.measure) (κ := ν.measure) h_νν
        exact distPotential_abs_sub_le h_int_ν p.1 p.2
      have h_μ : |distPotential μ.measure p.1 - distPotential μ.measure p.2| ≤ dist p.1 p.2 := by
        have h_int_μ : ∀ z, MeasureTheory.Integrable (fun y => dist z y) μ.measure :=
          integrability_slice (κ' := μ.measure) (κ := μ.measure) h_μμ
        exact distPotential_abs_sub_le h_int_μ p.1 p.2
      have h_triangle : |g p.1 - g p.2| ≤
          |distPotential ν.measure p.1 - distPotential ν.measure p.2| +
          |distPotential μ.measure p.1 - distPotential μ.measure p.2| := by
        calc
          |g p.1 - g p.2| = |(distPotential ν.measure p.1 - distPotential μ.measure p.1) -
              (distPotential ν.measure p.2 - distPotential μ.measure p.2)| := rfl
          _ = |(distPotential ν.measure p.1 - distPotential ν.measure p.2) -
              (distPotential μ.measure p.1 - distPotential μ.measure p.2)| := by ring_nf
          _ ≤ |distPotential ν.measure p.1 - distPotential ν.measure p.2| +
              |distPotential μ.measure p.1 - distPotential μ.measure p.2| :=
            abs_sub _ _
      calc
        |g p.1 - g p.2| ≤ |distPotential ν.measure p.1 - distPotential ν.measure p.2| +
            |distPotential μ.measure p.1 - distPotential μ.measure p.2| := h_triangle
        _ ≤ dist p.1 p.2 + dist p.1 p.2 := add_le_add h_ν h_μ
        _ = 2 * dist p.1 p.2 := by ring
    -- convert to ENNReal: ENNReal.ofReal |g p.1 - g p.2| ≤ 2 * edist p.1 p.2
    have h_pointwise : ∀ p : Ω × Ω,
        ENNReal.ofReal (|g p.1 - g p.2|) ≤ (2 : ℝ≥0∞) * edist p.1 p.2 := by
      intro p
      calc
        ENNReal.ofReal (|g p.1 - g p.2|) ≤ ENNReal.ofReal (2 * dist p.1 p.2) :=
          ENNReal.ofReal_le_ofReal (h_pointwise_bound p)
        _ = ENNReal.ofReal (2 : ℝ) * ENNReal.ofReal (dist p.1 p.2) := by
          rw [ENNReal.ofReal_mul (by norm_num : (0 : ℝ) ≤ 2)]
        _ = (2 : ℝ≥0∞) * edist p.1 p.2 := by
          rw [edist_dist]
          norm_num
    -- integrate the pointwise bound
    have h_lintegral_le : ∫⁻ p, ENNReal.ofReal (|g p.1 - g p.2|) ∂π ≤
        ∫⁻ p, (2 : ℝ≥0∞) * edist p.1 p.2 ∂π :=
      lintegral_mono h_pointwise
    -- pull out constant 2
    have h_two_ne_top : (2 : ℝ≥0∞) ≠ ⊤ := by norm_num
    have h_lintegral_const : (∫⁻ p, (2 : ℝ≥0∞) * edist p.1 p.2 ∂π) =
        (2 : ℝ≥0∞) * (∫⁻ p, edist p.1 p.2 ∂π) :=
      MeasureTheory.lintegral_const_mul' (2 : ℝ≥0∞) (fun (p : Ω × Ω) => edist p.1 p.2) h_two_ne_top
    rw [h_lintegral_const] at h_lintegral_le
    -- Hölder: ∫⁻ edist * 1 ≤ (∫⁻ edist²)^(1/2) * (∫⁻ 1²)^(1/2)
    have h_holder : (∫⁻ p, edist p.1 p.2 ∂π) ≤
        (∫⁻ p, (edist p.1 p.2) ^ (2 : ℝ) ∂π) ^ (1 / (2 : ℝ)) *
        (∫⁻ p, (1 : ℝ≥0∞) ^ (2 : ℝ) ∂π) ^ (1 / (2 : ℝ)) := by
      have h_meas : AEMeasurable (fun p : Ω × Ω => edist p.1 p.2) π :=
        measurable_edist.aemeasurable
      have h_meas_one : AEMeasurable (fun _ : Ω × Ω => (1 : ℝ≥0∞)) π :=
        aemeasurable_const
      have hpq : Real.HolderConjugate (2 : ℝ) (2 : ℝ) := Real.HolderConjugate.two_two
      have h := ENNReal.lintegral_mul_le_Lp_mul_Lq π hpq h_meas h_meas_one
      simpa [Pi.mul_apply, one_mul] using h
    -- simplify the second factor: (∫⁻ 1^2)^(1/2) = 1
    have h_second_factor : (∫⁻ p, (1 : ℝ≥0∞) ^ (2 : ℝ) ∂π) ^ (1 / (2 : ℝ)) = (1 : ℝ≥0∞) := by
      have h_int_one : (∫⁻ p, (1 : ℝ≥0∞) ∂π) = π Set.univ :=
        MeasureTheory.lintegral_one
      have h_univ : π Set.univ = 1 := measure_univ
      calc
        (∫⁻ p, (1 : ℝ≥0∞) ^ (2 : ℝ) ∂π) ^ (1 / (2 : ℝ)) =
            (∫⁻ p, (1 : ℝ≥0∞) ∂π) ^ (1 / (2 : ℝ)) := by simp
        _ = (π Set.univ) ^ (1 / (2 : ℝ)) := by rw [h_int_one]
        _ = (1 : ℝ≥0∞) ^ (1 / (2 : ℝ)) := by rw [h_univ]
        _ = (1 : ℝ≥0∞) := by simp
    -- Hölder gives: ∫⁻ edist ≤ (∫⁻ edist^(2:ℝ))^(1/2)
    have h_holder_simp : (∫⁻ p, edist p.1 p.2 ∂π) ≤
        (∫⁻ p, (edist p.1 p.2) ^ (2 : ℝ) ∂π) ^ (1 / (2 : ℝ)) := by
      calc
        (∫⁻ p, edist p.1 p.2 ∂π) ≤
            (∫⁻ p, (edist p.1 p.2) ^ (2 : ℝ) ∂π) ^ (1 / (2 : ℝ)) *
            (∫⁻ p, (1 : ℝ≥0∞) ^ (2 : ℝ) ∂π) ^ (1 / (2 : ℝ)) := h_holder
        _ = (∫⁻ p, (edist p.1 p.2) ^ (2 : ℝ) ∂π) ^ (1 / (2 : ℝ)) * (1 : ℝ≥0∞) := by rw [h_second_factor]
        _ = (∫⁻ p, (edist p.1 p.2) ^ (2 : ℝ) ∂π) ^ (1 / (2 : ℝ)) := by simp
    -- square both sides: (∫⁻ edist)^2 ≤ ((∫⁻ edist^(2:ℝ))^(1/2))^2 = ∫⁻ edist^(2:ℝ)
    have h_edist_sq_bound : (∫⁻ p, edist p.1 p.2 ∂π)^2 ≤
        ∫⁻ p, (edist p.1 p.2) ^ (2 : ℝ) ∂π := by
      have h := pow_le_pow_left' h_holder_simp 2
      -- h : (∫⁻ p, edist p.1 p.2 ∂π)^2 ≤ ((∫⁻ p, (edist p.1 p.2) ^ (2 : ℝ) ∂π) ^ (1 / (2 : ℝ))) ^ 2
      -- Now simplify the RHS: (x^(1/2))^2 = x^(1/2 * 2) = x^1 = x
      -- using ENNReal.rpow_mul
      have h_simp : ((∫⁻ p, (edist p.1 p.2) ^ (2 : ℝ) ∂π) ^ (1 / (2 : ℝ))) ^ 2 =
          ∫⁻ p, (edist p.1 p.2) ^ (2 : ℝ) ∂π := by
        set x := ∫⁻ p, (edist p.1 p.2) ^ (2 : ℝ) ∂π
        calc
          (x ^ (1 / (2 : ℝ))) ^ 2 = (x ^ (1 / (2 : ℝ))) ^ (2 : ℝ) := by norm_num
          _ = x ^ ((1 / (2 : ℝ)) * (2 : ℝ)) := by rw [← ENNReal.rpow_mul]
          _ = x ^ (1 : ℝ) := by ring_nf
          _ = x := by simp
      rw [h_simp] at h
      exact h
    -- convert edist^(2:ℝ) to edist^2 (Nat exponent) for the goal
    have h_edist_sq_int : (∫⁻ p, (edist p.1 p.2) ^ (2 : ℝ) ∂π) =
        (∫⁻ p, (edist p.1 p.2)^2 ∂π) := by
      refine MeasureTheory.lintegral_congr fun p => ?_
      simp
    -- chain everything together
    calc
      (ENNReal.ofReal D)^2 ≤ (ENNReal.ofReal (∫ p, |g p.1 - g p.2| ∂π))^2 :=
        pow_le_pow_left' h_ofReal_le 2
      _ = (∫⁻ p, ENNReal.ofReal (|g p.1 - g p.2|) ∂π)^2 := by rw [h_lintegral_eq]
      _ ≤ (2 * ∫⁻ p, edist p.1 p.2 ∂π)^2 := pow_le_pow_left' h_lintegral_le 2
      _ = (∫⁻ p, (2 : ℝ≥0∞) * edist p.1 p.2 ∂π)^2 := by rw [h_lintegral_const]
      _ = ((2 : ℝ≥0∞) * (∫⁻ p, edist p.1 p.2 ∂π))^2 := by rw [h_lintegral_const]
      _ = (4 : ℝ≥0∞) * ((∫⁻ p, edist p.1 p.2 ∂π)^2) := by ring
      _ ≤ (4 : ℝ≥0∞) * (∫⁻ p, (edist p.1 p.2) ^ (2 : ℝ) ∂π) := by
        gcongr
      _ = (4 : ℝ≥0∞) * (∫⁻ p, (edist p.1 p.2)^2 ∂π) := by rw [h_edist_sq_int]
      _ = 4 * ∫⁻ p, (edist p.1 p.2)^2 ∂π := rfl

/-! ### H7: master theorem — energy distance is bounded by twice the Wasserstein distance -/

/-- H7: `energy_le_two_wasserstein`: D_E²(μ, ν) ≤ 2·W₂(μ, ν).

    Proof: Set D := energyDistanceSq ….  Case D ≤ 0: trivial since Wasserstein distance ≥ 0.
    Case 0 < D: from H5, every coupling π satisfies (ENNReal.ofReal D)² ≤ 4·cost(π).
    Taking sInf gives (ENNReal.ofReal D)² ≤ 4·W₂².  Apply `ENNReal.toReal` (monotone,
    valid because both sides are finite by `WassersteinGeometry.wassersteinDistanceSq_lt_top`) to get D² ≤ 4·t where t = W₂².toReal.
    Then D ≤ 2·√t = 2·W₂ (definitionally).

    Reference: Székely & Rizzo (2023), §3; Panaretos & Zemel (2020), Ch. 2. -/
theorem energy_le_two_wasserstein
    [OpensMeasurableSpace Ω] [SecondCountableTopology Ω]
    (μ ν : WassersteinGeometry.WassersteinMeasure Ω)
    (h_μν : EnergyStatistics.FiniteDistMoment (toEnergyProb μ) (toEnergyProb ν))
    (h_μμ : EnergyStatistics.FiniteDistMoment (toEnergyProb μ) (toEnergyProb μ))
    (h_νν : EnergyStatistics.FiniteDistMoment (toEnergyProb ν) (toEnergyProb ν)) :
    EnergyStatistics.energyDistanceSq (toEnergyProb μ) (toEnergyProb ν) ≤
      2 * WassersteinGeometry.WassersteinDistance μ ν := by
  haveI : MeasureTheory.IsProbabilityMeasure μ.measure := ⟨μ.is_probability⟩
  haveI : MeasureTheory.IsProbabilityMeasure ν.measure := ⟨ν.is_probability⟩
  set D := EnergyStatistics.energyDistanceSq (toEnergyProb μ) (toEnergyProb ν) with hD
  by_cases hD_le : D ≤ 0
  · -- D ≤ 0 ≤ 2 * WassersteinDistance (since WassersteinDistance ≥ 0)
    have h_wass_nonneg : 0 ≤ WassersteinGeometry.WassersteinDistance μ ν := by
      unfold WassersteinGeometry.WassersteinDistance
      refine Real.rpow_nonneg ENNReal.toReal_nonneg _
    nlinarith
  · -- 0 < D
    have hD_pos : 0 < D := lt_of_not_ge hD_le
    -- From H5: for all π ∈ couplingSet, (ENNReal.ofReal D)^2 ≤ 4 * cost(π)
    -- Take sInf over π to get (ENNReal.ofReal D)^2 ≤ 4 * WassersteinDistanceSq
    have h_sq_le : (ENNReal.ofReal D)^2 ≤
        (4 : ℝ≥0∞) * WassersteinGeometry.WassersteinDistanceSq μ ν := by
      -- For each coupling π, we have (ENNReal.ofReal D)^2 ≤ 4 * cost(π)
      -- So (ENNReal.ofReal D)^2 / 4 ≤ cost(π) for each π
      -- Hence (ENNReal.ofReal D)^2 / 4 ≤ sInf {cost(π) | π ∈ couplingSet}
      -- = WassersteinDistanceSq
      -- Then multiply by 4
      have h_div : (ENNReal.ofReal D)^2 / (4 : ℝ≥0∞) ≤ WassersteinGeometry.WassersteinDistanceSq μ ν := by
        refine le_sInf ?_
        rintro c ⟨π, hπ, rfl⟩
        have h_bound := per_coupling_sq_bound μ ν h_μν h_μμ h_νν hπ
        -- h_bound : (ENNReal.ofReal D)^2 ≤ 4 * cost(π); rewrite as a/4 ≤ c via div_le_iff
        have h4_ne_zero : (4 : ℝ≥0∞) ≠ 0 := by norm_num
        have h4_ne_top : (4 : ℝ≥0∞) ≠ ⊤ := by norm_num
        rw [ENNReal.div_le_iff h4_ne_zero h4_ne_top]
        rw [mul_comm]
        exact h_bound
      have h4_ne_zero : (4 : ℝ≥0∞) ≠ 0 := by norm_num
      have h4_ne_top : (4 : ℝ≥0∞) ≠ ⊤ := by norm_num
      have h := (ENNReal.div_le_iff h4_ne_zero h4_ne_top).mp h_div
      rw [mul_comm] at h
      exact h
    -- Apply ENNReal.toReal to both sides (both sides are finite)
    have h_fin_sq : (ENNReal.ofReal D)^2 ≠ ⊤ := by
      have h_fin : ENNReal.ofReal D ≠ ⊤ := ENNReal.ofReal_ne_top
      -- ENNReal.pow_ne_top : x ≠ ⊤ → x^n ≠ ⊤
      exact ENNReal.pow_ne_top h_fin
    have h_fin_rhs : (4 : ℝ≥0∞) * WassersteinGeometry.WassersteinDistanceSq μ ν ≠ ⊤ := by
      have h_wdsq_lt_top : WassersteinGeometry.WassersteinDistanceSq μ ν < ⊤ :=
        WassersteinGeometry.wassersteinDistanceSq_lt_top μ ν
      have h4_ne_top : (4 : ℝ≥0∞) ≠ ⊤ := by norm_num
      exact ENNReal.mul_ne_top h4_ne_top (ne_top_of_lt h_wdsq_lt_top)
    have h_toReal_le : ENNReal.toReal ((ENNReal.ofReal D)^2) ≤
        ENNReal.toReal ((4 : ℝ≥0∞) * WassersteinGeometry.WassersteinDistanceSq μ ν) :=
      (ENNReal.toReal_le_toReal h_fin_sq h_fin_rhs).mpr h_sq_le
    -- Simplify LHS: ENNReal.toReal ((ENNReal.ofReal D)^2) = D^2
    have h_lhs : ENNReal.toReal ((ENNReal.ofReal D)^2) = D ^ 2 := by
      calc
        ENNReal.toReal ((ENNReal.ofReal D)^2) = (ENNReal.toReal (ENNReal.ofReal D)) ^ 2 := by
          rw [ENNReal.toReal_pow]
        _ = D ^ 2 := by
          rw [ENNReal.toReal_ofReal (le_of_lt hD_pos)]
    -- Simplify RHS: ENNReal.toReal (4 * WassersteinDistanceSq) = 4 * (WassersteinDistanceSq).toReal
    have h_wdsq_ne_top : WassersteinGeometry.WassersteinDistanceSq μ ν ≠ ⊤ :=
      ne_top_of_lt (WassersteinGeometry.wassersteinDistanceSq_lt_top μ ν)
    have h_rhs : ENNReal.toReal ((4 : ℝ≥0∞) * WassersteinGeometry.WassersteinDistanceSq μ ν) =
        (4 : ℝ) * ENNReal.toReal (WassersteinGeometry.WassersteinDistanceSq μ ν) := by
      rw [ENNReal.toReal_mul, ENNReal.toReal_ofNat]
    rw [h_lhs, h_rhs] at h_toReal_le
    -- Now we have: D^2 ≤ 4 * t where t = (WassersteinDistanceSq).toReal
    -- D ≤ 2 * sqrt(t)
    -- And WassersteinDistance = t^(1/2) = sqrt(t)
    set t := ENNReal.toReal (WassersteinGeometry.WassersteinDistanceSq μ ν) with ht
    have ht_nonneg : 0 ≤ t := ENNReal.toReal_nonneg
    have hD_sq_le : D ^ 2 ≤ (4 : ℝ) * t := h_toReal_le
    -- D ≤ 2 * sqrt(t)
    have hD_le_2_sqrt_t : D ≤ (2 : ℝ) * Real.sqrt t := by
      have h_sqrt_D : D = Real.sqrt (D ^ 2) := by rw [Real.sqrt_sq (le_of_lt hD_pos)]
      have h_sqrt_le : Real.sqrt (D ^ 2) ≤ Real.sqrt ((4 : ℝ) * t) :=
        Real.sqrt_le_sqrt hD_sq_le
      have h_sqrt_mul : Real.sqrt ((4 : ℝ) * t) = Real.sqrt (4 : ℝ) * Real.sqrt t := by
        rw [Real.sqrt_mul (show 0 ≤ (4 : ℝ) from by norm_num) t]
      have h_sqrt_four : Real.sqrt (4 : ℝ) = (2 : ℝ) := by
        calc
          Real.sqrt (4 : ℝ) = Real.sqrt ((2 : ℝ)^2) := by norm_num
          _ = |(2 : ℝ)| := Real.sqrt_sq_eq_abs _
          _ = (2 : ℝ) := abs_of_pos (by norm_num : 0 < (2 : ℝ))
      calc
        D = Real.sqrt (D ^ 2) := h_sqrt_D
        _ ≤ Real.sqrt ((4 : ℝ) * t) := h_sqrt_le
        _ = Real.sqrt (4 : ℝ) * Real.sqrt t := h_sqrt_mul
        _ = (2 : ℝ) * Real.sqrt t := by rw [h_sqrt_four]
    -- WassersteinDistance = sqrt(t)
    have h_wass_eq : WassersteinGeometry.WassersteinDistance μ ν = Real.sqrt t := by
      unfold WassersteinGeometry.WassersteinDistance
      -- WassersteinDistance = ENNReal.toReal (WassersteinDistanceSq) ^ (1/2 : ℝ)
      -- Real.sqrt t = t ^ (1/2 : ℝ)
      -- And t = ENNReal.toReal (WassersteinDistanceSq)
      -- So both sides are t^(1/2)
      -- Use Real.sqrt_eq_rpow
      rw [ht, Real.sqrt_eq_rpow]
    rw [h_wass_eq]
    simpa using hD_le_2_sqrt_t

/-! ### Ball inclusion (Paper 5 energy-robustness T6) -/

/--
**Corollary (Energy Bound inside a Wasserstein Ball).** (Paper 5 energy-robustness T6, scalar form.)

If `ν` lies in the 2-Wasserstein ball of radius `δ` around `μ`, then the squared energy
distance is at most `2δ` — immediate from `energy_le_two_wasserstein`. One-sided by design:
no reverse bound `W ≲ 𝓔` holds in general (`kantorovich_duality` is out-of-window), so
energy-ball robustness certificates are *conservative relative to* Wasserstein-DRO ones,
never tighter by theorem.
-/
theorem energyDistanceSq_le_of_wasserstein_le
    [OpensMeasurableSpace Ω] [SecondCountableTopology Ω]
    (μ ν : WassersteinGeometry.WassersteinMeasure Ω) {δ : ℝ}
    (h_μν : EnergyStatistics.FiniteDistMoment (toEnergyProb μ) (toEnergyProb ν))
    (h_μμ : EnergyStatistics.FiniteDistMoment (toEnergyProb μ) (toEnergyProb μ))
    (h_νν : EnergyStatistics.FiniteDistMoment (toEnergyProb ν) (toEnergyProb ν))
    (h_ball : WassersteinGeometry.WassersteinDistance μ ν ≤ δ) :
    EnergyStatistics.energyDistanceSq (toEnergyProb μ) (toEnergyProb ν) ≤ 2 * δ := by
  have h := energy_le_two_wasserstein μ ν h_μν h_μμ h_νν
  linarith

/--
**Corollary (Wasserstein Ball ⊆ Energy Ball).** (Paper 5 energy-robustness T6, set form.)

Among measures with the requisite finite first distance moments, the 2-Wasserstein ball of
radius `δ` around `μ` is contained in the squared-energy ball of radius `2δ`. Consequently,
for any objective, the energy-ball worst case at calibrated radius `2δ` upper-bounds the
`W₂`-ball worst case at radius `δ`: energy-DRO certificates *cover* Wasserstein-DRO ones
(Blanchet–Chen–Zhou 2022) on this hypothesis class.
-/
theorem wasserstein_ball_subset_energy_ball
    [OpensMeasurableSpace Ω] [SecondCountableTopology Ω]
    (μ : WassersteinGeometry.WassersteinMeasure Ω) (δ : ℝ) :
    {ν : WassersteinGeometry.WassersteinMeasure Ω |
        EnergyStatistics.FiniteDistMoment (toEnergyProb μ) (toEnergyProb ν) ∧
          EnergyStatistics.FiniteDistMoment (toEnergyProb μ) (toEnergyProb μ) ∧
            EnergyStatistics.FiniteDistMoment (toEnergyProb ν) (toEnergyProb ν) ∧
              WassersteinGeometry.WassersteinDistance μ ν ≤ δ} ⊆
      {ν : WassersteinGeometry.WassersteinMeasure Ω |
        EnergyStatistics.energyDistanceSq (toEnergyProb μ) (toEnergyProb ν) ≤ 2 * δ} := by
  rintro ν ⟨h_μν, h_μμ, h_νν, h_ball⟩
  exact energyDistanceSq_le_of_wasserstein_le μ ν h_μν h_μμ h_νν h_ball
