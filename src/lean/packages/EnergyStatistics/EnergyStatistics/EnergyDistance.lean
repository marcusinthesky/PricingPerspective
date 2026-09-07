import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.Probability.Independence.Basic
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Positivity
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.MeasureTheory.Integral.Bochner.L1
import EnergyStatistics.Defs

set_option linter.style.longLine false

/-!
# Energy Distance

This module formalizes the energy distance between probability distributions,
following Székely & Rizzo (2023), "The Energy of Data and Distance Correlation".

## Main definitions

* `energyDistanceSq`: The squared energy distance D_E²(μ, ν)

## Main theorems

* `energy_distance_nonneg`: Energy distance is nonnegative
* `energyDistanceSq_cnd`: population conditional negative definiteness of the energy matrix
* `energyDistanceSq_simplexMixture`: mixture-energy expansion `2c'w − w'Qw − b` (Paper 2 sign)
* `energy_distance_zero_iff_of_strong`: zero-iff characterization under `StrongDistNegativeType`
* `energy_distance_zero_iff_refuted`: the naive zero-iff over a bare `PseudoMetricSpace` is false

## References

* Székely, G. J., & Rizzo, M. L. (2023). The Energy of Data and Distance Correlation.
  CRC Press. Chapters 3, 10.
-/

namespace EnergyStatistics

/-- The squared energy distance between two probability measures.

    D_E²(μ, ν) = 2·E[‖X - Y‖] - E[‖X - X'‖] - E[‖Y - Y'‖]

    where X, X' ~ μ iid and Y, Y' ~ ν iid.

    This is the standard population energy definition from
    Székely & Rizzo (2023), Definition 3.3.

    Reference: Székely & Rizzo (2023), Definition 3.3, eq. (3.3). -/
noncomputable def energyDistanceSq
    {α : Type*} [MeasurableSpace α] [PseudoMetricSpace α]
    (μ ν : ProbabilityMeasure α) : ℝ :=
  2 * ∫ x, ∫ y, dist x y ∂ν.measure ∂μ.measure
    - ∫ x, ∫ y, dist x y ∂μ.measure ∂μ.measure
    - ∫ x, ∫ y, dist x y ∂ν.measure ∂ν.measure

/-- **Strong negative type.** A space has strong negative type when it has negative type and the
    population energy functional separates all probability measures for which the three distance
    moments are finite.

    This is deliberately a measure-level property relative to the supplied `MeasurableSpace α`:
    both `ProbabilityMeasure α` and `FiniteDistMoment` use that measurable structure. Intended
    metric-space consumers should provide a Borel-compatible structure (normally `BorelSpace α`);
    the definition does not claim invariance under an arbitrary incompatible measurable structure.
    The finite-support strictness predicate
    `StrictDistNegativeType` is weaker on infinite spaces and cannot discharge population
    separation: Lyons (2013), Remark 3.3 gives a strict-negative-type space that is not of strong
    negative type. Proving that a particular ground space satisfies this definition is a separate
    theorem; no Euclidean or Hilbert instance is asserted here.

    Reference: Lyons (2013), Definition 3.1, Proposition 3.1 and Remark 3.3; Sejdinovic et al.
    (2013), Definition 28 and Proposition 29. -/
def StrongDistNegativeType
    (α : Type*) [MeasurableSpace α] [PseudoMetricSpace α] : Prop :=
  DistNegativeType α ∧
    ∀ (μ ν : ProbabilityMeasure α),
      FiniteDistMoment μ ν → FiniteDistMoment μ μ → FiniteDistMoment ν ν →
        energyDistanceSq μ ν = 0 → μ = ν

section EnergyNonneg

open MeasureTheory ProbabilityTheory Filter

/-- Pair marginal of an iid product: two distinct coordinates of `Measure.pi (fun _ => μ)`
are jointly distributed as `μ ⊗ μ`. -/
private lemma pi_map_pair {α : Type*} [MeasurableSpace α] {μ : Measure α}
    [IsProbabilityMeasure μ] {n : ℕ} {i j : Fin n} (hij : i ≠ j) :
    (Measure.pi fun _ : Fin n => μ).map (fun ω => (ω i, ω j)) = μ.prod μ := by
  have hindep : IndepFun (fun ω : Fin n → α => ω i) (fun ω : Fin n → α => ω j)
      (Measure.pi fun _ : Fin n => μ) :=
    (iIndepFun_pi (X := fun _ : Fin n => id) fun _ => aemeasurable_id).indepFun hij
  rw [hindep.map_prod_eq_prod_map_map (measurable_pi_apply i).aemeasurable
    (measurable_pi_apply j).aemeasurable,
    (measurePreserving_eval _ i).map_eq, (measurePreserving_eval _ j).map_eq]

/-- Transport of the distance integrand through a distribution identity `P.map φ = Q`:
integrability and the value of the integral both descend from `Q` to `P`. -/
private lemma integrable_integral_dist_comp {α Ω : Type*} [MeasurableSpace α]
    [PseudoMetricSpace α] [MeasurableSpace Ω] {P : Measure Ω} {Q : Measure (α × α)}
    {φ : Ω → α × α} (hφ : AEMeasurable φ P) (hmap : P.map φ = Q)
    (hQ : Integrable (fun z : α × α => dist z.1 z.2) Q) :
    Integrable (fun ω => dist (φ ω).1 (φ ω).2) P ∧
      (∫ ω, dist (φ ω).1 (φ ω).2 ∂P) = ∫ z : α × α, dist z.1 z.2 ∂Q := by
  subst hmap
  exact ⟨(integrable_map_measure hQ.aestronglyMeasurable hφ).mp hQ,
    (integral_map hφ hQ.aestronglyMeasurable).symm⟩

/-- Pointwise block form of conditional negative definiteness: for two `n`-tuples `a b` and
weights `+1` on the `a`-block, `−1` on the `b`-block,

`∑ᵢⱼ d(aᵢ,aⱼ) + ∑ᵢⱼ d(bᵢ,bⱼ) ≤ 2·∑ᵢⱼ d(aᵢ,bⱼ)`. -/
private lemma cnd_block_sums {α : Type*} [PseudoMetricSpace α] (h_neg : DistNegativeType α)
    {n : ℕ} (a b : Fin n → α) :
    (∑ i, ∑ j, dist (a i) (a j)) + (∑ i, ∑ j, dist (b i) (b j))
      ≤ 2 * ∑ i, ∑ j, dist (a i) (b j) := by
  have hsum : (∑ i, Fin.addCases (motive := fun _ => ℝ)
      (fun _ : Fin n => 1) (fun _ : Fin n => -1) i) = 0 := by
    rw [Fin.sum_univ_add]
    simp only [Fin.addCases_left, Fin.addCases_right]
    simp
  have h0 := h_neg (n + n) (Fin.addCases (motive := fun _ => α) a b)
    (Fin.addCases (motive := fun _ => ℝ) (fun _ => 1) fun _ => -1) hsum
  rw [Fin.sum_univ_add] at h0
  simp only [Fin.sum_univ_add, Fin.addCases_left, Fin.addCases_right, one_mul,
    neg_mul, Finset.sum_neg_distrib] at h0
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib] at h0
  have hswap : (∑ i, ∑ j, dist (b i) (a j)) = ∑ i, ∑ j, dist (a i) (b j) := by
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => dist_comm _ _
  rw [hswap] at h0
  simp only [Finset.sum_neg_distrib] at h0
  linarith

/-- The `n`-sample bound behind `energy_distance_nonneg`: integrating the discrete CND
inequality over `μ^⊗n ⊗ ν^⊗n` and counting off-diagonal pairs gives

`n(n−1)·(I(μ,μ) + I(ν,ν)) ≤ 2n²·I(μ,ν)`

where `I(κ,λ) = ∫ dist d(κ⊗λ)`.  Letting `n → ∞` yields the energy inequality. -/
private lemma energy_sample_bound {α : Type*} [MeasurableSpace α] [PseudoMetricSpace α]
    (h_neg : DistNegativeType α) (μ ν : ProbabilityMeasure α)
    (h_μν : FiniteDistMoment μ ν) (h_μμ : FiniteDistMoment μ μ) (h_νν : FiniteDistMoment ν ν)
    (n : ℕ) :
    (n : ℝ) * ((n : ℝ) - 1) *
        ((∫ z : α × α, dist z.1 z.2 ∂(μ.measure.prod μ.measure))
          + ∫ z : α × α, dist z.1 z.2 ∂(ν.measure.prod ν.measure))
      ≤ 2 * (n : ℝ) ^ 2 * ∫ z : α × α, dist z.1 z.2 ∂(μ.measure.prod ν.measure) := by
  classical
  set Iμμ := ∫ z : α × α, dist z.1 z.2 ∂(μ.measure.prod μ.measure) with hIμμ
  set Iνν := ∫ z : α × α, dist z.1 z.2 ∂(ν.measure.prod ν.measure) with hIνν
  set Iμν := ∫ z : α × α, dist z.1 z.2 ∂(μ.measure.prod ν.measure) with hIμν
  set P : Measure ((Fin n → α) × (Fin n → α)) :=
    (Measure.pi fun _ => μ.measure).prod (Measure.pi fun _ => ν.measure) with hPdef
  haveI : IsProbabilityMeasure P := by rw [hPdef]; infer_instance
  -- distribution identities for the coordinate pairs
  have hmapAA : ∀ {i j : Fin n}, i ≠ j →
      P.map (fun ω : (Fin n → α) × (Fin n → α) => (ω.1 i, ω.1 j))
        = μ.measure.prod μ.measure := by
    intro i j hij
    have h1 : (fun ω : (Fin n → α) × (Fin n → α) => (ω.1 i, ω.1 j))
        = (fun a : Fin n → α => (a i, a j)) ∘ Prod.fst := rfl
    rw [h1, ← Measure.map_map (by fun_prop) measurable_fst, hPdef, Measure.map_fst_prod]
    simp only [measure_univ, one_smul]
    exact pi_map_pair hij
  have hmapBB : ∀ {i j : Fin n}, i ≠ j →
      P.map (fun ω : (Fin n → α) × (Fin n → α) => (ω.2 i, ω.2 j))
        = ν.measure.prod ν.measure := by
    intro i j hij
    have h1 : (fun ω : (Fin n → α) × (Fin n → α) => (ω.2 i, ω.2 j))
        = (fun a : Fin n → α => (a i, a j)) ∘ Prod.snd := rfl
    rw [h1, ← Measure.map_map (by fun_prop) measurable_snd, hPdef, Measure.map_snd_prod]
    simp only [measure_univ, one_smul]
    exact pi_map_pair hij
  have hmapAB : ∀ (i j : Fin n),
      P.map (fun ω : (Fin n → α) × (Fin n → α) => (ω.1 i, ω.2 j))
        = μ.measure.prod ν.measure := by
    intro i j
    have h1 : (fun ω : (Fin n → α) × (Fin n → α) => (ω.1 i, ω.2 j))
        = Prod.map (Function.eval i) (Function.eval j) := rfl
    rw [h1, hPdef, ← Measure.map_prod_map _ _ (measurable_pi_apply i) (measurable_pi_apply j),
      (measurePreserving_eval _ i).map_eq, (measurePreserving_eval _ j).map_eq]
  -- integrability and values of the atomic terms
  have hAA_int : ∀ i j : Fin n,
      Integrable (fun ω : (Fin n → α) × (Fin n → α) => dist (ω.1 i) (ω.1 j)) P := by
    intro i j
    by_cases hij : i = j
    · subst hij
      simp [dist_self]
    · exact (integrable_integral_dist_comp (by fun_prop) (hmapAA hij) h_μμ).1
  have hAA_val : ∀ i j : Fin n,
      (∫ ω, dist (ω.1 i) (ω.1 j) ∂P) = if i = j then 0 else Iμμ := by
    intro i j
    by_cases hij : i = j
    · subst hij; simp [dist_self]
    · rw [if_neg hij]
      exact (integrable_integral_dist_comp (by fun_prop) (hmapAA hij) h_μμ).2
  have hBB_int : ∀ i j : Fin n,
      Integrable (fun ω : (Fin n → α) × (Fin n → α) => dist (ω.2 i) (ω.2 j)) P := by
    intro i j
    by_cases hij : i = j
    · subst hij
      simp [dist_self]
    · exact (integrable_integral_dist_comp (by fun_prop) (hmapBB hij) h_νν).1
  have hBB_val : ∀ i j : Fin n,
      (∫ ω, dist (ω.2 i) (ω.2 j) ∂P) = if i = j then 0 else Iνν := by
    intro i j
    by_cases hij : i = j
    · subst hij; simp [dist_self]
    · rw [if_neg hij]
      exact (integrable_integral_dist_comp (by fun_prop) (hmapBB hij) h_νν).2
  have hAB_int : ∀ i j : Fin n,
      Integrable (fun ω : (Fin n → α) × (Fin n → α) => dist (ω.1 i) (ω.2 j)) P :=
    fun i j => (integrable_integral_dist_comp (by fun_prop) (hmapAB i j) h_μν).1
  have hAB_val : ∀ i j : Fin n, (∫ ω, dist (ω.1 i) (ω.2 j) ∂P) = Iμν :=
    fun i j => (integrable_integral_dist_comp (by fun_prop) (hmapAB i j) h_μν).2
  -- integrability of the block sums
  have hA_int : Integrable
      (fun ω : (Fin n → α) × (Fin n → α) => ∑ i, ∑ j, dist (ω.1 i) (ω.1 j)) P :=
    integrable_finsetSum _ fun i _ => integrable_finsetSum _ fun j _ => hAA_int i j
  have hB_int : Integrable
      (fun ω : (Fin n → α) × (Fin n → α) => ∑ i, ∑ j, dist (ω.2 i) (ω.2 j)) P :=
    integrable_finsetSum _ fun i _ => integrable_finsetSum _ fun j _ => hBB_int i j
  have hC_int : Integrable
      (fun ω : (Fin n → α) × (Fin n → α) => ∑ i, ∑ j, dist (ω.1 i) (ω.2 j)) P :=
    integrable_finsetSum _ fun i _ => integrable_finsetSum _ fun j _ => hAB_int i j
  -- values of the block-sum integrals
  have hA_val : (∫ ω, ∑ i, ∑ j, dist (ω.1 i) (ω.1 j) ∂P) = (n : ℝ) * (((n : ℝ) - 1) * Iμμ) := by
    rw [integral_finsetSum _ fun i _ => integrable_finsetSum _ fun j _ => hAA_int i j]
    have h1 : ∀ i : Fin n,
        (∫ ω, ∑ j, dist (ω.1 i) (ω.1 j) ∂P) = ((n : ℝ) - 1) * Iμμ := by
      intro i
      rw [integral_finsetSum _ fun j _ => hAA_int i j]
      simp_rw [hAA_val i]
      have hsplit : ∀ j : Fin n,
          (if i = j then (0 : ℝ) else Iμμ) = Iμμ - (if i = j then Iμμ else 0) := by
        intro j; split <;> ring
      simp_rw [hsplit]
      rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.sum_ite_eq, if_pos (Finset.mem_univ i)]
      simp only [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      ring
    simp_rw [h1]
    rw [Finset.sum_const]
    simp only [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hB_val : (∫ ω, ∑ i, ∑ j, dist (ω.2 i) (ω.2 j) ∂P) = (n : ℝ) * (((n : ℝ) - 1) * Iνν) := by
    rw [integral_finsetSum _ fun i _ => integrable_finsetSum _ fun j _ => hBB_int i j]
    have h1 : ∀ i : Fin n,
        (∫ ω, ∑ j, dist (ω.2 i) (ω.2 j) ∂P) = ((n : ℝ) - 1) * Iνν := by
      intro i
      rw [integral_finsetSum _ fun j _ => hBB_int i j]
      simp_rw [hBB_val i]
      have hsplit : ∀ j : Fin n,
          (if i = j then (0 : ℝ) else Iνν) = Iνν - (if i = j then Iνν else 0) := by
        intro j; split <;> ring
      simp_rw [hsplit]
      rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.sum_ite_eq, if_pos (Finset.mem_univ i)]
      simp only [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      ring
    simp_rw [h1]
    rw [Finset.sum_const]
    simp only [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hC_val : (∫ ω, ∑ i, ∑ j, dist (ω.1 i) (ω.2 j) ∂P) = (n : ℝ) * ((n : ℝ) * Iμν) := by
    rw [integral_finsetSum _ fun i _ => integrable_finsetSum _ fun j _ => hAB_int i j]
    have h1 : ∀ i : Fin n, (∫ ω, ∑ j, dist (ω.1 i) (ω.2 j) ∂P) = (n : ℝ) * Iμν := by
      intro i
      rw [integral_finsetSum _ fun j _ => hAB_int i j]
      simp_rw [hAB_val i]
      rw [Finset.sum_const]
      simp only [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    simp_rw [h1]
    rw [Finset.sum_const]
    simp only [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  -- integrate the pointwise CND inequality
  have hmono : (∫ ω, ((∑ i, ∑ j, dist (ω.1 i) (ω.1 j)) + ∑ i, ∑ j, dist (ω.2 i) (ω.2 j)) ∂P)
      ≤ ∫ ω, 2 * ∑ i, ∑ j, dist (ω.1 i) (ω.2 j) ∂P :=
    integral_mono (hA_int.add hB_int) (hC_int.const_mul 2)
      (fun ω => cnd_block_sums h_neg ω.1 ω.2)
  rw [integral_add hA_int hB_int, integral_const_mul, hA_val, hB_val, hC_val] at hmono
  calc (n : ℝ) * ((n : ℝ) - 1) * (Iμμ + Iνν)
      = (n : ℝ) * (((n : ℝ) - 1) * Iμμ) + (n : ℝ) * (((n : ℝ) - 1) * Iνν) := by ring
    _ ≤ 2 * ((n : ℝ) * ((n : ℝ) * Iμν)) := hmono
    _ = 2 * (n : ℝ) ^ 2 * Iμν := by ring

/-- Energy distance is nonnegative **on spaces of negative type**.

    D_E²(μ, ν) ≥ 0

    The negative-type hypothesis `DistNegativeType α` is necessary: over a bare
    `PseudoMetricSpace` the statement is false (circle with geodesic distance,
    Székely & Rizzo 2023 §3.2). Restated 2026-07 per ROADMAP §3.2 #5 — the
    hypothesis is discharged for Euclidean/Hilbert spaces by Schoenberg's
    theorem (future work: formalize Lyons 2013 or use the existing
    `InnerProductSpace` CND instance).

    The moment hypotheses are essential: under Lean's junk-zero Bochner
    convention the statement is FALSE without them (e.g. α = ℝ, ν = ½(δ₀+δ₁),
    μ Cauchy: both μ-integrals junk-zero and energyDistanceSq = −½).

    **Proof** (sampling argument): for each `n`, integrating the discrete CND
    inequality (`h_neg` with `2n` points, weights `±1`) over `μ^⊗n ⊗ ν^⊗n` and
    counting off-diagonal pairs gives `n(n−1)(I(μ,μ)+I(ν,ν)) ≤ 2n²·I(μ,ν)`
    (`energy_sample_bound`); dividing by `n²` and letting `n → ∞` yields
    `I(μ,μ) + I(ν,ν) ≤ 2·I(μ,ν)`.  The finite-sample analogue is
    `v_statistic_nonneg`.

    Reference: Székely & Rizzo (2023), Theorem 3.2, §3.3; Lyons (2013), §3. -/
theorem energy_distance_nonneg
    {α : Type*} [MeasurableSpace α] [PseudoMetricSpace α]
    (h_neg : DistNegativeType α)
    (μ ν : ProbabilityMeasure α)
    (h_μν : FiniteDistMoment μ ν) (h_μμ : FiniteDistMoment μ μ) (h_νν : FiniteDistMoment ν ν) :
    0 ≤ energyDistanceSq μ ν := by
  unfold energyDistanceSq
  -- rewrite the iterated integrals as product-measure integrals
  have hEμν : (∫ z : α × α, dist z.1 z.2 ∂(μ.measure.prod ν.measure))
      = ∫ x, ∫ y, dist x y ∂ν.measure ∂μ.measure := MeasureTheory.integral_prod _ h_μν
  have hEμμ : (∫ z : α × α, dist z.1 z.2 ∂(μ.measure.prod μ.measure))
      = ∫ x, ∫ y, dist x y ∂μ.measure ∂μ.measure := MeasureTheory.integral_prod _ h_μμ
  have hEνν : (∫ z : α × α, dist z.1 z.2 ∂(ν.measure.prod ν.measure))
      = ∫ x, ∫ y, dist x y ∂ν.measure ∂ν.measure := MeasureTheory.integral_prod _ h_νν
  rw [← hEμν, ← hEμμ, ← hEνν]
  set Iμμ := ∫ z : α × α, dist z.1 z.2 ∂(μ.measure.prod μ.measure) with hIμμ
  set Iνν := ∫ z : α × α, dist z.1 z.2 ∂(ν.measure.prod ν.measure) with hIνν
  set Iμν := ∫ z : α × α, dist z.1 z.2 ∂(μ.measure.prod ν.measure) with hIμν
  -- the n → ∞ limit of the sampling bound
  have hkey : Iμμ + Iνν ≤ 2 * Iμν := by
    have htend : Tendsto (fun k : ℕ => ((k : ℝ) - 1) / k * (Iμμ + Iνν)) atTop
        (nhds (Iμμ + Iνν)) := by
      have h1 : Tendsto (fun k : ℕ => ((k : ℝ) - 1) / k) atTop (nhds 1) := by
        have h2 : Tendsto (fun k : ℕ => 1 - 1 / (k : ℝ)) atTop (nhds (1 - 0)) :=
          tendsto_const_nhds.sub tendsto_one_div_atTop_nhds_zero_nat
        rw [sub_zero] at h2
        refine h2.congr' ?_
        filter_upwards [eventually_ne_atTop 0] with k hk
        have hk' : (k : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hk
        field_simp
      simpa using h1.mul_const (Iμμ + Iνν)
    refine le_of_tendsto htend ?_
    filter_upwards [eventually_ge_atTop 1] with k hk
    have hb := energy_sample_bound h_neg μ ν h_μν h_μμ h_νν k
    have hkpos : (0 : ℝ) < (k : ℝ) := by exact_mod_cast hk
    have hk2 : (0 : ℝ) < (k : ℝ) ^ 2 := by positivity
    rw [div_mul_eq_mul_div, div_le_iff₀ hkpos]
    nlinarith [hb, hkpos]
  linarith [hkey]

/-- Conditional negative definiteness over an arbitrary finite index type: `DistNegativeType`
transported along `Fintype.equivFin`.  For any finite family of points `x : ι → α` and
zero-sum weights `w`, `∑ᵢⱼ wᵢ wⱼ d(xᵢ,xⱼ) ≤ 0`. -/
private lemma cnd_of_fintype {α : Type*} [PseudoMetricSpace α] (h_neg : DistNegativeType α)
    {ι : Type*} [Fintype ι] (x : ι → α) (w : ι → ℝ) (hw : ∑ i, w i = 0) :
    ∑ i, ∑ j, w i * w j * dist (x i) (x j) ≤ 0 := by
  classical
  let e := Fintype.equivFin ι
  have hw' : ∑ k, w (e.symm k) = 0 := by rw [Equiv.sum_comp e.symm w]; exact hw
  have h := h_neg (Fintype.card ι) (fun k => x (e.symm k)) (fun k => w (e.symm k)) hw'
  calc ∑ i, ∑ j, w i * w j * dist (x i) (x j)
      = ∑ k, ∑ l, w (e.symm k) * w (e.symm l) * dist (x (e.symm k)) (x (e.symm l)) := by
        rw [← Equiv.sum_comp e.symm (fun i => ∑ j, w i * w j * dist (x i) (x j))]
        refine Finset.sum_congr rfl fun k _ => ?_
        rw [← Equiv.sum_comp e.symm (fun j => w (e.symm k) * w j * dist (x (e.symm k)) (x j))]
    _ ≤ 0 := h

/-- Heterogeneous pair marginal: two distinct coordinates of a `Measure.pi` of (possibly
different) probability measures are jointly distributed as the product of the two factors. -/
private lemma pi_map_pair_hetero {ι : Type*} [Fintype ι] {α : Type*} [MeasurableSpace α]
    (ρ : ι → Measure α) [∀ i, IsProbabilityMeasure (ρ i)] {i j : ι} (hij : i ≠ j) :
    (Measure.pi ρ).map (fun ω => (ω i, ω j)) = (ρ i).prod (ρ j) := by
  have hindep : IndepFun (fun ω : ι → α => ω i) (fun ω : ι → α => ω j) (Measure.pi ρ) :=
    (iIndepFun_pi (X := fun _ : ι => id) fun _ => aemeasurable_id).indepFun hij
  rw [hindep.map_prod_eq_prod_map_map (measurable_pi_apply i).aemeasurable
    (measurable_pi_apply j).aemeasurable,
    (measurePreserving_eval ρ i).map_eq, (measurePreserving_eval ρ j).map_eq]

/-- The K-block sampling bound behind `energyDistanceSq_cnd`.  Drawing `n` iid points from each
of the `K` blocks, assigning weight `u k` to every point of block `k` (total weight
`n·∑u = 0`), and integrating the discrete CND inequality (`h_neg` at the `K·n` points) over
`⨂ (ν k)^{⊗n}` gives, after counting same/cross pairs,

`n²·(∑ₖₗ uₖuₗ I_{kl}) ≤ n·(∑ₖ uₖ² I_{kk})`,   `I_{kl} = ∫∫ dist d(ν_k ⊗ ν_l)`.

Dividing by `n²` and letting `n → ∞` (in `energyDistanceSq_cnd`) yields `∑ₖₗ uₖuₗ I_{kl} ≤ 0`. -/
private lemma energyDistanceSq_sample_bound {α : Type*} [MeasurableSpace α] [PseudoMetricSpace α]
    (h_neg : DistNegativeType α) {K : ℕ} (ν : Fin K → ProbabilityMeasure α)
    (hmom : ∀ k l, FiniteDistMoment (ν k) (ν l))
    (u : Fin K → ℝ) (hsum : ∑ k, u k = 0) (n : ℕ) :
    (n : ℝ) ^ 2 *
        (∑ k, ∑ l, u k * u l *
          ∫ z : α × α, dist z.1 z.2 ∂((ν k).measure.prod (ν l).measure))
      ≤ (n : ℝ) *
        ∑ k, u k ^ 2 * ∫ z : α × α, dist z.1 z.2 ∂((ν k).measure.prod (ν k).measure) := by
  classical
  set I : Fin K → Fin K → ℝ :=
    fun k l => ∫ z : α × α, dist z.1 z.2 ∂((ν k).measure.prod (ν l).measure) with hI
  have hfold : ∀ k l, (∫ z : α × α, dist z.1 z.2 ∂((ν k).measure.prod (ν l).measure)) = I k l :=
    fun k l => by rw [hI]
  simp only [hfold]
  haveI hProb : ∀ p : Fin K × Fin n, IsProbabilityMeasure ((ν p.1).measure) :=
    fun p => (ν p.1).isProbabilityMeasure
  set P : Measure ((Fin K × Fin n) → α) := Measure.pi (fun p => (ν p.1).measure) with hP
  haveI : IsProbabilityMeasure P := by rw [hP]; infer_instance
  -- atomic integral values and integrability
  have hatom_val : ∀ p q : Fin K × Fin n,
      (∫ ω, dist (ω p) (ω q) ∂P) = if p = q then 0 else I p.1 q.1 := by
    intro p q
    by_cases hpq : p = q
    · subst hpq; simp [dist_self]
    · rw [if_neg hpq, ← hfold p.1 q.1]
      have hmap : P.map (fun ω : (Fin K × Fin n) → α => (ω p, ω q))
          = (ν p.1).measure.prod (ν q.1).measure := by
        rw [hP]; exact pi_map_pair_hetero (fun p => (ν p.1).measure) hpq
      exact (integrable_integral_dist_comp (by fun_prop) hmap (hmom p.1 q.1)).2
  have hatom_int : ∀ p q : Fin K × Fin n, Integrable (fun ω => dist (ω p) (ω q)) P := by
    intro p q
    by_cases hpq : p = q
    · subst hpq; simp [dist_self]
    · have hmap : P.map (fun ω : (Fin K × Fin n) → α => (ω p, ω q))
          = (ν p.1).measure.prod (ν q.1).measure := by
        rw [hP]; exact pi_map_pair_hetero (fun p => (ν p.1).measure) hpq
      exact (integrable_integral_dist_comp (by fun_prop) hmap (hmom p.1 q.1)).1
  -- total-weight zero for the K·n points
  have hwsum : ∑ p : Fin K × Fin n, u p.1 = 0 := by
    rw [Fintype.sum_prod_type]
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    rw [← Finset.mul_sum, hsum, mul_zero]
  -- value of the integrated double sum
  have hval : (∫ ω, ∑ p, ∑ q, u p.1 * u q.1 * dist (ω p) (ω q) ∂P)
      = (n : ℝ) ^ 2 * (∑ k, ∑ l, u k * u l * I k l) - (n : ℝ) * ∑ k, u k ^ 2 * I k k := by
    have hrow : ∀ p : Fin K × Fin n,
        (∫ ω, ∑ q, u p.1 * u q.1 * dist (ω p) (ω q) ∂P)
          = ∑ q, u p.1 * u q.1 * (if p = q then (0 : ℝ) else I p.1 q.1) := by
      intro p
      rw [integral_finsetSum _ (fun q _ => (hatom_int p q).const_mul _)]
      refine Finset.sum_congr rfl fun q _ => ?_
      rw [integral_const_mul, hatom_val p q]
    rw [integral_finsetSum _
      (fun p _ => integrable_finsetSum _ (fun q _ => (hatom_int p q).const_mul _))]
    simp_rw [hrow]
    have hsplit : ∀ p q : Fin K × Fin n,
        u p.1 * u q.1 * (if p = q then (0 : ℝ) else I p.1 q.1)
          = u p.1 * u q.1 * I p.1 q.1 - (if p = q then u p.1 ^ 2 * I p.1 p.1 else 0) := by
      intro p q
      by_cases hpq : p = q
      · subst hpq; rw [if_pos rfl, if_pos rfl]; ring
      · simp [hpq]
    simp_rw [hsplit, Finset.sum_sub_distrib]
    have hfirst : (∑ p : Fin K × Fin n, ∑ q : Fin K × Fin n, u p.1 * u q.1 * I p.1 q.1)
        = (n : ℝ) ^ 2 * ∑ k, ∑ l, u k * u l * I k l := by
      simp_rw [Fintype.sum_prod_type]
      simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      conv_rhs => rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun k _ => ?_
      rw [← Finset.mul_sum]
      ring
    have hsecond : (∑ p : Fin K × Fin n, ∑ q : Fin K × Fin n,
          (if p = q then u p.1 ^ 2 * I p.1 p.1 else 0))
        = (n : ℝ) * ∑ k, u k ^ 2 * I k k := by
      have hinner : ∀ p : Fin K × Fin n,
          (∑ q : Fin K × Fin n, (if p = q then u p.1 ^ 2 * I p.1 p.1 else 0))
            = u p.1 ^ 2 * I p.1 p.1 := by
        intro p
        rw [Finset.sum_ite_eq, if_pos (Finset.mem_univ p)]
      simp_rw [hinner, Fintype.sum_prod_type]
      simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      rw [← Finset.mul_sum]
    rw [hfirst, hsecond]
  -- integrate the pointwise CND inequality
  have hle0 : (∫ ω, ∑ p, ∑ q, u p.1 * u q.1 * dist (ω p) (ω q) ∂P) ≤ 0 :=
    integral_nonpos (fun ω => cnd_of_fintype h_neg ω (fun p => u p.1) hwsum)
  rw [hval] at hle0
  linarith [hle0]

/-- **Population conditional negative definiteness of the energy-distance matrix.**

For a space of negative type (`DistNegativeType α`), a finite family `ν : Fin K →
ProbabilityMeasure α` with pairwise finite distance moments, and zero-sum weights
`u : Fin K → ℝ`, the weighted double sum of pairwise squared energy distances is nonpositive:

`∑ₖₗ uₖ uₗ · 𝓔²(ν_k, ν_l) ≤ 0`.

This is the K-block generalization of `energy_distance_nonneg` (which is the two-block, weights
`±1` case): the algebra `∑ₖₗ uₖuₗ 𝓔²_{kl} = 2·∑ₖₗ uₖuₗ I_{kl}` (the `I_{kk}`, `I_{ll}`
self-terms cancel under `∑u = 0`), and `∑ₖₗ uₖuₗ I_{kl} ≤ 0` by the sampling bound
`energyDistanceSq_sample_bound` in the `n → ∞` limit.  Both the negative-type and the
finite-moment hypotheses are essential (see `energy_distance_nonneg`).  Consumed by Paper 2's
mixture-QP convexity (`Q_concave_on_simplex` at `D = √𝓔`).

Reference: Székely & Rizzo (2023), §3.2, §10; Lyons (2013), §3. -/
theorem energyDistanceSq_cnd {α : Type*} [MeasurableSpace α] [PseudoMetricSpace α]
    (h_neg : DistNegativeType α) {K : ℕ} (ν : Fin K → ProbabilityMeasure α)
    (hmom : ∀ k l, FiniteDistMoment (ν k) (ν l))
    (u : Fin K → ℝ) (hsum : ∑ k, u k = 0) :
    ∑ k, ∑ l, u k * u l * energyDistanceSq (ν k) (ν l) ≤ 0 := by
  classical
  set I : Fin K → Fin K → ℝ :=
    fun k l => ∫ z : α × α, dist z.1 z.2 ∂((ν k).measure.prod (ν l).measure) with hI
  have hfold : ∀ k l, (∫ z : α × α, dist z.1 z.2 ∂((ν k).measure.prod (ν l).measure)) = I k l :=
    fun k l => by rw [hI]
  -- energy distance in terms of the product-measure integrals `I`
  have hE : ∀ k l, energyDistanceSq (ν k) (ν l) = 2 * I k l - I k k - I l l := by
    intro k l
    unfold energyDistanceSq
    rw [← MeasureTheory.integral_prod _ (hmom k l), ← MeasureTheory.integral_prod _ (hmom k k),
      ← MeasureTheory.integral_prod _ (hmom l l), hfold k l, hfold k k, hfold l l]
  -- reduce the weighted double sum to `2 * ∑ₖₗ uₖuₗ I_{kl}` using `∑u = 0`
  have hmid : ∑ k, ∑ l, u k * u l * I k k = 0 := by
    have h1 : ∀ k, (∑ l, u k * u l * I k k) = (u k * I k k) * ∑ l, u l := by
      intro k; rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun l _ => by ring
    simp_rw [h1, hsum, mul_zero, Finset.sum_const_zero]
  have hlast : ∑ k, ∑ l, u k * u l * I l l = 0 := by
    have h1 : ∀ k, (∑ l, u k * u l * I l l) = u k * ∑ l, u l * I l l := by
      intro k; rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun l _ => by ring
    simp_rw [h1, ← Finset.sum_mul, hsum, zero_mul]
  have hSeq : ∑ k, ∑ l, u k * u l * energyDistanceSq (ν k) (ν l)
      = 2 * ∑ k, ∑ l, u k * u l * I k l := by
    calc ∑ k, ∑ l, u k * u l * energyDistanceSq (ν k) (ν l)
        = ∑ k, ∑ l, (2 * (u k * u l * I k l) - u k * u l * I k k - u k * u l * I l l) := by
          simp_rw [hE]
          exact Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun l _ => by ring
      _ = (∑ k, ∑ l, 2 * (u k * u l * I k l)) - (∑ k, ∑ l, u k * u l * I k k)
            - (∑ k, ∑ l, u k * u l * I l l) := by simp_rw [Finset.sum_sub_distrib]
      _ = 2 * ∑ k, ∑ l, u k * u l * I k l := by
          rw [hmid, hlast]; simp_rw [← Finset.mul_sum]; ring
  rw [hSeq]
  -- the `n → ∞` limit of the sampling bound
  have hbound : ∀ m : ℕ,
      (m : ℝ) ^ 2 * (∑ k, ∑ l, u k * u l * I k l) ≤ (m : ℝ) * ∑ k, u k ^ 2 * I k k := by
    intro m
    have hb := energyDistanceSq_sample_bound h_neg ν hmom u hsum m
    simp only [hfold] at hb
    exact hb
  have hS : ∑ k, ∑ l, u k * u l * I k l ≤ 0 := by
    have htend : Tendsto (fun m : ℕ => (∑ k, u k ^ 2 * I k k) / (m : ℝ)) atTop (nhds 0) := by
      have h0 := tendsto_one_div_atTop_nhds_zero_nat.const_mul (∑ k, u k ^ 2 * I k k)
      simp only [mul_zero] at h0
      refine h0.congr fun m => ?_
      rw [mul_one_div]
    refine ge_of_tendsto htend ?_
    filter_upwards [eventually_ge_atTop 1] with m hm
    have hb := hbound m
    have hmpos : (0 : ℝ) < (m : ℝ) := by exact_mod_cast hm
    rw [le_div_iff₀ hmpos]
    nlinarith [hb, hmpos]
  linarith [hS]

/-! ### C2: simplex-mixture energy expansion + convexity (Paper 2)

**Adjudicated sign convention (Paper 2 methodology).** Expanding the squared energy distance
of `μ` against a simplex mixture `mix w ν = ∑ₖ w_k ν_k` gives, with cross moments
`c_k = ∫ dist d(μ ⊗ ν_k)`, Gram `Q_{kl} = ∫ dist d(ν_k ⊗ ν_l)`, and constant `b = ∫ dist d(μ ⊗ μ)`,

  `𝓔²(μ, mix w ν) = 2 c'w − w'Qw − b`.

The mixture-optimization QP is therefore `argmin_w (2 c'w − w'Qw)` — i.e. the quadratic matrix is
`−Q`, **not** `argmin_w (w'Qw − 2 c'w)` (its negative). Convexity of the objective along the
simplex holds because the only non-affine term is `−w'Qw`, and `Q` (the distance-cross-moment
Gram) is **conditionally negative definite on zero-sum vectors** (`energyCrossGram_cnd`), so
`−w'Qw` is convex — this is the correct mechanism, **not** "Q is positive semidefinite".  This
docstring adjudicates `02_long_short_testing/chapters/methodology.typ:32-45`; the manuscript prose
is fixed in the F(P2) lane (do not edit the Typst here).  The simplex-convexity corollary
`energyDistanceSq_simplexMixture_convex` lives in `Connections/EnergyStatistics.lean`.

Reference: Székely & Rizzo (2023), §3, §10. -/

/-- **Simplex mixture** of a finite family of probability measures.  Given weights `w : Fin K → ℝ`
on the probability simplex (`w ≥ 0`, `∑ w = 1`) and probability measures `ν`, the convex
combination `∑ₖ ENNReal.ofReal (w_k) • ν_k` is again a probability measure. -/
noncomputable def simplexMixture {α : Type*} [MeasurableSpace α] {K : ℕ}
    (w : Fin K → ℝ) (ν : Fin K → ProbabilityMeasure α)
    (hw_nonneg : ∀ k, 0 ≤ w k) (hw_sum : ∑ k, w k = 1) : ProbabilityMeasure α where
  measure := ∑ k, ENNReal.ofReal (w k) • (ν k).measure
  is_probability := by
    rw [MeasureTheory.Measure.finsetSum_apply]
    have hstep : ∀ k ∈ (Finset.univ : Finset (Fin K)),
        (ENNReal.ofReal (w k) • (ν k).measure) Set.univ = ENNReal.ofReal (w k) := by
      intro k _
      rw [MeasureTheory.Measure.smul_apply, (ν k).is_probability, smul_eq_mul, mul_one]
    rw [Finset.sum_congr rfl hstep, ← ENNReal.ofReal_sum_of_nonneg (fun i _ => hw_nonneg i),
      hw_sum, ENNReal.ofReal_one]

/-- Integration against a simplex mixture is the convex combination of the component integrals:
`∫ g d(mix w ν) = ∑ₖ w_k ∫ g dν_k`, whenever `g` is integrable against every component. -/
theorem integral_simplexMixture {α : Type*} [MeasurableSpace α] {K : ℕ}
    (w : Fin K → ℝ) (ν : Fin K → ProbabilityMeasure α)
    (hw_nonneg : ∀ k, 0 ≤ w k) (hw_sum : ∑ k, w k = 1)
    (g : α → ℝ) (hg : ∀ k, Integrable g (ν k).measure) :
    (∫ a, g a ∂(simplexMixture w ν hw_nonneg hw_sum).measure)
      = ∑ k, w k * ∫ a, g a ∂(ν k).measure := by
  have hf : ∀ k ∈ (Finset.univ : Finset (Fin K)),
      Integrable g (ENNReal.ofReal (w k) • (ν k).measure) :=
    fun k _ => (hg k).smul_measure ENNReal.ofReal_ne_top
  show ∫ a, g a ∂(∑ k, ENNReal.ofReal (w k) • (ν k).measure)
      = ∑ k, w k * ∫ a, g a ∂(ν k).measure
  rw [integral_finsetSum_measure hf]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [integral_smul_measure, ENNReal.toReal_ofReal (hw_nonneg k), smul_eq_mul]

/-- Cross moment `c_k = ∫ dist d(μ ⊗ ν_k)` in the mixture-energy expansion. -/
noncomputable def mixC {α : Type*} [MeasurableSpace α] [PseudoMetricSpace α]
    (μ : ProbabilityMeasure α) {K : ℕ} (ν : Fin K → ProbabilityMeasure α) (k : Fin K) : ℝ :=
  ∫ z : α × α, dist z.1 z.2 ∂(μ.measure.prod (ν k).measure)

/-- Distance-cross-moment Gram entry `Q_{kl} = ∫ dist d(ν_k ⊗ ν_l)` in the mixture-energy
expansion. -/
noncomputable def mixQ {α : Type*} [MeasurableSpace α] [PseudoMetricSpace α]
    {K : ℕ} (ν : Fin K → ProbabilityMeasure α) (k l : Fin K) : ℝ :=
  ∫ z : α × α, dist z.1 z.2 ∂((ν k).measure.prod (ν l).measure)

/-- Constant `b = ∫ dist d(μ ⊗ μ)` in the mixture-energy expansion. -/
noncomputable def mixB {α : Type*} [MeasurableSpace α] [PseudoMetricSpace α]
    (μ : ProbabilityMeasure α) : ℝ :=
  ∫ z : α × α, dist z.1 z.2 ∂(μ.measure.prod μ.measure)

/-- Fiberwise expansion of the inner distance integral against a simplex mixture: for a.e. base
point `x` (under any fixed `κ` with `FiniteDistMoment κ (ν l)` for all `l`), the inner integral
`∫ y, dist x y d(mix)` equals `∑ₗ w_l ∫ y, dist x y dν_l`. -/
private lemma simplexMixture_inner_ae {α : Type*} [MeasurableSpace α] [PseudoMetricSpace α]
    (κ : ProbabilityMeasure α) {K : ℕ} (w : Fin K → ℝ) (ν : Fin K → ProbabilityMeasure α)
    (hw_nonneg : ∀ k, 0 ≤ w k) (hw_sum : ∑ k, w k = 1)
    (hmom : ∀ l, FiniteDistMoment κ (ν l)) :
    (fun x => ∫ y, dist x y ∂(simplexMixture w ν hw_nonneg hw_sum).measure)
      =ᵐ[κ.measure] fun x => ∑ l, w l * ∫ y, dist x y ∂(ν l).measure := by
  have hfib : ∀ᵐ x ∂κ.measure, ∀ l, Integrable (fun y => dist x y) (ν l).measure :=
    ae_all_iff.mpr fun l => Integrable.prod_right_ae (hmom l)
  filter_upwards [hfib] with x hx
  exact integral_simplexMixture w ν hw_nonneg hw_sum (fun y => dist x y) hx

/-- The fiberwise-expanded inner integral `x ↦ ∑ₗ w_l ∫ y, dist x y dν_l` is integrable against
`κ`, being a finite combination of the `Integrable.integral_prod_left` images of the pairwise
finite-moment hypotheses. -/
private lemma simplexMixture_inner_integrable {α : Type*} [MeasurableSpace α] [PseudoMetricSpace α]
    (κ : ProbabilityMeasure α) {K : ℕ} (w : Fin K → ℝ) (ν : Fin K → ProbabilityMeasure α)
    (hmom : ∀ l, FiniteDistMoment κ (ν l)) :
    Integrable (fun x => ∑ l, w l * ∫ y, dist x y ∂(ν l).measure) κ.measure :=
  integrable_finsetSum Finset.univ
    (fun l _ => (Integrable.integral_prod_left (hmom l)).const_mul (w l))

/-- Integrability of the inner mixture integral `x ↦ ∫ y, dist x y d(mix)` against `κ`, obtained
by transporting `simplexMixture_inner_integrable` across the a.e. identity
`simplexMixture_inner_ae`. -/
private lemma integrable_inner_dist_mix {α : Type*} [MeasurableSpace α] [PseudoMetricSpace α]
    (κ : ProbabilityMeasure α) {K : ℕ} (w : Fin K → ℝ) (ν : Fin K → ProbabilityMeasure α)
    (hw_nonneg : ∀ k, 0 ≤ w k) (hw_sum : ∑ k, w k = 1)
    (hmom : ∀ l, FiniteDistMoment κ (ν l)) :
    Integrable (fun x => ∫ y, dist x y ∂(simplexMixture w ν hw_nonneg hw_sum).measure) κ.measure :=
  (simplexMixture_inner_integrable κ w ν hmom).congr
    (simplexMixture_inner_ae κ w ν hw_nonneg hw_sum hmom).symm

/-- Reduction of the (inner-mixture) iterated distance integral to the component cross moments:
`∫ x, ∫ y, dist x y d(mix) dκ = ∑ₗ w_l ∫ dist d(κ ⊗ ν_l)`. -/
private lemma integral_dist_mix_right {α : Type*} [MeasurableSpace α] [PseudoMetricSpace α]
    (κ : ProbabilityMeasure α) {K : ℕ} (w : Fin K → ℝ) (ν : Fin K → ProbabilityMeasure α)
    (hw_nonneg : ∀ k, 0 ≤ w k) (hw_sum : ∑ k, w k = 1)
    (hmom : ∀ l, FiniteDistMoment κ (ν l)) :
    (∫ x, ∫ y, dist x y ∂(simplexMixture w ν hw_nonneg hw_sum).measure ∂κ.measure)
      = ∑ l, w l * ∫ z : α × α, dist z.1 z.2 ∂(κ.measure.prod (ν l).measure) := by
  rw [integral_congr_ae (simplexMixture_inner_ae κ w ν hw_nonneg hw_sum hmom),
    integral_finsetSum Finset.univ
      (fun l _ => (Integrable.integral_prod_left (hmom l)).const_mul (w l))]
  refine Finset.sum_congr rfl fun l _ => ?_
  rw [integral_const_mul, ← integral_prod _ (hmom l)]

/-- **C2 — mixture-energy expansion.** The squared energy distance of `μ` against a simplex
mixture `mix w ν` decomposes into the cross, Gram, and constant terms:

  `𝓔²(μ, mix w ν) = 2·(∑ₖ w_k c_k) − (∑ₖ ∑ₗ w_k w_l Q_{kl}) − b`,

with `c_k = mixC μ ν k`, `Q_{kl} = mixQ ν k l`, `b = mixB μ`.  This is the identity that fixes the
Paper-2 mixture-QP sign convention (see the section docstring); the simplex-convexity corollary
is `energyDistanceSq_simplexMixture_convex` (`Connections/EnergyStatistics.lean`).

Reference: Székely & Rizzo (2023), §3, §10. -/
theorem energyDistanceSq_simplexMixture {α : Type*} [MeasurableSpace α] [PseudoMetricSpace α]
    {K : ℕ} (μ : ProbabilityMeasure α) (w : Fin K → ℝ) (ν : Fin K → ProbabilityMeasure α)
    (hw_nonneg : ∀ k, 0 ≤ w k) (hw_sum : ∑ k, w k = 1)
    (h_μμ : FiniteDistMoment μ μ) (h_μν : ∀ k, FiniteDistMoment μ (ν k))
    (h_νν : ∀ k l, FiniteDistMoment (ν k) (ν l)) :
    energyDistanceSq μ (simplexMixture w ν hw_nonneg hw_sum)
      = 2 * (∑ k, w k * mixC μ ν k) - (∑ k, ∑ l, w k * w l * mixQ ν k l) - mixB μ := by
  unfold energyDistanceSq mixC mixQ mixB
  have e1 : (∫ x, ∫ y, dist x y ∂(simplexMixture w ν hw_nonneg hw_sum).measure ∂μ.measure)
      = ∑ k, w k * ∫ z : α × α, dist z.1 z.2 ∂(μ.measure.prod (ν k).measure) :=
    integral_dist_mix_right μ w ν hw_nonneg hw_sum h_μν
  have e2 : (∫ x, ∫ y, dist x y ∂μ.measure ∂μ.measure)
      = ∫ z : α × α, dist z.1 z.2 ∂(μ.measure.prod μ.measure) :=
    (integral_prod _ h_μμ).symm
  have e3 : (∫ x, ∫ y, dist x y ∂(simplexMixture w ν hw_nonneg hw_sum).measure
        ∂(simplexMixture w ν hw_nonneg hw_sum).measure)
      = ∑ k, ∑ l, w k * w l * ∫ z : α × α, dist z.1 z.2 ∂((ν k).measure.prod (ν l).measure) := by
    rw [integral_simplexMixture w ν hw_nonneg hw_sum
      (fun x => ∫ y, dist x y ∂(simplexMixture w ν hw_nonneg hw_sum).measure)
      (fun k => integrable_inner_dist_mix (ν k) w ν hw_nonneg hw_sum (h_νν k))]
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [integral_dist_mix_right (ν k) w ν hw_nonneg hw_sum (h_νν k), Finset.mul_sum]
    exact Finset.sum_congr rfl fun l _ => by ring
  rw [e1, e2, e3]
  ring

/-- **Distance-cross-moment Gram conditional negative definiteness.** For a space of negative type
and a finite family `ν` with pairwise finite distance moments, the raw distance Gram
`Q_{kl} = ∫ dist d(ν_k ⊗ ν_l)` is conditionally negative definite on zero-sum weights:

  `∑ₖ ∑ₗ u_k u_l Q_{kl} ≤ 0`   whenever `∑ u = 0`.

This is the `n → ∞` limit of the K-block sampling bound `energyDistanceSq_sample_bound`; it is the
CND fact behind the convexity of the mixture-energy QP objective `−w'Qw` (Paper 2), i.e. the
correct replacement for "Q is positive semidefinite" (see `energyDistanceSq_simplexMixture`).

Reference: Székely & Rizzo (2023), §3.2, §10; Lyons (2013), §3. -/
theorem energyCrossGram_cnd {α : Type*} [MeasurableSpace α] [PseudoMetricSpace α]
    (h_neg : DistNegativeType α) {K : ℕ} (ν : Fin K → ProbabilityMeasure α)
    (hmom : ∀ k l, FiniteDistMoment (ν k) (ν l))
    (u : Fin K → ℝ) (hsum : ∑ k, u k = 0) :
    ∑ k, ∑ l, u k * u l *
        ∫ z : α × α, dist z.1 z.2 ∂((ν k).measure.prod (ν l).measure) ≤ 0 := by
  classical
  set S := ∑ k, ∑ l, u k * u l *
      ∫ z : α × α, dist z.1 z.2 ∂((ν k).measure.prod (ν l).measure) with hS_def
  set T := ∑ k, u k ^ 2 *
      ∫ z : α × α, dist z.1 z.2 ∂((ν k).measure.prod (ν k).measure) with hT_def
  have hbound : ∀ m : ℕ, (m : ℝ) ^ 2 * S ≤ (m : ℝ) * T := by
    intro m
    have hb := energyDistanceSq_sample_bound h_neg ν hmom u hsum m
    rw [← hS_def, ← hT_def] at hb
    exact hb
  have htend : Tendsto (fun m : ℕ => T / (m : ℝ)) atTop (nhds 0) := by
    have h0 := tendsto_one_div_atTop_nhds_zero_nat.const_mul T
    simp only [mul_zero] at h0
    refine h0.congr fun m => ?_
    rw [mul_one_div]
  refine ge_of_tendsto htend ?_
  filter_upwards [eventually_ge_atTop 1] with m hm
  have hb := hbound m
  have hmpos : (0 : ℝ) < (m : ℝ) := by exact_mod_cast hm
  rw [le_div_iff₀ hmpos]
  nlinarith [hb, hmpos]

/-- **Energy distance characterizes equality of distributions — under strong negative type.**

    `D_E²(μ, ν) = 0 ↔ μ = ν`, given `StrongDistNegativeType α` and pairwise finite distance
    moments.

    The earlier unconditional form (over a bare `PseudoMetricSpace`, no negative-type hypothesis)
    is **false** and has been retired: `energy_distance_zero_iff_refuted` exhibits a two-point
    carrier with the identically-zero pseudometric on which `𝓔 ≡ 0` for distinct point masses.
    The measure-level rigidity clause of `StrongDistNegativeType` is precisely what rules this out.

    The forward implication is the population-separation clause of the explicit hypothesis. The
    reverse implication is unconditional and follows by substitution in `energyDistanceSq`.
    Establishing strong negative type for Euclidean or Hilbert ground spaces is not part of this
    theorem; it requires a separate characteristic-function or Cramér--Wold argument.

    Reference: Székely & Rizzo (2023), Theorem 3.1; Lyons (2013), Theorem 3.16. -/
theorem energy_distance_zero_iff_of_strong
    {α : Type*} [MeasurableSpace α] [PseudoMetricSpace α]
    (h_strong : StrongDistNegativeType α)
    (μ ν : ProbabilityMeasure α)
    (h_μν : FiniteDistMoment μ ν) (h_μμ : FiniteDistMoment μ μ) (h_νν : FiniteDistMoment ν ν) :
    energyDistanceSq μ ν = 0 ↔ μ = ν := by
  refine ⟨fun h => ?_, fun h => ?_⟩
  · exact h_strong.2 μ ν h_μν h_μμ h_νν h
  · subst h
    unfold energyDistanceSq
    ring

end EnergyNonneg

end EnergyStatistics

/-! ### C6: `α = 2` degeneracy (append-only, does not touch anything above)

The energy-distance family generalizes to `𝓔_α(μ, ν) := 2·E[‖X-Y‖^α] -
E[‖X-X'‖^α] - E[‖Y-Y'‖^α]`; only `α ∈ (0, 2)` gives a genuine (conditionally
negative definite / metric-generating) member of the family. At `α = 2` the
functional degenerates: it is a function of means alone and can vanish for
distinct distributions. We exhibit the standard counterexample: `δ₀` versus
the two-point mixture `½(δ₋₁ + δ₁)` (equal means, distinct distributions),
for which the squared-distance (`α = 2`) energy functional vanishes.

Reference: Székely & Rizzo (2023) §3 (the `α`-parametrized family and its
degeneracy at `α = 2`); this module's `energyDistanceSq` above is the `α = 1`
member. -/

namespace EnergyStatistics

open MeasureTheory
open scoped ENNReal

/-- The `α = 2` (squared-distance) energy functional: the same
    two-term-minus-two-terms shape as `energyDistanceSq`, but with squared
    Euclidean distance `(x - y)^2` in place of the metric `dist x y`. Stated
    over bare `Measure ℝ` (rather than the bundled `ProbabilityMeasure`) since
    the counterexample computation only needs the underlying integrals. -/
noncomputable def energySqDistFunctional (μ ν : Measure ℝ) : ℝ :=
  2 * (∫ x, ∫ y, (x - y) ^ 2 ∂ν ∂μ)
    - (∫ x, ∫ y, (x - y) ^ 2 ∂μ ∂μ)
    - (∫ x, ∫ y, (x - y) ^ 2 ∂ν ∂ν)

/-- The uniform two-point mixture `½(δ_a + δ_b)` as a bare measure. -/
noncomputable def twoPointMixture (a b : ℝ) : Measure ℝ :=
  (2⁻¹ : ℝ≥0∞) • Measure.dirac a + (2⁻¹ : ℝ≥0∞) • Measure.dirac b

/-- Integrating any measurable `f` against the two-point mixture averages its
    values at the two support points. -/
theorem integral_twoPointMixture (a b : ℝ) (f : ℝ → ℝ) (hf : Measurable f) :
    ∫ y, f y ∂(twoPointMixture a b) = (f a + f b) / 2 := by
  have hfa : Integrable f (Measure.dirac a) := integrable_dirac (by finiteness)
  have hfb : Integrable f (Measure.dirac b) := integrable_dirac (by finiteness)
  unfold twoPointMixture
  rw [integral_add_measure (hfa.smul_measure (by norm_num)) (hfb.smul_measure (by norm_num)),
    integral_smul_measure, integral_smul_measure,
    integral_dirac' f a hf.stronglyMeasurable, integral_dirac' f b hf.stronglyMeasurable]
  norm_num
  ring

/-- The two-point mixture `½(δ₋₁ + δ₁)` is not the point mass at `0`: it puts
    zero mass on `{0}` while `δ₀` puts full mass there. -/
theorem twoPointMixture_ne_dirac_zero :
    twoPointMixture (-1 : ℝ) 1 ≠ Measure.dirac (0 : ℝ) := by
  intro h
  have h1 : (Measure.dirac (0 : ℝ)) {(0 : ℝ)} = 1 := by simp
  have h2 : (twoPointMixture (-1 : ℝ) 1) {(0 : ℝ)} = 0 := by
    unfold twoPointMixture
    simp [Measure.dirac_apply']
  rw [← h, h2] at h1
  norm_num at h1

/-- **C6 (`α = 2` degeneracy counterexample).** The squared-distance energy
    functional vanishes between `δ₀` and the equal-mean, distinct two-point
    mixture `½(δ₋₁ + δ₁)`: the `α = 2` member of the energy-distance family
    fails to distinguish distributions in general, unlike `α = 1`
    (`energyDistanceSq`, whose zero-iff-equal characterization is
    `energy_distance_zero_iff_of_strong` above, itself hypothesis-conditional
    under `StrongDistNegativeType`). -/
theorem energySqDistFunctional_alpha_two_degenerate :
    energySqDistFunctional (Measure.dirac (0 : ℝ)) (twoPointMixture (-1 : ℝ) 1) = 0 ∧
      Measure.dirac (0 : ℝ) ≠ twoPointMixture (-1 : ℝ) 1 := by
  refine ⟨?_, fun h => twoPointMixture_ne_dirac_zero h.symm⟩
  have hinner :
      (fun x : ℝ => ∫ y, (x - y) ^ 2 ∂(twoPointMixture (-1 : ℝ) 1)) =
        fun x : ℝ => ((x - (-1)) ^ 2 + (x - 1) ^ 2) / 2 := by
    funext x
    exact integral_twoPointMixture (-1) 1 (fun y => (x - y) ^ 2) (by fun_prop)
  have hA : ∫ x, ∫ y, (x - y) ^ 2 ∂(twoPointMixture (-1 : ℝ) 1) ∂(Measure.dirac (0 : ℝ)) = 1 := by
    rw [hinner, integral_dirac' _ 0 (by fun_prop)]
    norm_num
  have hB : ∫ x, ∫ y, (x - y) ^ 2 ∂(Measure.dirac (0 : ℝ)) ∂(Measure.dirac (0 : ℝ)) = 0 := by
    have hinnerB : (fun x : ℝ => ∫ y, (x - y) ^ 2 ∂(Measure.dirac (0 : ℝ))) =
        fun x : ℝ => (x - 0) ^ 2 := by
      funext x
      exact integral_dirac' (fun y => (x - y) ^ 2) 0 (by fun_prop)
    rw [hinnerB, integral_dirac' _ 0 (by fun_prop)]
    norm_num
  have hC : ∫ x, ∫ y, (x - y) ^ 2 ∂(twoPointMixture (-1 : ℝ) 1) ∂(twoPointMixture (-1 : ℝ) 1) = 2 := by
    rw [hinner, integral_twoPointMixture (-1) 1
      (fun x => ((x - (-1)) ^ 2 + (x - 1) ^ 2) / 2) (by fun_prop)]
    norm_num
  unfold energySqDistFunctional
  rw [hA, hB, hC]
  norm_num

end EnergyStatistics

/-! ### E1: refutation of the naive zero-iff over a bare `PseudoMetricSpace`

The retired `energy_distance_zero_iff` claimed `𝓔²(μ,ν) = 0 ↔ μ = ν` for *every*
`PseudoMetricSpace` with no negative-type hypothesis. That is false: on a two-point carrier
equipped with the identically-zero pseudometric, `𝓔²` vanishes identically, yet distinct point
masses are distinct measures. The honest restatement is `energy_distance_zero_iff_of_strong`
(under `StrongDistNegativeType`); this section certifies the refutation so the false schema is
not silently re-introduced.

Reference: Székely & Rizzo (2023), §3.1 (negative type is required); Lyons (2013), §3. -/

namespace EnergyStatistics

open MeasureTheory

/-- Two-point carrier (`Bool`) with the identically-zero pseudometric and the top measurable
    structure — the witness that refutes the naive zero-iff. -/
def ZeroTwoPoint : Type := Bool

instance : MeasurableSpace ZeroTwoPoint := ⊤

instance : MeasurableSingletonClass ZeroTwoPoint := ⟨fun _ => trivial⟩

noncomputable instance : PseudoMetricSpace ZeroTwoPoint where
  dist _ _ := 0
  dist_self _ := rfl
  dist_comm _ _ := rfl
  dist_triangle _ _ _ := by norm_num

/-- **E1 — refutation of the naive zero-iff.** The universally-quantified characterization
    `𝓔²(μ,ν) = 0 ↔ μ = ν` over an arbitrary `PseudoMetricSpace` (no negative-type hypothesis) is
    false: the two-point carrier `ZeroTwoPoint` with the zero pseudometric makes `𝓔²` vanish for
    the distinct point masses `δ_false ≠ δ_true`. This is why `energy_distance_zero_iff_of_strong`
    carries the `StrongDistNegativeType` hypothesis; the naive form is a landmine and has been
    removed. -/
theorem energy_distance_zero_iff_refuted :
    ¬ ∀ (α : Type) [MeasurableSpace α] [PseudoMetricSpace α] (μ ν : ProbabilityMeasure α),
        energyDistanceSq μ ν = 0 ↔ μ = ν := by
  intro h
  have hdist : ∀ x y : ZeroTwoPoint, dist x y = 0 := fun _ _ => rfl
  refine absurd
    ((h ZeroTwoPoint ⟨Measure.dirac false, by simp⟩ ⟨Measure.dirac true, by simp⟩).mp ?_) ?_
  · -- `𝓔² = 0`: the pseudometric is identically zero, so every distance integral vanishes.
    unfold energyDistanceSq
    simp_rw [hdist]
    simp
  · -- `δ_false ≠ δ_true`: they disagree on the (measurable) singleton `{false}`.
    intro hEq
    have hm : (Measure.dirac (false : ZeroTwoPoint)) = Measure.dirac true :=
      congrArg ProbabilityMeasure.measure hEq
    have h1 : (Measure.dirac (false : ZeroTwoPoint)) {false} = 1 := by
      rw [Measure.dirac_apply' _ (measurableSet_singleton false)]; simp
    rw [hm, Measure.dirac_apply' _ (measurableSet_singleton false)] at h1
    simp at h1

end EnergyStatistics
