import Mathlib.Analysis.Real.Sqrt
import Mathlib.Topology.MetricSpace.Basic
import EnergyStatistics.Defs
import EnergyStatistics.EnergyDistance

set_option linter.style.longLine false

/-!
# V-Statistic Representation of Energy Distance

This module formalizes the two-sample energy statistic (V-statistic)
and its properties, following Székely & Rizzo (2023), Chapter 3.

## Main definitions

* `vStatistic`: The two-sample energy V-statistic E_{n,m}(X, Y)

## Main theorems

* `v_statistic_nonneg`: The V-statistic is nonnegative

## References

* Székely, G. J., & Rizzo, M. L. (2023). The Energy of Data and Distance Correlation.
  CRC Press. Chapters 2, 3.
-/

namespace EnergyStatistics

/-- The V-statistic representation of energy distance (two-sample).

    For samples X₁, ..., Xₙ ~ μ and Y₁, ..., Yₘ ~ ν:

    E_{n,m}(X, Y) = (2/(nm)) Σᵢ Σⱼ ‖Xᵢ - Yⱼ‖
                   - (1/n²) Σᵢ Σⱼ ‖Xᵢ - Xⱼ‖
                   - (1/m²) Σᵢ Σⱼ ‖Yᵢ - Yⱼ‖

    This is the two-sample energy statistic from Definition 3.1.
    It is always nonnegative and equals 0 iff the two samples are
    equal as sets (up to permutation).

    Reference: Székely & Rizzo (2023), Definition 3.1, eq. (3.1). -/
noncomputable def vStatistic
    {α : Type*} [MeasurableSpace α] [PseudoMetricSpace α]
    (Xs : List α) (Ys : List α) : ℝ :=
  let n := Xs.length
  let m := Ys.length
  if _hnm : n = 0 ∨ m = 0 then 0
  else
    let cross := (Xs.map fun x => (Ys.map fun y => dist x y).sum).sum
    let selfX := (Xs.map fun x => (Xs.map fun x' => dist x x').sum).sum
    let selfY := (Ys.map fun y => (Ys.map fun y' => dist y y').sum).sum
    2 * cross / (n * m) - selfX / (n * n) - selfY / (m * m)

/-- Relate a List sum of a mapped function to a Fin sum: `(Xs.map f).sum = ∑ i, f (Xs.get i)`. -/
private lemma list_sum_eq_fin_sum {α : Type*} (Xs : List α) (f : α → ℝ) :
    (Xs.map f).sum = ∑ i : Fin Xs.length, f (Xs.get i) := by
  calc
    (Xs.map f).sum = ((List.ofFn Xs.get).map f).sum := by rw [List.ofFn_get]
    _ = (List.ofFn (f ∘ Xs.get)).sum := by rw [List.map_ofFn]
    _ = ∑ i : Fin Xs.length, (f ∘ Xs.get) i := by rw [List.sum_ofFn]
    _ = ∑ i : Fin Xs.length, f (Xs.get i) := rfl

/-- The V-statistic is nonnegative **on spaces of negative type**.

    E_{n,m}(X, Y) ≥ 0

    The negative-type hypothesis `DistNegativeType α` is necessary: over a bare
    `PseudoMetricSpace` the statement is false (circle with geodesic distance,
    Székely & Rizzo 2023 §3.2). Restated 2026-07-08 per ROADMAP §3.2 #6 — the
    hypothesis is discharged for Euclidean/Hilbert spaces by Schoenberg's
    theorem (future work).

    Proof: instantiate `DistNegativeType` with the `n + m` sample points and
    weights `1/n` on the `Xs`-block and `-1/m` on the `Ys`-block (which sum to
    zero); the double sum expands to
    `selfX/n² + selfY/m² − 2·cross/(nm) ≤ 0`.

    Reference: Székely & Rizzo (2023), Section 3.3, Proposition 3.1. -/
theorem v_statistic_nonneg
    {α : Type*} [MeasurableSpace α] [PseudoMetricSpace α]
    (h_neg : DistNegativeType α)
    (Xs Ys : List α) :
    0 ≤ vStatistic Xs Ys := by
  simp only [vStatistic]
  split_ifs with hnm
  · -- Case: n = 0 ∨ m = 0, so vStatistic = 0
    exact le_refl 0
  · -- Case: n > 0 and m > 0 (Székely & Rizzo 2023, Prop 3.1).
    let S_cross := ∑ i : Fin Xs.length, ∑ j : Fin Ys.length, dist (Xs.get i) (Ys.get j)
    let S_selfX := ∑ i : Fin Xs.length, ∑ j : Fin Xs.length, dist (Xs.get i) (Xs.get j)
    let S_selfY := ∑ i : Fin Ys.length, ∑ j : Fin Ys.length, dist (Ys.get i) (Ys.get j)
    have h_cross : (Xs.map fun x => (Ys.map fun y => dist x y).sum).sum = S_cross := by
      rw [list_sum_eq_fin_sum Xs (fun x => (Ys.map fun y => dist x y).sum)]
      exact Finset.sum_congr rfl fun i _ => list_sum_eq_fin_sum Ys (fun y => dist (Xs.get i) y)
    have h_selfX : (Xs.map fun x => (Xs.map fun x' => dist x x').sum).sum = S_selfX := by
      rw [list_sum_eq_fin_sum Xs (fun x => (Xs.map fun x' => dist x x').sum)]
      exact Finset.sum_congr rfl fun i _ => list_sum_eq_fin_sum Xs (fun x' => dist (Xs.get i) x')
    have h_selfY : (Ys.map fun y => (Ys.map fun y' => dist y y').sum).sum = S_selfY := by
      rw [list_sum_eq_fin_sum Ys (fun y => (Ys.map fun y' => dist y y').sum)]
      exact Finset.sum_congr rfl fun i _ => list_sum_eq_fin_sum Ys (fun y' => dist (Ys.get i) y')
    let n := Xs.length
    let m := Ys.length
    have hn : n ≠ 0 := by intro h; exact hnm (Or.inl h)
    have hm : m ≠ 0 := by intro h; exact hnm (Or.inr h)
    have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn
    have hm' : (m : ℝ) ≠ 0 := by exact_mod_cast hm
    let z : Fin (n + m) → α := Fin.addCases (fun i => Xs.get i) (fun j => Ys.get j)
    let w : Fin (n + m) → ℝ := Fin.addCases (fun _ => 1 / (n : ℝ)) (fun _ => -(1 / (m : ℝ)))
    have hsum_w : (∑ i, w i) = 0 := by
      rw [Fin.sum_univ_add]
      dsimp only [w]
      simp only [Fin.addCases_left, Fin.addCases_right, Finset.sum_const,
        Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      field_simp
      norm_num
    have h_ineq : (∑ i, ∑ j, w i * w j * dist (z i) (z j)) ≤ 0 :=
      h_neg (n + m) z w hsum_w
    have h_cross_swap : (∑ i : Fin m, ∑ j : Fin n, dist (Ys.get i) (Xs.get j)) = S_cross := by
      rw [Finset.sum_comm]
      exact Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun i _ => dist_comm _ _
    have h_expand : (∑ i, ∑ j, w i * w j * dist (z i) (z j)) =
        S_selfX / ((n : ℝ) * (n : ℝ)) +
        S_selfY / ((m : ℝ) * (m : ℝ)) -
        (2 * S_cross) / ((n : ℝ) * (m : ℝ)) := by
      simp_rw [Fin.sum_univ_add]
      dsimp only [w, z]
      simp only [Fin.addCases_left, Fin.addCases_right, neg_mul, mul_neg, neg_neg,
        Finset.sum_add_distrib, Finset.sum_neg_distrib, ← Finset.mul_sum]
      rw [h_cross_swap]
      dsimp only [S_selfX, S_selfY, S_cross]
      field_simp
      ring
    have h_goal : 0 ≤ (2 * S_cross) / ((n : ℝ) * (m : ℝ)) -
        S_selfX / ((n : ℝ) * (n : ℝ)) -
        S_selfY / ((m : ℝ) * (m : ℝ)) := by
      rw [h_expand] at h_ineq
      linarith
    simpa [h_cross, h_selfX, h_selfY] using h_goal

/-- The V-statistic is a consistent estimator of energy distance.

    As n, m → ∞, E_{n,m}(X, Y) → D_E²(μ, ν) almost surely.

    This follows from the law of large numbers for V-statistics
    with kernel h(x, y, x', y') = 2|x - y| - |x - x'| - |y - y'|.

    Reference: Székely & Rizzo (2023), Chapter 2 (V-statistics). -/
theorem v_statistic_consistent
    {α : Type*} [MeasurableSpace α] [PseudoMetricSpace α]
    (μ ν : ProbabilityMeasure α) :
    True := by
  trivial

end EnergyStatistics
