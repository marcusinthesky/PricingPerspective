import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Positivity
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.MeasureTheory.Function.L2Space
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.Probability.Independence.Basic
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Analysis.Matrix.Order
import Mathlib.LinearAlgebra.Matrix.Hadamard
import Mathlib.Analysis.SpecificLimits.Basic
import EnergyStatistics.Defs
import EnergyStatistics.DistanceCovariance.Basic

set_option linter.style.longLine false

namespace EnergyStatistics

/-!
### The sampling route to `dcov_nonneg`

The remaining machinery integrates the discrete Schur inequality
(`dcov_schur_sum_nonneg`) over `n` iid draws from `π` and lets `n → ∞`.  The key
organizational device: rewrite the empirically double-centered raw-distance sum in terms
of the **population-centered** kernels `centeredDistFst`/`centeredDistSnd` (double
centering annihilates additive one-variable shifts, `dcov_centered_entry_shift`).  These
kernels are *degenerate* — `∫ centeredDistFst π (z, ·) dπ = 0` a.e.
(`integral_centeredDistFst_right_eq_zero`) — so all coincidence patterns of the resulting
V-statistic expectation with a "private" index vanish, and

  `E[F_n] = Σ₁ − Σ₂/n − Σ₂'/n + Σ₃/n²`,   `Σ₁ = Σ₂ = Σ₂' = n(n−1)·dCov² + n·C₀`,
  `Σ₃ = n·(C₀ + 2(n−1)·dCov² + (n−1)·DX·DY)`,

where `C₀ = ∫ A(z,z)B(z,z) dπ` and `DX, DY` are the mean marginal distances.  Pointwise
`0 ≤ F_n`, so `0 ≤ n·E[F_n] = n(n−1)(n−2)·dCov² + O(n²)`, and `dCov² ≥ 0` follows for
large `n`.

Reference: Székely & Rizzo (2023), Ch. 12; Lyons (2013), Thm 3.20.
-/

section DcovSampling

open MeasureTheory ProbabilityTheory

/-- Transport of an integrand through a distribution identity `P.map φ = Q`:
    integrability and the value of the integral both descend from `Q` to `P`. -/
private lemma dcov_transport {Ω γ : Type*} [MeasurableSpace Ω] [MeasurableSpace γ]
    {P : Measure Ω} {Q : Measure γ} {φ : Ω → γ} (hφ : AEMeasurable φ P) (hmap : P.map φ = Q)
    {f : γ → ℝ} (hf : Integrable f Q) :
    Integrable (fun ω => f (φ ω)) P ∧ (∫ ω, f (φ ω) ∂P) = ∫ x, f x ∂Q := by
  subst hmap
  exact ⟨(integrable_map_measure hf.aestronglyMeasurable hφ).mp hf,
    (integral_map hφ hf.aestronglyMeasurable).symm⟩

/-- Pair marginal of an iid product: two distinct coordinates of `Measure.pi (fun _ => μ)`
    are jointly distributed as `μ ⊗ μ`. -/
private lemma dcov_pi_map_pair {γ : Type*} [MeasurableSpace γ] {μ : Measure γ}
    [IsProbabilityMeasure μ] {n : ℕ} {i j : Fin n} (hij : i ≠ j) :
    (Measure.pi fun _ : Fin n => μ).map (fun ω => (ω i, ω j)) = μ.prod μ := by
  have hindep : IndepFun (fun ω : Fin n → γ => ω i) (fun ω : Fin n → γ => ω j)
      (Measure.pi fun _ : Fin n => μ) :=
    (iIndepFun_pi (X := fun _ : Fin n => id) fun _ => aemeasurable_id).indepFun hij
  rw [hindep.map_prod_eq_prod_map_map (measurable_pi_apply i).aemeasurable
    (measurable_pi_apply j).aemeasurable,
    (measurePreserving_eval _ i).map_eq, (measurePreserving_eval _ j).map_eq]

/-- Triple marginal of an iid product: three distinct coordinates are jointly
    distributed as `μ ⊗ (μ ⊗ μ)` (nested pairing `(ω i, (ω j, ω k))`). -/
private lemma dcov_pi_map_triple {γ : Type*} [MeasurableSpace γ] {μ : Measure γ}
    [IsProbabilityMeasure μ] {n : ℕ} {i j k : Fin n}
    (hij : i ≠ j) (hik : i ≠ k) (hjk : j ≠ k) :
    (Measure.pi fun _ : Fin n => μ).map (fun ω => (ω i, (ω j, ω k)))
      = μ.prod (μ.prod μ) := by
  have hindep : IndepFun (fun ω : Fin n → γ => ω i) (fun ω : Fin n → γ => (ω j, ω k))
      (Measure.pi fun _ : Fin n => μ) :=
    ((iIndepFun_pi (X := fun _ : Fin n => id) fun _ =>
        aemeasurable_id).indepFun_prodMk (fun r => measurable_pi_apply r)
      j k i hij.symm hik.symm).symm
  rw [hindep.map_prod_eq_prod_map_map (measurable_pi_apply i).aemeasurable
    (by fun_prop : Measurable fun ω : Fin n → γ => (ω j, ω k)).aemeasurable,
    (measurePreserving_eval _ i).map_eq, dcov_pi_map_pair hjk]

/-- Quadruple marginal of an iid product: four distinct coordinates are jointly
    distributed as `(μ ⊗ μ) ⊗ (μ ⊗ μ)` (pairing `((ω i, ω j), (ω k, ω l))`). -/
private lemma dcov_pi_map_quad {γ : Type*} [MeasurableSpace γ] {μ : Measure γ}
    [IsProbabilityMeasure μ] {n : ℕ} {i j k l : Fin n}
    (hij : i ≠ j) (hik : i ≠ k) (hil : i ≠ l) (hjk : j ≠ k) (hjl : j ≠ l) (hkl : k ≠ l) :
    (Measure.pi fun _ : Fin n => μ).map (fun ω => ((ω i, ω j), (ω k, ω l)))
      = (μ.prod μ).prod (μ.prod μ) := by
  have hindep : IndepFun (fun ω : Fin n → γ => (ω i, ω j)) (fun ω : Fin n → γ => (ω k, ω l))
      (Measure.pi fun _ : Fin n => μ) :=
    (iIndepFun_pi (X := fun _ : Fin n => id) fun _ =>
        aemeasurable_id).indepFun_prodMk_prodMk (fun r => measurable_pi_apply r)
      i j k l hik hil hjk hjl
  rw [hindep.map_prod_eq_prod_map_map
    (by fun_prop : Measurable fun ω : Fin n → γ => (ω i, ω j)).aemeasurable
    (by fun_prop : Measurable fun ω : Fin n → γ => (ω k, ω l)).aemeasurable,
    dcov_pi_map_pair hij, dcov_pi_map_pair hkl]

/-- The doubly-centered first-coordinate kernel is symmetric in its two arguments. -/
private lemma centeredDistFst_symm {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    [PseudoMetricSpace α] [PseudoMetricSpace β]
    (π : JointProbabilityMeasure α β) (z w : α × β) :
    centeredDistFst π (z, w) = centeredDistFst π (w, z) := by
  simp only [centeredDistFst]
  rw [dist_comm]
  ring

/-- The doubly-centered second-coordinate kernel is symmetric in its two arguments. -/
private lemma centeredDistSnd_symm {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    [PseudoMetricSpace α] [PseudoMetricSpace β]
    (π : JointProbabilityMeasure α β) (z w : α × β) :
    centeredDistSnd π (z, w) = centeredDistSnd π (w, z) := by
  simp only [centeredDistSnd]
  rw [dist_comm]
  ring

/-- Empirical double centering is invariant under additive one-variable shifts
    `m i j ↦ m i j - u i - u j + c` of the matrix entries. -/
private lemma dcov_centered_entry_shift {n : ℕ} (hn : (n : ℝ) ≠ 0)
    (m : Fin n → Fin n → ℝ) (u : Fin n → ℝ) (c : ℝ) (i j : Fin n) :
    m i j - (∑ k, m i k) / n - (∑ k, m k j) / n + (∑ k, ∑ l, m k l) / (n ^ 2)
      = (m i j - u i - u j + c) - (∑ k, (m i k - u i - u k + c)) / n
        - (∑ k, (m k j - u k - u j + c)) / n
        + (∑ k, ∑ l, (m k l - u k - u l + c)) / (n ^ 2) := by
  have h1 : (∑ k, (m i k - u i - u k + c))
      = (∑ k, m i k) - (n : ℝ) * u i - (∑ k, u k) + (n : ℝ) * c := by
    rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.sum_sub_distrib,
      Finset.sum_const, Finset.sum_const]
    simp only [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have h2 : (∑ k, (m k j - u k - u j + c))
      = (∑ k, m k j) - (∑ k, u k) - (n : ℝ) * u j + (n : ℝ) * c := by
    rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.sum_sub_distrib,
      Finset.sum_const, Finset.sum_const]
    simp only [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have h3 : (∑ k, ∑ l, (m k l - u k - u l + c))
      = (∑ k, ∑ l, m k l) - (n : ℝ) * (∑ k, u k) - (n : ℝ) * (∑ k, u k)
        + (n : ℝ) * ((n : ℝ) * c) := by
    have hrow : ∀ k : Fin n, (∑ l, (m k l - u k - u l + c))
        = (∑ l, m k l) - (n : ℝ) * u k - (∑ l, u l) + (n : ℝ) * c := by
      intro k
      rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.sum_sub_distrib,
        Finset.sum_const, Finset.sum_const]
      simp only [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    simp only [hrow]
    rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.sum_sub_distrib,
      Finset.sum_const, Finset.sum_const, ← Finset.mul_sum]
    simp only [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  rw [h1, h2, h3]
  field_simp
  ring

/-- Row sums of an empirically double-centered matrix vanish. -/
private lemma dcov_centered_row_sum {n : ℕ} (hn : (n : ℝ) ≠ 0)
    (a : Fin n → Fin n → ℝ) (i : Fin n) :
    (∑ j, (a i j - (∑ k, a i k) / n - (∑ k, a k j) / n + (∑ k, ∑ l, a k l) / (n ^ 2)))
      = 0 := by
  rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.sum_sub_distrib,
    Finset.sum_const, Finset.sum_const]
  have hswap : (∑ j, (∑ k, a k j) / (n : ℝ)) = (∑ k, ∑ l, a k l) / (n : ℝ) := by
    rw [← Finset.sum_div, Finset.sum_comm]
  rw [hswap]
  simp only [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  field_simp
  ring

/-- Column sums of an empirically double-centered matrix vanish. -/
private lemma dcov_centered_col_sum {n : ℕ} (hn : (n : ℝ) ≠ 0)
    (a : Fin n → Fin n → ℝ) (j : Fin n) :
    (∑ i, (a i j - (∑ k, a i k) / n - (∑ k, a k j) / n + (∑ k, ∑ l, a k l) / (n ^ 2)))
      = 0 := by
  rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.sum_sub_distrib,
    Finset.sum_const, Finset.sum_const]
  have hswap : (∑ i, (∑ k, a i k) / (n : ℝ)) = (∑ k, ∑ l, a k l) / (n : ℝ) := by
    rw [← Finset.sum_div]
  rw [hswap]
  simp only [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  field_simp
  ring

/-- Expansion of the doubly-centered Hadamard sum: since centering is an orthogonal
    projection (row and column sums of a centered matrix vanish), the centered-times-
    centered sum collapses to centered-times-raw and then expands into four raw sums. -/
private lemma dcov_centered_hadamard_expand {n : ℕ} (hn : (n : ℝ) ≠ 0)
    (a b : Fin n → Fin n → ℝ) :
    (∑ i, ∑ j,
        (a i j - (∑ k, a i k) / n - (∑ k, a k j) / n + (∑ k, ∑ l, a k l) / (n ^ 2))
        * (b i j - (∑ k, b i k) / n - (∑ k, b k j) / n + (∑ k, ∑ l, b k l) / (n ^ 2)))
      = (∑ i, ∑ j, a i j * b i j)
        - (∑ i, ∑ j, ∑ k, a i k * b i j) / n
        - (∑ i, ∑ j, ∑ k, a k j * b i j) / n
        + (∑ i, ∑ k, ∑ j, ∑ l, a i j * b k l) / (n ^ 2) := by
  set A' : Fin n → Fin n → ℝ := fun i j =>
    a i j - (∑ k, a i k) / n - (∑ k, a k j) / n + (∑ k, ∑ l, a k l) / (n ^ 2) with hA'
  -- Step 1: kill the centering of `b` against the vanishing row/column sums of `A'`.
  have hstep1 : (∑ i, ∑ j, A' i j
        * (b i j - (∑ k, b i k) / n - (∑ k, b k j) / n + (∑ k, ∑ l, b k l) / (n ^ 2)))
      = ∑ i, ∑ j, A' i j * b i j := by
    have hexp : ∀ i j, A' i j
        * (b i j - (∑ k, b i k) / n - (∑ k, b k j) / n + (∑ k, ∑ l, b k l) / (n ^ 2))
        = A' i j * b i j - A' i j * ((∑ k, b i k) / n) - A' i j * ((∑ k, b k j) / n)
          + A' i j * ((∑ k, ∑ l, b k l) / (n ^ 2)) := by
      intro i j; ring
    simp only [hexp, Finset.sum_add_distrib, Finset.sum_sub_distrib]
    have h2 : (∑ i, ∑ j, A' i j * ((∑ k, b i k) / n)) = 0 := by
      refine Finset.sum_eq_zero fun i _ => ?_
      rw [← Finset.sum_mul, dcov_centered_row_sum hn a i, zero_mul]
    have h3 : (∑ i, ∑ j, A' i j * ((∑ k, b k j) / n)) = 0 := by
      rw [Finset.sum_comm]
      refine Finset.sum_eq_zero fun j _ => ?_
      rw [← Finset.sum_mul, dcov_centered_col_sum hn a j, zero_mul]
    have h4 : (∑ i, ∑ j, A' i j * ((∑ k, ∑ l, b k l) / (n ^ 2))) = 0 := by
      refine Finset.sum_eq_zero fun i _ => ?_
      rw [← Finset.sum_mul, dcov_centered_row_sum hn a i, zero_mul]
    rw [h2, h3, h4]
    ring
  rw [hstep1]
  -- Step 2: expand the centering of `a` against the raw `b`.
  have hexp2 : ∀ i j : Fin n, A' i j * b i j
      = a i j * b i j - (∑ k, a i k * b i j) / n - (∑ k, a k j * b i j) / n
        + ((∑ k, ∑ l, a k l) / (n ^ 2)) * b i j := by
    intro i j
    simp only [hA']
    rw [← Finset.sum_mul, ← Finset.sum_mul]
    ring
  simp only [hexp2, Finset.sum_add_distrib, Finset.sum_sub_distrib]
  have e2 : (∑ i, ∑ j, (∑ k, a i k * b i j) / n)
      = (∑ i, ∑ j, ∑ k, a i k * b i j) / n := by
    simp only [← Finset.sum_div]
  have e3 : (∑ i, ∑ j, (∑ k, a k j * b i j) / n)
      = (∑ i, ∑ j, ∑ k, a k j * b i j) / n := by
    simp only [← Finset.sum_div]
  have e4 : (∑ i, ∑ j, ((∑ k, ∑ l, a k l) / (n ^ 2)) * b i j)
      = (∑ i, ∑ k, ∑ j, ∑ l, a i j * b k l) / (n ^ 2) := by
    have hconst : (∑ i, ∑ j, ((∑ k, ∑ l, a k l) / (n ^ 2)) * b i j)
        = ((∑ k, ∑ l, a k l) / (n ^ 2)) * (∑ i, ∑ j, b i j) := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [Finset.mul_sum]
    rw [hconst, div_mul_eq_mul_div]
    refine congrArg (· / ((n : ℝ) ^ 2)) ?_
    rw [Finset.sum_mul_sum]
    refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun k _ => ?_
    rw [Finset.sum_mul_sum]
  rw [e2, e3, e4]

variable {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
  [PseudoMetricSpace α] [PseudoMetricSpace β]

/-- Degeneracy of the doubly-centered first-coordinate kernel: integrating out the second
    argument gives zero, for a.e. first argument. -/
private lemma integral_centeredDistFst_right_eq_zero
    (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π) :
    ∀ᵐ z ∂π.measure, (∫ w, centeredDistFst π (z, w) ∂π.measure) = 0 := by
  have hg_int : Integrable (fun p : α × β => ∫ q, dist p.1 q.1 ∂π.measure) π.measure :=
    hm.fst.integral_prod_left
  filter_upwards [hm.fst.prod_right_ae] with z hz
  have hz' : Integrable (fun w : α × β => dist z.1 w.1) π.measure := hz
  have e1 : (∫ w, centeredDistFst π (z, w) ∂π.measure)
      = ∫ w, (dist z.1 w.1 - (∫ q, dist z.1 q.1 ∂π.measure)
          - (∫ q, dist w.1 q.1 ∂π.measure)
          + ∫ p, ∫ q, dist p.1 q.1 ∂π.measure ∂π.measure) ∂π.measure := rfl
  have e2 : (∫ w, (dist z.1 w.1 - (∫ q, dist z.1 q.1 ∂π.measure)
        - (∫ q, dist w.1 q.1 ∂π.measure)
        + ∫ p, ∫ q, dist p.1 q.1 ∂π.measure ∂π.measure) ∂π.measure)
      = (∫ w, (dist z.1 w.1 - (∫ q, dist z.1 q.1 ∂π.measure)
          - (∫ q, dist w.1 q.1 ∂π.measure)) ∂π.measure)
        + ∫ _w, (∫ p, ∫ q, dist p.1 q.1 ∂π.measure ∂π.measure) ∂π.measure :=
    integral_add ((hz'.sub (integrable_const _)).sub hg_int) (integrable_const _)
  have e3 : (∫ w, (dist z.1 w.1 - (∫ q, dist z.1 q.1 ∂π.measure)
        - (∫ q, dist w.1 q.1 ∂π.measure)) ∂π.measure)
      = (∫ w, (dist z.1 w.1 - (∫ q, dist z.1 q.1 ∂π.measure)) ∂π.measure)
        - ∫ w, (∫ q, dist w.1 q.1 ∂π.measure) ∂π.measure :=
    integral_sub (hz'.sub (integrable_const _)) hg_int
  have e4 : (∫ w, (dist z.1 w.1 - (∫ q, dist z.1 q.1 ∂π.measure)) ∂π.measure)
      = (∫ w, dist z.1 w.1 ∂π.measure) - ∫ _w, (∫ q, dist z.1 q.1 ∂π.measure) ∂π.measure :=
    integral_sub hz' (integrable_const _)
  rw [e1, e2, e3, e4, integral_const, integral_const]
  simp only [probReal_univ, one_smul]
  ring

/-- Degeneracy of the doubly-centered second-coordinate kernel (mirror). -/
private lemma integral_centeredDistSnd_right_eq_zero
    (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π) :
    ∀ᵐ z ∂π.measure, (∫ w, centeredDistSnd π (z, w) ∂π.measure) = 0 := by
  have hg_int : Integrable (fun p : α × β => ∫ q, dist p.2 q.2 ∂π.measure) π.measure :=
    hm.snd.integral_prod_left
  filter_upwards [hm.snd.prod_right_ae] with z hz
  have hz' : Integrable (fun w : α × β => dist z.2 w.2) π.measure := hz
  have e1 : (∫ w, centeredDistSnd π (z, w) ∂π.measure)
      = ∫ w, (dist z.2 w.2 - (∫ q, dist z.2 q.2 ∂π.measure)
          - (∫ q, dist w.2 q.2 ∂π.measure)
          + ∫ p, ∫ q, dist p.2 q.2 ∂π.measure ∂π.measure) ∂π.measure := rfl
  have e2 : (∫ w, (dist z.2 w.2 - (∫ q, dist z.2 q.2 ∂π.measure)
        - (∫ q, dist w.2 q.2 ∂π.measure)
        + ∫ p, ∫ q, dist p.2 q.2 ∂π.measure ∂π.measure) ∂π.measure)
      = (∫ w, (dist z.2 w.2 - (∫ q, dist z.2 q.2 ∂π.measure)
          - (∫ q, dist w.2 q.2 ∂π.measure)) ∂π.measure)
        + ∫ _w, (∫ p, ∫ q, dist p.2 q.2 ∂π.measure ∂π.measure) ∂π.measure :=
    integral_add ((hz'.sub (integrable_const _)).sub hg_int) (integrable_const _)
  have e3 : (∫ w, (dist z.2 w.2 - (∫ q, dist z.2 q.2 ∂π.measure)
        - (∫ q, dist w.2 q.2 ∂π.measure)) ∂π.measure)
      = (∫ w, (dist z.2 w.2 - (∫ q, dist z.2 q.2 ∂π.measure)) ∂π.measure)
        - ∫ w, (∫ q, dist w.2 q.2 ∂π.measure) ∂π.measure :=
    integral_sub (hz'.sub (integrable_const _)) hg_int
  have e4 : (∫ w, (dist z.2 w.2 - (∫ q, dist z.2 q.2 ∂π.measure)) ∂π.measure)
      = (∫ w, dist z.2 w.2 ∂π.measure) - ∫ _w, (∫ q, dist z.2 q.2 ∂π.measure) ∂π.measure :=
    integral_sub hz' (integrable_const _)
  rw [e1, e2, e3, e4, integral_const, integral_const]
  simp only [probReal_univ, one_smul]
  ring

/-- Total degeneracy: the doubly-centered first-coordinate kernel integrates to zero
    over `π ⊗ π`. -/
private lemma integral_centeredDistFst_prod_eq_zero
    (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π) :
    (∫ z, centeredDistFst π z ∂(π.measure.prod π.measure)) = 0 := by
  have hg_int : Integrable (fun p : α × β => ∫ q, dist p.1 q.1 ∂π.measure) π.measure :=
    hm.fst.integral_prod_left
  have hga : Integrable (fun z : (α × β) × (α × β) => ∫ q, dist z.1.1 q.1 ∂π.measure)
      (π.measure.prod π.measure) := hg_int.comp_fst π.measure
  have hgb : Integrable (fun z : (α × β) × (α × β) => ∫ q, dist z.2.1 q.1 ∂π.measure)
      (π.measure.prod π.measure) := hg_int.comp_snd π.measure
  have hcong : (∫ z, centeredDistFst π z ∂(π.measure.prod π.measure))
      = ∫ z, (dist z.1.1 z.2.1 - (∫ q, dist z.1.1 q.1 ∂π.measure)
          - (∫ q, dist z.2.1 q.1 ∂π.measure)
          + ∫ p, ∫ q, dist p.1 q.1 ∂π.measure ∂π.measure) ∂(π.measure.prod π.measure) :=
    integral_congr_ae (Filter.Eventually.of_forall fun z => by simp only [centeredDistFst])
  have e1 : (∫ z, (dist z.1.1 z.2.1 - (∫ q, dist z.1.1 q.1 ∂π.measure)
        - (∫ q, dist z.2.1 q.1 ∂π.measure)
        + ∫ p, ∫ q, dist p.1 q.1 ∂π.measure ∂π.measure) ∂(π.measure.prod π.measure))
      = (∫ z, (dist z.1.1 z.2.1 - (∫ q, dist z.1.1 q.1 ∂π.measure)
          - (∫ q, dist z.2.1 q.1 ∂π.measure)) ∂(π.measure.prod π.measure))
        + ∫ _z, (∫ p, ∫ q, dist p.1 q.1 ∂π.measure ∂π.measure)
            ∂(π.measure.prod π.measure) :=
    integral_add ((hm.fst.sub hga).sub hgb) (integrable_const _)
  have e2 : (∫ z, (dist z.1.1 z.2.1 - (∫ q, dist z.1.1 q.1 ∂π.measure)
        - (∫ q, dist z.2.1 q.1 ∂π.measure)) ∂(π.measure.prod π.measure))
      = (∫ z, (dist z.1.1 z.2.1 - (∫ q, dist z.1.1 q.1 ∂π.measure))
            ∂(π.measure.prod π.measure))
        - ∫ z, (∫ q, dist z.2.1 q.1 ∂π.measure) ∂(π.measure.prod π.measure) :=
    integral_sub (hm.fst.sub hga) hgb
  have e3 : (∫ z, (dist z.1.1 z.2.1 - (∫ q, dist z.1.1 q.1 ∂π.measure))
        ∂(π.measure.prod π.measure))
      = (∫ z, dist z.1.1 z.2.1 ∂(π.measure.prod π.measure))
        - ∫ z, (∫ q, dist z.1.1 q.1 ∂π.measure) ∂(π.measure.prod π.measure) :=
    integral_sub hm.fst hga
  rw [hcong, e1, e2, e3, integral_const]
  have h1 : (∫ z, dist z.1.1 z.2.1 ∂(π.measure.prod π.measure))
      = ∫ p, ∫ q, dist p.1 q.1 ∂π.measure ∂π.measure := integral_prod _ hm.fst
  have h2 : (∫ z, (∫ q, dist z.1.1 q.1 ∂π.measure) ∂(π.measure.prod π.measure))
      = ∫ p, ∫ q, dist p.1 q.1 ∂π.measure ∂π.measure := by
    have h := integral_fun_fst (μ := π.measure) (ν := π.measure)
      (fun p : α × β => ∫ q, dist p.1 q.1 ∂π.measure)
    rw [h]; simp only [probReal_univ, one_smul]
  have h3 : (∫ z, (∫ q, dist z.2.1 q.1 ∂π.measure) ∂(π.measure.prod π.measure))
      = ∫ p, ∫ q, dist p.1 q.1 ∂π.measure ∂π.measure := by
    have h := integral_fun_snd (μ := π.measure) (ν := π.measure)
      (fun p : α × β => ∫ q, dist p.1 q.1 ∂π.measure)
    rw [h]; simp only [probReal_univ, one_smul]
  rw [h1, h2, h3]
  simp only [probReal_univ, one_smul]
  ring

/-- Total degeneracy for the second-coordinate kernel (mirror). -/
private lemma integral_centeredDistSnd_prod_eq_zero
    (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π) :
    (∫ z, centeredDistSnd π z ∂(π.measure.prod π.measure)) = 0 := by
  have hg_int : Integrable (fun p : α × β => ∫ q, dist p.2 q.2 ∂π.measure) π.measure :=
    hm.snd.integral_prod_left
  have hga : Integrable (fun z : (α × β) × (α × β) => ∫ q, dist z.1.2 q.2 ∂π.measure)
      (π.measure.prod π.measure) := hg_int.comp_fst π.measure
  have hgb : Integrable (fun z : (α × β) × (α × β) => ∫ q, dist z.2.2 q.2 ∂π.measure)
      (π.measure.prod π.measure) := hg_int.comp_snd π.measure
  have hcong : (∫ z, centeredDistSnd π z ∂(π.measure.prod π.measure))
      = ∫ z, (dist z.1.2 z.2.2 - (∫ q, dist z.1.2 q.2 ∂π.measure)
          - (∫ q, dist z.2.2 q.2 ∂π.measure)
          + ∫ p, ∫ q, dist p.2 q.2 ∂π.measure ∂π.measure) ∂(π.measure.prod π.measure) :=
    integral_congr_ae (Filter.Eventually.of_forall fun z => by simp only [centeredDistSnd])
  have e1 : (∫ z, (dist z.1.2 z.2.2 - (∫ q, dist z.1.2 q.2 ∂π.measure)
        - (∫ q, dist z.2.2 q.2 ∂π.measure)
        + ∫ p, ∫ q, dist p.2 q.2 ∂π.measure ∂π.measure) ∂(π.measure.prod π.measure))
      = (∫ z, (dist z.1.2 z.2.2 - (∫ q, dist z.1.2 q.2 ∂π.measure)
          - (∫ q, dist z.2.2 q.2 ∂π.measure)) ∂(π.measure.prod π.measure))
        + ∫ _z, (∫ p, ∫ q, dist p.2 q.2 ∂π.measure ∂π.measure)
            ∂(π.measure.prod π.measure) :=
    integral_add ((hm.snd.sub hga).sub hgb) (integrable_const _)
  have e2 : (∫ z, (dist z.1.2 z.2.2 - (∫ q, dist z.1.2 q.2 ∂π.measure)
        - (∫ q, dist z.2.2 q.2 ∂π.measure)) ∂(π.measure.prod π.measure))
      = (∫ z, (dist z.1.2 z.2.2 - (∫ q, dist z.1.2 q.2 ∂π.measure))
            ∂(π.measure.prod π.measure))
        - ∫ z, (∫ q, dist z.2.2 q.2 ∂π.measure) ∂(π.measure.prod π.measure) :=
    integral_sub (hm.snd.sub hga) hgb
  have e3 : (∫ z, (dist z.1.2 z.2.2 - (∫ q, dist z.1.2 q.2 ∂π.measure))
        ∂(π.measure.prod π.measure))
      = (∫ z, dist z.1.2 z.2.2 ∂(π.measure.prod π.measure))
        - ∫ z, (∫ q, dist z.1.2 q.2 ∂π.measure) ∂(π.measure.prod π.measure) :=
    integral_sub hm.snd hga
  rw [hcong, e1, e2, e3, integral_const]
  have h1 : (∫ z, dist z.1.2 z.2.2 ∂(π.measure.prod π.measure))
      = ∫ p, ∫ q, dist p.2 q.2 ∂π.measure ∂π.measure := integral_prod _ hm.snd
  have h2 : (∫ z, (∫ q, dist z.1.2 q.2 ∂π.measure) ∂(π.measure.prod π.measure))
      = ∫ p, ∫ q, dist p.2 q.2 ∂π.measure ∂π.measure := by
    have h := integral_fun_fst (μ := π.measure) (ν := π.measure)
      (fun p : α × β => ∫ q, dist p.2 q.2 ∂π.measure)
    rw [h]; simp only [probReal_univ, one_smul]
  have h3 : (∫ z, (∫ q, dist z.2.2 q.2 ∂π.measure) ∂(π.measure.prod π.measure))
      = ∫ p, ∫ q, dist p.2 q.2 ∂π.measure ∂π.measure := by
    have h := integral_fun_snd (μ := π.measure) (ν := π.measure)
      (fun p : α × β => ∫ q, dist p.2 q.2 ∂π.measure)
    rw [h]; simp only [probReal_univ, one_smul]
  rw [h1, h2, h3]
  simp only [probReal_univ, one_smul]
  ring

/-- The diagonal of the doubly-centered first-coordinate kernel is square-integrable:
    `z ↦ centeredDistFst π (z, z) = DX − 2·gX z` with `gX ∈ L²(π)` by Jensen. -/
private lemma memLp_two_centeredDistFst_diag
    (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π) :
    MemLp (fun z : α × β => centeredDistFst π (z, z)) 2 π.measure := by
  have hgX_int : Integrable (fun p : α × β => ∫ q, dist p.1 q.1 ∂π.measure) π.measure :=
    hm.fst.integral_prod_left
  have hgXsq_int : Integrable
      (fun p : α × β => (∫ q, dist p.1 q.1 ∂π.measure) ^ 2) π.measure := by
    have hbound : Integrable
        (fun p : α × β => ∫ q, dist p.1 q.1 ^ 2 ∂π.measure) π.measure :=
      hm.fstSq.integral_prod_left
    have hmeas : AEStronglyMeasurable
        (fun p : α × β => (∫ q, dist p.1 q.1 ∂π.measure) ^ 2) π.measure := by
      refine (hgX_int.aestronglyMeasurable.mul hgX_int.aestronglyMeasurable).congr ?_
      filter_upwards with p
      simp [pow_two]
    refine hbound.mono' hmeas ?_
    filter_upwards [hm.fst.prod_right_ae, hm.fstSq.prod_right_ae] with p hp1 hp2
    have hj := sq_integral_le (fun q : α × β => dist p.1 q.1) hp1 hp2
    calc ‖(∫ q, dist p.1 q.1 ∂π.measure) ^ 2‖
        = (∫ q, dist p.1 q.1 ∂π.measure) ^ 2 := by
          rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
      _ ≤ ∫ q, dist p.1 q.1 ^ 2 ∂π.measure := hj
  have hgX_mem : MemLp (fun p : α × β => ∫ q, dist p.1 q.1 ∂π.measure) 2 π.measure :=
    (memLp_two_iff_integrable_sq hgX_int.aestronglyMeasurable).mpr hgXsq_int
  have hEq : (fun z : α × β => centeredDistFst π (z, z))
      = fun z : α × β => (0 : ℝ) - (∫ q, dist z.1 q.1 ∂π.measure)
          - (∫ q, dist z.1 q.1 ∂π.measure)
          + ∫ p, ∫ q, dist p.1 q.1 ∂π.measure ∂π.measure := by
    funext z
    simp only [centeredDistFst, dist_self]
  rw [hEq]
  exact (((memLp_const 0).sub hgX_mem).sub hgX_mem).add (memLp_const _)

/-- The diagonal of the doubly-centered second-coordinate kernel is square-integrable. -/
private lemma memLp_two_centeredDistSnd_diag
    (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π) :
    MemLp (fun z : α × β => centeredDistSnd π (z, z)) 2 π.measure := by
  have hgY_int : Integrable (fun p : α × β => ∫ q, dist p.2 q.2 ∂π.measure) π.measure :=
    hm.snd.integral_prod_left
  have hgYsq_int : Integrable
      (fun p : α × β => (∫ q, dist p.2 q.2 ∂π.measure) ^ 2) π.measure := by
    have hbound : Integrable
        (fun p : α × β => ∫ q, dist p.2 q.2 ^ 2 ∂π.measure) π.measure :=
      hm.sndSq.integral_prod_left
    have hmeas : AEStronglyMeasurable
        (fun p : α × β => (∫ q, dist p.2 q.2 ∂π.measure) ^ 2) π.measure := by
      refine (hgY_int.aestronglyMeasurable.mul hgY_int.aestronglyMeasurable).congr ?_
      filter_upwards with p
      simp [pow_two]
    refine hbound.mono' hmeas ?_
    filter_upwards [hm.snd.prod_right_ae, hm.sndSq.prod_right_ae] with p hp1 hp2
    have hj := sq_integral_le (fun q : α × β => dist p.2 q.2) hp1 hp2
    calc ‖(∫ q, dist p.2 q.2 ∂π.measure) ^ 2‖
        = (∫ q, dist p.2 q.2 ∂π.measure) ^ 2 := by
          rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
      _ ≤ ∫ q, dist p.2 q.2 ^ 2 ∂π.measure := hj
  have hgY_mem : MemLp (fun p : α × β => ∫ q, dist p.2 q.2 ∂π.measure) 2 π.measure :=
    (memLp_two_iff_integrable_sq hgY_int.aestronglyMeasurable).mpr hgYsq_int
  have hEq : (fun z : α × β => centeredDistSnd π (z, z))
      = fun z : α × β => (0 : ℝ) - (∫ q, dist z.2 q.2 ∂π.measure)
          - (∫ q, dist z.2 q.2 ∂π.measure)
          + ∫ p, ∫ q, dist p.2 q.2 ∂π.measure ∂π.measure := by
    funext z
    simp only [centeredDistSnd, dist_self]
  rw [hEq]
  exact (((memLp_const 0).sub hgY_mem).sub hgY_mem).add (memLp_const _)

/-- Mean of the diagonal of the doubly-centered first-coordinate kernel:
    `∫ A(z,z) dπ = −DX`. -/
private lemma integral_centeredDistFst_diag
    (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π) :
    (∫ z, centeredDistFst π (z, z) ∂π.measure)
      = -(∫ p, ∫ q, dist p.1 q.1 ∂π.measure ∂π.measure) := by
  have hgX_int : Integrable (fun p : α × β => ∫ q, dist p.1 q.1 ∂π.measure) π.measure :=
    hm.fst.integral_prod_left
  have hcong : (∫ z, centeredDistFst π (z, z) ∂π.measure)
      = ∫ z, ((0 : ℝ) - (∫ q, dist z.1 q.1 ∂π.measure) - (∫ q, dist z.1 q.1 ∂π.measure)
          + ∫ p, ∫ q, dist p.1 q.1 ∂π.measure ∂π.measure) ∂π.measure :=
    integral_congr_ae (Filter.Eventually.of_forall fun z => by
      simp only [centeredDistFst, dist_self])
  have e1 : (∫ z, ((0 : ℝ) - (∫ q, dist z.1 q.1 ∂π.measure)
        - (∫ q, dist z.1 q.1 ∂π.measure)
        + ∫ p, ∫ q, dist p.1 q.1 ∂π.measure ∂π.measure) ∂π.measure)
      = (∫ z, ((0 : ℝ) - (∫ q, dist z.1 q.1 ∂π.measure)
          - (∫ q, dist z.1 q.1 ∂π.measure)) ∂π.measure)
        + ∫ _z, (∫ p, ∫ q, dist p.1 q.1 ∂π.measure ∂π.measure) ∂π.measure :=
    integral_add (((integrable_const 0).sub hgX_int).sub hgX_int) (integrable_const _)
  have e2 : (∫ z, ((0 : ℝ) - (∫ q, dist z.1 q.1 ∂π.measure)
        - (∫ q, dist z.1 q.1 ∂π.measure)) ∂π.measure)
      = (∫ z, ((0 : ℝ) - (∫ q, dist z.1 q.1 ∂π.measure)) ∂π.measure)
        - ∫ z, (∫ q, dist z.1 q.1 ∂π.measure) ∂π.measure :=
    integral_sub ((integrable_const 0).sub hgX_int) hgX_int
  have e3 : (∫ z, ((0 : ℝ) - (∫ q, dist z.1 q.1 ∂π.measure)) ∂π.measure)
      = (∫ _z, (0 : ℝ) ∂π.measure) - ∫ z, (∫ q, dist z.1 q.1 ∂π.measure) ∂π.measure :=
    integral_sub (integrable_const 0) hgX_int
  rw [hcong, e1, e2, e3, integral_const, integral_const]
  simp only [probReal_univ, one_smul, smul_zero]
  ring

/-- Mean of the diagonal of the doubly-centered second-coordinate kernel:
    `∫ B(z,z) dπ = −DY`. -/
private lemma integral_centeredDistSnd_diag
    (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π) :
    (∫ z, centeredDistSnd π (z, z) ∂π.measure)
      = -(∫ p, ∫ q, dist p.2 q.2 ∂π.measure ∂π.measure) := by
  have hgY_int : Integrable (fun p : α × β => ∫ q, dist p.2 q.2 ∂π.measure) π.measure :=
    hm.snd.integral_prod_left
  have hcong : (∫ z, centeredDistSnd π (z, z) ∂π.measure)
      = ∫ z, ((0 : ℝ) - (∫ q, dist z.2 q.2 ∂π.measure) - (∫ q, dist z.2 q.2 ∂π.measure)
          + ∫ p, ∫ q, dist p.2 q.2 ∂π.measure ∂π.measure) ∂π.measure :=
    integral_congr_ae (Filter.Eventually.of_forall fun z => by
      simp only [centeredDistSnd, dist_self])
  have e1 : (∫ z, ((0 : ℝ) - (∫ q, dist z.2 q.2 ∂π.measure)
        - (∫ q, dist z.2 q.2 ∂π.measure)
        + ∫ p, ∫ q, dist p.2 q.2 ∂π.measure ∂π.measure) ∂π.measure)
      = (∫ z, ((0 : ℝ) - (∫ q, dist z.2 q.2 ∂π.measure)
          - (∫ q, dist z.2 q.2 ∂π.measure)) ∂π.measure)
        + ∫ _z, (∫ p, ∫ q, dist p.2 q.2 ∂π.measure ∂π.measure) ∂π.measure :=
    integral_add (((integrable_const 0).sub hgY_int).sub hgY_int) (integrable_const _)
  have e2 : (∫ z, ((0 : ℝ) - (∫ q, dist z.2 q.2 ∂π.measure)
        - (∫ q, dist z.2 q.2 ∂π.measure)) ∂π.measure)
      = (∫ z, ((0 : ℝ) - (∫ q, dist z.2 q.2 ∂π.measure)) ∂π.measure)
        - ∫ z, (∫ q, dist z.2 q.2 ∂π.measure) ∂π.measure :=
    integral_sub ((integrable_const 0).sub hgY_int) hgY_int
  have e3 : (∫ z, ((0 : ℝ) - (∫ q, dist z.2 q.2 ∂π.measure)) ∂π.measure)
      = (∫ _z, (0 : ℝ) ∂π.measure) - ∫ z, (∫ q, dist z.2 q.2 ∂π.measure) ∂π.measure :=
    integral_sub (integrable_const 0) hgY_int
  rw [hcong, e1, e2, e3, integral_const, integral_const]
  simp only [probReal_univ, one_smul, smul_zero]
  ring

/-!
#### Canonical pattern integrands

Every per-tuple expectation of the sampled Schur sum reduces (through the marginal maps
above) to one of nine canonical integrands over `π`, `π ⊗ π`, `π ⊗ (π ⊗ π)` or
`(π ⊗ π) ⊗ (π ⊗ π)`.  Each is integrable (L² × L² via Hölder, pulling `MemLp` back
through measure-preserving projections), and each evaluates by Fubini plus degeneracy.
-/

/-- `f₁`: the diagonal product `A(z,z)·B(z,z)` is integrable over `π`. -/
private lemma dcov_int_diagAB (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π) :
    Integrable (fun z : α × β => centeredDistFst π (z, z) * centeredDistSnd π (z, z))
      π.measure :=
  (memLp_two_centeredDistFst_diag π hm).integrable_mul (memLp_two_centeredDistSnd_diag π hm)

/-- `f₂`: the kernel product `A·B` is integrable over `π ⊗ π`. -/
private lemma dcov_int_AB (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π) :
    Integrable (fun z : (α × β) × (α × β) => centeredDistFst π z * centeredDistSnd π z)
      (π.measure.prod π.measure) :=
  (memLp_two_centeredDistFst π hm).integrable_mul (memLp_two_centeredDistSnd π hm)

/-- `f₃`: `A(x,x)·B(x,y)` is integrable over `π ⊗ π` and integrates to `0`
    (degeneracy of `B` in its second argument). -/
private lemma dcov_int_diagA_B (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π) :
    Integrable (fun p : (α × β) × (α × β) => centeredDistFst π (p.1, p.1) * centeredDistSnd π p)
      (π.measure.prod π.measure) := by
  have h1 : MemLp (fun p : (α × β) × (α × β) => centeredDistFst π (p.1, p.1)) 2
      (π.measure.prod π.measure) :=
    (memLp_two_centeredDistFst_diag π hm).comp_measurePreserving measurePreserving_fst
  exact h1.integrable_mul (memLp_two_centeredDistSnd π hm)

private lemma dcov_val_diagA_B (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π) :
    (∫ p, centeredDistFst π (p.1, p.1) * centeredDistSnd π p ∂(π.measure.prod π.measure))
      = 0 := by
  have hprod : (∫ p, centeredDistFst π (p.1, p.1) * centeredDistSnd π p
        ∂(π.measure.prod π.measure))
      = ∫ x, ∫ y, centeredDistFst π (x, x) * centeredDistSnd π (x, y) ∂π.measure ∂π.measure :=
    integral_prod _ (dcov_int_diagA_B π hm)
  have hinner : ∀ᵐ x ∂π.measure,
      (∫ y, centeredDistFst π (x, x) * centeredDistSnd π (x, y) ∂π.measure) = 0 := by
    filter_upwards [integral_centeredDistSnd_right_eq_zero π hm] with x hx
    rw [integral_const_mul, hx, mul_zero]
  have hzero : (∫ x, ∫ y, centeredDistFst π (x, x) * centeredDistSnd π (x, y)
        ∂π.measure ∂π.measure) = ∫ _x, (0 : ℝ) ∂π.measure :=
    integral_congr_ae hinner
  rw [hprod, hzero, integral_zero]

/-- `f₄`: `A(x,y)·B(x,x)` is integrable over `π ⊗ π` and integrates to `0`
    (degeneracy of `A` in its second argument). -/
private lemma dcov_int_A_diagB (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π) :
    Integrable (fun p : (α × β) × (α × β) => centeredDistFst π p * centeredDistSnd π (p.1, p.1))
      (π.measure.prod π.measure) := by
  have h2 : MemLp (fun p : (α × β) × (α × β) => centeredDistSnd π (p.1, p.1)) 2
      (π.measure.prod π.measure) :=
    (memLp_two_centeredDistSnd_diag π hm).comp_measurePreserving measurePreserving_fst
  exact (memLp_two_centeredDistFst π hm).integrable_mul h2

private lemma dcov_val_A_diagB (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π) :
    (∫ p, centeredDistFst π p * centeredDistSnd π (p.1, p.1) ∂(π.measure.prod π.measure))
      = 0 := by
  have hprod : (∫ p, centeredDistFst π p * centeredDistSnd π (p.1, p.1)
        ∂(π.measure.prod π.measure))
      = ∫ x, ∫ y, centeredDistFst π (x, y) * centeredDistSnd π (x, x) ∂π.measure ∂π.measure :=
    integral_prod _ (dcov_int_A_diagB π hm)
  have hinner : ∀ᵐ x ∂π.measure,
      (∫ y, centeredDistFst π (x, y) * centeredDistSnd π (x, x) ∂π.measure) = 0 := by
    filter_upwards [integral_centeredDistFst_right_eq_zero π hm] with x hx
    rw [integral_mul_const, hx, zero_mul]
  have hzero : (∫ x, ∫ y, centeredDistFst π (x, y) * centeredDistSnd π (x, x)
        ∂π.measure ∂π.measure) = ∫ _x, (0 : ℝ) ∂π.measure :=
    integral_congr_ae hinner
  rw [hprod, hzero, integral_zero]

/-- `f₅`: `A(x,y)·B(x,z)` is integrable over `π ⊗ (π ⊗ π)` and integrates to `0`. -/
private lemma dcov_int_triple (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π) :
    Integrable (fun q : (α × β) × ((α × β) × (α × β)) =>
        centeredDistFst π (q.1, q.2.1) * centeredDistSnd π (q.1, q.2.2))
      (π.measure.prod (π.measure.prod π.measure)) := by
  have h1 : MemLp (fun q : (α × β) × ((α × β) × (α × β)) =>
      centeredDistFst π (q.1, q.2.1)) 2 (π.measure.prod (π.measure.prod π.measure)) :=
    (memLp_two_centeredDistFst π hm).comp_measurePreserving
      ((MeasurePreserving.id π.measure).prod measurePreserving_fst)
  have h2 : MemLp (fun q : (α × β) × ((α × β) × (α × β)) =>
      centeredDistSnd π (q.1, q.2.2)) 2 (π.measure.prod (π.measure.prod π.measure)) :=
    (memLp_two_centeredDistSnd π hm).comp_measurePreserving
      ((MeasurePreserving.id π.measure).prod measurePreserving_snd)
  exact h1.integrable_mul h2

private lemma dcov_val_triple (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π) :
    (∫ q, centeredDistFst π (q.1, q.2.1) * centeredDistSnd π (q.1, q.2.2)
        ∂(π.measure.prod (π.measure.prod π.measure))) = 0 := by
  have hprod : (∫ q, centeredDistFst π (q.1, q.2.1) * centeredDistSnd π (q.1, q.2.2)
        ∂(π.measure.prod (π.measure.prod π.measure)))
      = ∫ x, ∫ yz, centeredDistFst π (x, yz.1) * centeredDistSnd π (x, yz.2)
          ∂(π.measure.prod π.measure) ∂π.measure :=
    integral_prod _ (dcov_int_triple π hm)
  have hinner : ∀ᵐ x ∂π.measure,
      (∫ yz, centeredDistFst π (x, yz.1) * centeredDistSnd π (x, yz.2)
        ∂(π.measure.prod π.measure)) = 0 := by
    filter_upwards [integral_centeredDistFst_right_eq_zero π hm] with x hx
    have h := integral_prod_mul (μ := π.measure) (ν := π.measure)
      (fun y => centeredDistFst π (x, y)) (fun z => centeredDistSnd π (x, z))
    rw [h, hx, zero_mul]
  have hzero : (∫ x, ∫ yz, centeredDistFst π (x, yz.1) * centeredDistSnd π (x, yz.2)
        ∂(π.measure.prod π.measure) ∂π.measure) = ∫ _x, (0 : ℝ) ∂π.measure :=
    integral_congr_ae hinner
  rw [hprod, hzero, integral_zero]

/-- `f₆`: `A(x,x)·B(y,y)` is integrable over `π ⊗ π` and integrates to `DX·DY`
    (product of the two diagonal means `(−DX)·(−DY)`). -/
private lemma dcov_int_diag_diag (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π) :
    Integrable (fun p : (α × β) × (α × β) =>
        centeredDistFst π (p.1, p.1) * centeredDistSnd π (p.2, p.2))
      (π.measure.prod π.measure) := by
  have h1 : MemLp (fun p : (α × β) × (α × β) => centeredDistFst π (p.1, p.1)) 2
      (π.measure.prod π.measure) :=
    (memLp_two_centeredDistFst_diag π hm).comp_measurePreserving measurePreserving_fst
  have h2 : MemLp (fun p : (α × β) × (α × β) => centeredDistSnd π (p.2, p.2)) 2
      (π.measure.prod π.measure) :=
    (memLp_two_centeredDistSnd_diag π hm).comp_measurePreserving measurePreserving_snd
  exact h1.integrable_mul h2

private lemma dcov_val_diag_diag (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π) :
    (∫ p, centeredDistFst π (p.1, p.1) * centeredDistSnd π (p.2, p.2)
        ∂(π.measure.prod π.measure))
      = (∫ p, ∫ q, dist p.1 q.1 ∂π.measure ∂π.measure)
        * (∫ p, ∫ q, dist p.2 q.2 ∂π.measure ∂π.measure) := by
  have h := integral_prod_mul (μ := π.measure) (ν := π.measure)
    (fun z : α × β => centeredDistFst π (z, z)) (fun z : α × β => centeredDistSnd π (z, z))
  rw [h, integral_centeredDistFst_diag π hm, integral_centeredDistSnd_diag π hm, neg_mul_neg]

/-- `f₇`: `A(x,x)·B(y,z)` is integrable over `π ⊗ (π ⊗ π)` and integrates to `0`. -/
private lemma dcov_int_diagA_pair (π : JointProbabilityMeasure α β)
    (hm : JointFiniteMoments π) :
    Integrable (fun q : (α × β) × ((α × β) × (α × β)) =>
        centeredDistFst π (q.1, q.1) * centeredDistSnd π q.2)
      (π.measure.prod (π.measure.prod π.measure)) := by
  have h1 : MemLp (fun q : (α × β) × ((α × β) × (α × β)) =>
      centeredDistFst π (q.1, q.1)) 2 (π.measure.prod (π.measure.prod π.measure)) :=
    (memLp_two_centeredDistFst_diag π hm).comp_measurePreserving measurePreserving_fst
  have h2 : MemLp (fun q : (α × β) × ((α × β) × (α × β)) =>
      centeredDistSnd π q.2) 2 (π.measure.prod (π.measure.prod π.measure)) :=
    (memLp_two_centeredDistSnd π hm).comp_measurePreserving measurePreserving_snd
  exact h1.integrable_mul h2

private lemma dcov_val_diagA_pair (π : JointProbabilityMeasure α β)
    (hm : JointFiniteMoments π) :
    (∫ q, centeredDistFst π (q.1, q.1) * centeredDistSnd π q.2
        ∂(π.measure.prod (π.measure.prod π.measure))) = 0 := by
  have h := integral_prod_mul (μ := π.measure) (ν := π.measure.prod π.measure)
    (fun z : α × β => centeredDistFst π (z, z))
    (fun w : (α × β) × (α × β) => centeredDistSnd π w)
  rw [h, integral_centeredDistSnd_prod_eq_zero π hm, mul_zero]

/-- `f₈`: `A(y,z)·B(x,x)` is integrable over `π ⊗ (π ⊗ π)` and integrates to `0`. -/
private lemma dcov_int_pair_diagB (π : JointProbabilityMeasure α β)
    (hm : JointFiniteMoments π) :
    Integrable (fun q : (α × β) × ((α × β) × (α × β)) =>
        centeredDistFst π q.2 * centeredDistSnd π (q.1, q.1))
      (π.measure.prod (π.measure.prod π.measure)) := by
  have h1 : MemLp (fun q : (α × β) × ((α × β) × (α × β)) =>
      centeredDistFst π q.2) 2 (π.measure.prod (π.measure.prod π.measure)) :=
    (memLp_two_centeredDistFst π hm).comp_measurePreserving measurePreserving_snd
  have h2 : MemLp (fun q : (α × β) × ((α × β) × (α × β)) =>
      centeredDistSnd π (q.1, q.1)) 2 (π.measure.prod (π.measure.prod π.measure)) :=
    (memLp_two_centeredDistSnd_diag π hm).comp_measurePreserving measurePreserving_fst
  exact h1.integrable_mul h2

private lemma dcov_val_pair_diagB (π : JointProbabilityMeasure α β)
    (hm : JointFiniteMoments π) :
    (∫ q, centeredDistFst π q.2 * centeredDistSnd π (q.1, q.1)
        ∂(π.measure.prod (π.measure.prod π.measure))) = 0 := by
  have hcomm : (∫ q, centeredDistFst π q.2 * centeredDistSnd π (q.1, q.1)
        ∂(π.measure.prod (π.measure.prod π.measure)))
      = ∫ q : (α × β) × ((α × β) × (α × β)),
          centeredDistSnd π (q.1, q.1) * centeredDistFst π q.2
          ∂(π.measure.prod (π.measure.prod π.measure)) :=
    integral_congr_ae (Filter.Eventually.of_forall fun q => by ring)
  have h := integral_prod_mul (μ := π.measure) (ν := π.measure.prod π.measure)
    (fun z : α × β => centeredDistSnd π (z, z))
    (fun w : (α × β) × (α × β) => centeredDistFst π w)
  rw [hcomm, h, integral_centeredDistFst_prod_eq_zero π hm, mul_zero]

/-- `f₉`: `A(x,y)·B(z,w)` is integrable over `(π ⊗ π) ⊗ (π ⊗ π)` and integrates to `0`. -/
private lemma dcov_int_quad (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π) :
    Integrable (fun q : ((α × β) × (α × β)) × ((α × β) × (α × β)) =>
        centeredDistFst π q.1 * centeredDistSnd π q.2)
      ((π.measure.prod π.measure).prod (π.measure.prod π.measure)) := by
  have h1 : MemLp (fun q : ((α × β) × (α × β)) × ((α × β) × (α × β)) =>
      centeredDistFst π q.1) 2
      ((π.measure.prod π.measure).prod (π.measure.prod π.measure)) :=
    (memLp_two_centeredDistFst π hm).comp_measurePreserving measurePreserving_fst
  have h2 : MemLp (fun q : ((α × β) × (α × β)) × ((α × β) × (α × β)) =>
      centeredDistSnd π q.2) 2
      ((π.measure.prod π.measure).prod (π.measure.prod π.measure)) :=
    (memLp_two_centeredDistSnd π hm).comp_measurePreserving measurePreserving_snd
  exact h1.integrable_mul h2

private lemma dcov_val_quad (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π) :
    (∫ q, centeredDistFst π q.1 * centeredDistSnd π q.2
        ∂((π.measure.prod π.measure).prod (π.measure.prod π.measure))) = 0 := by
  have h := integral_prod_mul (μ := π.measure.prod π.measure)
    (ν := π.measure.prod π.measure)
    (fun w : (α × β) × (α × β) => centeredDistFst π w)
    (fun w : (α × β) × (α × β) => centeredDistSnd π w)
  rw [h, integral_centeredDistFst_prod_eq_zero π hm, zero_mul]

/-!
#### Per-tuple expectations over the iid sample

For each index tuple, the expectation of the corresponding product of centered kernels
over `π^⊗ⁿ` is computed by transporting the canonical integrand through the marginal
maps.  Degeneracy kills every pattern with a private index; only the diagonal (`C₀`),
matched-pair (`dCov²`) and split-diagonal (`DX·DY`) patterns survive.
-/

/-- Expectation of `A(ωᵢ,ωⱼ)·B(ωᵢ,ωⱼ)` over the iid sample. -/
private lemma dcov_tuple_pair (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π)
    {n : ℕ} (i j : Fin n) :
    Integrable (fun ω : Fin n → α × β =>
        centeredDistFst π (ω i, ω j) * centeredDistSnd π (ω i, ω j))
      (Measure.pi fun _ : Fin n => π.measure)
    ∧ (∫ ω, centeredDistFst π (ω i, ω j) * centeredDistSnd π (ω i, ω j)
        ∂(Measure.pi fun _ : Fin n => π.measure))
      = if i = j
        then ∫ z, centeredDistFst π (z, z) * centeredDistSnd π (z, z) ∂π.measure
        else distanceCovarianceSq π := by
  by_cases hij : i = j
  · subst hij
    rw [if_pos rfl]
    obtain ⟨h1, h2⟩ := dcov_transport ((measurable_pi_apply i).aemeasurable)
      ((measurePreserving_eval (fun _ : Fin n => π.measure) i).map_eq) (dcov_int_diagAB π hm)
    exact ⟨h1, h2⟩
  · rw [if_neg hij]
    obtain ⟨h1, h2⟩ := dcov_transport (P := Measure.pi fun _ : Fin n => π.measure)
      (φ := fun ω => (ω i, ω j)) (by fun_prop) (dcov_pi_map_pair hij) (dcov_int_AB π hm)
    exact ⟨h1, h2.trans (dcov_eq_integral_mul π hm).symm⟩

/-- Expectation of `A(ωᵢ,ωₖ)·B(ωᵢ,ωⱼ)` over the iid sample (row-mean triple family). -/
private lemma dcov_tuple_rowA (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π)
    {n : ℕ} (i j k : Fin n) :
    Integrable (fun ω : Fin n → α × β =>
        centeredDistFst π (ω i, ω k) * centeredDistSnd π (ω i, ω j))
      (Measure.pi fun _ : Fin n => π.measure)
    ∧ (∫ ω, centeredDistFst π (ω i, ω k) * centeredDistSnd π (ω i, ω j)
        ∂(Measure.pi fun _ : Fin n => π.measure))
      = if k = j
        then (if i = j
          then ∫ z, centeredDistFst π (z, z) * centeredDistSnd π (z, z) ∂π.measure
          else distanceCovarianceSq π)
        else 0 := by
  by_cases hkj : k = j
  · subst hkj
    rw [if_pos rfl]
    exact dcov_tuple_pair π hm i k
  · rw [if_neg hkj]
    by_cases hik : i = k
    · -- `A(ωᵢ,ωᵢ)·B(ωᵢ,ωⱼ)`, `i ≠ j`
      subst hik
      obtain ⟨h1, h2⟩ := dcov_transport (P := Measure.pi fun _ : Fin n => π.measure)
        (φ := fun ω => (ω i, ω j)) (by fun_prop) (dcov_pi_map_pair hkj)
        (dcov_int_diagA_B π hm)
      exact ⟨h1, h2.trans (dcov_val_diagA_B π hm)⟩
    · by_cases hij : i = j
      · -- `A(ωᵢ,ωₖ)·B(ωᵢ,ωᵢ)`, `i ≠ k`
        subst hij
        obtain ⟨h1, h2⟩ := dcov_transport (P := Measure.pi fun _ : Fin n => π.measure)
          (φ := fun ω => (ω i, ω k)) (by fun_prop) (dcov_pi_map_pair hik)
          (dcov_int_A_diagB π hm)
        exact ⟨h1, h2.trans (dcov_val_A_diagB π hm)⟩
      · -- all three distinct: `A(ωᵢ,ωₖ)·B(ωᵢ,ωⱼ)` with private indices on both sides
        obtain ⟨h1, h2⟩ := dcov_transport (P := Measure.pi fun _ : Fin n => π.measure)
          (φ := fun ω => (ω i, (ω k, ω j))) (by fun_prop)
          (dcov_pi_map_triple hik hij hkj) (dcov_int_triple π hm)
        exact ⟨h1, h2.trans (dcov_val_triple π hm)⟩

/-- Expectation of `A(ωₖ,ωⱼ)·B(ωᵢ,ωⱼ)` over the iid sample (column-mean triple family). -/
private lemma dcov_tuple_colA (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π)
    {n : ℕ} (i j k : Fin n) :
    Integrable (fun ω : Fin n → α × β =>
        centeredDistFst π (ω k, ω j) * centeredDistSnd π (ω i, ω j))
      (Measure.pi fun _ : Fin n => π.measure)
    ∧ (∫ ω, centeredDistFst π (ω k, ω j) * centeredDistSnd π (ω i, ω j)
        ∂(Measure.pi fun _ : Fin n => π.measure))
      = if k = i
        then (if i = j
          then ∫ z, centeredDistFst π (z, z) * centeredDistSnd π (z, z) ∂π.measure
          else distanceCovarianceSq π)
        else 0 := by
  by_cases hki : k = i
  · subst hki
    rw [if_pos rfl]
    exact dcov_tuple_pair π hm k j
  · rw [if_neg hki]
    by_cases hjk : j = k
    · -- `A(ωⱼ,ωⱼ)·B(ωᵢ,ωⱼ)` with `j ≠ i`: symmetrize `B` and use the diag-`A` pattern
      subst hjk
      have hfun : (fun ω : Fin n → α × β =>
          centeredDistFst π (ω j, ω j) * centeredDistSnd π (ω i, ω j))
          = fun ω => centeredDistFst π (ω j, ω j) * centeredDistSnd π (ω j, ω i) := by
        funext ω
        rw [centeredDistSnd_symm π (ω i) (ω j)]
      rw [hfun]
      have hji : j ≠ i := hki
      obtain ⟨h1, h2⟩ := dcov_transport (P := Measure.pi fun _ : Fin n => π.measure)
        (φ := fun ω => (ω j, ω i)) (by fun_prop) (dcov_pi_map_pair hji)
        (dcov_int_diagA_B π hm)
      exact ⟨h1, h2.trans (dcov_val_diagA_B π hm)⟩
    · by_cases hji : j = i
      · -- `A(ωₖ,ωᵢ)·B(ωᵢ,ωᵢ)` with `i ≠ k`: symmetrize `A`
        subst hji
        have hfun : (fun ω : Fin n → α × β =>
            centeredDistFst π (ω k, ω j) * centeredDistSnd π (ω j, ω j))
            = fun ω => centeredDistFst π (ω j, ω k) * centeredDistSnd π (ω j, ω j) := by
          funext ω
          rw [centeredDistFst_symm π (ω k) (ω j)]
        rw [hfun]
        have hjk' : j ≠ k := fun h => hki h.symm
        obtain ⟨h1, h2⟩ := dcov_transport (P := Measure.pi fun _ : Fin n => π.measure)
          (φ := fun ω => (ω j, ω k)) (by fun_prop) (dcov_pi_map_pair hjk')
          (dcov_int_A_diagB π hm)
        exact ⟨h1, h2.trans (dcov_val_A_diagB π hm)⟩
      -- (fallthrough continues below)
      · -- all three distinct: symmetrize both kernels
        have hfun : (fun ω : Fin n → α × β =>
            centeredDistFst π (ω k, ω j) * centeredDistSnd π (ω i, ω j))
            = fun ω => centeredDistFst π (ω j, ω k) * centeredDistSnd π (ω j, ω i) := by
          funext ω
          rw [centeredDistFst_symm π (ω k) (ω j), centeredDistSnd_symm π (ω i) (ω j)]
        rw [hfun]
        have hjk' : j ≠ k := hjk
        obtain ⟨h1, h2⟩ := dcov_transport (P := Measure.pi fun _ : Fin n => π.measure)
          (φ := fun ω => (ω j, (ω k, ω i))) (by fun_prop)
          (dcov_pi_map_triple hjk' hji (fun h => hki h)) (dcov_int_triple π hm)
        exact ⟨h1, h2.trans (dcov_val_triple π hm)⟩

/-- Expectation of `A(ωᵢ,ωⱼ)·B(ωₖ,ωₗ)` over the iid sample (grand-mean quadruple family). -/
private lemma dcov_tuple_quad (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π)
    {n : ℕ} (i j k l : Fin n) :
    Integrable (fun ω : Fin n → α × β =>
        centeredDistFst π (ω i, ω j) * centeredDistSnd π (ω k, ω l))
      (Measure.pi fun _ : Fin n => π.measure)
    ∧ (∫ ω, centeredDistFst π (ω i, ω j) * centeredDistSnd π (ω k, ω l)
        ∂(Measure.pi fun _ : Fin n => π.measure))
      = if i = j
        then (if k = l
          then (if i = k
            then ∫ z, centeredDistFst π (z, z) * centeredDistSnd π (z, z) ∂π.measure
            else (∫ p, ∫ q, dist p.1 q.1 ∂π.measure ∂π.measure)
              * (∫ p, ∫ q, dist p.2 q.2 ∂π.measure ∂π.measure))
          else 0)
        else (if (k = i ∧ l = j) ∨ (k = j ∧ l = i) then distanceCovarianceSq π else 0) := by
  by_cases hij : i = j
  · subst hij
    rw [if_pos rfl]
    by_cases hkl : k = l
    · subst hkl
      rw [if_pos rfl]
      by_cases hik : i = k
      · subst hik
        rw [if_pos rfl]
        obtain ⟨h1, h2⟩ := dcov_transport ((measurable_pi_apply i).aemeasurable)
          ((measurePreserving_eval (fun _ : Fin n => π.measure) i).map_eq)
          (dcov_int_diagAB π hm)
        exact ⟨h1, h2⟩
      · rw [if_neg hik]
        obtain ⟨h1, h2⟩ := dcov_transport (P := Measure.pi fun _ : Fin n => π.measure)
          (φ := fun ω => (ω i, ω k)) (by fun_prop) (dcov_pi_map_pair hik)
          (dcov_int_diag_diag π hm)
        exact ⟨h1, h2.trans (dcov_val_diag_diag π hm)⟩
    · rw [if_neg hkl]
      by_cases hki : k = i
      · -- `A(ωᵢ,ωᵢ)·B(ωᵢ,ωₗ)`, `i ≠ l`
        subst hki
        have hkl' : k ≠ l := hkl
        obtain ⟨h1, h2⟩ := dcov_transport (P := Measure.pi fun _ : Fin n => π.measure)
          (φ := fun ω => (ω k, ω l)) (by fun_prop) (dcov_pi_map_pair hkl')
          (dcov_int_diagA_B π hm)
        exact ⟨h1, h2.trans (dcov_val_diagA_B π hm)⟩
      · by_cases hli : l = i
        · -- `A(ωᵢ,ωᵢ)·B(ωₖ,ωᵢ)`: symmetrize `B`
          subst hli
          have hfun : (fun ω : Fin n → α × β =>
              centeredDistFst π (ω l, ω l) * centeredDistSnd π (ω k, ω l))
              = fun ω => centeredDistFst π (ω l, ω l) * centeredDistSnd π (ω l, ω k) := by
            funext ω
            rw [centeredDistSnd_symm π (ω k) (ω l)]
          rw [hfun]
          have hlk : l ≠ k := fun h => hkl h.symm
          obtain ⟨h1, h2⟩ := dcov_transport (P := Measure.pi fun _ : Fin n => π.measure)
            (φ := fun ω => (ω l, ω k)) (by fun_prop) (dcov_pi_map_pair hlk)
            (dcov_int_diagA_B π hm)
          exact ⟨h1, h2.trans (dcov_val_diagA_B π hm)⟩
        · -- `A(ωᵢ,ωᵢ)·B(ωₖ,ωₗ)` with `i ∉ {k, l}`
          have hik' : i ≠ k := fun h => hki h.symm
          have hil' : i ≠ l := fun h => hli h.symm
          obtain ⟨h1, h2⟩ := dcov_transport (P := Measure.pi fun _ : Fin n => π.measure)
            (φ := fun ω => (ω i, (ω k, ω l))) (by fun_prop)
            (dcov_pi_map_triple hik' hil' hkl) (dcov_int_diagA_pair π hm)
          exact ⟨h1, h2.trans (dcov_val_diagA_pair π hm)⟩
  · rw [if_neg hij]
    by_cases hkl : k = l
    · subst hkl
      rw [if_neg (by rintro (⟨h1, h2⟩ | ⟨h1, h2⟩)
        <;> [exact hij (h1.symm.trans h2); exact hij (h2.symm.trans h1)])]
      by_cases hki : k = i
      · -- `A(ωᵢ,ωⱼ)·B(ωᵢ,ωᵢ)`
        subst hki
        obtain ⟨h1, h2⟩ := dcov_transport (P := Measure.pi fun _ : Fin n => π.measure)
          (φ := fun ω => (ω k, ω j)) (by fun_prop) (dcov_pi_map_pair hij)
          (dcov_int_A_diagB π hm)
        exact ⟨h1, h2.trans (dcov_val_A_diagB π hm)⟩
      · by_cases hkj : k = j
        · -- `A(ωᵢ,ωⱼ)·B(ωⱼ,ωⱼ)`: symmetrize `A`
          subst hkj
          have hfun : (fun ω : Fin n → α × β =>
              centeredDistFst π (ω i, ω k) * centeredDistSnd π (ω k, ω k))
              = fun ω => centeredDistFst π (ω k, ω i) * centeredDistSnd π (ω k, ω k) := by
            funext ω
            rw [centeredDistFst_symm π (ω i) (ω k)]
          rw [hfun]
          have hki' : k ≠ i := hki
          obtain ⟨h1, h2⟩ := dcov_transport (P := Measure.pi fun _ : Fin n => π.measure)
            (φ := fun ω => (ω k, ω i)) (by fun_prop) (dcov_pi_map_pair hki')
            (dcov_int_A_diagB π hm)
          exact ⟨h1, h2.trans (dcov_val_A_diagB π hm)⟩
        · -- `A(ωᵢ,ωⱼ)·B(ωₖ,ωₖ)` with `k ∉ {i, j}`
          obtain ⟨h1, h2⟩ := dcov_transport (P := Measure.pi fun _ : Fin n => π.measure)
            (φ := fun ω => (ω k, (ω i, ω j))) (by fun_prop)
            (dcov_pi_map_triple hki hkj hij) (dcov_int_pair_diagB π hm)
          exact ⟨h1, h2.trans (dcov_val_pair_diagB π hm)⟩
    · by_cases hki : k = i
      · by_cases hlj : l = j
        · -- matched pair `(k,l) = (i,j)`
          subst hki; subst hlj
          rw [if_pos (Or.inl ⟨rfl, rfl⟩)]
          obtain ⟨h1, h2⟩ := dcov_tuple_pair π hm k l
          exact ⟨h1, h2.trans (if_neg hij)⟩
        · -- `A(ωᵢ,ωⱼ)·B(ωᵢ,ωₗ)`, `l ∉ {i, j}`
          subst hki
          rw [if_neg (by rintro (⟨h1, h2⟩ | ⟨h1, h2⟩)
            <;> [exact hlj h2; exact hij h1])]
          have hkj : k ≠ j := fun h => hij h
          have hkl' : k ≠ l := hkl
          have hjl : j ≠ l := fun h => hlj h.symm
          obtain ⟨h1, h2⟩ := dcov_transport (P := Measure.pi fun _ : Fin n => π.measure)
            (φ := fun ω => (ω k, (ω j, ω l))) (by fun_prop)
            (dcov_pi_map_triple hkj hkl' hjl) (dcov_int_triple π hm)
          exact ⟨h1, h2.trans (dcov_val_triple π hm)⟩
      · by_cases hkj : k = j
        · by_cases hli : l = i
          · -- transposed pair `(k,l) = (j,i)`: symmetrize `B`
            subst hkj; subst hli
            rw [if_pos (Or.inr ⟨rfl, rfl⟩)]
            have hfun : (fun ω : Fin n → α × β =>
                centeredDistFst π (ω l, ω k) * centeredDistSnd π (ω k, ω l))
                = fun ω => centeredDistFst π (ω l, ω k) * centeredDistSnd π (ω l, ω k) := by
              funext ω
              rw [centeredDistSnd_symm π (ω k) (ω l)]
            rw [hfun]
            obtain ⟨h1, h2⟩ := dcov_tuple_pair π hm l k
            exact ⟨h1, h2.trans (if_neg hij)⟩
          · -- `A(ωᵢ,ωⱼ)·B(ωⱼ,ωₗ)`: symmetrize `A`
            subst hkj
            rw [if_neg (by rintro (⟨h1, h2⟩ | ⟨h1, h2⟩)
              <;> [exact hki h1; exact hli h2])]
            have hfun : (fun ω : Fin n → α × β =>
                centeredDistFst π (ω i, ω k) * centeredDistSnd π (ω k, ω l))
                = fun ω => centeredDistFst π (ω k, ω i) * centeredDistSnd π (ω k, ω l) := by
              funext ω
              rw [centeredDistFst_symm π (ω i) (ω k)]
            rw [hfun]
            have hki' : k ≠ i := hki
            have hil : i ≠ l := fun h => hli (h.symm)
            obtain ⟨h1, h2⟩ := dcov_transport (P := Measure.pi fun _ : Fin n => π.measure)
              (φ := fun ω => (ω k, (ω i, ω l))) (by fun_prop)
              (dcov_pi_map_triple hki' hkl hil) (dcov_int_triple π hm)
            exact ⟨h1, h2.trans (dcov_val_triple π hm)⟩
        · rw [if_neg (by rintro (⟨h1, h2⟩ | ⟨h1, h2⟩) <;> [exact hki h1; exact hkj h1])]
          by_cases hli : l = i
          · -- `A(ωᵢ,ωⱼ)·B(ωₖ,ωᵢ)`: symmetrize `B`
            subst hli
            have hfun : (fun ω : Fin n → α × β =>
                centeredDistFst π (ω l, ω j) * centeredDistSnd π (ω k, ω l))
                = fun ω => centeredDistFst π (ω l, ω j) * centeredDistSnd π (ω l, ω k) := by
              funext ω
              rw [centeredDistSnd_symm π (ω k) (ω l)]
            rw [hfun]
            have hlj' : l ≠ j := hij
            have hlk : l ≠ k := fun h => hkl h.symm
            have hjk : j ≠ k := fun h => hkj h.symm
            obtain ⟨h1, h2⟩ := dcov_transport (P := Measure.pi fun _ : Fin n => π.measure)
              (φ := fun ω => (ω l, (ω j, ω k))) (by fun_prop)
              (dcov_pi_map_triple hlj' hlk hjk) (dcov_int_triple π hm)
            exact ⟨h1, h2.trans (dcov_val_triple π hm)⟩
          · by_cases hlj : l = j
            · -- `A(ωᵢ,ωⱼ)·B(ωₖ,ωⱼ)`: symmetrize both
              subst hlj
              have hfun : (fun ω : Fin n → α × β =>
                  centeredDistFst π (ω i, ω l) * centeredDistSnd π (ω k, ω l))
                  = fun ω => centeredDistFst π (ω l, ω i) * centeredDistSnd π (ω l, ω k) := by
                funext ω
                rw [centeredDistFst_symm π (ω i) (ω l), centeredDistSnd_symm π (ω k) (ω l)]
              rw [hfun]
              have hli' : l ≠ i := fun h => hij h.symm
              have hlk : l ≠ k := fun h => hkl h.symm
              have hik : i ≠ k := fun h => hki h.symm
              obtain ⟨h1, h2⟩ := dcov_transport (P := Measure.pi fun _ : Fin n => π.measure)
                (φ := fun ω => (ω l, (ω i, ω k))) (by fun_prop)
                (dcov_pi_map_triple hli' hlk hik) (dcov_int_triple π hm)
              exact ⟨h1, h2.trans (dcov_val_triple π hm)⟩
            · -- all four distinct
              have hik : i ≠ k := fun h => hki h.symm
              have hil : i ≠ l := fun h => hli h.symm
              have hjk : j ≠ k := fun h => hkj h.symm
              have hjl : j ≠ l := fun h => hlj h.symm
              obtain ⟨h1, h2⟩ := dcov_transport (P := Measure.pi fun _ : Fin n => π.measure)
                (φ := fun ω => ((ω i, ω j), (ω k, ω l))) (by fun_prop)
                (dcov_pi_map_quad hij hik hil hjk hjl hkl) (dcov_int_quad π hm)
              exact ⟨h1, h2.trans (dcov_val_quad π hm)⟩

end DcovSampling

open MeasureTheory ProbabilityTheory

/-- The empirically double-centered Hadamard sum `F_n` over `n` iid draws `ω : Fin n → α × β`:
    the discrete Schur/Gram quadratic form of `dcov_schur_sum_nonneg` specialized to the
    marginal samples `x i = (ω i).1`, `y i = (ω i).2`.  Pointwise nonnegative
    (`dcovSampleStat_nonneg`); its expectation over `π^⊗n` drives the sampling route to
    `dcov_nonneg` (see the `DcovSampling` section docstring above, `:1436`). -/
private noncomputable def dcovSampleStat {α β : Type*}
    [PseudoMetricSpace α] [PseudoMetricSpace β] {n : ℕ} (ω : Fin n → α × β) : ℝ :=
  ∑ i, ∑ j,
    (dist (ω i).1 (ω j).1 - (∑ k, dist (ω i).1 (ω k).1) / n
        - (∑ k, dist (ω k).1 (ω j).1) / n
        + (∑ k, ∑ l, dist (ω k).1 (ω l).1) / (n ^ 2))
      * (dist (ω i).2 (ω j).2 - (∑ k, dist (ω i).2 (ω k).2) / n
          - (∑ k, dist (ω k).2 (ω j).2) / n
          + (∑ k, ∑ l, dist (ω k).2 (ω l).2) / (n ^ 2))

/-- `F_n` is pointwise nonnegative, by the discrete Schur/Gram step
    (`dcov_schur_sum_nonneg`) applied to the coordinate projections of the sample. -/
private lemma dcovSampleStat_nonneg {α β : Type*}
    [PseudoMetricSpace α] [PseudoMetricSpace β]
    (h_negα : DistNegativeType α) (h_negβ : DistNegativeType β)
    {n : ℕ} (ω : Fin n → α × β) :
    0 ≤ dcovSampleStat ω :=
  dcov_schur_sum_nonneg h_negα h_negβ (fun i => (ω i).1) (fun i => (ω i).2)

section DcovExpectation

variable {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
  [PseudoMetricSpace α] [PseudoMetricSpace β]

/-- Raw-to-centered rewrite of the sample statistic: empirical double centering is invariant
    under the additive one-variable shift taking raw distances to the population-centered
    kernels (`dcov_centered_entry_shift`), and the centered Hadamard sum then expands into
    the four canonical sums of `dcov_centered_hadamard_expand`. -/
private lemma dcovSampleStat_eq_centered (π : JointProbabilityMeasure α β) {n : ℕ}
    (hn : (n : ℝ) ≠ 0) (ω : Fin n → α × β) :
    dcovSampleStat ω
      = (∑ i, ∑ j, centeredDistFst π (ω i, ω j) * centeredDistSnd π (ω i, ω j))
        - (∑ i, ∑ j, ∑ k, centeredDistFst π (ω i, ω k) * centeredDistSnd π (ω i, ω j)) / n
        - (∑ i, ∑ j, ∑ k, centeredDistFst π (ω k, ω j) * centeredDistSnd π (ω i, ω j)) / n
        + (∑ i, ∑ k, ∑ j, ∑ l,
            centeredDistFst π (ω i, ω j) * centeredDistSnd π (ω k, ω l)) / (n ^ 2) := by
  have hAeq : ∀ i j : Fin n,
      dist (ω i).1 (ω j).1 - (∑ k, dist (ω i).1 (ω k).1) / n
          - (∑ k, dist (ω k).1 (ω j).1) / n + (∑ k, ∑ l, dist (ω k).1 (ω l).1) / (n ^ 2)
        = centeredDistFst π (ω i, ω j) - (∑ k, centeredDistFst π (ω i, ω k)) / n
          - (∑ k, centeredDistFst π (ω k, ω j)) / n
          + (∑ k, ∑ l, centeredDistFst π (ω k, ω l)) / (n ^ 2) := fun i j =>
    dcov_centered_entry_shift hn (fun a b => dist (ω a).1 (ω b).1)
      (fun a => ∫ q, dist (ω a).1 q.1 ∂π.measure)
      (∫ p, ∫ q, dist p.1 q.1 ∂π.measure ∂π.measure) i j
  have hBeq : ∀ i j : Fin n,
      dist (ω i).2 (ω j).2 - (∑ k, dist (ω i).2 (ω k).2) / n
          - (∑ k, dist (ω k).2 (ω j).2) / n + (∑ k, ∑ l, dist (ω k).2 (ω l).2) / (n ^ 2)
        = centeredDistSnd π (ω i, ω j) - (∑ k, centeredDistSnd π (ω i, ω k)) / n
          - (∑ k, centeredDistSnd π (ω k, ω j)) / n
          + (∑ k, ∑ l, centeredDistSnd π (ω k, ω l)) / (n ^ 2) := fun i j =>
    dcov_centered_entry_shift hn (fun a b => dist (ω a).2 (ω b).2)
      (fun a => ∫ q, dist (ω a).2 q.2 ∂π.measure)
      (∫ p, ∫ q, dist p.2 q.2 ∂π.measure ∂π.measure) i j
  have hstep : dcovSampleStat ω
      = ∑ i, ∑ j,
          (centeredDistFst π (ω i, ω j) - (∑ k, centeredDistFst π (ω i, ω k)) / n
            - (∑ k, centeredDistFst π (ω k, ω j)) / n
            + (∑ k, ∑ l, centeredDistFst π (ω k, ω l)) / (n ^ 2))
          * (centeredDistSnd π (ω i, ω j) - (∑ k, centeredDistSnd π (ω i, ω k)) / n
            - (∑ k, centeredDistSnd π (ω k, ω j)) / n
            + (∑ k, ∑ l, centeredDistSnd π (ω k, ω l)) / (n ^ 2)) := by
    unfold dcovSampleStat
    exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by
      rw [hAeq i j, hBeq i j]
  exact hstep.trans (dcov_centered_hadamard_expand hn
    (fun i j => centeredDistFst π (ω i, ω j)) (fun i j => centeredDistSnd π (ω i, ω j)))

/-- Row count: summing `if i = j then c else d` over `j` gives `c + (n − 1)·d`. -/
private lemma dcov_count_row {n : ℕ} (c d : ℝ) (i : Fin n) :
    (∑ j : Fin n, if i = j then c else d) = c + ((n : ℝ) - 1) * d := by
  have hsplit : ∀ j : Fin n, (if i = j then c else d) = (if i = j then c - d else 0) + d :=
    fun j => by by_cases h : i = j <;> simp [h]
  rw [Finset.sum_congr rfl fun j _ => hsplit j, Finset.sum_add_distrib, Finset.sum_ite_eq,
    Finset.sum_const]
  simp only [Finset.mem_univ, if_true, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  ring

/-- Pair count: summing `if i = j then c else d` over both indices gives `n·c + n(n−1)·d`
    — `n` diagonal tuples and `n(n−1)` off-diagonal tuples. -/
private lemma dcov_count_pair {n : ℕ} (c d : ℝ) :
    (∑ i : Fin n, ∑ j : Fin n, if i = j then c else d)
      = n * c + n * ((n : ℝ) - 1) * d := by
  rw [Finset.sum_congr rfl fun i _ => dcov_count_row c d i, Finset.sum_const]
  simp only [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  ring

/-- Stage E2a: expectation of the matched-pair sum `∑ᵢ ∑ⱼ A(ωᵢ,ωⱼ)·B(ωᵢ,ωⱼ)` over `π^⊗ⁿ`:
    the `n` diagonal terms contribute `C₀` each, the `n(n−1)` off-diagonal terms `dCov²`
    each (`dcov_tuple_pair`). -/
private lemma dcov_exp_sum_pair (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π)
    {n : ℕ} :
    Integrable (fun ω : Fin n → α × β =>
        ∑ i, ∑ j, centeredDistFst π (ω i, ω j) * centeredDistSnd π (ω i, ω j))
      (Measure.pi fun _ : Fin n => π.measure)
    ∧ (∫ ω, ∑ i, ∑ j, centeredDistFst π (ω i, ω j) * centeredDistSnd π (ω i, ω j)
          ∂(Measure.pi fun _ : Fin n => π.measure))
      = n * (∫ z, centeredDistFst π (z, z) * centeredDistSnd π (z, z) ∂π.measure)
        + n * ((n : ℝ) - 1) * distanceCovarianceSq π := by
  classical
  constructor
  · exact integrable_finsetSum _ fun i _ => integrable_finsetSum _ fun j _ =>
      (dcov_tuple_pair π hm i j).1
  · calc (∫ ω, ∑ i, ∑ j, centeredDistFst π (ω i, ω j) * centeredDistSnd π (ω i, ω j)
            ∂(Measure.pi fun _ : Fin n => π.measure))
        = ∑ i, ∫ ω, ∑ j, centeredDistFst π (ω i, ω j) * centeredDistSnd π (ω i, ω j)
            ∂(Measure.pi fun _ : Fin n => π.measure) :=
          integral_finsetSum _ fun i _ => integrable_finsetSum _ fun j _ =>
            (dcov_tuple_pair π hm i j).1
      _ = ∑ i, ∑ j, ∫ ω, centeredDistFst π (ω i, ω j) * centeredDistSnd π (ω i, ω j)
            ∂(Measure.pi fun _ : Fin n => π.measure) :=
          Finset.sum_congr rfl fun i _ => integral_finsetSum _ fun j _ =>
            (dcov_tuple_pair π hm i j).1
      _ = ∑ i, ∑ j, if i = j
            then ∫ z, centeredDistFst π (z, z) * centeredDistSnd π (z, z) ∂π.measure
            else distanceCovarianceSq π :=
          Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ =>
            (dcov_tuple_pair π hm i j).2
      _ = n * (∫ z, centeredDistFst π (z, z) * centeredDistSnd π (z, z) ∂π.measure)
            + n * ((n : ℝ) - 1) * distanceCovarianceSq π := dcov_count_pair _ _

/-- Stage E2b (row-mean family): expectation of `∑ᵢ ∑ⱼ ∑ₖ A(ωᵢ,ωₖ)·B(ωᵢ,ωⱼ)` over `π^⊗ⁿ`.
    Degeneracy (`dcov_tuple_rowA`) kills every term with `k ≠ j`, collapsing the triple sum
    to the matched-pair count `n·C₀ + n(n−1)·dCov²`. -/
private lemma dcov_exp_sum_rowA (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π)
    {n : ℕ} :
    Integrable (fun ω : Fin n → α × β =>
        ∑ i, ∑ j, ∑ k, centeredDistFst π (ω i, ω k) * centeredDistSnd π (ω i, ω j))
      (Measure.pi fun _ : Fin n => π.measure)
    ∧ (∫ ω, ∑ i, ∑ j, ∑ k, centeredDistFst π (ω i, ω k) * centeredDistSnd π (ω i, ω j)
          ∂(Measure.pi fun _ : Fin n => π.measure))
      = n * (∫ z, centeredDistFst π (z, z) * centeredDistSnd π (z, z) ∂π.measure)
        + n * ((n : ℝ) - 1) * distanceCovarianceSq π := by
  classical
  constructor
  · exact integrable_finsetSum _ fun i _ => integrable_finsetSum _ fun j _ =>
      integrable_finsetSum _ fun k _ => (dcov_tuple_rowA π hm i j k).1
  · calc (∫ ω, ∑ i, ∑ j, ∑ k, centeredDistFst π (ω i, ω k) * centeredDistSnd π (ω i, ω j)
            ∂(Measure.pi fun _ : Fin n => π.measure))
        = ∑ i, ∫ ω, ∑ j, ∑ k, centeredDistFst π (ω i, ω k) * centeredDistSnd π (ω i, ω j)
            ∂(Measure.pi fun _ : Fin n => π.measure) :=
          integral_finsetSum _ fun i _ => integrable_finsetSum _ fun j _ =>
            integrable_finsetSum _ fun k _ => (dcov_tuple_rowA π hm i j k).1
      _ = ∑ i, ∑ j, ∫ ω, ∑ k, centeredDistFst π (ω i, ω k) * centeredDistSnd π (ω i, ω j)
            ∂(Measure.pi fun _ : Fin n => π.measure) :=
          Finset.sum_congr rfl fun i _ => integral_finsetSum _ fun j _ =>
            integrable_finsetSum _ fun k _ => (dcov_tuple_rowA π hm i j k).1
      _ = ∑ i, ∑ j, ∑ k, ∫ ω, centeredDistFst π (ω i, ω k) * centeredDistSnd π (ω i, ω j)
            ∂(Measure.pi fun _ : Fin n => π.measure) :=
          Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ =>
            integral_finsetSum _ fun k _ => (dcov_tuple_rowA π hm i j k).1
      _ = ∑ i, ∑ j, ∑ k, if k = j
            then (if i = j
              then ∫ z, centeredDistFst π (z, z) * centeredDistSnd π (z, z) ∂π.measure
              else distanceCovarianceSq π)
            else 0 :=
          Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ =>
            Finset.sum_congr rfl fun k _ => (dcov_tuple_rowA π hm i j k).2
      _ = ∑ i, ∑ j, if i = j
            then ∫ z, centeredDistFst π (z, z) * centeredDistSnd π (z, z) ∂π.measure
            else distanceCovarianceSq π :=
          Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by simp
      _ = n * (∫ z, centeredDistFst π (z, z) * centeredDistSnd π (z, z) ∂π.measure)
            + n * ((n : ℝ) - 1) * distanceCovarianceSq π := dcov_count_pair _ _

/-- Stage E2b (column-mean family): expectation of `∑ᵢ ∑ⱼ ∑ₖ A(ωₖ,ωⱼ)·B(ωᵢ,ωⱼ)` over
    `π^⊗ⁿ`. Degeneracy (`dcov_tuple_colA`) kills every term with `k ≠ i`, collapsing the
    triple sum to the matched-pair count `n·C₀ + n(n−1)·dCov²`. -/
private lemma dcov_exp_sum_colA (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π)
    {n : ℕ} :
    Integrable (fun ω : Fin n → α × β =>
        ∑ i, ∑ j, ∑ k, centeredDistFst π (ω k, ω j) * centeredDistSnd π (ω i, ω j))
      (Measure.pi fun _ : Fin n => π.measure)
    ∧ (∫ ω, ∑ i, ∑ j, ∑ k, centeredDistFst π (ω k, ω j) * centeredDistSnd π (ω i, ω j)
          ∂(Measure.pi fun _ : Fin n => π.measure))
      = n * (∫ z, centeredDistFst π (z, z) * centeredDistSnd π (z, z) ∂π.measure)
        + n * ((n : ℝ) - 1) * distanceCovarianceSq π := by
  classical
  constructor
  · exact integrable_finsetSum _ fun i _ => integrable_finsetSum _ fun j _ =>
      integrable_finsetSum _ fun k _ => (dcov_tuple_colA π hm i j k).1
  · calc (∫ ω, ∑ i, ∑ j, ∑ k, centeredDistFst π (ω k, ω j) * centeredDistSnd π (ω i, ω j)
            ∂(Measure.pi fun _ : Fin n => π.measure))
        = ∑ i, ∫ ω, ∑ j, ∑ k, centeredDistFst π (ω k, ω j) * centeredDistSnd π (ω i, ω j)
            ∂(Measure.pi fun _ : Fin n => π.measure) :=
          integral_finsetSum _ fun i _ => integrable_finsetSum _ fun j _ =>
            integrable_finsetSum _ fun k _ => (dcov_tuple_colA π hm i j k).1
      _ = ∑ i, ∑ j, ∫ ω, ∑ k, centeredDistFst π (ω k, ω j) * centeredDistSnd π (ω i, ω j)
            ∂(Measure.pi fun _ : Fin n => π.measure) :=
          Finset.sum_congr rfl fun i _ => integral_finsetSum _ fun j _ =>
            integrable_finsetSum _ fun k _ => (dcov_tuple_colA π hm i j k).1
      _ = ∑ i, ∑ j, ∑ k, ∫ ω, centeredDistFst π (ω k, ω j) * centeredDistSnd π (ω i, ω j)
            ∂(Measure.pi fun _ : Fin n => π.measure) :=
          Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ =>
            integral_finsetSum _ fun k _ => (dcov_tuple_colA π hm i j k).1
      _ = ∑ i, ∑ j, ∑ k, if k = i
            then (if i = j
              then ∫ z, centeredDistFst π (z, z) * centeredDistSnd π (z, z) ∂π.measure
              else distanceCovarianceSq π)
            else 0 :=
          Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ =>
            Finset.sum_congr rfl fun k _ => (dcov_tuple_colA π hm i j k).2
      _ = ∑ i, ∑ j, if i = j
            then ∫ z, centeredDistFst π (z, z) * centeredDistSnd π (z, z) ∂π.measure
            else distanceCovarianceSq π :=
          Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by simp
      _ = n * (∫ z, centeredDistFst π (z, z) * centeredDistSnd π (z, z) ∂π.measure)
            + n * ((n : ℝ) - 1) * distanceCovarianceSq π := dcov_count_pair _ _

/-- Stage E2c (grand-mean quadruple family): expectation of
    `∑ᵢ ∑ₖ ∑ⱼ ∑ₗ A(ωᵢ,ωⱼ)·B(ωₖ,ωₗ)` over `π^⊗ⁿ`. By `dcov_tuple_quad` only three
    coincidence classes survive degeneracy: the `n` fully-diagonal tuples (`C₀` each), the
    `n(n−1)` split-diagonal tuples (`DX·DY` each), and the `2n(n−1)` matched/transposed
    pairs (`dCov²` each). -/
private lemma dcov_exp_sum_quad (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π)
    {n : ℕ} :
    Integrable (fun ω : Fin n → α × β =>
        ∑ i, ∑ k, ∑ j, ∑ l, centeredDistFst π (ω i, ω j) * centeredDistSnd π (ω k, ω l))
      (Measure.pi fun _ : Fin n => π.measure)
    ∧ (∫ ω, ∑ i, ∑ k, ∑ j, ∑ l,
            centeredDistFst π (ω i, ω j) * centeredDistSnd π (ω k, ω l)
          ∂(Measure.pi fun _ : Fin n => π.measure))
      = n * (∫ z, centeredDistFst π (z, z) * centeredDistSnd π (z, z) ∂π.measure)
        + n * ((n : ℝ) - 1) * ((∫ p, ∫ q, dist p.1 q.1 ∂π.measure ∂π.measure)
            * (∫ p, ∫ q, dist p.2 q.2 ∂π.measure ∂π.measure))
        + n * ((n : ℝ) - 1) * (2 * distanceCovarianceSq π) := by
  classical
  constructor
  · exact integrable_finsetSum _ fun i _ => integrable_finsetSum _ fun k _ =>
      integrable_finsetSum _ fun j _ => integrable_finsetSum _ fun l _ =>
        (dcov_tuple_quad π hm i j k l).1
  · set C₀ := ∫ z, centeredDistFst π (z, z) * centeredDistSnd π (z, z) ∂π.measure with hC₀
    set DXY := (∫ p, ∫ q, dist p.1 q.1 ∂π.measure ∂π.measure)
      * (∫ p, ∫ q, dist p.2 q.2 ∂π.measure ∂π.measure) with hDXY
    set D := distanceCovarianceSq π with hD
    have hV : ∀ i j k l : Fin n,
        (∫ ω, centeredDistFst π (ω i, ω j) * centeredDistSnd π (ω k, ω l)
            ∂(Measure.pi fun _ : Fin n => π.measure))
          = if i = j then (if k = l then (if i = k then C₀ else DXY) else 0)
            else (if (k = i ∧ l = j) ∨ (k = j ∧ l = i) then D else 0) := by
      intro i j k l
      rw [hC₀, hDXY, hD]
      exact (dcov_tuple_quad π hm i j k l).2
    have hinner : ∀ i j : Fin n,
        (∑ k : Fin n, ∑ l : Fin n,
          (if i = j then (if k = l then (if i = k then C₀ else DXY) else 0)
            else (if (k = i ∧ l = j) ∨ (k = j ∧ l = i) then D else 0)))
          = if i = j then C₀ + ((n : ℝ) - 1) * DXY else 2 * D := by
      intro i j
      by_cases hij : i = j
      · rw [if_pos hij]
        have hentry : ∀ k l : Fin n,
            (if i = j then (if k = l then (if i = k then C₀ else DXY) else 0)
              else (if (k = i ∧ l = j) ∨ (k = j ∧ l = i) then D else 0))
              = if k = l then (if i = k then C₀ else DXY) else 0 :=
          fun k l => if_pos hij
        rw [Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun l _ => hentry k l]
        have hrow : ∀ k : Fin n,
            (∑ l : Fin n, if k = l then (if i = k then C₀ else DXY) else 0)
              = if i = k then C₀ else DXY := fun k => by simp
        rw [Finset.sum_congr rfl fun k _ => hrow k]
        exact dcov_count_row _ _ i
      · rw [if_neg hij]
        have hentry : ∀ k l : Fin n,
            (if i = j then (if k = l then (if i = k then C₀ else DXY) else 0)
              else (if (k = i ∧ l = j) ∨ (k = j ∧ l = i) then D else 0))
              = (if k = i ∧ l = j then D else 0) + (if k = j ∧ l = i then D else 0) := by
          intro k l
          rw [if_neg hij]
          by_cases h1 : k = i ∧ l = j
          · rw [if_pos (Or.inl h1 : (k = i ∧ l = j) ∨ (k = j ∧ l = i)), if_pos h1,
              if_neg (fun h2 : k = j ∧ l = i => hij (h1.1.symm.trans h2.1)), add_zero]
          · by_cases h2 : k = j ∧ l = i
            · rw [if_pos (Or.inr h2 : (k = i ∧ l = j) ∨ (k = j ∧ l = i)), if_neg h1,
                if_pos h2, zero_add]
            · rw [if_neg (fun h : (k = i ∧ l = j) ∨ (k = j ∧ l = i) => h.elim h1 h2),
                if_neg h1, if_neg h2, add_zero]
        rw [Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun l _ => hentry k l]
        have hsplit : ∀ k : Fin n,
            (∑ l : Fin n,
              ((if k = i ∧ l = j then D else 0) + (if k = j ∧ l = i then D else 0)))
              = (∑ l : Fin n, if k = i ∧ l = j then D else 0)
                + ∑ l : Fin n, if k = j ∧ l = i then D else 0 :=
          fun k => Finset.sum_add_distrib
        rw [Finset.sum_congr rfl fun k _ => hsplit k, Finset.sum_add_distrib]
        have hs : ∀ a b : Fin n,
            (∑ k : Fin n, ∑ l : Fin n, if k = a ∧ l = b then D else 0) = D := by
          intro a b
          have hk : ∀ k : Fin n, (∑ l : Fin n, if k = a ∧ l = b then D else 0)
              = if k = a then D else 0 := by
            intro k
            by_cases hka : k = a
            · simp [hka]
            · simp [hka]
          rw [Finset.sum_congr rfl fun k _ => hk k]
          simp
        rw [hs i j, hs j i]
        ring
    calc (∫ ω, ∑ i, ∑ k, ∑ j, ∑ l,
            centeredDistFst π (ω i, ω j) * centeredDistSnd π (ω k, ω l)
          ∂(Measure.pi fun _ : Fin n => π.measure))
        = ∑ i, ∫ ω, ∑ k, ∑ j, ∑ l,
            centeredDistFst π (ω i, ω j) * centeredDistSnd π (ω k, ω l)
            ∂(Measure.pi fun _ : Fin n => π.measure) :=
          integral_finsetSum _ fun i _ => integrable_finsetSum _ fun k _ =>
            integrable_finsetSum _ fun j _ => integrable_finsetSum _ fun l _ =>
              (dcov_tuple_quad π hm i j k l).1
      _ = ∑ i, ∑ k, ∫ ω, ∑ j, ∑ l,
            centeredDistFst π (ω i, ω j) * centeredDistSnd π (ω k, ω l)
            ∂(Measure.pi fun _ : Fin n => π.measure) :=
          Finset.sum_congr rfl fun i _ => integral_finsetSum _ fun k _ =>
            integrable_finsetSum _ fun j _ => integrable_finsetSum _ fun l _ =>
              (dcov_tuple_quad π hm i j k l).1
      _ = ∑ i, ∑ k, ∑ j, ∫ ω, ∑ l,
            centeredDistFst π (ω i, ω j) * centeredDistSnd π (ω k, ω l)
            ∂(Measure.pi fun _ : Fin n => π.measure) :=
          Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun k _ =>
            integral_finsetSum _ fun j _ => integrable_finsetSum _ fun l _ =>
              (dcov_tuple_quad π hm i j k l).1
      _ = ∑ i, ∑ k, ∑ j, ∑ l, ∫ ω,
            centeredDistFst π (ω i, ω j) * centeredDistSnd π (ω k, ω l)
            ∂(Measure.pi fun _ : Fin n => π.measure) :=
          Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun k _ =>
            Finset.sum_congr rfl fun j _ => integral_finsetSum _ fun l _ =>
              (dcov_tuple_quad π hm i j k l).1
      _ = ∑ i, ∑ k, ∑ j, ∑ l,
            (if i = j then (if k = l then (if i = k then C₀ else DXY) else 0)
              else (if (k = i ∧ l = j) ∨ (k = j ∧ l = i) then D else 0)) :=
          Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun k _ =>
            Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun l _ => hV i j k l
      _ = ∑ i, ∑ j, ∑ k, ∑ l,
            (if i = j then (if k = l then (if i = k then C₀ else DXY) else 0)
              else (if (k = i ∧ l = j) ∨ (k = j ∧ l = i) then D else 0)) :=
          Finset.sum_congr rfl fun i _ => Finset.sum_comm
      _ = ∑ i, ∑ j, if i = j then C₀ + ((n : ℝ) - 1) * DXY else 2 * D :=
          Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => hinner i j
      _ = n * (C₀ + ((n : ℝ) - 1) * DXY) + n * ((n : ℝ) - 1) * (2 * D) :=
          dcov_count_pair _ _
      _ = n * C₀ + n * ((n : ℝ) - 1) * DXY + n * ((n : ℝ) - 1) * (2 * D) := by ring

/-- Stage E2c assembly: the exact closed form of `E[F_n]`, matching the sketch at the
    `DcovSampling` section header: `E[F_n] = Σ₁ − Σ₂/n − Σ₂′/n + Σ₃/n²` with
    `Σ₁ = Σ₂ = Σ₂′ = n·C₀ + n(n−1)·dCov²` and
    `Σ₃ = n·C₀ + n(n−1)·DX·DY + 2n(n−1)·dCov²`. -/
private lemma dcov_integral_sampleStat (π : JointProbabilityMeasure α β)
    (hm : JointFiniteMoments π) {n : ℕ} (hn : (n : ℝ) ≠ 0) :
    (∫ ω, dcovSampleStat ω ∂(Measure.pi fun _ : Fin n => π.measure))
      = (n * (∫ z, centeredDistFst π (z, z) * centeredDistSnd π (z, z) ∂π.measure)
          + n * ((n : ℝ) - 1) * distanceCovarianceSq π)
        - (n * (∫ z, centeredDistFst π (z, z) * centeredDistSnd π (z, z) ∂π.measure)
            + n * ((n : ℝ) - 1) * distanceCovarianceSq π) / n
        - (n * (∫ z, centeredDistFst π (z, z) * centeredDistSnd π (z, z) ∂π.measure)
            + n * ((n : ℝ) - 1) * distanceCovarianceSq π) / n
        + (n * (∫ z, centeredDistFst π (z, z) * centeredDistSnd π (z, z) ∂π.measure)
            + n * ((n : ℝ) - 1) * ((∫ p, ∫ q, dist p.1 q.1 ∂π.measure ∂π.measure)
              * (∫ p, ∫ q, dist p.2 q.2 ∂π.measure ∂π.measure))
            + n * ((n : ℝ) - 1) * (2 * distanceCovarianceSq π)) / n ^ 2 := by
  obtain ⟨hi1, hv1⟩ := dcov_exp_sum_pair π hm (n := n)
  obtain ⟨hi2, hv2⟩ := dcov_exp_sum_rowA π hm (n := n)
  obtain ⟨hi3, hv3⟩ := dcov_exp_sum_colA π hm (n := n)
  obtain ⟨hi4, hv4⟩ := dcov_exp_sum_quad π hm (n := n)
  calc (∫ ω, dcovSampleStat ω ∂(Measure.pi fun _ : Fin n => π.measure))
      = ∫ ω, ((∑ i, ∑ j, centeredDistFst π (ω i, ω j) * centeredDistSnd π (ω i, ω j))
          - (∑ i, ∑ j, ∑ k,
              centeredDistFst π (ω i, ω k) * centeredDistSnd π (ω i, ω j)) / n
          - (∑ i, ∑ j, ∑ k,
              centeredDistFst π (ω k, ω j) * centeredDistSnd π (ω i, ω j)) / n
          + (∑ i, ∑ k, ∑ j, ∑ l,
              centeredDistFst π (ω i, ω j) * centeredDistSnd π (ω k, ω l)) / (n ^ 2))
          ∂(Measure.pi fun _ : Fin n => π.measure) :=
        integral_congr_ae (Filter.Eventually.of_forall fun ω =>
          dcovSampleStat_eq_centered π hn ω)
    _ = (∫ ω, ∑ i, ∑ j, centeredDistFst π (ω i, ω j) * centeredDistSnd π (ω i, ω j)
            ∂(Measure.pi fun _ : Fin n => π.measure))
        - (∫ ω, ∑ i, ∑ j, ∑ k,
              centeredDistFst π (ω i, ω k) * centeredDistSnd π (ω i, ω j)
            ∂(Measure.pi fun _ : Fin n => π.measure)) / n
        - (∫ ω, ∑ i, ∑ j, ∑ k,
              centeredDistFst π (ω k, ω j) * centeredDistSnd π (ω i, ω j)
            ∂(Measure.pi fun _ : Fin n => π.measure)) / n
        + (∫ ω, ∑ i, ∑ k, ∑ j, ∑ l,
              centeredDistFst π (ω i, ω j) * centeredDistSnd π (ω k, ω l)
            ∂(Measure.pi fun _ : Fin n => π.measure)) / n ^ 2 := by
        have h_ab : Integrable (fun ω : Fin n → α × β =>
            (∑ i, ∑ j, centeredDistFst π (ω i, ω j) * centeredDistSnd π (ω i, ω j))
              - (∑ i, ∑ j, ∑ k,
                  centeredDistFst π (ω i, ω k) * centeredDistSnd π (ω i, ω j)) / n)
            (Measure.pi fun _ : Fin n => π.measure) := hi1.sub (hi2.div_const _)
        have h_abc : Integrable (fun ω : Fin n → α × β =>
            (∑ i, ∑ j, centeredDistFst π (ω i, ω j) * centeredDistSnd π (ω i, ω j))
              - (∑ i, ∑ j, ∑ k,
                  centeredDistFst π (ω i, ω k) * centeredDistSnd π (ω i, ω j)) / n
              - (∑ i, ∑ j, ∑ k,
                  centeredDistFst π (ω k, ω j) * centeredDistSnd π (ω i, ω j)) / n)
            (Measure.pi fun _ : Fin n => π.measure) := h_ab.sub (hi3.div_const _)
        rw [integral_add h_abc (hi4.div_const _), integral_sub h_ab (hi3.div_const _),
          integral_sub hi1 (hi2.div_const _), integral_div, integral_div, integral_div]
    _ = (n * (∫ z, centeredDistFst π (z, z) * centeredDistSnd π (z, z) ∂π.measure)
          + n * ((n : ℝ) - 1) * distanceCovarianceSq π)
        - (n * (∫ z, centeredDistFst π (z, z) * centeredDistSnd π (z, z) ∂π.measure)
            + n * ((n : ℝ) - 1) * distanceCovarianceSq π) / n
        - (n * (∫ z, centeredDistFst π (z, z) * centeredDistSnd π (z, z) ∂π.measure)
            + n * ((n : ℝ) - 1) * distanceCovarianceSq π) / n
        + (n * (∫ z, centeredDistFst π (z, z) * centeredDistSnd π (z, z) ∂π.measure)
            + n * ((n : ℝ) - 1) * ((∫ p, ∫ q, dist p.1 q.1 ∂π.measure ∂π.measure)
              * (∫ p, ∫ q, dist p.2 q.2 ∂π.measure ∂π.measure))
            + n * ((n : ℝ) - 1) * (2 * distanceCovarianceSq π)) / n ^ 2 := by
        rw [hv1, hv2, hv3, hv4]

end DcovExpectation

open Filter in
/-- Stage E3 (real-analysis core): the exact closed form of `E[F_n]`
    (`dcov_integral_sampleStat`), divided by its leading coefficient `n(n−1)`, converges to
    the population value `D` as `n → ∞`.  Writing the closed form as
    `D·n² + (C₀−3D)·n + (−2C₀+4D+DXY) + (C₀−DXY−2D)/n` and normalizing, every lower-order term
    (the `C₀`, `DXY` contributions and the `1/n`, `1/n²`, `1/n³` corrections) vanishes, leaving
    exactly `D`.  Instantiated with `D = distanceCovarianceSq π` this drives `dcov_nonneg`. -/
private lemma dcov_closed_form_tendsto (C₀ DXY D : ℝ) :
    Tendsto (fun n : ℕ =>
        ((n * C₀ + n * ((n : ℝ) - 1) * D)
            - (n * C₀ + n * ((n : ℝ) - 1) * D) / n
            - (n * C₀ + n * ((n : ℝ) - 1) * D) / n
            + (n * C₀ + n * ((n : ℝ) - 1) * DXY + n * ((n : ℝ) - 1) * (2 * D)) / n ^ 2)
          / ((n : ℝ) * ((n : ℝ) - 1)))
      atTop (nhds D) := by
  have hu : Tendsto (fun n : ℕ => 1 / (n : ℝ)) atTop (nhds 0) :=
    tendsto_one_div_atTop_nhds_zero_nat
  have h1 : Tendsto (fun n : ℕ => (C₀ - 3 * D) * (1 / (n : ℝ))) atTop (nhds 0) := by
    simpa using hu.const_mul (C₀ - 3 * D)
  have h2 : Tendsto (fun n : ℕ => (-2 * C₀ + 4 * D + DXY) * (1 / (n : ℝ)) ^ 2) atTop
      (nhds 0) := by
    simpa using (hu.pow 2).const_mul (-2 * C₀ + 4 * D + DXY)
  have h3 : Tendsto (fun n : ℕ => (C₀ - DXY - 2 * D) * (1 / (n : ℝ)) ^ 3) atTop (nhds 0) := by
    simpa using (hu.pow 3).const_mul (C₀ - DXY - 2 * D)
  have hnum : Tendsto (fun n : ℕ => D + (C₀ - 3 * D) * (1 / (n : ℝ))
      + (-2 * C₀ + 4 * D + DXY) * (1 / (n : ℝ)) ^ 2
      + (C₀ - DXY - 2 * D) * (1 / (n : ℝ)) ^ 3) atTop (nhds D) := by
    simpa using ((tendsto_const_nhds.add h1).add h2).add h3
  have hden : Tendsto (fun n : ℕ => 1 - 1 / (n : ℝ)) atTop (nhds 1) := by
    simpa using tendsto_const_nhds.sub hu
  have hquot : Tendsto (fun n : ℕ => (D + (C₀ - 3 * D) * (1 / (n : ℝ))
      + (-2 * C₀ + 4 * D + DXY) * (1 / (n : ℝ)) ^ 2
      + (C₀ - DXY - 2 * D) * (1 / (n : ℝ)) ^ 3) / (1 - 1 / (n : ℝ))) atTop (nhds D) := by
    have h := hnum.div hden (by norm_num)
    rw [div_one] at h
    exact h
  refine Tendsto.congr' ?_ hquot
  filter_upwards [eventually_ge_atTop 2] with n hn
  have hn2 : (2 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  have hn0 : (n : ℝ) ≠ 0 := by positivity
  have hn1 : (n : ℝ) - 1 ≠ 0 := by nlinarith
  have h1n : (1 : ℝ) - 1 / (n : ℝ) ≠ 0 := by
    have : 1 / (n : ℝ) ≤ 1 / 2 := by
      rw [div_le_div_iff₀ (by positivity) (by norm_num)]; linarith
    nlinarith
  field_simp
  ring

open MeasureTheory ProbabilityTheory Filter in
/-- **dCov² is nonnegative.** The population squared distance covariance
    `dCov²(X, Y) = S₁ + S₂ − 2·S₃` is nonnegative whenever both marginals are of negative type
    and `π` has finite joint distance moments.

    **Proof** (sampling / V-statistic route): the empirically double-centered Hadamard sum
    `F_n` (`dcovSampleStat`) over `n` iid draws is pointwise a discrete Schur/Gram quadratic
    form, hence `0 ≤ F_n` (`dcovSampleStat_nonneg`); integrating over `π^⊗ⁿ` gives
    `0 ≤ E[F_n]` (`integral_nonneg`). The closed form `E[F_n]` (`dcov_integral_sampleStat`),
    divided by its leading coefficient `n(n−1)`, converges to `dCov²`
    (`dcov_closed_form_tendsto`); passing to the limit with `le_of_tendsto` yields
    `0 ≤ dCov²`. Mirrors `energy_distance_nonneg`.

    Reference: Székely & Rizzo (2023), Proposition 12.1 / §12; Lyons (2013), §3. -/
theorem dcov_nonneg {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    [PseudoMetricSpace α] [PseudoMetricSpace β]
    (h_negα : DistNegativeType α) (h_negβ : DistNegativeType β)
    (π : JointProbabilityMeasure α β) (hm : JointFiniteMoments π) :
    0 ≤ distanceCovarianceSq π := by
  have htend : Tendsto (fun n : ℕ =>
      (∫ ω, dcovSampleStat ω ∂(Measure.pi fun _ : Fin n => π.measure))
        / ((n : ℝ) * ((n : ℝ) - 1)))
      atTop (nhds (distanceCovarianceSq π)) := by
    refine Tendsto.congr' ?_ (dcov_closed_form_tendsto
      (∫ z, centeredDistFst π (z, z) * centeredDistSnd π (z, z) ∂π.measure)
      ((∫ p, ∫ q, dist p.1 q.1 ∂π.measure ∂π.measure)
        * (∫ p, ∫ q, dist p.2 q.2 ∂π.measure ∂π.measure))
      (distanceCovarianceSq π))
    filter_upwards [eventually_ne_atTop 0] with n hn
    have hn0 : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn
    rw [dcov_integral_sampleStat π hm hn0]
  refine ge_of_tendsto htend ?_
  filter_upwards [eventually_ge_atTop 2] with n hn
  have hnum : 0 ≤ ∫ ω, dcovSampleStat ω ∂(Measure.pi fun _ : Fin n => π.measure) :=
    integral_nonneg fun ω => dcovSampleStat_nonneg h_negα h_negβ ω
  have hden : (0 : ℝ) ≤ (n : ℝ) * ((n : ℝ) - 1) := by
    have h2 : (2 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
    nlinarith
  exact div_nonneg hnum hden

end EnergyStatistics
