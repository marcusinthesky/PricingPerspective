import PricingPerspective.Exposure.Defs
import PricingPerspective.Exposure.Energy
import EnergyStatistics.EnergyDistance
import EnergyStatistics.MeanEmbedding
import StatadicFormal.Geometry.NegativeType.Basic
import EnergyStatistics.VStatistic

/-!
# Energy Statistics Connection

Connection between PricingPerspective and EnergyStatistics packages.
-/

namespace PricingPerspective.Exposure.EnergyEmbedding

/-! ### Mean-embedding polarization (Paper 3 §methodology, C7)

Citable corollary of `norm_sub_sq_real`: the mean-embedding inner product is
recovered from norms alone, `⟪μ_i, μ_j⟫ = (‖μ_i‖² + ‖μ_j‖² − ‖μ_i − μ_j‖²) / 2`.
This identity supplies the mean-embedding polarization used in Paper 3.
-/
section MeanEmbeddingPolarization

open scoped InnerProductSpace

/-- Real polarization identity for mean embeddings: the covariance inner
    product between two mean embeddings `μ_i, μ_j` is determined by their
    norms and the norm of their difference. -/
theorem mean_embedding_polarization
    {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] (μ_i μ_j : E) :
    ⟪μ_i, μ_j⟫_ℝ = (‖μ_i‖ ^ 2 + ‖μ_j‖ ^ 2 - ‖μ_i - μ_j‖ ^ 2) / 2 := by
  have h := norm_sub_sq_real μ_i μ_j
  linarith

end MeanEmbeddingPolarization

/-! ### Paper 1 half-normalized mean embedding (G1)

`EnergyStatistics.energyEmbed` absorbs the factor of two into its point
embedding and therefore satisfies `‖m μ - m ν‖² = 𝓔²(μ,ν)`.  Appendix E of
Paper 1 instead uses the half-normalized kernel.  The declarations below make
that convention change explicit by scaling the existing embedding by `1/√2`.
-/
section HalfNormalizedMeanEmbedding

open MeasureTheory

noncomputable section

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [FiniteDimensional ℝ E] [MeasurableSpace E] [BorelSpace E]

/--
The Paper 1 mean embedding with the half-normalized distance-induced kernel.

The scale `√2 / 2 = 1 / √2` is applied outside the canonical
`EnergyStatistics.energyEmbed`, preserving that package's normalization.
-/
def halfNormalizedEnergyEmbed (μ : EnergyStatistics.ProbabilityMeasure E) :
    Lp ℝ 2 ((ProbabilityTheory.stdGaussian E).prod (volume : Measure ℝ)) :=
  (Real.sqrt 2 / 2) • EnergyStatistics.energyEmbed μ

/--
The half-normalized mean embedding has squared distance equal to one half of
the squared energy distance used by `EnergyStatistics`.
-/
theorem norm_sub_halfNormalizedEnergyEmbed_sq
    (μ ν : EnergyStatistics.ProbabilityMeasure E)
    (hμ : Integrable (fun x : E => ‖x‖) μ.measure)
    (hν : Integrable (fun y : E => ‖y‖) ν.measure) :
    ‖PricingPerspective.Exposure.EnergyEmbedding.halfNormalizedEnergyEmbed μ - PricingPerspective.Exposure.EnergyEmbedding.halfNormalizedEnergyEmbed ν‖ ^ 2 =
      (1 / 2 : ℝ) * EnergyStatistics.energyDistanceSq μ ν := by
  rw [PricingPerspective.Exposure.EnergyEmbedding.halfNormalizedEnergyEmbed, PricingPerspective.Exposure.EnergyEmbedding.halfNormalizedEnergyEmbed, ← smul_sub, norm_smul]
  rw [Real.norm_of_nonneg (div_nonneg (Real.sqrt_nonneg 2) (by norm_num))]
  rw [mul_pow, EnergyStatistics.norm_sub_energyEmbed_sq μ ν hμ hν]
  have hsqrt : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  rw [div_pow, hsqrt]
  ring

/--
**G1 (Paper 1 `√2` convention).**

Under finite first moments, population energy distance is `√2` times the norm
between the half-normalized mean embeddings.  This is the exact formal bridge
between Appendix E's kernel convention and the canonical energy embedding.
-/
theorem energyDistance_eq_sqrt_two_mul_norm_sub_halfNormalizedEnergyEmbed
    (μ ν : EnergyStatistics.ProbabilityMeasure E)
    (hμ : Integrable (fun x : E => ‖x‖) μ.measure)
    (hν : Integrable (fun y : E => ‖y‖) ν.measure) :
    Real.sqrt (EnergyStatistics.energyDistanceSq μ ν) =
      Real.sqrt 2 * ‖PricingPerspective.Exposure.EnergyEmbedding.halfNormalizedEnergyEmbed μ - PricingPerspective.Exposure.EnergyEmbedding.halfNormalizedEnergyEmbed ν‖ := by
  have hbase := EnergyStatistics.norm_sub_energyEmbed_sq μ ν hμ hν
  calc
    Real.sqrt (EnergyStatistics.energyDistanceSq μ ν) =
        Real.sqrt (‖EnergyStatistics.energyEmbed μ - EnergyStatistics.energyEmbed ν‖ ^ 2) := by
      rw [hbase]
    _ = ‖EnergyStatistics.energyEmbed μ - EnergyStatistics.energyEmbed ν‖ := by
      rw [Real.sqrt_sq_eq_abs, abs_of_nonneg (norm_nonneg _)]
    _ = Real.sqrt 2 * ‖PricingPerspective.Exposure.EnergyEmbedding.halfNormalizedEnergyEmbed μ - PricingPerspective.Exposure.EnergyEmbedding.halfNormalizedEnergyEmbed ν‖ := by
      rw [PricingPerspective.Exposure.EnergyEmbedding.halfNormalizedEnergyEmbed, PricingPerspective.Exposure.EnergyEmbedding.halfNormalizedEnergyEmbed, ← smul_sub, norm_smul]
      rw [Real.norm_of_nonneg (div_nonneg (Real.sqrt_nonneg 2) (by norm_num))]
      have hsqrt : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
      have hcoef : Real.sqrt 2 * (Real.sqrt 2 / 2) = 1 := by
        nlinarith [Real.sqrt_nonneg 2]
      rw [← mul_assoc, hcoef, one_mul]

/--
**Operator form of bounded transmission (Paper 1, sufficiency direction).**

If exposures are generated from the half-normalized energy embedding by a
bounded linear operator `A`, the resulting exposure distance obeys the
transmission bound with constant `‖A‖ / √2`:

  `‖A (Φ μ) - A (Φ ν)‖ ≤ (‖A‖ / √2) * 𝓔(μ, ν)`.

The `√2` is the same convention constant fixed by the half-normalized kernel
and carried by `PricingPerspective.Exposure.EnergyEmbedding.energyDistance_eq_sqrt_two_mul_norm_sub_halfNormalizedEnergyEmbed`;
this declaration checks that it survives composition with the operator, which
is the step where the constant is easiest to mis-normalize.

Scope. This is universally quantified over `A`: it certifies the composition
and its constant, and it is the sufficiency half of the paper's operator
proposition. It asserts no existence — that a particular economically
meaningful `A` exists is a modelling claim the manuscript makes in its own
voice, not a checked one. The necessity direction, and hence minimality of the
constant, is not formalized here.
-/
theorem energy_transmission_of_bounded_operator
    {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]
    (A : Lp ℝ 2 ((ProbabilityTheory.stdGaussian E).prod (volume : Measure ℝ)) →L[ℝ] F)
    (μ ν : EnergyStatistics.ProbabilityMeasure E)
    (hμ : Integrable (fun x : E => ‖x‖) μ.measure)
    (hν : Integrable (fun y : E => ‖y‖) ν.measure) :
    ‖A (PricingPerspective.Exposure.EnergyEmbedding.halfNormalizedEnergyEmbed μ) - A (PricingPerspective.Exposure.EnergyEmbedding.halfNormalizedEnergyEmbed ν)‖ ≤
      (‖A‖ / Real.sqrt 2) * Real.sqrt (EnergyStatistics.energyDistanceSq μ ν) := by
  have hE := PricingPerspective.Exposure.EnergyEmbedding.energyDistance_eq_sqrt_two_mul_norm_sub_halfNormalizedEnergyEmbed μ ν hμ hν
  have h2 : (0 : ℝ) < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
  rw [← map_sub, hE]
  calc ‖A (PricingPerspective.Exposure.EnergyEmbedding.halfNormalizedEnergyEmbed μ - PricingPerspective.Exposure.EnergyEmbedding.halfNormalizedEnergyEmbed ν)‖
      ≤ ‖A‖ * ‖PricingPerspective.Exposure.EnergyEmbedding.halfNormalizedEnergyEmbed μ - PricingPerspective.Exposure.EnergyEmbedding.halfNormalizedEnergyEmbed ν‖ :=
        A.le_opNorm _
    _ = ‖A‖ / Real.sqrt 2 *
        (Real.sqrt 2 * ‖PricingPerspective.Exposure.EnergyEmbedding.halfNormalizedEnergyEmbed μ - PricingPerspective.Exposure.EnergyEmbedding.halfNormalizedEnergyEmbed ν‖) := by
        field_simp

/-! ### Finite-sample embedding identity (G3) -/

/--
The empirical mean of the canonical energy point embeddings for a finite
sample.  Nonempty sample sizes are imposed by the identities below rather than
by this total definition.
-/
def empiricalEnergyPointMean {n : ℕ} (x : Fin n → E) :
    Lp ℝ 2 ((ProbabilityTheory.stdGaussian E).prod (volume : Measure ℝ)) :=
  (n : ℝ)⁻¹ • ∑ i, EnergyStatistics.energyPointEmbed (x i)

/--
The empirical mean embedding under Paper 1's half-normalized kernel convention.
-/
def halfNormalizedEmpiricalEnergyPointMean {n : ℕ} (x : Fin n → E) :
    Lp ℝ 2 ((ProbabilityTheory.stdGaussian E).prod (volume : Measure ℝ)) :=
  (Real.sqrt 2 / 2) • PricingPerspective.Exposure.EnergyEmbedding.empiricalEnergyPointMean x

omit [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
    [MeasurableSpace E] [BorelSpace E] in
private lemma sum_energyPointKernel_eq {n m : ℕ} (x : Fin n → E) (y : Fin m → E) :
    ∑ i, ∑ j, (‖x i‖ + ‖y j‖ - dist (x i) (y j)) =
      (m : ℝ) * ∑ i, ‖x i‖ + (n : ℝ) * ∑ j, ‖y j‖ -
        ∑ i, ∑ j, dist (x i) (y j) := by
  simp_rw [Finset.sum_sub_distrib, Finset.sum_add_distrib]
  simp [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  simp_rw [← Finset.mul_sum]

private lemma inner_empiricalEnergyPointMean {n m : ℕ}
    (x : Fin n → E) (y : Fin m → E) :
    inner ℝ (PricingPerspective.Exposure.EnergyEmbedding.empiricalEnergyPointMean x) (PricingPerspective.Exposure.EnergyEmbedding.empiricalEnergyPointMean y) =
      (n : ℝ)⁻¹ * (m : ℝ)⁻¹ *
        ∑ i, ∑ j, (‖x i‖ + ‖y j‖ - dist (x i) (y j)) := by
  rw [PricingPerspective.Exposure.EnergyEmbedding.empiricalEnergyPointMean, PricingPerspective.Exposure.EnergyEmbedding.empiricalEnergyPointMean]
  simp_rw [real_inner_smul_left, real_inner_smul_right, sum_inner, inner_sum,
    EnergyStatistics.inner_energyPointEmbed]
  ring

/--
For two nonempty finite samples, the usual energy V-statistic radicand equals
the squared distance between their canonical empirical energy embeddings.
-/
theorem finiteSampleEnergyRadicand_eq_norm_sub_empiricalEnergyPointMean_sq
    {n m : ℕ} (x : Fin n → E) (y : Fin m → E)
    (hn : n ≠ 0) (hm : m ≠ 0) :
    2 * (∑ i, ∑ j, dist (x i) (y j)) / ((n : ℝ) * (m : ℝ)) -
        (∑ i, ∑ j, dist (x i) (x j)) / ((n : ℝ) * (n : ℝ)) -
        (∑ i, ∑ j, dist (y i) (y j)) / ((m : ℝ) * (m : ℝ)) =
      ‖PricingPerspective.Exposure.EnergyEmbedding.empiricalEnergyPointMean x - PricingPerspective.Exposure.EnergyEmbedding.empiricalEnergyPointMean y‖ ^ 2 := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn
  have hm' : (m : ℝ) ≠ 0 := by exact_mod_cast hm
  rw [norm_sub_sq_real,
    ← real_inner_self_eq_norm_sq (PricingPerspective.Exposure.EnergyEmbedding.empiricalEnergyPointMean x),
    ← real_inner_self_eq_norm_sq (PricingPerspective.Exposure.EnergyEmbedding.empiricalEnergyPointMean y),
    PricingPerspective.Exposure.EnergyEmbedding.inner_empiricalEnergyPointMean, PricingPerspective.Exposure.EnergyEmbedding.inner_empiricalEnergyPointMean,
    PricingPerspective.Exposure.EnergyEmbedding.inner_empiricalEnergyPointMean, PricingPerspective.Exposure.EnergyEmbedding.sum_energyPointKernel_eq,
    PricingPerspective.Exposure.EnergyEmbedding.sum_energyPointKernel_eq, PricingPerspective.Exposure.EnergyEmbedding.sum_energyPointKernel_eq]
  field_simp
  ring

/--
**G3 (Finite-Sample Radicand Embedding Identity).**

For two nonempty finite samples, the energy V-statistic radicand is twice the
squared distance between their half-normalized empirical mean embeddings.  In
particular it is nonnegative; no clipping convention is involved, and the
within-sample sums retain their diagonal terms exactly as a V-statistic does.
-/
theorem finiteSampleEnergyRadicand_eq_two_mul_norm_sub_halfNormalizedMean_sq
    {n m : ℕ} (x : Fin n → E) (y : Fin m → E)
    (hn : n ≠ 0) (hm : m ≠ 0) :
    2 * (∑ i, ∑ j, dist (x i) (y j)) / ((n : ℝ) * (m : ℝ)) -
        (∑ i, ∑ j, dist (x i) (x j)) / ((n : ℝ) * (n : ℝ)) -
        (∑ i, ∑ j, dist (y i) (y j)) / ((m : ℝ) * (m : ℝ)) =
      2 * ‖PricingPerspective.Exposure.EnergyEmbedding.halfNormalizedEmpiricalEnergyPointMean x -
        PricingPerspective.Exposure.EnergyEmbedding.halfNormalizedEmpiricalEnergyPointMean y‖ ^ 2 := by
  rw [PricingPerspective.Exposure.EnergyEmbedding.finiteSampleEnergyRadicand_eq_norm_sub_empiricalEnergyPointMean_sq x y hn hm]
  rw [PricingPerspective.Exposure.EnergyEmbedding.halfNormalizedEmpiricalEnergyPointMean, PricingPerspective.Exposure.EnergyEmbedding.halfNormalizedEmpiricalEnergyPointMean,
    ← smul_sub, norm_smul]
  rw [Real.norm_of_nonneg (div_nonneg (Real.sqrt_nonneg 2) (by norm_num))]
  have hsqrt : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  nlinarith

end

end HalfNormalizedMeanEmbedding

/-! ### Carrying the energy-kernel embedding to measures (Paper 3)

`Exposure/Energy/Core.lean` builds `hilbertEnergyKernelEmbedding`, whose `energy_sq`
slot is `‖· − ·‖²` on a real inner-product space. The declarations below carry that
instance from *exposures* to *probability measures* through
`EnergyStatistics.energyEmbed`, constructed in `MeanEmbedding.lean`.

**Constant.** Sejdinovic et al. (2013) Thm 22 reads `D_{E,ρ} = 2γ_k²` for a kernel `k`
that generates `ρ` via `ρ(z,w) = k(z,z) + k(w,w) − 2k(z,w)` — the *half-normalized*
kernel. `EnergyStatistics.energyPointEmbed` absorbs a `√(2/c)` rescale and satisfies
`⟪Φx, Φy⟫ = ‖x‖ + ‖y‖ − dist x y` (`inner_energyPointEmbed`), i.e. the **unhalved**
kernel at reference point `0`. That doubles `γ²`, so the constant here is `1`, not `2`.
Paper 1's `√2` lives in `PricingPerspective.Exposure.EnergyEmbedding.halfNormalizedEnergyEmbed` above, not in this section.

**Not established here.** Three gaps, none of them closed by the two theorems below:

* *Generality.* Thm 22 holds on any semimetric space of negative type. What is proved
  here is the finite-dimensional real inner-product case, via the explicit
  Gaussian-projection construction in `StatadicFormal.Geometry.NegativeType.Basic` — not the general
  strong-negative-type result.
* *The economic reading.* `sys_cov` is an `L²` inner product of mean embeddings. That
  it is the **systematic covariance of returns** is a modelling claim formalized
  nowhere in this development, and one the paper's own GMM/Wolak tests reject.
* *Injectivity.* Equality of the two functionals is not separation of measures; that is
  the strong-negative-type question, left open.

**Why there is no `EnergyKernelEmbedding (ProbabilityMeasure E)` instance.** The
structure's `identity` field is unconditional (`∀ β_i β_j`), while
`norm_sub_energyEmbed_sq` carries essential finite-first-moment hypotheses — Thm 22
needs them too (Sejdinovic Remark 23). So `identity` cannot be discharged from it at
`energy_sq := energyDistanceSq`. Whether the identity instead holds *vacuously* off the
finite-moment set — where `energyEmbed` and the three integrals in `energyDistanceSq`
each independently degenerate to `0` under Bochner's junk-value convention — was not
determined. Constructing the instance therefore needs a finite-first-moment subtype,
which is deliberately not introduced here.

References: Sejdinovic et al. (2013) Thm 22 and Remark 23; Lyons (2013) §3.
-/
section EnergyKernelEmbeddingBridge

open MeasureTheory

noncomputable section

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [FiniteDimensional ℝ E] [MeasurableSpace E] [BorelSpace E]

/--
**s4 slot alignment.** Evaluated at canonical mean embeddings, the `energy_sq` slot of
`hilbertEnergyKernelEmbedding` *is* the population squared energy distance:

  `hilbertEnergyKernelEmbedding.energy_sq (Φ μ) (Φ ν) = 𝓔²(μ, ν)`.

This is `EnergyStatistics.norm_sub_energyEmbed_sq` read through the structure's slot, and
it is deliberately **not** named `energy_sq_eq_energyDistanceSq`: the field `energy_sq` is
not being identified with `energyDistanceSq`: only its value at embedded measures is,
and only under finite first moments. See the section docstring for what that leaves open.
-/
theorem energy_sq_energyEmbed_eq_energyDistanceSq
    (μ ν : EnergyStatistics.ProbabilityMeasure E)
    (hμ : Integrable (fun x : E => ‖x‖) μ.measure)
    (hν : Integrable (fun y : E => ‖y‖) ν.measure) :
    (PricingPerspective.Exposure.Energy.hilbertEnergyKernelEmbedding
        (E := Lp ℝ 2 ((ProbabilityTheory.stdGaussian E).prod (volume : Measure ℝ)))).energy_sq
        (EnergyStatistics.energyEmbed μ) (EnergyStatistics.energyEmbed ν) =
      EnergyStatistics.energyDistanceSq μ ν :=
  EnergyStatistics.norm_sub_energyEmbed_sq μ ν hμ hν

/--
**The identity layer at the measure level.**

Applying `energy_to_covariance_exact_identity` at `hilbertEnergyKernelEmbedding` and the
slot alignment above gives the exact decomposition of the *population squared energy
distance* into embedding variances and covariance:

  `𝓔²(μ, ν) = ‖Φ μ‖² + ‖Φ ν‖² − 2⟪Φ μ, Φ ν⟫`.

At exposures, `energy_sq` is `‖β_i − β_j‖²` by construction. Here the left-hand side
is a genuine energy distance between probability measures.

**Scope.** `⟪Φ μ, Φ ν⟫` is an `L²` inner product of mean embeddings. Reading it as
systematic return covariance is the paper's modelling claim, is not formalized anywhere in
this development, and is the claim its own GMM/Wolak tests reject. This theorem licenses
the *algebra*, not that reading.
-/
theorem energyDistanceSq_eq_energyEmbed_covariance_form
    (μ ν : EnergyStatistics.ProbabilityMeasure E)
    (hμ : Integrable (fun x : E => ‖x‖) μ.measure)
    (hν : Integrable (fun y : E => ‖y‖) ν.measure) :
    EnergyStatistics.energyDistanceSq μ ν =
      ‖EnergyStatistics.energyEmbed μ‖ ^ 2 + ‖EnergyStatistics.energyEmbed ν‖ ^ 2 -
        2 * inner ℝ (EnergyStatistics.energyEmbed μ) (EnergyStatistics.energyEmbed ν) := by
  rw [← EnergyStatistics.norm_sub_energyEmbed_sq μ ν hμ hν]
  exact PricingPerspective.Exposure.Energy.energy_to_covariance_exact_identity_hilbert
    (EnergyStatistics.energyEmbed μ) (EnergyStatistics.energyEmbed ν)

end

end EnergyKernelEmbeddingBridge

/-! ### Euclidean energy V-statistic nonnegativity

    Uses Statadic’s canonical Schoenberg theorem directly to discharge the
    negative-type hypothesis for Euclidean/Hilbert-space samples.
-/
section EuclideanNegativeType

/--
**G3 (Finite-Sample Energy Radicand Nonnegativity).**

For samples in any real Hilbert space, the two-sample energy V-statistic is
nonnegative.  The result specializes `EnergyStatistics.v_statistic_nonneg`
with the proved Schoenberg negative-type instance, so the within-sample
diagonals remain exactly those of the manuscript's V-statistic rather than a
clipped numerical convention.
-/
theorem hilbert_vStatistic_nonneg
    {E : Type*} [MeasurableSpace E] [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (Xs Ys : List E) :
    0 ≤ EnergyStatistics.vStatistic Xs Ys :=
  EnergyStatistics.v_statistic_nonneg
    (StatadicFormal.Geometry.distNegativeType_of_innerProductSpace E) Xs Ys

end EuclideanNegativeType

/-! ### B-core corollaries: energy-matrix CND and mixture-QP convexity

Corollaries of `EnergyStatistics.energyDistanceSq_cnd` (population conditional negative
definiteness of the energy-distance matrix) for mixture optimization.
-/
section EnergyMatrixCND

open EnergyStatistics

variable {α : Type*} [MeasurableSpace α] [PseudoMetricSpace α]

/-- **Cross-moment Gram CND.** The energy-distance
Gram matrix `[𝓔²(ν_k, ν_l)]` is conditionally negative definite on zero-sum vectors: for a
negative-type space and pairwise finite distance moments, `∑ₖₗ uₖuₗ 𝓔²(ν_k,ν_l) ≤ 0` whenever
`∑u = 0`.  Restatement of `EnergyStatistics.energyDistanceSq_cnd` for manuscript citation: the
mixture-QP objective `−w'Qw` is thus convex along the simplex — the mechanism the manuscript
should state in place of "Q is positive semidefinite". -/
theorem energyDistanceGram_cnd
    (h_neg : StatadicFormal.Geometry.DistNegativeType α) {K : ℕ} (ν : Fin K → ProbabilityMeasure α)
    (hmom : ∀ k l, FiniteDistMoment (ν k) (ν l))
    (u : Fin K → ℝ) (hsum : ∑ k, u k = 0) :
    ∑ k, ∑ l, u k * u l * energyDistanceSq (ν k) (ν l) ≤ 0 :=
  energyDistanceSq_cnd h_neg ν hmom u hsum

/-- CND of the square-root energy matrix `D_{kl} = √𝓔²(ν_k, ν_l)`: since `(√𝓔²)² = 𝓔²`
(by `Real.sq_sqrt` and `energy_distance_nonneg`), the squared-distance matrix `[D_{kl}²]`
coincides with the energy Gram, so `energyDistanceSq_cnd` supplies the CND inequality in the
exact `∑ₖₗ uₖuₗ D_{kl}² ≤ 0` shape required by `Q_concave_on_simplex.hCND`. -/
theorem energyDistanceSqrt_cnd
    (h_neg : StatadicFormal.Geometry.DistNegativeType α) {K : ℕ} (ν : Fin K → ProbabilityMeasure α)
    (hmom : ∀ k l, FiniteDistMoment (ν k) (ν l))
    (u : Fin K → ℝ) (hsum : ∑ k, u k = 0) :
    ∑ k, ∑ l, u k * u l * Real.sqrt (energyDistanceSq (ν k) (ν l)) ^ 2 ≤ 0 := by
  have hsq : ∀ k l, Real.sqrt (energyDistanceSq (ν k) (ν l)) ^ 2
      = energyDistanceSq (ν k) (ν l) :=
    fun k l => Real.sq_sqrt
      (energy_distance_nonneg h_neg (ν k) (ν l) (hmom k l) (hmom k k) (hmom l l))
  simp_rw [hsq]
  exact energyDistanceSq_cnd h_neg ν hmom u hsum

/-- **Mixture-QP convexity for the energy matrix.** Feeding `PricingPerspective.Exposure.EnergyEmbedding.energyDistanceSqrt_cnd`
into `Q_concave_on_simplex`: the price-free quadratic `Q(w) = ∑ₖₗ wₖwₗ 𝓔²(ν_k,ν_l)` (with chord
matrix `D_{kl} = √𝓔²`) is concave along the sum-one simplex, so the mixture-energy objective is
a concave program (every local optimum is global). -/
theorem energyMatrix_Q_concave_on_simplex
    (h_neg : StatadicFormal.Geometry.DistNegativeType α) {K : ℕ} (ν : Fin K → ProbabilityMeasure α)
    (hmom : ∀ k l, FiniteDistMoment (ν k) (ν l))
    (w₁ w₂ : Fin K → ℝ) (hsum₁ : ∑ i, w₁ i = 1) (hsum₂ : ∑ i, w₂ i = 1)
    (θ : ℝ) (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) :
    θ * (∑ i, ∑ j, w₁ i * w₁ j * Real.sqrt (energyDistanceSq (ν i) (ν j)) ^ 2) +
        (1 - θ) * (∑ i, ∑ j, w₂ i * w₂ j * Real.sqrt (energyDistanceSq (ν i) (ν j)) ^ 2) ≤
      ∑ i, ∑ j, (θ * w₁ i + (1 - θ) * w₂ i) * (θ * w₁ j + (1 - θ) * w₂ j) *
        Real.sqrt (energyDistanceSq (ν i) (ν j)) ^ 2 :=
  PricingPerspective.Exposure.Energy.Q_concave_on_simplex
    (fun i j => Real.sqrt (energyDistanceSq (ν i) (ν j)))
    (fun u hu => PricingPerspective.Exposure.EnergyEmbedding.energyDistanceSqrt_cnd h_neg ν hmom u hu) w₁ w₂ hsum₁ hsum₂ θ hθ0 hθ1

/-- **C2 — simplex convexity of the mixture-energy objective.** Consuming the
mixture-energy expansion `energyDistanceSq_simplexMixture` and the distance-Gram CND
`energyCrossGram_cnd`, the map `w ↦ 𝓔²(μ, mix w ν)` is convex along the probability simplex: for
simplex weights `w₁, w₂` and `θ ∈ [0,1]`,

  `𝓔²(μ, mix (θ w₁ + (1−θ) w₂) ν) ≤ θ·𝓔²(μ, mix w₁ ν) + (1−θ)·𝓔²(μ, mix w₂ ν)`.

The cross term `2 c'w` is affine and the constant `b` is fixed, so the only content is that the
quadratic `−w'Qw` is convex — equivalently the distance-cross-moment Gram `Q` is CND on zero-sum
vectors (`energyCrossGram_cnd`, applied through `Q_concave_on_simplex` with `D_{kl} = √Q_{kl}`,
`Q_{kl} ≥ 0`). The combined-weight nonnegativity/normalization (`hcnn`, `hcs`) are logically
forced by the simplex hypotheses on `w₁, w₂` and `θ`; they are taken as explicit arguments only
because they appear in the mixture term of the statement. -/
theorem energyDistanceSq_simplexMixture_convex
    (μ : ProbabilityMeasure α) (h_neg : StatadicFormal.Geometry.DistNegativeType α)
    {K : ℕ} (ν : Fin K → ProbabilityMeasure α)
    (h_μμ : FiniteDistMoment μ μ) (h_μν : ∀ k, FiniteDistMoment μ (ν k))
    (h_νν : ∀ k l, FiniteDistMoment (ν k) (ν l))
    (w₁ w₂ : Fin K → ℝ)
    (h1nn : ∀ k, 0 ≤ w₁ k) (h1s : ∑ k, w₁ k = 1)
    (h2nn : ∀ k, 0 ≤ w₂ k) (h2s : ∑ k, w₂ k = 1)
    (θ : ℝ) (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1)
    (hcnn : ∀ k, 0 ≤ θ * w₁ k + (1 - θ) * w₂ k)
    (hcs : ∑ k, (θ * w₁ k + (1 - θ) * w₂ k) = 1) :
    energyDistanceSq μ (simplexMixture (fun k => θ * w₁ k + (1 - θ) * w₂ k) ν hcnn hcs)
      ≤ θ * energyDistanceSq μ (simplexMixture w₁ ν h1nn h1s)
        + (1 - θ) * energyDistanceSq μ (simplexMixture w₂ ν h2nn h2s) := by
  rw [energyDistanceSq_simplexMixture μ (fun k => θ * w₁ k + (1 - θ) * w₂ k) ν hcnn hcs
        h_μμ h_μν h_νν,
      energyDistanceSq_simplexMixture μ w₁ ν h1nn h1s h_μμ h_μν h_νν,
      energyDistanceSq_simplexMixture μ w₂ ν h2nn h2s h_μμ h_μν h_νν]
  -- linearity of the cross term `∑ w_k c_k`
  have hClin : (∑ k, (θ * w₁ k + (1 - θ) * w₂ k) * mixC μ ν k)
      = θ * (∑ k, w₁ k * mixC μ ν k) + (1 - θ) * (∑ k, w₂ k * mixC μ ν k) := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun k _ => by ring
  rw [hClin]
  -- concavity of the Gram quadratic `∑∑ w_k w_l Q_{kl}` via `Q_concave_on_simplex`
  have hQnn : ∀ k l, 0 ≤ mixQ ν k l :=
    fun k l => MeasureTheory.integral_nonneg fun _ => dist_nonneg
  have hsq : ∀ k l, Real.sqrt (mixQ ν k l) ^ 2 = mixQ ν k l := fun k l => Real.sq_sqrt (hQnn k l)
  have hCND : ∀ u : Fin K → ℝ, (∑ i, u i) = 0 →
      ∑ i, ∑ j, u i * u j * Real.sqrt (mixQ ν i j) ^ 2 ≤ 0 := by
    intro u hu
    simp_rw [hsq, mixQ]
    exact energyCrossGram_cnd h_neg ν h_νν u hu
  have hconc := PricingPerspective.Exposure.Energy.Q_concave_on_simplex
    (fun k l => Real.sqrt (mixQ ν k l)) hCND w₁ w₂ h1s h2s θ hθ0 hθ1
  simp_rw [hsq] at hconc
  nlinarith [hconc]

end EnergyMatrixCND

end PricingPerspective.Exposure.EnergyEmbedding
