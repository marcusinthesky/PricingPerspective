import PricingPerspective.Discrete.Spatial
import Mathlib.LinearAlgebra.Matrix.ToLinearEquiv
import Mathlib.Topology.Instances.Matrix

/-!
# Spatial asymptotics: segment/stationary structure and the strong-interaction limit

Paper 5 rung `C2` (plan `paper5/t06-spatial-asymptotics-rungs`): the "memorable theorems" of the
spatial-lag program, built on the finite-`ρ` spine of `Discrete/Spatial.lean`
(`leontief`, `spatialCov`, `spatialCov_factor_model`). One-writer-per-file: this module CONSUMES
`Discrete/Spatial.lean` and never edits it.

Two results.

* **S6a — segment/stationary structure.** For a row-stochastic `W` (which the energy-implied `W♭`
  is by construction), a single irreducible/primitive block is modelled by the whole index type
  `ι` (the block-diagonal decomposition of the market into segments is structural/prose). The
  Perron–Frobenius theorem for primitive nonnegative matrices supplies a stationary distribution
  `π` and the power limit `Wᵏ → 𝟙 πᵀ`; that limit is packaged as `IsPerronLimit W π`.
  A quantitative geometric bound is packaged separately as `HasGeometricPerronBound` and is
  machine-proved sufficient for `IsPerronLimit`. Conditional on that limit,
  `replication_centrality` shows every `k`-step
  interaction cascade `Wᵏ *ᵥ x` converges to the *constant* vector `(π ⬝ᵥ x) • 𝟙`, whose common
  entry is the `π`-weighted average of the starting vector `x`, and
  `perron_stationary` extracts the left-eigenvector (stationarity) relation `π ᵥ* W = π`. The
  elementary stochastic algebra — `𝟙` as a right eigenvector, closure of row-stochasticity under
  products/powers, and idempotence of the rank-one projection `𝟙 πᵀ` — is proved in full (no proof
  holes) with no Perron–Frobenius input.

* **S6b — rank-one common-mode limit.** As `ρ ↑ 1`, `Σ_SAR = (1−ρW)⁻¹ V (1−ρW)⁻ᵀ` develops the
  dominant rank-one component `(1−ρ)⁻²(πᵀ V π) 𝟙𝟙ᵀ`. The exact projection collapse
  `𝟙 πᵀ · V · (𝟙 πᵀ)ᵀ = (πᵀ V π) 𝟙𝟙ᵀ` (`projCov_collapse`) is elementary. Conditional on the
  resolvent limit `(1−ρ)(1−ρW)⁻¹ → 𝟙 πᵀ` (packaged as the named hypothesis `IsResolventLimit W π`,
  consumed by `sigmaSAR_rankOne_limit`, and for row-stochastic `W` *discharged in-file* from the
  power limit by `isResolventLimit_of_isPerronLimit`), the rescaled spatial covariance
  `(1−ρ)² Σ_SAR` converges to that rank-one matrix.
  `capm_market_loading`/`capm_market_variance`/`capm_beta_collapse` machine-check the purely
  algebraic fact that under a rank-one covariance `c 𝟙𝟙ᵀ` with `c ≠ 0`, every asset's beta against
  the common mode equals one, for *any* fully-invested weight vector `wM` (`∑ wM = 1`);
  `capm_recovered_in_limit` instantiates `c = πᵀ V π`.

  **Scope guard — what S6b does not establish.** The beta collapse is the whole formal content.
  Nothing here identifies `π` (or any other vector) as a market portfolio, a traded basket, or a
  tracking device; `wM` is an arbitrary fully-invested vector, unrelated to `π`, and `π` enters
  `capm_recovered_in_limit` only through the scalar `c = πᵀ V π`. Nothing here asserts aggregate
  wealth, market clearing, mean-variance efficiency, a nonzero risk premium, a beta-pricing
  equilibrium, or the recovery of the classical CAPM. `π` is a limit object of the interaction
  cascade only — never a priced characteristic.

## Perron–Frobenius disclosure

Full Perron–Frobenius for primitive nonnegative matrices is not developed in-repo. The qualitative
power limit is disclosed as `IsPerronLimit W π`. The quantitative predicate
`HasGeometricPerronBound W π C q` records an auditable contraction premise, and
`isPerronLimit_of_geometric_bound` proves that it discharges the qualitative limit when
`0 ≤ q < 1`. The companion resolvent predicate is not assumed independently: for row-stochastic
`W` it is proved from the power limit by elementary linear algebra. The surrounding algebra is
proved in full and adds no non-constructive declaration to the inventory.
-/

namespace PricingPerspective.Connections.SpatialAsymptotics

open Matrix Filter PricingPerspective.Discrete.Spatial
open scoped Matrix.Norms.Operator Topology

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-! ## Elementary row-stochastic and rank-one algebra (Perron–Frobenius-free) -/

/-- The all-ones vector is a right eigenvector of a row-stochastic matrix: `W *ᵥ 𝟙 = 𝟙`. -/
theorem rowStochastic_mulVec_one {W : Matrix ι ι ℝ} (hW : ∀ i, ∑ j, W i j = 1) :
    W *ᵥ (1 : ι → ℝ) = 1 := by
  funext i
  simp only [Matrix.mulVec, dotProduct, Pi.one_apply, mul_one]
  exact hW i

/-- Row-stochasticity is closed under matrix multiplication: the product of two row-stochastic
matrices is row-stochastic. -/
theorem rowStochastic_mul {W W' : Matrix ι ι ℝ} (hW : ∀ i, ∑ j, W i j = 1)
    (hW' : ∀ i, ∑ j, W' i j = 1) (i : ι) : ∑ j, (W * W') i j = 1 := by
  simp only [Matrix.mul_apply]
  rw [Finset.sum_comm]
  have hstep : ∀ k, ∑ j, W i k * W' k j = W i k := fun k => by
    rw [← Finset.mul_sum, hW' k, mul_one]
  rw [Finset.sum_congr rfl fun k _ => hstep k]
  exact hW i

/-- Row-stochasticity is closed under powers: `Wᵏ` is row-stochastic for every `k`, so every
`k`-step interaction cascade preserves total mass (the network multiplier is mass-conserving). -/
theorem rowStochastic_pow {W : Matrix ι ι ℝ} (hW : ∀ i, ∑ j, W i j = 1) :
    ∀ (k : ℕ) (i : ι), ∑ j, (W ^ k) i j = 1 := by
  intro k
  induction k with
  | zero => intro i; simp [pow_zero, Matrix.one_apply, Finset.sum_ite_eq]
  | succ n ih => intro i; rw [pow_succ]; exact rowStochastic_mul ih hW i

/-- Stationary-interaction-centrality kernel: the rank-one projection `𝟙 πᵀ` sends every vector to a
multiple of the all-ones vector `𝟙`, namely `(𝟙 πᵀ) *ᵥ x = (π ⬝ᵥ x) • 𝟙` — a constant vector whose
common entry is the `π`-weighted average of `x`. -/
theorem vecMulVec_one_mulVec (π x : ι → ℝ) :
    vecMulVec (1 : ι → ℝ) π *ᵥ x = (π ⬝ᵥ x) • (1 : ι → ℝ) := by
  rw [Matrix.vecMulVec_mulVec, op_smul_eq_smul]

/-- The rank-one projection `𝟙 πᵀ` is idempotent when `π` is a distribution (`∑ π = 1`); it is the
limiting one-step-equals-`k`-step barycentric interaction operator. -/
theorem vecMulVec_one_idempotent {π : ι → ℝ} (hπ : ∑ i, π i = 1) :
    vecMulVec (1 : ι → ℝ) π * vecMulVec (1 : ι → ℝ) π = vecMulVec (1 : ι → ℝ) π := by
  rw [Matrix.vecMulVec_mul_vecMulVec, dotProduct_one, hπ, one_smul]

/-- Cancellation from the left `𝟙` factor: `𝟙 aᵀ = 𝟙 bᵀ` forces `a = b` (needs a nonempty market). -/
theorem vecMulVec_one_left_inj [Nonempty ι] {a b : ι → ℝ}
    (h : vecMulVec (1 : ι → ℝ) a = vecMulVec (1 : ι → ℝ) b) : a = b := by
  funext j
  have hj : vecMulVec (1 : ι → ℝ) a (Classical.arbitrary ι) j
      = vecMulVec (1 : ι → ℝ) b (Classical.arbitrary ι) j := by rw [h]
  simpa only [Matrix.vecMulVec_apply, Pi.one_apply, one_mul] using hj

/-- Exact projection collapse of the innovation covariance (the S6b rank-one core, no limit):
`(𝟙 πᵀ) V (𝟙 πᵀ)ᵀ = (πᵀ V π) 𝟙𝟙ᵀ`. -/
theorem projCov_collapse (π : ι → ℝ) (V : Matrix ι ι ℝ) :
    vecMulVec (1 : ι → ℝ) π * V * (vecMulVec (1 : ι → ℝ) π)ᵀ
      = (π ⬝ᵥ (V *ᵥ π)) • vecMulVec (1 : ι → ℝ) 1 := by
  rw [Matrix.transpose_vecMulVec, Matrix.vecMulVec_mul, Matrix.vecMulVec_mul_vecMulVec]
  ext i j
  simp only [Matrix.vecMulVec_apply, Matrix.smul_apply, Pi.smul_apply, Pi.one_apply, smul_eq_mul,
    mul_one, one_mul]
  exact (dotProduct_mulVec π V π).symm

/-! ## The Perron–Frobenius hypotheses (disclosed) -/

/-- **Perron–Frobenius primitive-convergence hypothesis** (assumed; NOT proved in-repo).

For a primitive (irreducible + aperiodic) row-stochastic `W`, the Perron–Frobenius theorem for
primitive nonnegative matrices gives a unique stationary distribution `π ≥ 0`, `∑ π = 1`, with
`Wᵏ → 𝟙 πᵀ` (entrywise, hence in the product topology). This predicate packages exactly that
limit. It is the sole Perron–Frobenius input to the S6a theorems below; each carries it explicitly
as a hypothesis, and nothing here asserts it. -/
def IsPerronLimit (W : Matrix ι ι ℝ) (π : ι → ℝ) : Prop :=
  Tendsto (fun k : ℕ => W ^ k) atTop (𝓝 (vecMulVec (1 : ι → ℝ) π))

/-- A quantitative, auditable replacement for an opaque Perron--Frobenius invocation: the powers
of `W` approach `𝟙 πᵀ` at a geometric rate.  For the empirical operator this premise can be
certified numerically (for example from a contraction coefficient or a positive matrix power),
while the theorem below checks that it is sufficient for every result stated through
`IsPerronLimit`.

The predicate deliberately does not claim that primitivity alone has been formalized in-repo; it
records the exact quantitative fact consumed by the SAR asymptotics. -/
def HasGeometricPerronBound (W : Matrix ι ι ℝ) (π : ι → ℝ) (C q : ℝ) : Prop :=
  ∀ k : ℕ, ‖W ^ k - vecMulVec (1 : ι → ℝ) π‖ ≤ C * q ^ k

/-- A geometric Perron bound with `0 ≤ q < 1` discharges the named Perron-limit
hypothesis.  This is the formal bridge from a finite-dimensional contraction certificate to the
qualitative power limit used by the spatial theory. -/
theorem isPerronLimit_of_geometric_bound {W : Matrix ι ι ℝ} {π : ι → ℝ} {C q : ℝ}
    (hq0 : 0 ≤ q) (hq1 : q < 1) (hgeom : HasGeometricPerronBound W π C q) :
    IsPerronLimit W π := by
  unfold IsPerronLimit
  apply tendsto_iff_norm_sub_tendsto_zero.mpr
  apply squeeze_zero' (Eventually.of_forall fun k => norm_nonneg (W ^ k - vecMulVec 1 π))
    (Eventually.of_forall hgeom)
  simpa using
    (tendsto_const_nhds.mul (tendsto_pow_atTop_nhds_zero_of_lt_one hq0 hq1) :
      Tendsto (fun k : ℕ => C * q ^ k) atTop (𝓝 (C * 0)))

/-- **Resolvent-limit predicate** (named hypothesis; for row-stochastic `W` it is *proved* in this
file, not assumed).

As `ρ ↑ 1`, the rescaled Leontief resolvent `(1−ρ)(1−ρW)⁻¹ = (1−ρ) • leontief ρ W` converges to the
rank-one matrix `𝟙 πᵀ`. This predicate packages exactly that limit, so that `sigmaSAR_rankOne_limit`
below can carry it explicitly in its signature.

It is **not** an independent Perron–Frobenius input: for a row-stochastic `W` it follows from the
power limit `IsPerronLimit W π` by elementary linear algebra — see
`isResolventLimit_of_isPerronLimit` — with no Abel/Tauberian summation and no spectral theory. -/
def IsResolventLimit (W : Matrix ι ι ℝ) (π : ι → ℝ) : Prop :=
  Tendsto (fun ρ : ℝ => (1 - ρ) • leontief ρ W) (𝓝[<] (1 : ℝ)) (𝓝 (vecMulVec (1 : ι → ℝ) π))

/-! ## S6a: segment/stationary structure -/

/-- **S6a (stationary interaction centrality).** Conditional on the Perron–Frobenius power limit,
every `k`-step interaction cascade `Wᵏ *ᵥ x` converges to the constant vector `(π ⬝ᵥ x) • 𝟙`, whose
common entry is the `π`-weighted average `π ⬝ᵥ x` of the starting vector `x`.

`x` is an arbitrary vector — it is not assumed normalized and need not be read as a portfolio — and
the limit is a constant vector, not a basket. The declaration name is historical; nothing here
identifies `π` as a portfolio or a traded basket. -/
theorem replication_centrality {W : Matrix ι ι ℝ} {π : ι → ℝ} (hPF : IsPerronLimit W π)
    (x : ι → ℝ) : Tendsto (fun k : ℕ => (W ^ k) *ᵥ x) atTop (𝓝 ((π ⬝ᵥ x) • (1 : ι → ℝ))) := by
  have hPF' : Tendsto (fun k : ℕ => W ^ k) atTop (𝓝 (vecMulVec (1 : ι → ℝ) π)) := hPF
  have hcont : Continuous (fun M : Matrix ι ι ℝ => M *ᵥ x) := by fun_prop
  have h := (hcont.tendsto _).comp hPF'
  rw [vecMulVec_one_mulVec] at h
  exact h

/-- **S6a (stationarity).** Conditional on the Perron–Frobenius power limit, the limit vector `π` is
stationary for the interaction chain — the left Perron eigenvector relation `π ᵥ* W = π`. This is
the genuine Perron–Frobenius content (the right-eigenvector relation `W *ᵥ 𝟙 = 𝟙` is elementary).

This statement gives the fixed-point relation only; that `π` is in addition nonnegative with unit
total mass — hence a distribution — is established separately by `perron_nonneg` and
`perron_sum_eq_one`, under further hypotheses on `W`. -/
theorem perron_stationary [Nonempty ι] {W : Matrix ι ι ℝ} {π : ι → ℝ}
    (hPF : IsPerronLimit W π) : π ᵥ* W = π := by
  have hPF' : Tendsto (fun k : ℕ => W ^ k) atTop (𝓝 (vecMulVec (1 : ι → ℝ) π)) := hPF
  have e1 : Tendsto (fun k : ℕ => W ^ k * W) atTop (𝓝 (vecMulVec (1 : ι → ℝ) π)) := by
    simpa only [Function.comp_def, pow_succ] using hPF'.comp (tendsto_add_atTop_nat 1)
  have hcont : Continuous (fun M : Matrix ι ι ℝ => M * W) := by fun_prop
  have e2 : Tendsto (fun k : ℕ => W ^ k * W) atTop (𝓝 (vecMulVec (1 : ι → ℝ) π * W)) :=
    (hcont.tendsto _).comp hPF'
  have hfix : vecMulVec (1 : ι → ℝ) π = vecMulVec (1 : ι → ℝ) π * W := tendsto_nhds_unique e1 e2
  rw [Matrix.vecMulVec_mul] at hfix
  exact (vecMulVec_one_left_inj hfix).symm

/-! ## S6b: strong-interaction limit and the rank-one common-mode limit -/

/-- Under a rank-one covariance `c 𝟙𝟙ᵀ`, every asset has the identical common-mode loading
`(c 𝟙𝟙ᵀ) *ᵥ wM = c 𝟙`, for *any* fully-invested weight vector `wM` (`∑ wM = 1`). `wM` is not
assumed to be, and is not identified as, a market portfolio. -/
theorem capm_market_loading {c : ℝ} {wM : ι → ℝ} (hsum : ∑ i, wM i = 1) :
    (c • vecMulVec (1 : ι → ℝ) 1) *ᵥ wM = c • (1 : ι → ℝ) := by
  rw [Matrix.smul_mulVec, Matrix.vecMulVec_mulVec, op_smul_eq_smul, one_dotProduct, hsum,
    one_smul]

/-- Common-mode variance under a rank-one covariance `c 𝟙𝟙ᵀ`: `wM ⬝ᵥ (c 𝟙𝟙ᵀ) wM = c` for any
fully-invested weight vector `wM`. -/
theorem capm_market_variance {c : ℝ} {wM : ι → ℝ} (hsum : ∑ i, wM i = 1) :
    wM ⬝ᵥ ((c • vecMulVec (1 : ι → ℝ) 1) *ᵥ wM) = c := by
  rw [capm_market_loading hsum, dotProduct_smul, smul_eq_mul]
  have hone : wM ⬝ᵥ (1 : ι → ℝ) = 1 := by simpa [dotProduct] using hsum
  rw [hone, mul_one]

/-- **S6b (beta collapse under a rank-one common mode).** Under a rank-one covariance `c 𝟙𝟙ᵀ`
with `c ≠ 0`, every *fully-invested* weight vector has beta one against the common mode. This is
elementary rank-one algebra. It is not a CAPM: no market portfolio is identified (the statement
holds for an arbitrary `wM` with `∑ wM = 1`), and no market clearing, mean-variance efficiency,
risk premium, or beta-pricing equilibrium is asserted. -/
theorem capm_beta_collapse {c : ℝ} {wM : ι → ℝ} (hsum : ∑ i, wM i = 1) (hc : c ≠ 0) (i : ι) :
    ((c • vecMulVec (1 : ι → ℝ) 1) *ᵥ wM) i / (wM ⬝ᵥ ((c • vecMulVec (1 : ι → ℝ) 1) *ᵥ wM)) = 1 := by
  rw [capm_market_variance hsum, capm_market_loading hsum, Pi.smul_apply, Pi.one_apply, smul_eq_mul,
    mul_one, div_self hc]

/-- **S6b (strong-interaction limit).** Conditional on the resolvent limit
`(1−ρ)(1−ρW)⁻¹ → 𝟙 πᵀ` as `ρ ↑ 1`, packaged as the named hypothesis `IsResolventLimit W π`, the
rescaled spatial covariance `(1−ρ)² Σ_SAR` converges to the dominant rank-one component
`(πᵀ V π) 𝟙𝟙ᵀ`. For row-stochastic `W` that hypothesis is NOT independent input: it follows from
the power limit `Wᵏ → 𝟙 πᵀ` by elementary linear algebra (see the section below), so prefer
`sigmaSAR_rankOne_limit_of_perron`, which assumes only the power limit. -/
theorem sigmaSAR_rankOne_limit {W V : Matrix ι ι ℝ} {π : ι → ℝ}
    (hres : IsResolventLimit W π) :
    Tendsto (fun ρ : ℝ => ((1 - ρ) ^ 2) • spatialCov ρ W V) (𝓝[<] (1 : ℝ))
      (𝓝 ((π ⬝ᵥ (V *ᵥ π)) • vecMulVec (1 : ι → ℝ) 1)) := by
  have hres' : Tendsto (fun ρ : ℝ => (1 - ρ) • leontief ρ W) (𝓝[<] (1 : ℝ))
      (𝓝 (vecMulVec (1 : ι → ℝ) π)) := hres
  have hcont : Continuous (fun M : Matrix ι ι ℝ => M * V * Mᵀ) := by fun_prop
  have h := (hcont.tendsto _).comp hres'
  have hfun : (fun ρ : ℝ => ((1 - ρ) ^ 2) • spatialCov ρ W V)
      = (fun M : Matrix ι ι ℝ => M * V * Mᵀ) ∘ (fun ρ : ℝ => (1 - ρ) • leontief ρ W) := by
    funext ρ
    simp only [Function.comp_apply, spatialCov, Matrix.transpose_smul, smul_mul_assoc,
      mul_smul_comm, smul_smul, pow_two]
  rw [hfun, show (π ⬝ᵥ (V *ᵥ π)) • vecMulVec (1 : ι → ℝ) 1
      = vecMulVec (1 : ι → ℝ) π * V * (vecMulVec (1 : ι → ℝ) π)ᵀ from (projCov_collapse π V).symm]
  exact h

/-- **S6b (beta collapse at the rank-one limit).** In the `ρ ↑ 1` rank-one limit the covariance
is `(πᵀ V π) 𝟙𝟙ᵀ`; when `πᵀ V π ≠ 0` every asset's beta against the emergent common mode equals
one. NOTE: the declaration name is historical and overstates the statement. Nothing here recovers
the classical CAPM and nothing identifies `π` as a market portfolio — `π` does not appear in the
argument at all, which fixes an arbitrary fully-invested `wM`. -/
theorem capm_recovered_in_limit {V : Matrix ι ι ℝ} {π wM : ι → ℝ} (hsum : ∑ i, wM i = 1)
    (hc : π ⬝ᵥ (V *ᵥ π) ≠ 0) (i : ι) :
    (((π ⬝ᵥ (V *ᵥ π)) • vecMulVec (1 : ι → ℝ) 1) *ᵥ wM) i
      / (wM ⬝ᵥ (((π ⬝ᵥ (V *ᵥ π)) • vecMulVec (1 : ι → ℝ) 1) *ᵥ wM)) = 1 :=
  capm_beta_collapse hsum hc i

/-! ## Discharging the resolvent hypothesis from the Perron power limit

`IsResolventLimit` is **not** independent Perron–Frobenius input. For a row-stochastic `W` it is a
*consequence* of `IsPerronLimit`, by elementary (Perron–Frobenius-free) linear algebra: writing
`P = 𝟙 πᵀ` and `N = W - P`, the relations `W P = P W = P = P²` make `P` an absorbing idempotent, so
`Nᵏ⁺¹ = Wᵏ⁺¹ - P → 0`, `1 - N` is invertible, and the resolvent splits exactly as
`(1 − ρW)⁻¹ = (1−ρ)⁻¹ P + (1 − ρN)⁻¹ (1 − P)`. Multiplying by `(1−ρ)` and letting `ρ ↑ 1` gives the
resolvent limit with no Abel/Tauberian summation and no spectral theory.

The general lemmas below are stated for an arbitrary absorbing idempotent `P`; the Perron
specialisation is `isResolventLimit_of_isPerronLimit`. -/

section RankOneResolvent

variable {W P : Matrix ι ι ℝ}

/-- An absorbing right factor is absorbed by every power: `W P = P` implies `Wᵏ P = P`. -/
theorem pow_mul_of_mul_eq (hWP : W * P = P) : ∀ k : ℕ, W ^ k * P = P := by
  intro k
  induction k with
  | zero => rw [pow_zero, Matrix.one_mul]
  | succ n ih => rw [pow_succ, Matrix.mul_assoc, hWP, ih]

/-- Powers of the defect `N = W - P` at an absorbing idempotent `P`: `Nᵏ⁺¹ = Wᵏ⁺¹ - P`. -/
theorem defect_pow (hWP : W * P = P) (hPW : P * W = P) (hPP : P * P = P) (k : ℕ) :
    (W - P) ^ (k + 1) = W ^ (k + 1) - P := by
  induction k with
  | zero => rw [pow_one, pow_one]
  | succ n ih =>
    have hstep : (W ^ (n + 1) - P) * (W - P)
        = W ^ (n + 1) * W - W ^ (n + 1) * P - (P * W - P * P) := by noncomm_ring
    rw [pow_succ, ih, hstep, pow_mul_of_mul_eq hWP, hPW, hPP, ← pow_succ]
    abel

/-- The defect powers vanish: if `Wᵏ → P` at an absorbing idempotent `P` then `(W − P)ᵏ → 0`. -/
theorem tendsto_defect_pow (hWP : W * P = P) (hPW : P * W = P) (hPP : P * P = P)
    (h : Tendsto (fun k : ℕ => W ^ k) atTop (𝓝 P)) :
    Tendsto (fun k : ℕ => (W - P) ^ k) atTop (𝓝 0) := by
  have hshift : Tendsto (fun k : ℕ => (W - P) ^ (k + 1)) atTop (𝓝 0) := by
    have h1 : Tendsto (fun k : ℕ => W ^ (k + 1)) atTop (𝓝 P) := by
      simpa only [Function.comp_def] using h.comp (tendsto_add_atTop_nat 1)
    have h2 : Tendsto (fun k : ℕ => W ^ (k + 1) - P) atTop (𝓝 (P - P)) :=
      h1.sub tendsto_const_nhds
    rw [sub_self] at h2
    exact h2.congr fun k => (defect_pow hWP hPW hPP k).symm
  exact (tendsto_add_atTop_iff_nat 1).mp hshift

/-- If the powers of `N` vanish then `1 − N` is nonsingular. Finite-dimensional and elementary:
`N *ᵥ v = v` forces `v = Nᵏ *ᵥ v → 0`. -/
theorem isUnit_det_one_sub_of_tendsto_pow_zero {N : Matrix ι ι ℝ}
    (h : Tendsto (fun k : ℕ => N ^ k) atTop (𝓝 0)) : IsUnit ((1 : Matrix ι ι ℝ) - N).det := by
  rw [isUnit_iff_ne_zero]
  intro hdet
  obtain ⟨v, hv, hvz⟩ := Matrix.exists_mulVec_eq_zero_iff.mpr hdet
  have hNv : N *ᵥ v = v := by
    rw [Matrix.sub_mulVec, Matrix.one_mulVec, sub_eq_zero] at hvz
    exact hvz.symm
  have hpow : ∀ k : ℕ, N ^ k *ᵥ v = v := by
    intro k
    induction k with
    | zero => rw [pow_zero, Matrix.one_mulVec]
    | succ n ih => rw [pow_succ, ← Matrix.mulVec_mulVec, hNv, ih]
  have hcont : Continuous fun M : Matrix ι ι ℝ => M *ᵥ v := by fun_prop
  have hlim : Tendsto (fun k : ℕ => N ^ k *ᵥ v) atTop (𝓝 ((0 : Matrix ι ι ℝ) *ᵥ v)) :=
    (hcont.tendsto _).comp h
  rw [Matrix.zero_mulVec, show (fun k : ℕ => N ^ k *ᵥ v) = fun _ : ℕ => v from funext hpow] at hlim
  exact hv (tendsto_nhds_unique tendsto_const_nhds hlim)

/-- **Exact resolvent split at an absorbing idempotent.** For `ρ ≠ 1` with `1 − ρ(W − P)`
nonsingular, `(1 − ρW)⁻¹ = (1−ρ)⁻¹ P + (1 − ρ(W − P))⁻¹ (1 − P)`. The `(1−ρ)⁻¹` pole is carried
entirely by the rank-one block; the complementary block stays regular at `ρ = 1`. -/
theorem leontief_eq_add_of_absorbing (hWP : W * P = P) (hPW : P * W = P) (hPP : P * P = P)
    {ρ : ℝ} (hρ : ρ ≠ 1) (hdet : IsUnit ((1 : Matrix ι ι ℝ) - ρ • (W - P)).det) :
    leontief ρ W = (1 - ρ)⁻¹ • P + ((1 : Matrix ι ι ℝ) - ρ • (W - P))⁻¹ * (1 - P) := by
  have h1ρ : (1 : ℝ) - ρ ≠ 0 := sub_ne_zero_of_ne (Ne.symm hρ)
  have hPN : P * (W - P) = 0 := by rw [Matrix.mul_sub, hPW, hPP, sub_self]
  have hRR : ((1 : Matrix ι ι ℝ) - ρ • (W - P)) * ((1 : Matrix ι ι ℝ) - ρ • (W - P))⁻¹ = 1 :=
    Matrix.mul_nonsing_inv _ hdet
  have hPR : P * ((1 : Matrix ι ι ℝ) - ρ • (W - P))⁻¹ = P := by
    have hstep : P * ((1 : Matrix ι ι ℝ) - ρ • (W - P)) = P := by
      rw [Matrix.mul_sub, Matrix.mul_one, Matrix.mul_smul, hPN, smul_zero, sub_zero]
    calc P * ((1 : Matrix ι ι ℝ) - ρ • (W - P))⁻¹
        = P * ((1 : Matrix ι ι ℝ) - ρ • (W - P)) * ((1 : Matrix ι ι ℝ) - ρ • (W - P))⁻¹ := by
          rw [hstep]
      _ = P * (((1 : Matrix ι ι ℝ) - ρ • (W - P)) * ((1 : Matrix ι ι ℝ) - ρ • (W - P))⁻¹) :=
          Matrix.mul_assoc _ _ _
      _ = P := by rw [hRR, Matrix.mul_one]
  have key1 : ((1 : Matrix ι ι ℝ) - ρ • W) * ((1 - ρ)⁻¹ • P) = P := by
    rw [Matrix.mul_smul, Matrix.sub_mul, Matrix.one_mul, Matrix.smul_mul, hWP,
      show P - ρ • P = (1 - ρ) • P from by rw [sub_smul, one_smul], smul_smul,
      inv_mul_cancel₀ h1ρ, one_smul]
  have hsplit : (1 : Matrix ι ι ℝ) - ρ • W = (1 : Matrix ι ι ℝ) - ρ • (W - P) - ρ • P := by
    rw [smul_sub]; abel
  have key2 : ((1 : Matrix ι ι ℝ) - ρ • W)
      * (((1 : Matrix ι ι ℝ) - ρ • (W - P))⁻¹ * (1 - P)) = 1 - P := by
    rw [hsplit, Matrix.sub_mul, ← Matrix.mul_assoc, hRR, Matrix.one_mul, Matrix.smul_mul,
      ← Matrix.mul_assoc, hPR, Matrix.mul_sub, Matrix.mul_one, hPP, sub_self, smul_zero, sub_zero]
  unfold leontief
  refine Matrix.inv_eq_right_inv ?_
  rw [Matrix.mul_add, key1, key2]
  abel

/-- **Exact finite-`ρ` remainder.** Away from the pole at `ρ = 1`, the difference between the
normalized SAR multiplier and its rank-one component is exactly the complementary resolvent,
scaled by `1-ρ`.  This identity turns the asymptotic statement into an auditable finite-`ρ`
quantity for the fitted model. -/
theorem normalized_leontief_sub_eq (hWP : W * P = P) (hPW : P * W = P)
    (hPP : P * P = P) {ρ : ℝ} (hρ : ρ ≠ 1)
    (hdet : IsUnit ((1 : Matrix ι ι ℝ) - ρ • (W - P)).det) :
    (1 - ρ) • leontief ρ W - P =
      (1 - ρ) • (((1 : Matrix ι ι ℝ) - ρ • (W - P))⁻¹ * (1 - P)) := by
  rw [leontief_eq_add_of_absorbing hWP hPW hPP hρ hdet, smul_add,
    smul_inv_smul₀ (sub_ne_zero_of_ne (Ne.symm hρ))]
  abel

/-- **Finite-`ρ` SAR error bound.** If the complementary resolvent is inside its Neumann region,
the normalized multiplier obeys

`‖(1-ρ)(1-ρW)⁻¹-P‖ ≤ |1-ρ| · (1-‖ρ(W-P)‖)⁻¹ · ‖1-P‖`.

Unlike the `ρ ↑ 1` limit, this inequality can be evaluated at the fitted `ρ`; no interpretation of
an asymptotic limit as a finite-sample equality is required. -/
theorem normalized_leontief_error_le [Nonempty ι] (hWP : W * P = P) (hPW : P * W = P)
    (hPP : P * P = P) {ρ : ℝ} (hρ : ρ ≠ 1)
    (hcontract : ‖ρ • (W - P)‖ < 1) :
    ‖(1 - ρ) • leontief ρ W - P‖ ≤
      |1 - ρ| * (1 - ‖ρ • (W - P)‖)⁻¹ * ‖(1 : Matrix ι ι ℝ) - P‖ := by
  have hdet : IsUnit ((1 : Matrix ι ι ℝ) - ρ • (W - P)).det := by
    rw [isUnit_iff_ne_zero]
    exact Matrix.det_ne_zero_of_left_inverse
      (Ring.inverse_mul_cancel _ (isUnit_one_sub_of_norm_lt_one hcontract))
  rw [normalized_leontief_sub_eq hWP hPW hPP hρ hdet, norm_smul, Real.norm_eq_abs]
  calc
    |1 - ρ| * ‖((1 : Matrix ι ι ℝ) - ρ • (W - P))⁻¹ * (1 - P)‖
        ≤ |1 - ρ| *
            (‖((1 : Matrix ι ι ℝ) - ρ • (W - P))⁻¹‖ * ‖(1 - P)‖) := by
          gcongr
          exact norm_mul_le _ _
    _ ≤ |1 - ρ| * ((1 - ‖ρ • (W - P)‖)⁻¹ * ‖(1 - P)‖) := by
          gcongr
          rw [Matrix.nonsing_inv_eq_ringInverse]
          exact norm_inverse_one_sub_le hcontract
    _ = |1 - ρ| * (1 - ‖ρ • (W - P)‖)⁻¹ * ‖(1 - P)‖ := by ring

/-- **Resolvent limit from the power limit** (the Perron-free core of S6b). If `Wᵏ → P` at an
absorbing idempotent `P`, then `(1−ρ)(1 − ρW)⁻¹ → P` as `ρ ↑ 1`. -/
theorem tendsto_resolvent_of_tendsto_pow (hWP : W * P = P) (hPW : P * W = P) (hPP : P * P = P)
    (h : Tendsto (fun k : ℕ => W ^ k) atTop (𝓝 P)) :
    Tendsto (fun ρ : ℝ => (1 - ρ) • leontief ρ W) (𝓝[<] (1 : ℝ)) (𝓝 P) := by
  have hdet1 : IsUnit ((1 : Matrix ι ι ℝ) - (W - P)).det :=
    isUnit_det_one_sub_of_tendsto_pow_zero (tendsto_defect_pow hWP hPW hPP h)
  have hmap : Continuous fun ρ : ℝ => (1 : Matrix ι ι ℝ) - ρ • (W - P) := by fun_prop
  have hne : ((1 : Matrix ι ι ℝ) - (1 : ℝ) • (W - P)).det ≠ 0 := by
    rw [one_smul]; exact isUnit_iff_ne_zero.mp hdet1
  have hinvcont : ContinuousAt (fun ρ : ℝ => ((1 : Matrix ι ι ℝ) - ρ • (W - P))⁻¹) 1 := by
    have hring : ContinuousAt Ring.inverse ((1 : Matrix ι ι ℝ) - (1 : ℝ) • (W - P)).det := by
      rw [Ring.inverse_eq_inv']
      exact continuousAt_inv₀ hne
    have hinv : ContinuousAt (fun M : Matrix ι ι ℝ => M⁻¹)
        ((1 : Matrix ι ι ℝ) - (1 : ℝ) • (W - P)) := continuousAt_matrix_inv _ hring
    exact hinv.tendsto.comp (hmap.tendsto (1 : ℝ))
  have hev : ∀ᶠ ρ : ℝ in 𝓝[<] (1 : ℝ),
      P + (1 - ρ) • (((1 : Matrix ι ι ℝ) - ρ • (W - P))⁻¹ * (1 - P))
        = (1 - ρ) • leontief ρ W := by
    have h1 : ∀ᶠ ρ : ℝ in 𝓝[<] (1 : ℝ), ρ ∈ Set.Iio (1 : ℝ) := self_mem_nhdsWithin
    have h2 : ∀ᶠ ρ : ℝ in 𝓝[<] (1 : ℝ), IsUnit ((1 : Matrix ι ι ℝ) - ρ • (W - P)).det := by
      have hop : ∀ᶠ ρ : ℝ in 𝓝 (1 : ℝ), ((1 : Matrix ι ι ℝ) - ρ • (W - P)).det ≠ 0 :=
        hmap.matrix_det.continuousAt.eventually_ne hne
      exact (hop.filter_mono nhdsWithin_le_nhds).mono fun _ hρ => isUnit_iff_ne_zero.mpr hρ
    filter_upwards [h1, h2] with ρ hρ hρdet
    have hlt : ρ < 1 := Set.mem_Iio.mp hρ
    rw [leontief_eq_add_of_absorbing hWP hPW hPP (ne_of_lt hlt) hρdet, smul_add,
      smul_inv_smul₀ (sub_ne_zero_of_ne (Ne.symm (ne_of_lt hlt)))]
  refine Filter.Tendsto.congr' hev ?_
  have hs : Tendsto (fun ρ : ℝ => 1 - ρ) (𝓝[<] (1 : ℝ)) (𝓝 (0 : ℝ)) := by
    have hc : ContinuousAt (fun ρ : ℝ => 1 - ρ) 1 := by fun_prop
    simpa using hc.tendsto.mono_left nhdsWithin_le_nhds
  have hm : Tendsto (fun ρ : ℝ => ((1 : Matrix ι ι ℝ) - ρ • (W - P))⁻¹ * (1 - P))
      (𝓝[<] (1 : ℝ)) (𝓝 (((1 : Matrix ι ι ℝ) - (1 : ℝ) • (W - P))⁻¹ * (1 - P))) :=
    (hinvcont.mul continuousAt_const).tendsto.mono_left nhdsWithin_le_nhds
  have hprod := hs.smul hm
  rw [zero_smul] at hprod
  simpa using tendsto_const_nhds.add hprod

end RankOneResolvent

/-! ## The Perron limit is a probability distribution, and it discharges S6b -/

section PerronDischarge

variable {W : Matrix ι ι ℝ} {π : ι → ℝ}

/-- Entrywise reading of the Perron power limit: `(Wᵏ) i j → π j`. -/
theorem tendsto_pow_entry (hPF : IsPerronLimit W π) (i j : ι) :
    Tendsto (fun k : ℕ => (W ^ k) i j) atTop (𝓝 (π j)) := by
  have hcont : Continuous fun M : Matrix ι ι ℝ => M i j := continuous_id.matrix_elem i j
  simpa only [Function.comp_def, Matrix.vecMulVec_apply, Pi.one_apply, one_mul] using
    (hcont.tendsto _).comp hPF

/-- The Perron limit is unique: `Wᵏ` cannot converge to two different rank-one projections. -/
theorem isPerronLimit_unique [Nonempty ι] {π' : ι → ℝ} (h : IsPerronLimit W π)
    (h' : IsPerronLimit W π') : π = π' :=
  vecMulVec_one_left_inj (tendsto_nhds_unique h h')

/-- **The Perron limit has unit total mass** (`∑ π = 1`), for row-stochastic `W`. Proved, not
assumed: `Wᵏ *ᵥ 𝟙 = 𝟙` for every `k`, so the limit `(π ⬝ᵥ 𝟙) • 𝟙` equals `𝟙`. -/
theorem perron_sum_eq_one [Nonempty ι] (hW : ∀ i, ∑ j, W i j = 1) (hPF : IsPerronLimit W π) :
    ∑ i, π i = 1 := by
  have hconst : ∀ k : ℕ, (W ^ k) *ᵥ (1 : ι → ℝ) = 1 := fun k =>
    rowStochastic_mulVec_one (rowStochastic_pow hW k)
  have h1 := replication_centrality hPF (1 : ι → ℝ)
  rw [show (fun k : ℕ => (W ^ k) *ᵥ (1 : ι → ℝ)) = fun _ : ℕ => (1 : ι → ℝ) from funext hconst]
    at h1
  have h2 : ((π ⬝ᵥ (1 : ι → ℝ)) • (1 : ι → ℝ)) = (1 : ι → ℝ) :=
    tendsto_nhds_unique h1 tendsto_const_nhds
  have h3 := congrFun h2 (Classical.arbitrary ι)
  simpa only [Pi.smul_apply, Pi.one_apply, smul_eq_mul, mul_one, dotProduct_one] using h3

/-- Entrywise nonnegativity passes to powers. -/
theorem pow_nonneg_entry (hW : ∀ i j, 0 ≤ W i j) : ∀ (k : ℕ) (i j : ι), 0 ≤ (W ^ k) i j := by
  intro k
  induction k with
  | zero => intro i j; rw [pow_zero, Matrix.one_apply]; split <;> norm_num
  | succ n ih =>
    intro i j
    rw [pow_succ, Matrix.mul_apply]
    exact Finset.sum_nonneg fun l _ => mul_nonneg (ih i l) (hW l j)

/-- **The Perron limit is nonnegative**, for an entrywise-nonnegative `W`. -/
theorem perron_nonneg [Nonempty ι] (hW : ∀ i j, 0 ≤ W i j) (hPF : IsPerronLimit W π) (j : ι) :
    0 ≤ π j :=
  ge_of_tendsto' (tendsto_pow_entry hPF (Classical.arbitrary ι) j) fun k =>
    pow_nonneg_entry hW k _ j

/-- **S6b's resolvent hypothesis is discharged at the Perron gate.** For a row-stochastic `W`,
`IsPerronLimit W π` implies `IsResolventLimit W π`: the Abel/Tauberian step is not an extra
Perron–Frobenius input but an elementary consequence of the power limit. -/
theorem isResolventLimit_of_isPerronLimit [Nonempty ι] (hW : ∀ i, ∑ j, W i j = 1)
    (hPF : IsPerronLimit W π) : IsResolventLimit W π :=
  tendsto_resolvent_of_tendsto_pow
    (by rw [Matrix.mul_vecMulVec, rowStochastic_mulVec_one hW])
    (by rw [Matrix.vecMulVec_mul, perron_stationary hPF])
    (vecMulVec_one_idempotent (perron_sum_eq_one hW hPF)) hPF

/-- **S6b at the Perron gate.** `sigmaSAR_rankOne_limit` with the resolvent hypothesis discharged:
for a row-stochastic `W`, the Perron power limit alone gives `(1−ρ)² Σ_SAR → (πᵀVπ) 𝟙𝟙ᵀ`. -/
theorem sigmaSAR_rankOne_limit_of_perron [Nonempty ι] {V : Matrix ι ι ℝ}
    (hW : ∀ i, ∑ j, W i j = 1) (hPF : IsPerronLimit W π) :
    Tendsto (fun ρ : ℝ => ((1 - ρ) ^ 2) • spatialCov ρ W V) (𝓝[<] (1 : ℝ))
      (𝓝 ((π ⬝ᵥ (V *ᵥ π)) • vecMulVec (1 : ι → ℝ) 1)) :=
  sigmaSAR_rankOne_limit (isResolventLimit_of_isPerronLimit hW hPF)

end PerronDischarge

end PricingPerspective.Connections.SpatialAsymptotics
