import PricingPerspective.Continuous.Energy.Core

/-!
# The operational (plug-in) distance-implied covariance and its bias

Every other theorem in the `Continuous.Energy` family concerns the *systematic
exposure geometry*: exposures `β` living in a Hilbert space, their inner
products, and bounds on those inner products under a Lipschitz identification.
Paper 3 does not deploy that object. It deploys a plug-in estimator that makes
two substitutions the geometry does not license:

1. **Total for systematic variance.** The identity's `sys_var (β i)` is
   replaced by an observed *total* return variance
   `σ_tot i = sys_var (β i) + σ_eps i`.
2. **Calibrated plug-in for the model energy.** The identity's
   `energy_sq (β i) (β j)` is replaced by `κ² * Dsq i j`, a single scale
   applied to an estimated squared energy-distance matrix.

This module quantifies exactly what those two substitutions cost. The results
are elementary — one linear rearrangement of `EnergyKernelEmbedding.identity`
— but they are the only formal statements in this development about the
estimator that is actually run, and they separate two error channels the
manuscript previously discussed only in prose.

## Main results

* `operational_covariance_bias` — the exact decomposition: the estimator's
  error against systematic covariance splits into a **residual pedestal**
  `½(σ_eps i + σ_eps j)` and a **calibration gap**
  `½(energy_sq (β i) (β j) − κ² * Dsq i j)`.
* `operational_covariance_exact_calibration` — with the gap closed, only the
  pedestal remains.
* `operational_self_covariance` — the gap being closed at `i = i` (with a
  zero-diagonal `Dsq`) forces `sys_cov (β i) (β i) = sys_var (β i)`, so the
  estimator's diagonal is the total variance exactly.
* `operational_quadratic_form_homoskedastic` — under a closed gap and a common
  residual variance `s`, the estimator's quadratic form exceeds the systematic
  one by exactly `s * (Σ w)²`.
* `operational_argmin_invariant_on_simplex` — consequently, on fully-invested
  weights the substitution is a pure **level shift**: it cannot change which
  portfolio minimizes the quadratic form.

## Interpretation

The last two are the economically interesting ones, and they cut against the
natural reading of the estimator's defects. Substituting total variances is
*harmless for portfolio choice* once the calibration is exact and residuals are
homoskedastic: it adds a constant to every fully-invested portfolio's variance
and shifts no argmin. It also cannot break positive semidefiniteness, since the
perturbation it induces is `s` times the all-ones form, which is itself
positive semidefinite.

Therefore any indefiniteness of the deployed matrix, and any change it induces
in the selected portfolio, is attributable to the **calibration gap** —
heterogeneity that one common `κ` cannot absorb — and not to the
total-variance substitution. This is the formal counterpart of the
specification tests rejecting the common-scale restriction.
-/

set_option linter.style.longLine false

namespace PricingPerspective.ContinuousAPT

-- No `NormedAddCommGroup`/`InnerProductSpace` instances are required anywhere in
-- this file. `EnergyKernelEmbedding` exposes `sys_var`, `sys_cov` and
-- `energy_sq` as plain real-valued fields, and every result below is a linear
-- rearrangement of its `identity`. The bias decomposition therefore needs no
-- Hilbert geometry at all — unlike the bounds in `Portfolio.lean`, which do.
variable {E : Type*} {ι : Type*}

/-- **Operational distance-implied covariance** (the deployed plug-in).

  `Σ̂(i,j) = ½ ((σ_tot i) + (σ_tot j) − κ² * Dsq i j)`,
  `σ_tot i = sys_var (β i) + σ_eps i`.

`σ_eps i` is asset `i`'s residual (idiosyncratic) variance and `Dsq` the
estimated squared energy-distance matrix. Nothing here is assumed about `Dsq`
beyond its being a real-valued function of pairs; in particular it is *not*
assumed to agree with `ke.energy_sq`, and the whole point of
`operational_covariance_bias` is to price that disagreement. -/
noncomputable def operationalCovariance
    (ke : EnergyKernelEmbedding E) (β : ι → E) (σ_eps : ι → ℝ)
    (κ : ℝ) (Dsq : ι → ι → ℝ) (i j : ι) : ℝ :=
  (1 / 2) * ((ke.sys_var (β i) + σ_eps i) + (ke.sys_var (β j) + σ_eps j)
    - κ ^ 2 * Dsq i j)

/--
**Theorem (Operational estimator bias decomposition).**

The plug-in estimator's error against the systematic covariance it is meant to
represent splits exactly into two additive channels:

  `Σ̂(i,j) − Cov_sys(i,j) = ½(σ_eps i + σ_eps j) + ½(energy_sq (β i) (β j) − κ² * Dsq i j)`

The first term is the **residual pedestal** contributed by anchoring the
diagonal on total rather than systematic variance. The second is the
**calibration gap** contributed by substituting one common scale `κ` times an
estimated distance matrix for the model's own energy functional. Neither term
is signed in general; both vanish exactly when their channel is clean.
-/
theorem operational_covariance_bias
    (ke : EnergyKernelEmbedding E) (β : ι → E) (σ_eps : ι → ℝ)
    (κ : ℝ) (Dsq : ι → ι → ℝ) (i j : ι) :
    operationalCovariance ke β σ_eps κ Dsq i j - ke.sys_cov (β i) (β j)
      = (1 / 2) * (σ_eps i + σ_eps j)
        + (1 / 2) * (ke.energy_sq (β i) (β j) - κ ^ 2 * Dsq i j) := by
  have h := ke.identity (β i) (β j)
  simp only [operationalCovariance]
  linarith

/-- With the calibration gap closed at the pair `(i, j)`, the estimator
overstates systematic covariance by exactly the pair's average residual
variance. -/
theorem operational_covariance_exact_calibration
    (ke : EnergyKernelEmbedding E) (β : ι → E) (σ_eps : ι → ℝ)
    (κ : ℝ) (Dsq : ι → ι → ℝ) (i j : ι)
    (hcal : κ ^ 2 * Dsq i j = ke.energy_sq (β i) (β j)) :
    operationalCovariance ke β σ_eps κ Dsq i j
      = ke.sys_cov (β i) (β j) + (1 / 2) * (σ_eps i + σ_eps j) := by
  have h := operational_covariance_bias ke β σ_eps κ Dsq i j
  rw [hcal] at h
  linarith

/-- A closed calibration gap on the diagonal of a zero-diagonal `Dsq` forces
the embedding's self-covariance to be its variance, and the estimator's
diagonal entry is then the asset's total variance exactly.

This is the precise sense in which the deployed matrix has a *correct*
diagonal and a biased off-diagonal block. -/
theorem operational_self_covariance
    (ke : EnergyKernelEmbedding E) (β : ι → E) (σ_eps : ι → ℝ)
    (κ : ℝ) (Dsq : ι → ι → ℝ) (i : ι)
    (hdiag : Dsq i i = 0)
    (hcal : κ ^ 2 * Dsq i i = ke.energy_sq (β i) (β i)) :
    ke.sys_cov (β i) (β i) = ke.sys_var (β i)
      ∧ operationalCovariance ke β σ_eps κ Dsq i i
          = ke.sys_var (β i) + σ_eps i := by
  have hzero : ke.energy_sq (β i) (β i) = 0 := by
    rw [← hcal, hdiag]; ring
  have hid := ke.identity (β i) (β i)
  rw [hzero] at hid
  refine ⟨by linarith, ?_⟩
  simp only [operationalCovariance, hdiag]
  ring

/--
**Theorem (Homoskedastic level shift).**

Under a closed calibration gap and a common residual variance `s`, the
estimator's quadratic form exceeds the systematic quadratic form by exactly
`s * (Σ w)²`:

  `Σᵢ Σⱼ wᵢ wⱼ Σ̂(i,j) = Σᵢ Σⱼ wᵢ wⱼ Cov_sys(i,j) + s * (Σᵢ wᵢ)²`.

The perturbation is `s` times the all-ones quadratic form, hence nonnegative:
the total-variance substitution *cannot* make the deployed matrix indefinite
on its own. -/
theorem operational_quadratic_form_homoskedastic {n : ℕ}
    (ke : EnergyKernelEmbedding E) (β : Fin n → E) (σ_eps : Fin n → ℝ)
    (κ s : ℝ) (Dsq : Fin n → Fin n → ℝ) (w : Fin n → ℝ)
    (hcal : ∀ i j, κ ^ 2 * Dsq i j = ke.energy_sq (β i) (β j))
    (hhom : ∀ i, σ_eps i = s) :
    ∑ i, ∑ j, w i * w j * operationalCovariance ke β σ_eps κ Dsq i j
      = (∑ i, ∑ j, w i * w j * ke.sys_cov (β i) (β j)) + s * (∑ i, w i) ^ 2 := by
  have hpoint : ∀ i j, operationalCovariance ke β σ_eps κ Dsq i j
      = ke.sys_cov (β i) (β j) + s := by
    intro i j
    have h := operational_covariance_exact_calibration ke β σ_eps κ Dsq i j (hcal i j)
    rw [h, hhom i, hhom j]; ring
  calc
    ∑ i, ∑ j, w i * w j * operationalCovariance ke β σ_eps κ Dsq i j
        = ∑ i, ∑ j, (w i * w j * ke.sys_cov (β i) (β j) + s * (w i * w j)) := by
          refine Finset.sum_congr rfl fun i _ => ?_
          refine Finset.sum_congr rfl fun j _ => ?_
          rw [hpoint i j]; ring
    _ = (∑ i, ∑ j, w i * w j * ke.sys_cov (β i) (β j))
          + ∑ i, ∑ j, s * (w i * w j) := by
          rw [← Finset.sum_add_distrib]
          refine Finset.sum_congr rfl fun i _ => ?_
          rw [← Finset.sum_add_distrib]
    _ = (∑ i, ∑ j, w i * w j * ke.sys_cov (β i) (β j)) + s * (∑ i, w i) ^ 2 := by
          congr 1
          have hprod : ∑ i, ∑ j, w i * w j = (∑ i, w i) ^ 2 := by
            rw [sq, Finset.sum_mul]
            simp only [Finset.mul_sum]
          calc ∑ i, ∑ j, s * (w i * w j)
              = s * ∑ i, ∑ j, w i * w j := by simp only [Finset.mul_sum]
            _ = s * (∑ i, w i) ^ 2 := by rw [hprod]

/--
**Corollary (Argmin invariance on fully-invested weights).**

For any two fully-invested weight vectors, the difference of their operational
quadratic forms equals the difference of their systematic quadratic forms. The
total-variance substitution is therefore a pure level shift on the simplex: it
changes the *value* of portfolio variance but never the *ranking*, so it cannot
move the minimum-variance portfolio.

Any difference the deployed estimator makes to portfolio selection is thus
carried entirely by the calibration gap. -/
theorem operational_argmin_invariant_on_simplex {n : ℕ}
    (ke : EnergyKernelEmbedding E) (β : Fin n → E) (σ_eps : Fin n → ℝ)
    (κ s : ℝ) (Dsq : Fin n → Fin n → ℝ) (v w : Fin n → ℝ)
    (hcal : ∀ i j, κ ^ 2 * Dsq i j = ke.energy_sq (β i) (β j))
    (hhom : ∀ i, σ_eps i = s)
    (hv : ∑ i, v i = 1) (hw : ∑ i, w i = 1) :
    (∑ i, ∑ j, v i * v j * operationalCovariance ke β σ_eps κ Dsq i j)
        - (∑ i, ∑ j, w i * w j * operationalCovariance ke β σ_eps κ Dsq i j)
      = (∑ i, ∑ j, v i * v j * ke.sys_cov (β i) (β j))
        - (∑ i, ∑ j, w i * w j * ke.sys_cov (β i) (β j)) := by
  rw [operational_quadratic_form_homoskedastic ke β σ_eps κ s Dsq v hcal hhom,
    operational_quadratic_form_homoskedastic ke β σ_eps κ s Dsq w hcal hhom,
    hv, hw]
  ring

end PricingPerspective.ContinuousAPT
