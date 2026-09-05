import PricingPerspective.Continuous.Energy

set_option linter.style.longLine false

/-!
# Robust bounds under energy-distance ambiguity (Paper 5 energy-robustness theorem ladder)

This module hosts the machine-checked propagation layer of Paper 5's energy-robustness arm
(*Robust Minimum-Variance Portfolios under Energy-Distance Ambiguity*,
`.context/plan/paper5/`):
`Σ_dist` is treated as the **center of a statistically calibrated ambiguity set**, and the radii
(V-statistic sampling error, GMM moment violations, certified frontier slack) are transported to
covariance brackets through the metric geometry already exposed by `Continuous/Energy.lean`.

The load-bearing discipline pinned here: **radii add on the metric `𝓔 = dist`, never on the
squared distance `𝓔²`** — the propagated squared bracket is the asymmetric
`[(𝓔 − ε)₊², (𝓔 + ε)²]`, not `𝓔² ± ε`.

## Theorem ladder (T1–T5 all PROVEN, no open goals, in this file, landed 2026-07-12)

* **T1 (this file, PROVEN).** Embedding-perturbation propagation:
  `inner_perturbation_bound` (bilinearity + Cauchy–Schwarz: an `ε`-perturbation of each
  embedding moves the covariance inner product by at most `ε_i‖μ_j‖ + ε_j‖μ_i‖ + ε_iε_j`),
  `dist_bracket_of_embedding_radius` (metric-level bracket `|dist' − dist| ≤ ε_i + ε_j`), and
  `energy_bracket_of_embedding_radius` (the asymmetric squared bracket).

* **T2 (this file, PROVEN 2026-07-12).** `covariance_bracket_of_embedding_radius` composes T1's
  distance bracket with the bi-Lipschitz envelope `[ℓ, L]` (via
  `systematic_covariance_lower_bound` / `systematic_covariance_upper_bound`,
  `Continuous/Energy/{Core,Portfolio}.lean`): under
  `ℓ·dist μ_i' μ_j' ≤ ‖β_i−β_j‖ ≤ L·dist μ_i' μ_j'`
  (bi-Lipschitz in the TRUE embedding distance) and radius uncertainty
  `‖μ_i'−μ_i‖ ≤ ε_i`, `‖μ_j'−μ_j‖ ≤ ε_j`, the covariance is bracketed by the polarization
  expression at the radius-inflated/deflated MEASURED distance: lower bound with
  `L·(dist μ_i μ_j + ε_i+ε_j)`, upper bound with `ℓ·max 0 (dist μ_i μ_j − (ε_i+ε_j))`. Radii add
  on the metric, never the squared distance — proved directly via `dist_bracket_of_embedding_radius`
  plus monotonicity of the (already-proven) pairwise covariance bounds; no new polarization
  machinery needed.

* **T3 (this file, PROVEN).** Long-only worst case is the upper box corner:
  `worst_case_variance_box_corner` (`w ≥ 0` and `S ≤ U` entrywise give
  `∑∑ wᵢwⱼSᵢⱼ ≤ ∑∑ wᵢwⱼUᵢⱼ`) and its portfolio form `portfolio_variance_le_box_corner` via
  `portfolio_norm_sq_eq_double_sum`. Note `U` need not be PSD: the bound is valid over
  box ∩ PSD ⊆ box regardless; PSD handling is an optimization concern, not a validity one.

* **T4 (this file, PROVEN 2026-07-12).** `portfolio_variance_robust_envelope` double-sums T2 over
  `Fin n` with per-asset radii `ε : Fin n → ℝ`, mirroring `portfolio_variance_envelope`
  (`Continuous/Energy/Portfolio.lean`) — `Finset.sum_le_sum` twice applied to T2's pair bounds, in
  both directions (lower via L-inflated measured distance, upper via ℓ-deflated).

* **T5 (this file, PROVEN 2026-07-12 — one subsection only).** Hilbert-ball linear DRO
  certificate, stated as two lemmas to avoid csSup plumbing: `inner_le_of_mem_closedBall`
  (`q ∈ closedBall p ε ⇒ ⟪q,f⟫ ≤ ⟪p,f⟫ + ε‖f‖`, Cauchy–Schwarz on `⟪q−p,f⟫`) and
  `exists_mem_closedBall_inner_eq` (attainment: witness `p + (ε/‖f‖)•f` when `f ≠ 0`, `p`
  otherwise). Together they machine-check `sup_{‖q−p‖≤ε} ⟪q,f⟫ = ⟪p,f⟫ + ε‖f‖`. One-sided
  *relaxation* certificate only: portfolio variance is quadratic in the distribution, `(w′x)²`
  has infinite Brownian-kernel RKHS norm on unbounded support, and the Hilbert ball strictly
  contains the embedded probability measures — the disclosures live in the paper's §T5
  subsection.

* **T6 (PROVEN in `Connections/EnergyWasserstein.lean`).** Ball inclusion
  `wasserstein_ball_subset_energy_ball`: `W₂(P,Q) ≤ δ ⇒ 𝓔²(P,Q) ≤ 2δ` via
  `energy_le_two_wasserstein`, so the calibrated energy ball contains the corresponding
  2-Wasserstein ball (one-sided; no reverse bound holds in general — `kantorovich_duality` is
  out-of-window). Lives in the `Connections` layer because it bridges packages.

* **T7 (upgrade path, not a dependency).** When the §4.6 keystone lands
  (`EnergyStatistics/MeanEmbedding.lean` s3/s4), T1–T4 become unconditional statements about
  probability measures; until then they are stated at the abstract Hilbert level, matching the
  `EnergyKernelEmbedding` axiom-conditional discipline of Paper 3.

* **T2′/T8 — observable composition and regret (PROVEN 2026-07-12).** Five theorems close the
  loop between T2's abstract bracket and the fully-observable, regret-bounded pipeline:
  `covariance_le_corner` (the no-information corner clamp `⟪β_i,β_j⟫ ≤ (1/2)(‖β_i‖²+‖β_j‖²)`,
  polarization + `sq_nonneg`, justifying the corner cap independent of any Lipschitz envelope),
  `covariance_bracket_with_misfit` (T2′: composes T2's bracket with a calibrated-mode misfit
  budget `|c − ⟪β_i,β_j⟫| ≤ η`), `covariance_bracket_of_variance_radius` (T2 restated against an
  observable variance proxy `s_i ≈ ‖β_i‖²` up to slack `δ_i`, covering stale-vol / price-light
  diagonal uncertainty), `covariance_bracket_observable` (T2′ + variance-radius fully composed:
  brackets a calibrated estimate `c` using only the observables `s_i, s_j, δ_i, δ_j, η,
  dist μ_i μ_j` — never the true exposure norms or the true covariance), and
  `robust_regret_le_bracket_width` (T8: chains T3's `worst_case_variance_box_corner` with the
  robust portfolio's optimality certificate `hopt` to bound the robust portfolio's TRUE variance
  regret against any comparator by the comparator-weighted bracket width
  `∑∑ w*ᵢw*ⱼ(Hiᵢⱼ − Loᵢⱼ)` — the materiality-to-premium link).

## References

* Goldfarb & Iyengar (2003). Robust portfolio selection problems. Math. Oper. Res. 28(1).
* Lobo & Boyd (2000). The worst-case risk of a portfolio.
* Blanchet, Chen & Zhou (2022). Distributionally robust mean-variance portfolio selection with
  Wasserstein distances. Management Science 68(9).
* Székely & Rizzo (2023). The Energy of Data and Distance Correlation. §3.
-/

namespace PricingPerspective.ContinuousAPT

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-! ### T1 — embedding-perturbation propagation -/

/--
**Theorem (Inner-Product Perturbation Bound).** (Paper 5 energy-robustness T1.)

If each mean embedding is known only up to a radius (`‖μ_i' − μ_i‖ ≤ ε_i`,
`‖μ_j' − μ_j‖ ≤ ε_j`), the covariance inner product moves by at most

  `|⟪μ_i', μ_j'⟫ − ⟪μ_i, μ_j⟫| ≤ ε_i‖μ_j‖ + ε_j‖μ_i‖ + ε_iε_j`,

first-order proportional to the assets' risk scales and radii. Pure bilinearity +
Cauchy–Schwarz; no probabilistic content.
-/
theorem inner_perturbation_bound (μ_i μ_j μ_i' μ_j' : E) {ε_i ε_j : ℝ}
    (h_i : ‖μ_i' - μ_i‖ ≤ ε_i) (h_j : ‖μ_j' - μ_j‖ ≤ ε_j) :
    |@Inner.inner ℝ E _ μ_i' μ_j' - @Inner.inner ℝ E _ μ_i μ_j| ≤
      ε_i * ‖μ_j‖ + ε_j * ‖μ_i‖ + ε_i * ε_j := by
  have hdecomp : @Inner.inner ℝ E _ μ_i' μ_j' - @Inner.inner ℝ E _ μ_i μ_j =
      @Inner.inner ℝ E _ (μ_i' - μ_i) μ_j + @Inner.inner ℝ E _ μ_i (μ_j' - μ_j) +
        @Inner.inner ℝ E _ (μ_i' - μ_i) (μ_j' - μ_j) := by
    simp only [inner_sub_left, inner_sub_right]
    ring
  have h1 : |@Inner.inner ℝ E _ (μ_i' - μ_i) μ_j| ≤ ε_i * ‖μ_j‖ :=
    (abs_real_inner_le_norm _ _).trans
      (mul_le_mul_of_nonneg_right h_i (norm_nonneg _))
  have h2 : |@Inner.inner ℝ E _ μ_i (μ_j' - μ_j)| ≤ ε_j * ‖μ_i‖ := by
    refine (abs_real_inner_le_norm _ _).trans ?_
    rw [mul_comm]
    exact mul_le_mul_of_nonneg_right h_j (norm_nonneg _)
  have h3 : |@Inner.inner ℝ E _ (μ_i' - μ_i) (μ_j' - μ_j)| ≤ ε_i * ε_j :=
    (abs_real_inner_le_norm _ _).trans
      (mul_le_mul h_i h_j (norm_nonneg _) ((norm_nonneg _).trans h_i))
  calc |@Inner.inner ℝ E _ μ_i' μ_j' - @Inner.inner ℝ E _ μ_i μ_j|
      = |@Inner.inner ℝ E _ (μ_i' - μ_i) μ_j + @Inner.inner ℝ E _ μ_i (μ_j' - μ_j) +
          @Inner.inner ℝ E _ (μ_i' - μ_i) (μ_j' - μ_j)| := by rw [hdecomp]
    _ ≤ |@Inner.inner ℝ E _ (μ_i' - μ_i) μ_j| + |@Inner.inner ℝ E _ μ_i (μ_j' - μ_j)| +
          |@Inner.inner ℝ E _ (μ_i' - μ_i) (μ_j' - μ_j)| := abs_add_three _ _ _
    _ ≤ ε_i * ‖μ_j‖ + ε_j * ‖μ_i‖ + ε_i * ε_j := add_le_add (add_le_add h1 h2) h3

omit [InnerProductSpace ℝ E] in
/--
**Theorem (Distance Bracket under Embedding Radii).** (Paper 5 energy-robustness T1, metric level.)

Radii add on the **metric**: `|dist μ_i' μ_j' − dist μ_i μ_j| ≤ ε_i + ε_j`. This is the
level at which ambiguity propagates; squaring afterwards gives the asymmetric bracket of
`energy_bracket_of_embedding_radius`. Adding radii to squared distances is wrong.
-/
theorem dist_bracket_of_embedding_radius (μ_i μ_j μ_i' μ_j' : E) {ε_i ε_j : ℝ}
    (h_i : ‖μ_i' - μ_i‖ ≤ ε_i) (h_j : ‖μ_j' - μ_j‖ ≤ ε_j) :
    |dist μ_i' μ_j' - dist μ_i μ_j| ≤ ε_i + ε_j := by
  have h := dist_dist_dist_le μ_i' μ_j' μ_i μ_j
  rw [Real.dist_eq] at h
  have hi : dist μ_i' μ_i ≤ ε_i := by rw [dist_eq_norm]; exact h_i
  have hj : dist μ_j' μ_j ≤ ε_j := by rw [dist_eq_norm]; exact h_j
  linarith

omit [InnerProductSpace ℝ E] in
/--
**Theorem (Squared Energy Bracket under Embedding Radii).** (Paper 5 energy-robustness T1, squared level.)

The propagated bracket on the squared distance is **asymmetric**:

  `(max 0 (dist μ_i μ_j − (ε_i + ε_j)))² ≤ (dist μ_i' μ_j')² ≤ (dist μ_i μ_j + (ε_i + ε_j))²`.

Under the keystone identity `dist` of embeddings *is* the energy metric `𝓔`, so this is the
`[(𝓔 − ε)₊², (𝓔 + ε)²]` bracket the radius-calibration code must implement — pinned here so
prose and code cannot silently add radii to `𝓔²`.
-/
theorem energy_bracket_of_embedding_radius (μ_i μ_j μ_i' μ_j' : E) {ε_i ε_j : ℝ}
    (h_i : ‖μ_i' - μ_i‖ ≤ ε_i) (h_j : ‖μ_j' - μ_j‖ ≤ ε_j) :
    max 0 (dist μ_i μ_j - (ε_i + ε_j)) ^ 2 ≤ dist μ_i' μ_j' ^ 2 ∧
      dist μ_i' μ_j' ^ 2 ≤ (dist μ_i μ_j + (ε_i + ε_j)) ^ 2 := by
  have habs := dist_bracket_of_embedding_radius μ_i μ_j μ_i' μ_j' h_i h_j
  have hlo : dist μ_i μ_j - (ε_i + ε_j) ≤ dist μ_i' μ_j' := by
    have := abs_le.mp habs
    linarith [this.1]
  have hhi : dist μ_i' μ_j' ≤ dist μ_i μ_j + (ε_i + ε_j) := by
    have := abs_le.mp habs
    linarith [this.2]
  constructor
  · have hmax_le : max 0 (dist μ_i μ_j - (ε_i + ε_j)) ≤ dist μ_i' μ_j' :=
      max_le dist_nonneg hlo
    exact pow_le_pow_left₀ (le_max_left _ _) hmax_le 2
  · exact pow_le_pow_left₀ dist_nonneg hhi 2

/-! ### T2 — composed entrywise bracket under bi-Lipschitz slack + radius uncertainty -/

/--
**Theorem (Composed Covariance Bracket under Bi-Lipschitz Slack and Embedding Radii).**
(Paper 5 energy-robustness T2, PROVEN 2026-07-12.)

Composes `dist_bracket_of_embedding_radius` (T1, metric-level radius propagation) with the
bi-Lipschitz systematic-covariance envelope (`systematic_covariance_lower_bound` /
`systematic_covariance_upper_bound`, `Continuous/Energy/{Core,Portfolio}.lean`): if the exposure
distance is bi-Lipschitz in the **true** (unobserved) embedding distance,

  `ℓ * dist μ_i' μ_j' ≤ ‖β_i − β_j‖ ≤ L * dist μ_i' μ_j'`,

and the true embeddings lie within radii `ε_i, ε_j` of the **measured** embeddings
(`‖μ_i' − μ_i‖ ≤ ε_i`, `‖μ_j' − μ_j‖ ≤ ε_j`), the systematic covariance is bracketed by the
polarization expression evaluated at the radius-inflated / radius-deflated MEASURED distance:

  `(1/2)(‖β_i‖² + ‖β_j‖² − (L·(dist μ_i μ_j + ε_i + ε_j))²) ≤ ⟪β_i, β_j⟫`
  `⟪β_i, β_j⟫ ≤ (1/2)(‖β_i‖² + ‖β_j‖² − (ℓ·max 0 (dist μ_i μ_j − (ε_i + ε_j)))²)`.

Radii add on the metric `dist`, never on the squared distance — mirroring T1's discipline: the
lower bound inflates the measured distance (weakening the Lipschitz cap forces a lower floor);
the upper bound deflates it via `max 0` (the bi-Lipschitz floor can only shrink, clamped at zero,
never go negative) before squaring.
-/
theorem covariance_bracket_of_embedding_radius
    (β_i β_j μ_i μ_j μ_i' μ_j' : E) (L ℓ : ℝ) {ε_i ε_j : ℝ}
    (hL : 0 ≤ L) (hℓ : 0 ≤ ℓ)
    (h_i : ‖μ_i' - μ_i‖ ≤ ε_i) (h_j : ‖μ_j' - μ_j‖ ≤ ε_j)
    (hLip : ‖β_i - β_j‖ ≤ L * dist μ_i' μ_j')
    (hLip_lower : ℓ * dist μ_i' μ_j' ≤ ‖β_i - β_j‖) :
    (1 / 2) * (‖β_i‖ ^ 2 + ‖β_j‖ ^ 2 - (L * (dist μ_i μ_j + (ε_i + ε_j))) ^ 2) ≤
        @Inner.inner ℝ E _ β_i β_j ∧
      @Inner.inner ℝ E _ β_i β_j ≤
        (1 / 2) * (‖β_i‖ ^ 2 + ‖β_j‖ ^ 2 -
          (ℓ * max 0 (dist μ_i μ_j - (ε_i + ε_j))) ^ 2) := by
  have hεi_nonneg : 0 ≤ ε_i := (norm_nonneg _).trans h_i
  have hεj_nonneg : 0 ≤ ε_j := (norm_nonneg _).trans h_j
  have hdist := dist_bracket_of_embedding_radius μ_i μ_j μ_i' μ_j' h_i h_j
  have hle := abs_le.mp hdist
  have hhi : dist μ_i' μ_j' ≤ dist μ_i μ_j + (ε_i + ε_j) := by linarith [hle.2]
  have hmax_le : max 0 (dist μ_i μ_j - (ε_i + ε_j)) ≤ dist μ_i' μ_j' :=
    max_le dist_nonneg (by linarith [hle.1])
  have hD_hi_nonneg : 0 ≤ dist μ_i μ_j + (ε_i + ε_j) := by
    have := dist_nonneg (x := μ_i) (y := μ_j)
    linarith
  have hLip' : ‖β_i - β_j‖ ≤ L * (dist μ_i μ_j + (ε_i + ε_j)) :=
    hLip.trans (mul_le_mul_of_nonneg_left hhi hL)
  have hLip_lower' : ℓ * max 0 (dist μ_i μ_j - (ε_i + ε_j)) ≤ ‖β_i - β_j‖ :=
    (mul_le_mul_of_nonneg_left hmax_le hℓ).trans hLip_lower
  exact ⟨systematic_covariance_lower_bound β_i β_j L (dist μ_i μ_j + (ε_i + ε_j)) hL
      hD_hi_nonneg hLip',
    systematic_covariance_upper_bound β_i β_j ℓ (max 0 (dist μ_i μ_j - (ε_i + ε_j))) hℓ
      (le_max_left _ _) hLip_lower'⟩

/-! ### T2′ — observable composition: corner clamp, misfit budget, and variance-radius brackets -/

/--
**Theorem (No-Information Corner Clamp).** (Paper 5 energy-robustness T2′, PROVEN 2026-07-12.)

The systematic covariance can never exceed the no-information corner value, regardless of any
Lipschitz envelope or distance bracket:

  `⟪β_i, β_j⟫ ≤ (1/2) * (‖β_i‖² + ‖β_j‖²)`.

This is the degeneration of `covariance_bracket_of_embedding_radius`'s upper bracket when the
bi-Lipschitz floor `ℓ` or the calibrated radius carries no information (e.g. `ℓ = 0`, or the
measured distance collapses under `max 0 (dist − (ε_i+ε_j))`), isolated as a standalone fact:
pure polarization (`norm_sub_sq_real`) plus `‖β_i − β_j‖² ≥ 0`. It justifies clamping the upper
covariance bracket at the corner independent of any energy-distance hypothesis.
-/
theorem covariance_le_corner (β_i β_j : E) :
    @Inner.inner ℝ E _ β_i β_j ≤ (1 / 2) * (‖β_i‖ ^ 2 + ‖β_j‖ ^ 2) := by
  have hpol : ‖β_i - β_j‖ ^ 2 = ‖β_i‖ ^ 2 - 2 * @Inner.inner ℝ E _ β_i β_j + ‖β_j‖ ^ 2 :=
    norm_sub_sq_real β_i β_j
  nlinarith [sq_nonneg ‖β_i - β_j‖]

/--
**Theorem (Composed Bracket under a Calibrated Misfit Budget).** (Paper 5 energy-robustness T2′, PROVEN 2026-07-12.)

Machine-checks the calibrated mode's misfit composition: if a fitted/estimated value `c` tracks
the true systematic covariance `⟪β_i, β_j⟫` only up to a misfit budget
`|c − ⟪β_i, β_j⟫| ≤ η` (e.g. GMM moment slack, calibration residual), the T2 bracket on
`⟪β_i, β_j⟫` transports to `c` by widening both sides by `η`:

  `(T2 lower) − η ≤ c ≤ (T2 upper) + η`.

Pure composition of `covariance_bracket_of_embedding_radius` with `abs_le.mp hmis`; no new
geometry.
-/
theorem covariance_bracket_with_misfit
    (c : ℝ) (β_i β_j μ_i μ_j μ_i' μ_j' : E) (L ℓ η : ℝ) {ε_i ε_j : ℝ}
    (hL : 0 ≤ L) (hℓ : 0 ≤ ℓ)
    (h_i : ‖μ_i' - μ_i‖ ≤ ε_i) (h_j : ‖μ_j' - μ_j‖ ≤ ε_j)
    (hLip : ‖β_i - β_j‖ ≤ L * dist μ_i' μ_j')
    (hLip_lower : ℓ * dist μ_i' μ_j' ≤ ‖β_i - β_j‖)
    (hmis : |c - @Inner.inner ℝ E _ β_i β_j| ≤ η) :
    (1 / 2) * (‖β_i‖ ^ 2 + ‖β_j‖ ^ 2 - (L * (dist μ_i μ_j + (ε_i + ε_j))) ^ 2) - η ≤ c ∧
      c ≤ (1 / 2) * (‖β_i‖ ^ 2 + ‖β_j‖ ^ 2 -
        (ℓ * max 0 (dist μ_i μ_j - (ε_i + ε_j))) ^ 2) + η := by
  have hbracket := covariance_bracket_of_embedding_radius β_i β_j μ_i μ_j μ_i' μ_j' L ℓ
    hL hℓ h_i h_j hLip hLip_lower
  have hmis' := abs_le.mp hmis
  exact ⟨by linarith [hbracket.1, hmis'.1], by linarith [hbracket.2, hmis'.2]⟩

/--
**Theorem (Covariance Bracket under Variance-Proxy Uncertainty).** (Paper 5 energy-robustness T2′,
PROVEN 2026-07-12.)

Covers vol-proxy / stale-variance uncertainty (the price-light certificate): the exposure norms
`‖β_i‖², ‖β_j‖²` are not directly observed, only tracked by variance proxies `s_i, s_j` up to a
slack `|s_i − ‖β_i‖²| ≤ δ_i`, `|s_j − ‖β_j‖²| ≤ δ_j` (e.g. a stale or infrequently-refreshed
volatility estimate). Restates `covariance_bracket_of_embedding_radius` with `‖β_i‖² + ‖β_j‖²`
replaced by the observable `s_i + s_j`, widening the bracket by the diagonal slack `δ_i + δ_j`
on each side.
-/
theorem covariance_bracket_of_variance_radius
    (s_i s_j : ℝ) (β_i β_j μ_i μ_j μ_i' μ_j' : E) (L ℓ : ℝ) {ε_i ε_j δ_i δ_j : ℝ}
    (hL : 0 ≤ L) (hℓ : 0 ≤ ℓ)
    (h_i : ‖μ_i' - μ_i‖ ≤ ε_i) (h_j : ‖μ_j' - μ_j‖ ≤ ε_j)
    (hLip : ‖β_i - β_j‖ ≤ L * dist μ_i' μ_j')
    (hLip_lower : ℓ * dist μ_i' μ_j' ≤ ‖β_i - β_j‖)
    (hs_i : |s_i - ‖β_i‖ ^ 2| ≤ δ_i) (hs_j : |s_j - ‖β_j‖ ^ 2| ≤ δ_j) :
    (1 / 2) * (s_i + s_j - (L * (dist μ_i μ_j + (ε_i + ε_j))) ^ 2) - (1 / 2) * (δ_i + δ_j) ≤
        @Inner.inner ℝ E _ β_i β_j ∧
      @Inner.inner ℝ E _ β_i β_j ≤
        (1 / 2) * (s_i + s_j - (ℓ * max 0 (dist μ_i μ_j - (ε_i + ε_j))) ^ 2) +
          (1 / 2) * (δ_i + δ_j) := by
  have hbracket := covariance_bracket_of_embedding_radius β_i β_j μ_i μ_j μ_i' μ_j' L ℓ
    hL hℓ h_i h_j hLip hLip_lower
  have hs_i' := abs_le.mp hs_i
  have hs_j' := abs_le.mp hs_j
  exact ⟨by linarith [hbracket.1, hs_i'.2, hs_j'.2], by linarith [hbracket.2, hs_i'.1, hs_j'.1]⟩

/--
**Theorem (Fully-Observable Composed Bracket).** (Paper 5 energy-robustness T2′, PROVEN 2026-07-12.)

Combines `covariance_bracket_with_misfit` and `covariance_bracket_of_variance_radius`: brackets
a calibrated estimate `c` using ONLY observables — the variance proxies `s_i, s_j` with slack
`δ_i, δ_j`, the misfit budget `η`, and the measured embedding distance `dist μ_i μ_j` with radii
`ε_i, ε_j` — never the true (unobserved) exposure norms `‖β_i‖, ‖β_j‖` or the true covariance
`⟪β_i, β_j⟫`. This is the paper-facing API: every quantity on both sides of the inequality is
something the calibration pipeline can actually compute.
-/
theorem covariance_bracket_observable
    (c s_i s_j : ℝ) (β_i β_j μ_i μ_j μ_i' μ_j' : E) (L ℓ η : ℝ) {ε_i ε_j δ_i δ_j : ℝ}
    (hL : 0 ≤ L) (hℓ : 0 ≤ ℓ)
    (h_i : ‖μ_i' - μ_i‖ ≤ ε_i) (h_j : ‖μ_j' - μ_j‖ ≤ ε_j)
    (hLip : ‖β_i - β_j‖ ≤ L * dist μ_i' μ_j')
    (hLip_lower : ℓ * dist μ_i' μ_j' ≤ ‖β_i - β_j‖)
    (hs_i : |s_i - ‖β_i‖ ^ 2| ≤ δ_i) (hs_j : |s_j - ‖β_j‖ ^ 2| ≤ δ_j)
    (hmis : |c - @Inner.inner ℝ E _ β_i β_j| ≤ η) :
    (1 / 2) * (s_i + s_j - (L * (dist μ_i μ_j + (ε_i + ε_j))) ^ 2)
        - (1 / 2) * (δ_i + δ_j) - η ≤ c ∧
      c ≤ (1 / 2) * (s_i + s_j - (ℓ * max 0 (dist μ_i μ_j - (ε_i + ε_j))) ^ 2)
        + (1 / 2) * (δ_i + δ_j) + η := by
  have hbracket := covariance_bracket_of_variance_radius s_i s_j β_i β_j μ_i μ_j μ_i' μ_j' L ℓ
    hL hℓ h_i h_j hLip hLip_lower hs_i hs_j
  have hmis' := abs_le.mp hmis
  exact ⟨by linarith [hbracket.1, hmis'.1], by linarith [hbracket.2, hmis'.2]⟩

/-! ### T3 — long-only worst case is the upper box corner -/

/--
**Theorem (Worst-Case Variance at the Box Corner).** (Paper 5 energy-robustness T3.)

For nonnegative weights and any covariance-candidate matrix dominated entrywise by `U`,
the quadratic form is dominated by the corner value: `∑∑ wᵢwⱼSᵢⱼ ≤ ∑∑ wᵢwⱼUᵢⱼ`.

Hence for long-only portfolios the supremum over the entrywise box `{S : L ≤ S ≤ U}` is
attained at `U`. `U` need not be PSD — the bound holds over box ∩ PSD ⊆ box a fortiori,
so PSD-ness only matters for the *optimization*, never for the validity of the cap.
-/
theorem worst_case_variance_box_corner {n : ℕ}
    (S U : Fin n → Fin n → ℝ) (w : Fin n → ℝ)
    (hw : ∀ i, 0 ≤ w i) (hSU : ∀ i j, S i j ≤ U i j) :
    ∑ i, ∑ j, w i * w j * S i j ≤ ∑ i, ∑ j, w i * w j * U i j := by
  refine Finset.sum_le_sum fun i _ => ?_
  refine Finset.sum_le_sum fun j _ => ?_
  exact mul_le_mul_of_nonneg_left (hSU i j) (mul_nonneg (hw i) (hw j))

/--
**Theorem (Portfolio Variance Capped by the Box Corner).** (Paper 5 energy-robustness T3, portfolio form.)

If every pairwise systematic covariance is dominated by the bracket corner
(`⟪βᵢ, βⱼ⟫ ≤ Uᵢⱼ` — e.g. `U = Σ_hi` from the T2 brackets), long-only portfolio systematic
variance is capped by the corner quadratic form: `‖Σᵢ wᵢβᵢ‖² ≤ ∑∑ wᵢwⱼUᵢⱼ`.
-/
theorem portfolio_variance_le_box_corner {n : ℕ}
    (β : Fin n → E) (w : Fin n → ℝ) (U : Fin n → Fin n → ℝ)
    (hw : ∀ i, 0 ≤ w i) (hU : ∀ i j, @Inner.inner ℝ E _ (β i) (β j) ≤ U i j) :
    ‖∑ i, w i • β i‖ ^ 2 ≤ ∑ i, ∑ j, w i * w j * U i j := by
  rw [portfolio_norm_sq_eq_double_sum w β]
  exact worst_case_variance_box_corner _ U w hw hU

/-! ### T8 — robust portfolio regret bounded by comparator bracket width -/

/--
**Theorem (Robust Portfolio Regret Bounded by Comparator Bracket Width).**
(Paper 5 energy-robustness T8, PROVEN 2026-07-12.)

Bounds the robust portfolio's variance regret against any feasible comparator by the
comparator-weighted bracket width — the materiality-to-premium link. If the robust weights
`w_rob` are chosen to minimize worst-case variance (`hopt`: the robust portfolio's corner
quadratic form `∑∑ w_robᵢw_robⱼHiᵢⱼ` is no worse than any comparator `w_star`'s corner quadratic
form `∑∑ w_starᵢw_starⱼHiᵢⱼ`), and the true covariance `S` lies in the box `[Lo, Hi]` entrywise,
then the robust portfolio's TRUE variance regret relative to the comparator is capped by the
comparator's bracket width:

  `∑∑ w_robᵢw_robⱼSᵢⱼ − ∑∑ w_starᵢw_starⱼSᵢⱼ ≤ ∑∑ w_starᵢw_starⱼ(Hiᵢⱼ − Loᵢⱼ)`.

Chains T3 (`worst_case_variance_box_corner`, twice — once on `S ≤ Hi` at `w_rob`, once on
`Hi − S ≤ Hi − Lo` at `w_star`) through the optimality certificate `hopt`. When the bracket is
tight (`Hi − Lo` small, i.e. the ambiguity radius is small / the calibration is materially
precise) the robust portfolio's regret against ANY comparator is small — this is the formal
materiality-to-premium link: wide brackets (uncertain calibration) can only be penalized by a
correspondingly wide regret cap, never by an unbounded one.
-/
theorem robust_regret_le_bracket_width {n : ℕ}
    (S Lo Hi : Fin n → Fin n → ℝ) (w_rob w_star : Fin n → ℝ)
    (hw_rob : ∀ i, 0 ≤ w_rob i) (hw_star : ∀ i, 0 ≤ w_star i)
    (hLo : ∀ i j, Lo i j ≤ S i j) (hHi : ∀ i j, S i j ≤ Hi i j)
    (hopt : ∑ i, ∑ j, w_rob i * w_rob j * Hi i j ≤ ∑ i, ∑ j, w_star i * w_star j * Hi i j) :
    ∑ i, ∑ j, w_rob i * w_rob j * S i j - ∑ i, ∑ j, w_star i * w_star j * S i j ≤
      ∑ i, ∑ j, w_star i * w_star j * (Hi i j - Lo i j) := by
  have hstep1 : ∑ i, ∑ j, w_rob i * w_rob j * S i j ≤ ∑ i, ∑ j, w_rob i * w_rob j * Hi i j :=
    worst_case_variance_box_corner S Hi w_rob hw_rob hHi
  have hstep2 : ∑ i, ∑ j, w_rob i * w_rob j * S i j ≤ ∑ i, ∑ j, w_star i * w_star j * Hi i j :=
    hstep1.trans hopt
  have hstep3 : ∑ i, ∑ j, w_star i * w_star j * Hi i j - ∑ i, ∑ j, w_star i * w_star j * S i j =
      ∑ i, ∑ j, w_star i * w_star j * (Hi i j - S i j) := by
    rw [← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [mul_sub]
  have hstep4 : ∑ i, ∑ j, w_star i * w_star j * (Hi i j - S i j) ≤
      ∑ i, ∑ j, w_star i * w_star j * (Hi i j - Lo i j) :=
    worst_case_variance_box_corner (fun i j => Hi i j - S i j) (fun i j => Hi i j - Lo i j)
      w_star hw_star (fun i j => by linarith [hLo i j])
  linarith [hstep2, hstep3, hstep4]

/-! ### T4 — robust portfolio variance envelope -/

/--
**Theorem (Robust Portfolio Variance Envelope).** (Paper 5 energy-robustness T4, PROVEN 2026-07-12.)

Double-sum packaging of T2 over `Fin n` with per-asset radii `ε : Fin n → ℝ`: mirrors
`portfolio_variance_envelope` (`Continuous/Energy/Portfolio.lean`), replacing each pairwise
systematic-covariance bracket by T2's radius-composed bracket evaluated at the measured
embeddings `μ`. `μ'` is the (unobserved) true embedding vector; `hrad` bounds each asset's
embedding radius; `hLip`/`hLip_lower` are stated in the true pairwise distance
`dist (μ' i) (μ' j)`, exactly as T2 requires.
-/
theorem portfolio_variance_robust_envelope {n : ℕ}
    (β μ μ' : Fin n → E) (w : Fin n → ℝ) (L ℓ : ℝ) (ε : Fin n → ℝ)
    (hw : ∀ i, 0 ≤ w i) (hL : 0 ≤ L) (hℓ : 0 ≤ ℓ)
    (hrad : ∀ i, ‖μ' i - μ i‖ ≤ ε i)
    (hLip : ∀ i j, ‖β i - β j‖ ≤ L * dist (μ' i) (μ' j))
    (hLip_lower : ∀ i j, ℓ * dist (μ' i) (μ' j) ≤ ‖β i - β j‖) :
    ∑ i, ∑ j, w i * w j *
        ((1 / 2) * (‖β i‖ ^ 2 + ‖β j‖ ^ 2 -
          (L * (dist (μ i) (μ j) + (ε i + ε j))) ^ 2)) ≤
      ‖∑ i, w i • β i‖ ^ 2 ∧
    ‖∑ i, w i • β i‖ ^ 2 ≤
      ∑ i, ∑ j, w i * w j *
        ((1 / 2) * (‖β i‖ ^ 2 + ‖β j‖ ^ 2 -
          (ℓ * max 0 (dist (μ i) (μ j) - (ε i + ε j))) ^ 2)) := by
  rw [portfolio_norm_sq_eq_double_sum w β]
  refine ⟨?_, ?_⟩
  · refine Finset.sum_le_sum fun i _ => ?_
    refine Finset.sum_le_sum fun j _ => ?_
    have hw_nonneg : 0 ≤ w i * w j := mul_nonneg (hw i) (hw j)
    exact mul_le_mul_of_nonneg_left
      ((covariance_bracket_of_embedding_radius (β i) (β j) (μ i) (μ j) (μ' i) (μ' j) L ℓ
        hL hℓ (hrad i) (hrad j) (hLip i j) (hLip_lower i j)).1) hw_nonneg
  · refine Finset.sum_le_sum fun i _ => ?_
    refine Finset.sum_le_sum fun j _ => ?_
    have hw_nonneg : 0 ≤ w i * w j := mul_nonneg (hw i) (hw j)
    exact mul_le_mul_of_nonneg_left
      ((covariance_bracket_of_embedding_radius (β i) (β j) (μ i) (μ j) (μ' i) (μ' j) L ℓ
        hL hℓ (hrad i) (hrad j) (hLip i j) (hLip_lower i j)).2) hw_nonneg

/-! ### T5 — Hilbert-ball linear DRO certificate -/

/--
**Theorem (Cauchy–Schwarz Linear Certificate over a Closed Ball).** (Paper 5 energy-robustness T5(a),
PROVEN 2026-07-12.)

For any `q` in the closed ball of radius `ε` around `p`, the linear functional `⟪·, f⟫` moves
by at most `ε‖f‖`: `⟪q, f⟫ ≤ ⟪p, f⟫ + ε‖f‖`. Pure Cauchy–Schwarz on `⟪q − p, f⟫`; no ordered-sup
API needed.
-/
theorem inner_le_of_mem_closedBall (p f : E) {ε : ℝ} {q : E}
    (hq : q ∈ Metric.closedBall p ε) :
    @Inner.inner ℝ E _ q f ≤ @Inner.inner ℝ E _ p f + ε * ‖f‖ := by
  have hdist : ‖q - p‖ ≤ ε := by
    have h := Metric.mem_closedBall.mp hq
    rwa [dist_eq_norm] at h
  have hexpand : @Inner.inner ℝ E _ (q - p) f =
      @Inner.inner ℝ E _ q f - @Inner.inner ℝ E _ p f := by
    rw [inner_sub_left]
  have h1 : @Inner.inner ℝ E _ (q - p) f ≤ |@Inner.inner ℝ E _ (q - p) f| := le_abs_self _
  have h2 : |@Inner.inner ℝ E _ (q - p) f| ≤ ‖q - p‖ * ‖f‖ := abs_real_inner_le_norm _ _
  have h3 : ‖q - p‖ * ‖f‖ ≤ ε * ‖f‖ := mul_le_mul_of_nonneg_right hdist (norm_nonneg _)
  linarith [hexpand, h1, h2, h3]

/--
**Theorem (Attainment of the Linear Certificate).** (Paper 5 energy-robustness T5(b), PROVEN 2026-07-12.)

The bound of `inner_le_of_mem_closedBall` is attained: for `ε ≥ 0` there is a `q` in the
closed ball of radius `ε` around `p` with `⟪q, f⟫ = ⟪p, f⟫ + ε‖f‖`. Witness
`p + (ε / ‖f‖) • f` when `f ≠ 0` (radial displacement in the direction of `f`), `p` itself
when `f = 0` (both sides collapse to `⟪p, f⟫`). Together with `inner_le_of_mem_closedBall`
this machine-checks `sup_{‖q−p‖≤ε} ⟪q, f⟫ = ⟪p, f⟫ + ε‖f‖` without ordered-sup plumbing.
-/
theorem exists_mem_closedBall_inner_eq (p f : E) {ε : ℝ} (hε : 0 ≤ ε) :
    ∃ q ∈ Metric.closedBall p ε, @Inner.inner ℝ E _ q f = @Inner.inner ℝ E _ p f + ε * ‖f‖ := by
  by_cases hf : f = 0
  · exact ⟨p, Metric.mem_closedBall_self hε, by simp [hf]⟩
  · have hfpos : 0 < ‖f‖ := norm_pos_iff.mpr hf
    refine ⟨p + (ε / ‖f‖) • f, ?_, ?_⟩
    · rw [Metric.mem_closedBall, dist_eq_norm, add_sub_cancel_left, norm_smul,
        Real.norm_of_nonneg (div_nonneg hε hfpos.le), div_mul_cancel₀ ε hfpos.ne']
    · rw [inner_add_left, real_inner_smul_left, real_inner_self_eq_norm_sq]
      field_simp

end PricingPerspective.ContinuousAPT
