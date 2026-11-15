import Architect
import PricingPerspective.Operating.RationalShare.EquilibriumResponse.OrthantComparativeStatics
import PricingPerspective.Operating.RationalShare.FirmResponse.IncumbentResponse
import PricingPerspective.Operating.RationalShare.SolverCertificates.ProjectedResidual

/-!
# Joint equilibrium residual and payoff-regret certificates

Numerical equilibrium solvers for the rational-share game return a feasible joint capacity
profile rather than an exact Nash equilibrium.  This module gives that computed profile two
real-arithmetic certificates.

First, the orthant stationarity residual keeps the full negative-payoff gradient at a positive
capacity coordinate and keeps only its negative part at a zero coordinate.  Its Euclidean norm
is a valid radial variational-inequality error.  Strong monotonicity therefore turns it into a
distance bound from the unique exact equilibrium.

Second, the exact unilateral payoff expansion in `Operating/RationalShare.lean` and the same
residual imply a payoff-regret bound.  This is deliberately not inferred from an optimizer's
stopping flag or from the natural projected-step residual: it uses the orthant KKT residual and
the declared own-payoff curvature directly.

All claims are over exact real arithmetic.  Evaluating the residual in floating point is a
separate numerical obligation.
-/

namespace PricingPerspective.Operating.RationalShare.SolverCertificates.EquilibriumRegretCertificate

open PricingPerspective.Operating.VariationalInequality
  PricingPerspective.Operating.RationalShare.FirmResponse
  PricingPerspective.Operating.RationalShare.FirmResponse.Response
  PricingPerspective.Operating.RationalShare.FirmResponse.IncumbentResponse
  PricingPerspective.Operating.RationalShare.EquilibriumResponse
  PricingPerspective.Operating.RationalShare.SolverCertificates.ProjectedResidual

open scoped RealInnerProductSpace

variable {κ ι α : Type*} [Fintype κ] [Fintype ι] [Fintype α]

/-- The negative own-payoff gradient for one firm, read from a joint profile as a one-firm
activity block. -/
noncomputable def firmPseudoGradientAt
    (F : Profile ι α → Profile ι α) (x : Profile ι α) (i : ι) : Response.Block α :=
  Response.ofActivity fun a ↦ F x (i, a)

/-- The orthant KKT residual for one firm's activity block at a joint profile.

The field is frozen at the supplied joint profile: this is a certificate of that firm's current
unilateral optimality, not a residual for a simultaneous update map. -/
noncomputable def firmStationarityResidual
    (F : Profile ι α → Profile ι α) (x : Profile ι α) (i : ι) : Response.Block α :=
  ProjectedResidual.stationarityResidual (fun _ ↦ firmPseudoGradientAt F x i)
    (IncumbentResponse.incumbentBlock x i)

/-- Coordinate formula for a firm's orthant stationarity residual. -/
@[simp]
theorem firmStationarityResidual_apply
    (F : Profile ι α → Profile ι α) (x : Profile ι α) (i : ι) (a : α) :
    firmStationarityResidual F x i ((), a) =
      if x (i, a) = 0 then min (F x (i, a)) 0 else F x (i, a) := rfl

/-- The firm's field extracted from the joint rational-share pseudo-gradient is exactly the
zero-proximal block gradient against the rivals present at that profile.

This is the bridge from the joint equilibrium solver's `F = -∇ᵢ payoffᵢ` to the already verified
one-firm frozen-rival response problem. -/
theorem firmPseudoGradientAt_pseudoGradient_eq_blockGradient
    [DecidableEq ι] [DecidableEq α]
    (N : Nodes κ α) (c : ι → α → ℝ) (H : ι → α → α → ℝ)
    {x : Profile ι α} (hx : x ∈ capacityOrthant) (i : ι) :
    firmPseudoGradientAt (pseudoGradient N c H) x i =
      Response.blockGradient N (N.rival x i)
        (Nodes.rival_pos_of_nonneg N hx i) (c i) (H i) 0
        (IncumbentResponse.incumbentBlock x i) (IncumbentResponse.incumbentBlock x i) := by
  ext ⟨u, a⟩
  cases u
  rw [Response.blockGradient_apply]
  change pseudoGradient N c H x (i, a) = _
  rw [pseudoGradient_apply]
  simp only [IncumbentResponse.incumbentBlock_apply, zero_mul, add_zero]
  congr 1
  apply Finset.sum_congr rfl
  intro l _
  rw [N.denom_eq_rival_add x i l, N.appeal_eq_blockAppeal]
  rfl

/-- The joint orthant stationarity residual of a firm-by-activity profile.

For the rational-share pseudo-gradient `F = -∇ᵢ payoffᵢ`, a positive coordinate retains `F`.
At a zero coordinate, a positive component already satisfies the nonnegative-orthant KKT
condition and is discarded, while a negative component is retained.  The exact zero test is a
real-arithmetic definition rather than a floating-point boundary heuristic. -/
noncomputable def equilibriumStationarityResidual
    (F : Profile ι α → Profile ι α) (x : Profile ι α) : Profile ι α :=
  WithLp.toLp 2 fun p ↦ if x p = 0 then min (F x p) 0 else F x p

/-- Coordinate formula for the joint orthant stationarity residual. -/
@[simp]
theorem equilibriumStationarityResidual_apply
    (F : Profile ι α → Profile ι α) (x : Profile ι α) (p : ι × α) :
    equilibriumStationarityResidual F x p =
      if x p = 0 then min (F x p) 0 else F x p := rfl

/-- Removing KKT-compatible boundary components can only lower the pairing with a feasible
orthant displacement. -/
theorem equilibriumStationarityResidual_inner_le
    (F : Profile ι α → Profile ι α) {x z : Profile ι α}
    (hz : z ∈ capacityOrthant) :
    ⟪equilibriumStationarityResidual F x, z - x⟫ ≤ ⟪F x, z - x⟫ := by
  rw [inner_eq_sum, inner_eq_sum]
  apply Finset.sum_le_sum
  intro p _
  change equilibriumStationarityResidual F x p * (z p - x p) ≤
    F x p * (z p - x p)
  rw [equilibriumStationarityResidual_apply]
  by_cases hzero : x p = 0
  · rw [if_pos hzero, hzero, sub_zero]
    by_cases hgrad : F x p ≤ 0
    · rw [min_eq_left hgrad]
    · have hgrad' : 0 ≤ F x p := le_of_lt (lt_of_not_ge hgrad)
      rw [min_eq_right hgrad', zero_mul]
      exact mul_nonneg hgrad' (hz p)
  · rw [if_neg hzero]

/-- A feasible joint profile satisfies the pseudo-gradient variational inequality up to radial
error equal to the norm of its orthant stationarity residual. -/
theorem approxSolvesVI_of_equilibriumStationarityResidual
    (F : Profile ι α → Profile ι α) {x : Profile ι α}
    (hx : x ∈ capacityOrthant) :
    ApproxSolvesVI F capacityOrthant ‖equilibriumStationarityResidual F x‖ x := by
  refine ⟨hx, fun z hz ↦ ?_⟩
  have hcauchy :=
    (abs_le.mp (abs_real_inner_le_norm (equilibriumStationarityResidual F x) (z - x))).1
  simpa only [neg_mul] using
    hcauchy.trans (equilibriumStationarityResidual_inner_le F hz)

/-- Strong monotonicity converts the joint orthant KKT residual into a distance certificate from
an exact variational equilibrium.  No optimizer stopping criterion or multistart distance enters
the bound. -/
theorem dist_exactVI_le_equilibriumStationarityResidual
    {F : Profile ι α → Profile ι α} {x y : Profile ι α} {mu : ℝ}
    (hmu : 0 < mu) (hF : StronglyMonotoneOn F capacityOrthant mu)
    (hx : x ∈ capacityOrthant) (hy : SolvesVI F capacityOrthant y) :
    ‖x - y‖ ≤ ‖equilibriumStationarityResidual F x‖ / mu := by
  exact approxSolvesVI_dist_le hmu (norm_nonneg _) hF
    (approxSolvesVI_of_equilibriumStationarityResidual F hx) hy

/-- A computed rational-share profile is within its joint KKT residual divided by curvature of
an exact orthant Nash equilibrium.

The pseudo-gradient in the residual is the closed-form negative own-payoff gradient certified by
`payoff_replaceFirm_eq`.  Symmetry and nonnegative linear costs are needed only to turn the exact
orthant Nash equilibrium into the corresponding variational-inequality solution. -/
@[blueprint "eq:certificate"
  (statement := /--
    Joint stationarity bounds equilibrium distance. Assumptions: Feasible profile and strongly
    monotone rational-share operator.
  -/)]
theorem dist_orthantEquilibrium_le_equilibriumStationarityResidual
    [DecidableEq ι]
    (N : Nodes κ α) (c : ι → α → ℝ) {H : ι → α → α → ℝ}
    {mu : ℝ} {x y : Profile ι α}
    (hsymm : ∀ i a b, H i a b = H i b a) (hmu : 0 < mu)
    (hH : ∀ i, ∀ u : α → ℝ,
      mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (hc : ∀ i a, 0 ≤ c i a) (hx : x ∈ capacityOrthant)
    (hy : IsNashOrthant N c H y) :
    ‖x - y‖ ≤
      ‖equilibriumStationarityResidual (pseudoGradient N c H) x‖ / mu := by
  have hyvi := solvesVI_capacityOrthant_of_isNashOrthant N c hsymm hmu hH hc hy
  have hmono := stronglyMonotoneOn_pseudoGradient_capacityOrthant N c hH
  exact dist_exactVI_le_equilibriumStationarityResidual hmu hmono hx hyvi

/-- Pairing a profile field with a unilateral deviation leaves exactly the deviating firm's
activity coordinates. -/
theorem inner_replaceFirm_sub [DecidableEq ι]
    (v x : Profile ι α) (i : ι) (q : α → ℝ) :
    ⟪v, replaceFirm x i q - x⟫ = ∑ a, v (i, a) * (q a - x (i, a)) := by
  rw [inner_eq_sum, Fintype.sum_prod_type]
  refine (Finset.sum_eq_single i ?_ ?_).trans ?_
  · intro j _ hji
    exact Finset.sum_eq_zero fun a _ ↦ by
      simp only [PiLp.sub_apply]
      rw [replaceFirm_of_ne x hji q a]
      simp
  · intro hi
    exact absurd (Finset.mem_univ i) hi
  · exact Finset.sum_congr rfl fun a _ ↦ by
      simp only [PiLp.sub_apply]
      rw [replaceFirm_self]

/-- The squared profile norm of a unilateral deviation is the squared Euclidean norm of the
deviating firm's activity block. -/
theorem norm_replaceFirm_sub_sq [DecidableEq ι]
    (x : Profile ι α) (i : ι) (q : α → ℝ) :
    ‖replaceFirm x i q - x‖ ^ 2 = ∑ a, (q a - x (i, a)) ^ 2 := by
  rw [norm_sq_eq_sum, Fintype.sum_prod_type]
  refine (Finset.sum_eq_single i ?_ ?_).trans ?_
  · intro j _ hji
    exact Finset.sum_eq_zero fun a _ ↦ by
      simp only [PiLp.sub_apply]
      rw [replaceFirm_of_ne x hji q a]
      simp
  · intro hi
    exact absurd (Finset.mem_univ i) hi
  · exact Finset.sum_congr rfl fun a _ ↦ by
      simp only [PiLp.sub_apply]
      rw [replaceFirm_self]

/-- Pairing a firm's extracted pseudo-gradient with a block deviation is the activity-coordinate
sum appearing in the exact unilateral payoff expansion. -/
theorem inner_firmPseudoGradientAt_sub
    (F : Profile ι α → Profile ι α) (x : Profile ι α) (i : ι) (q : α → ℝ) :
    ⟪firmPseudoGradientAt F x i,
        Response.ofActivity q - IncumbentResponse.incumbentBlock x i⟫ =
      ∑ a, F x (i, a) * (q a - x (i, a)) := by
  rw [inner_eq_sum, Fintype.sum_prod_type]
  simp only [Fintype.sum_unique]
  exact Finset.sum_congr rfl fun a _ ↦ by
    simp only [firmPseudoGradientAt, Response.ofActivity, PiLp.sub_apply]
    rw [show (default : Unit) = () from Subsingleton.elim _ _,
      IncumbentResponse.incumbentBlock_apply]

/-- The squared norm of a one-firm block deviation is the sum of squared activity-coordinate
deviations. -/
theorem norm_firmBlock_sub_sq (x : Profile ι α) (i : ι) (q : α → ℝ) :
    ‖Response.ofActivity q - IncumbentResponse.incumbentBlock x i‖ ^ 2 =
      ∑ a, (q a - x (i, a)) ^ 2 := by
  rw [norm_sq_eq_sum, Fintype.sum_prod_type]
  simp only [Fintype.sum_unique]
  exact Finset.sum_congr rfl fun a _ ↦ by
    simp only [Response.ofActivity, PiLp.sub_apply]
    rw [show (default : Unit) = () from Subsingleton.elim _ _,
      IncumbentResponse.incumbentBlock_apply]

/-- Every feasible unilateral payoff gain for firm `i` is bounded by that firm's squared block
KKT residual divided by twice its own-payoff curvature.

The residual is computed from the firm's activity block only.  Its field is the corresponding
slice of the joint negative pseudo-gradient, and
`firmPseudoGradientAt_pseudoGradient_eq_blockGradient` identifies that slice with the verified
zero-proximal frozen-rival block gradient. -/
@[blueprint "app:regret-proof"
  (statement := /--
    Firm stationarity residual bounds unilateral regret. Assumptions: Own strong concavity,
    feasibility, and frozen rivals.
  -/)]
theorem unilateralPayoffGain_le_sq_norm_firmStationarityResidual_div
    [DecidableEq ι]
    (N : Nodes κ α) (c : ι → α → ℝ) {H : ι → α → α → ℝ}
    (hsymm : ∀ i a b, H i a b = H i b a) {mu : ℝ} (hmu : 0 < mu)
    {x : Profile ι α} (hx : x ∈ capacityOrthant) (i : ι) (q : α → ℝ)
    (hq : ∀ a, 0 ≤ q a)
    (hH : ∀ u : α → ℝ,
      mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b) :
    payoff N c H i (replaceFirm x i q) - payoff N c H i x ≤
      ‖firmStationarityResidual (pseudoGradient N c H) x i‖ ^ 2 / (2 * mu) := by
  let xb : Response.Block α := IncumbentResponse.incumbentBlock x i
  let z : Response.Block α := Response.ofActivity q
  let d : Response.Block α := z - xb
  let e : Response.Block α := firmStationarityResidual (pseudoGradient N c H) x i
  have hz : z ∈ Response.orthant := by
    rintro ⟨u, a⟩
    cases u
    simpa [z, Response.ofActivity] using hq a
  have hpay := payoff_replaceFirm_eq N c hsymm i hx hq
  have hquad := hH (fun a ↦ q a - x (i, a))
  have hrem := remainder_nonneg' N i hx hq
  have hinner :
      (∑ a, pseudoGradient N c H x (i, a) * (q a - x (i, a))) =
        ⟪firmPseudoGradientAt (pseudoGradient N c H) x i, d⟫ := by
    simpa only [d, z, xb] using
      (inner_firmPseudoGradientAt_sub (pseudoGradient N c H) x i q).symm
  have hnorm : (∑ a, (q a - x (i, a)) ^ 2) = ‖d‖ ^ 2 := by
    simpa only [d, z, xb] using (norm_firmBlock_sub_sq x i q).symm
  have hcurvature :
      payoff N c H i (replaceFirm x i q) - payoff N c H i x ≤
        -⟪firmPseudoGradientAt (pseudoGradient N c H) x i, d⟫
          - mu / 2 * ‖d‖ ^ 2 := by
    rw [hinner] at hpay
    rw [hnorm] at hquad
    linarith
  have hresidual := ProjectedResidual.stationarityResidual_inner_le
    (fun _ ↦ firmPseudoGradientAt (pseudoGradient N c H) x i)
    (x := xb) (z := z) hz
  change ⟪e, d⟫ ≤
    ⟪firmPseudoGradientAt (pseudoGradient N c H) x i, d⟫ at hresidual
  have hcauchy : -⟪e, d⟫ ≤ ‖e‖ * ‖d‖ := by
    exact (neg_le_abs _).trans (abs_real_inner_le_norm e d)
  have hradial :
      payoff N c H i (replaceFirm x i q) - payoff N c H i x ≤
        ‖e‖ * ‖d‖ - mu / 2 * ‖d‖ ^ 2 := by
    linarith
  have hyoung : ‖e‖ * ‖d‖ - mu / 2 * ‖d‖ ^ 2 ≤ ‖e‖ ^ 2 / (2 * mu) := by
    apply (le_div_iff₀ (mul_pos two_pos hmu)).2
    nlinarith [sq_nonneg (‖e‖ - mu * ‖d‖)]
  simpa only [e] using hradial.trans hyoung

/-- Every feasible unilateral payoff gain is also bounded by the squared joint KKT residual
divided by twice the firm's own-payoff curvature.

The curvature premise is imposed on the firm's quadratic cost block.  The share-revenue
remainder has the favorable sign by the exact rational-share expansion, so the negative payoff
is at least as strongly convex along unilateral deviations.  Consequently this is a genuine
payoff-regret certificate; it is not an interpretation of a gradient tolerance. -/
theorem unilateralPayoffGain_le_sq_norm_jointStationarityResidual_div
    [DecidableEq ι]
    (N : Nodes κ α) (c : ι → α → ℝ) {H : ι → α → α → ℝ}
    (hsymm : ∀ i a b, H i a b = H i b a) {mu : ℝ} (hmu : 0 < mu)
    {x : Profile ι α} (hx : x ∈ capacityOrthant) (i : ι) (q : α → ℝ)
    (hq : ∀ a, 0 ≤ q a)
    (hH : ∀ u : α → ℝ,
      mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b) :
    payoff N c H i (replaceFirm x i q) - payoff N c H i x ≤
      ‖equilibriumStationarityResidual (pseudoGradient N c H) x‖ ^ 2 / (2 * mu) := by
  let z : Profile ι α := replaceFirm x i q
  let d : Profile ι α := z - x
  let e : Profile ι α := equilibriumStationarityResidual (pseudoGradient N c H) x
  have hz : z ∈ capacityOrthant := by
    rintro ⟨j, a⟩
    by_cases hji : j = i
    · subst j
      simpa [z] using hq a
    · simpa [z, replaceFirm_of_ne x hji] using hx (j, a)
  have hpay := payoff_replaceFirm_eq N c hsymm i hx hq
  have hquad := hH (fun a ↦ q a - x (i, a))
  have hrem := remainder_nonneg' N i hx hq
  have hinner :
      (∑ a, pseudoGradient N c H x (i, a) * (q a - x (i, a))) =
        ⟪pseudoGradient N c H x, d⟫ := by
    simpa only [d, z] using
      (inner_replaceFirm_sub (pseudoGradient N c H x) x i q).symm
  have hnorm : (∑ a, (q a - x (i, a)) ^ 2) = ‖d‖ ^ 2 := by
    simpa only [d, z] using (norm_replaceFirm_sub_sq x i q).symm
  have hcurvature :
      payoff N c H i z - payoff N c H i x ≤
        -⟪pseudoGradient N c H x, d⟫ - mu / 2 * ‖d‖ ^ 2 := by
    simpa only [z] using (show
      payoff N c H i (replaceFirm x i q) - payoff N c H i x ≤
        -⟪pseudoGradient N c H x, d⟫ - mu / 2 * ‖d‖ ^ 2 by
          rw [hinner] at hpay
          rw [hnorm] at hquad
          linarith)
  have hresidual := equilibriumStationarityResidual_inner_le
    (pseudoGradient N c H) (x := x) (z := z) hz
  change ⟪e, d⟫ ≤ ⟪pseudoGradient N c H x, d⟫ at hresidual
  have hcauchy : -⟪e, d⟫ ≤ ‖e‖ * ‖d‖ := by
    exact (neg_le_abs _).trans (abs_real_inner_le_norm e d)
  have hradial :
      payoff N c H i z - payoff N c H i x ≤ ‖e‖ * ‖d‖ - mu / 2 * ‖d‖ ^ 2 := by
    linarith
  have hyoung : ‖e‖ * ‖d‖ - mu / 2 * ‖d‖ ^ 2 ≤ ‖e‖ ^ 2 / (2 * mu) := by
    apply (le_div_iff₀ (mul_pos two_pos hmu)).2
    nlinarith [sq_nonneg (‖e‖ - mu * ‖d‖)]
  simpa only [z, e] using hradial.trans hyoung

end PricingPerspective.Operating.RationalShare.SolverCertificates.EquilibriumRegretCertificate
