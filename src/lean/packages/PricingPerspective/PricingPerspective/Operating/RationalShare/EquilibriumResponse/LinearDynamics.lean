import PricingPerspective.Operating.RationalShare.EquilibriumResponse.LinearResponse

/-!
# Linearised anchored dynamics: propagation through the firm configuration

Around an equilibrium with a fixed active set, the anchored stage condition
`F(q_{t+1}; r_{t+1}) + kappa (q_{t+1} - q_t) = 0` linearises to

    P ((J + kappa I) δq_{t+1}) = P (kappa δq_t + 𝒢 δr_{t+1}),    P δq_{t+1} = δq_{t+1}.

This is the contemporaneous-rival timing of `AnchoredAdjustment.lean`. This module shows:

* each linearised step exists and is unique, because `J + kappa I` has modulus `mu + kappa`;
* the configuration map `A_kappa` contracts at `kappa / (mu + kappa)`. After a one-period
  opportunity impulse, the displacement decays geometrically. The disturbance propagates through
  the firm configuration, not through its own dynamics;
* under a permanent shift the path converges at the same rate to the static linear response
  `J⁻¹ 𝒢 δr` of `LinearResponse.lean`. Short-run adjustment ends at the comparative-static
  equilibrium.

The linearisation itself, as a derivative statement about the nonlinear anchored map, is not
claimed here.
-/

namespace PricingPerspective.Operating.RationalShare.EquilibriumResponse

open PricingPerspective.Operating.VariationalInequality

open scoped RealInnerProductSpace

section Restricted

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable {T P : E →ₗ[ℝ] E} {mu kappa : ℝ}

/-- An orthogonal projection does not increase norms. -/
theorem IsOrthoProjection.norm_le (hP : IsOrthoProjection P) (u : E) : ‖P u‖ ≤ ‖u‖ := by
  have h : ‖P u‖ ^ 2 ≤ ‖u‖ * ‖P u‖ := by
    calc ‖P u‖ ^ 2 = ⟪P u, P u⟫ := (real_inner_self_eq_norm_sq _).symm
      _ = ⟪u, P (P u)⟫ := hP.symm _ _
      _ = ⟪u, P u⟫ := by rw [hP.idem]
      _ ≤ ‖u‖ * ‖P u‖ := real_inner_le_norm _ _
  rcases eq_or_lt_of_le (norm_nonneg (P u)) with h0 | hpos
  · rw [← h0]; exact norm_nonneg _
  · nlinarith

/-- The anchor adds `kappa` to the restricted modulus. -/
theorem coercive_add_smul_id
    (hT : ∀ v, P v = v → mu * ‖v‖ ^ 2 ≤ ⟪T v, v⟫) :
    ∀ v, P v = v → (mu + kappa) * ‖v‖ ^ 2 ≤ ⟪(T + kappa • (LinearMap.id : E →ₗ[ℝ] E)) v, v⟫ := by
  intro v hv
  have h := hT v hv
  rw [LinearMap.add_apply, inner_add_left, LinearMap.smul_apply, LinearMap.id_apply,
    real_inner_smul_left, real_inner_self_eq_norm_sq]
  nlinarith [sq_nonneg ‖v‖]

/-- `w` is the linearised anchored step from `v` under forcing `g`. -/
def AnchoredStep (T P : E →ₗ[ℝ] E) (kappa : ℝ) (g v w : E) : Prop :=
  SolvesRestricted (T + kappa • (LinearMap.id : E →ₗ[ℝ] E)) P (kappa • v + g) w

/-- **Each linearised step exists and is unique.** -/
theorem existsUnique_anchoredStep [FiniteDimensional ℝ E] (hP : IsOrthoProjection P)
    (hmu : 0 < mu) (hk : 0 ≤ kappa) (hT : ∀ v, P v = v → mu * ‖v‖ ^ 2 ≤ ⟪T v, v⟫)
    (g v : E) : ∃! w, AnchoredStep T P kappa g v w :=
  existsUnique_solvesRestricted hP (by linarith) (coercive_add_smul_id hT) _

/-- **The configuration map contracts.** Two steps under the same forcing move by at most
`kappa / (mu + kappa)` times the distance between their starting configurations. -/
theorem anchoredStep_dist_le (hP : IsOrthoProjection P) (hmu : 0 < mu) (hk : 0 ≤ kappa)
    (hT : ∀ v, P v = v → mu * ‖v‖ ^ 2 ≤ ⟪T v, v⟫) {g v v' w w' : E}
    (hw : AnchoredStep T P kappa g v w) (hw' : AnchoredStep T P kappa g v' w') :
    ‖w - w'‖ ≤ kappa / (mu + kappa) * ‖v - v'‖ := by
  have hpos : 0 < mu + kappa := by linarith
  have hd := (SolvesRestricted.sub hw hw').norm_le hP hpos (coercive_add_smul_id hT)
  have hforce : kappa • v + g - (kappa • v' + g) = kappa • (v - v') := by
    rw [smul_sub]; abel
  rw [hforce, map_smul, norm_smul, Real.norm_of_nonneg hk] at hd
  calc ‖w - w'‖ ≤ kappa * ‖P (v - v')‖ / (mu + kappa) := hd
    _ ≤ kappa * ‖v - v'‖ / (mu + kappa) :=
        div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left (hP.norm_le _) hk) hpos.le
    _ = kappa / (mu + kappa) * ‖v - v'‖ := by ring

/-- The static restricted response is a fixed point of the anchored step. -/
theorem anchoredStep_self {g u : E} (hu : SolvesRestricted T P g u) :
    AnchoredStep T P kappa g u u := by
  refine ⟨hu.1, ?_⟩
  rw [LinearMap.add_apply, map_add, hu.2, LinearMap.smul_apply, LinearMap.id_apply, map_add]
  abel

/-- **Convergence to the static response.** Under a permanent forcing `g`, the linearised path
approaches the static restricted response at rate `kappa / (mu + kappa)`. -/
theorem anchoredPath_dist_le (hP : IsOrthoProjection P) (hmu : 0 < mu) (hk : 0 ≤ kappa)
    (hT : ∀ v, P v = v → mu * ‖v‖ ^ 2 ≤ ⟪T v, v⟫) {g u : E}
    (hu : SolvesRestricted T P g u) (v : ℕ → E)
    (hv : ∀ t, AnchoredStep T P kappa g (v t) (v (t + 1))) (t : ℕ) :
    ‖v t - u‖ ≤ (kappa / (mu + kappa)) ^ t * ‖v 0 - u‖ := by
  have hrho : 0 ≤ kappa / (mu + kappa) := div_nonneg hk (by linarith)
  induction t with
  | zero => simp
  | succ t ih =>
    calc ‖v (t + 1) - u‖ ≤ kappa / (mu + kappa) * ‖v t - u‖ :=
          anchoredStep_dist_le hP hmu hk hT (hv t) (anchoredStep_self hu)
      _ ≤ kappa / (mu + kappa) * ((kappa / (mu + kappa)) ^ t * ‖v 0 - u‖) :=
          mul_le_mul_of_nonneg_left ih hrho
      _ = (kappa / (mu + kappa)) ^ (t + 1) * ‖v 0 - u‖ := by ring

/-- **Impulse decay.** With no further forcing, the configuration displacement decays
geometrically. -/
theorem anchoredImpulse_norm_le (hP : IsOrthoProjection P) (hmu : 0 < mu) (hk : 0 ≤ kappa)
    (hT : ∀ v, P v = v → mu * ‖v‖ ^ 2 ≤ ⟪T v, v⟫) (v : ℕ → E)
    (hv : ∀ t, AnchoredStep T P kappa 0 (v t) (v (t + 1))) (t : ℕ) :
    ‖v t‖ ≤ (kappa / (mu + kappa)) ^ t * ‖v 0‖ := by
  have hzero : SolvesRestricted T P 0 0 := ⟨map_zero P, by rw [map_zero, map_zero]⟩
  simpa using anchoredPath_dist_le hP hmu hk hT hzero v hv t

end Restricted

variable {κ ι α : Type*} [Fintype κ] [Fintype ι] [Fintype α]
variable (N : Nodes κ α) (H : ι → α → α → ℝ) (x : Profile ι α)

/-- **Linearised anchored dynamics converge to the static response.** Around a nonnegative
profile, for any active set and a permanent opportunity change `δ`, the linearised anchored path
approaches the linear response `J⁻¹ 𝒢 δ` at rate `kappa / (mu + kappa)`. -/
theorem linearAnchoredPath_dist_le {mu kappa : ℝ} (hmu : 0 < mu) (hk : 0 ≤ kappa)
    (hH : ∀ i, ∀ u : α → ℝ, mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (hx : ∀ p, 0 ≤ x p) (act : ι × α → Prop) [DecidablePred act] {δ : κ → ℝ}
    {u : Profile ι α}
    (hu : SolvesRestricted (jac N H x) (restrictL act) (opportunityImpulse N x δ) u)
    (v : ℕ → Profile ι α)
    (hv : ∀ t, AnchoredStep (jac N H x) (restrictL act) kappa (opportunityImpulse N x δ)
      (v t) (v (t + 1))) (t : ℕ) :
    ‖v t - u‖ ≤ (kappa / (mu + kappa)) ^ t * ‖v 0 - u‖ :=
  anchoredPath_dist_le (restrictL_isOrthoProjection act) hmu hk
    (fun w _ ↦ jac_coercive N H x hH hx w) hu v hv t

/-- **A one-period opportunity impulse decays through the configuration.** After the impulse,
with no further opportunity change, the displacement shrinks by at least
`kappa / (mu + kappa)` per period. -/
theorem linearAnchoredImpulse_norm_le {mu kappa : ℝ} (hmu : 0 < mu) (hk : 0 ≤ kappa)
    (hH : ∀ i, ∀ u : α → ℝ, mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (hx : ∀ p, 0 ≤ x p) (act : ι × α → Prop) [DecidablePred act]
    (v : ℕ → Profile ι α)
    (hv : ∀ t, AnchoredStep (jac N H x) (restrictL act) kappa 0 (v t) (v (t + 1))) (t : ℕ) :
    ‖v t‖ ≤ (kappa / (mu + kappa)) ^ t * ‖v 0‖ :=
  anchoredImpulse_norm_le (restrictL_isOrthoProjection act) hmu hk
    (fun w _ ↦ jac_coercive N H x hH hx w) v hv t

end PricingPerspective.Operating.RationalShare.EquilibriumResponse
