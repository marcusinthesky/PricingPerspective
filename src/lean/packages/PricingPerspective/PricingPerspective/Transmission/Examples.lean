import PricingPerspective.Transmission.Factor
import PricingPerspective.Transmission.Portfolio
import PricingPerspective.Transmission.Transfer

/-!
# Random exposure smoke checks

These examples are compile-time executable checks for the model's nested vector rung:
a unit empirical cloud has a unique coupling, and the definitions reduce to the usual
scalar loading covariance and squared distance.
-/

namespace PricingPerspective.RandomExposure

private noncomputable def unitCoupling :
    Coupling (fun _ : Unit => (1 : ℝ)) (fun _ : Unit => (1 : ℝ)) :=
  { w := fun _ _ => 1
    nonneg := by intros; norm_num
    marginal_fst := by intro a; simp
    marginal_snd := by intro b; simp }

private noncomputable def cloudA : RandomExposureCloud Unit ℝ ℝ :=
  { characteristic :=
      { weight := fun _ => 1
        point := fun _ => 2
        nonneg := by intro; norm_num
        mass_one := by simp }
    slack := fun _ => (1 / 2 : ℝ)
    radius := 1 / 2
    radius_nonneg := by norm_num
    slack_bound := by intro; norm_num [Real.norm_eq_abs] }

private noncomputable def cloudB : RandomExposureCloud Unit ℝ ℝ :=
  { characteristic :=
      { weight := fun _ => 1
        point := fun _ => -1
        nonneg := by intro; norm_num
        mass_one := by simp }
    slack := fun _ => (-1 / 2 : ℝ)
    radius := 1 / 2
    radius_nonneg := by norm_num
    slack_bound := by intro; norm_num [Real.norm_eq_abs] }

private def sharedMap : ℝ → ℝ := fun x => 2 * x

private noncomputable def symmetricFactor : IsotropicFactor (Fin 2) ℝ :=
  { weight := fun _ => 1 / 2
    value := fun k => if k = 0 then 1 else -1
    nonneg := by intro; norm_num
    mass_one := by simp
    mean_zero := by simp [Fin.sum_univ_two]
    isotropic := by
      intro u v
      simp [Fin.sum_univ_two]
      ring }

private def leftResidual : Unit → Fin 2 → ℝ :=
  fun _ k => if k = 0 then 1 else -1

private def rightResidual : Unit → Fin 2 → ℝ :=
  fun _ _ => 0

private noncomputable def orthogonalResiduals :
    ResidualCrossOrthogonality symmetricFactor
      (p := fun _ : Unit => (1 : ℝ)) (q := fun _ : Unit => (1 : ℝ))
      leftResidual rightResidual :=
  { left_mean_zero := by
      intro a
      simp [residualMean, weightedMean, symmetricFactor, leftResidual]
    right_mean_zero := by
      intro b
      simp [residualMean, weightedMean, symmetricFactor, rightResidual]
    cross_zero := by
      intro a b
      simp [weightedMean, symmetricFactor, leftResidual, rightResidual]
  }

example :
    factorReturnCovarianceWithResidual unitCoupling symmetricFactor
        (fun _ : Unit => (0 : ℝ)) (fun _ : Unit => (0 : ℝ))
        leftResidual rightResidual = 0 := by
  have h :
      factorReturnCovarianceWithResidual unitCoupling symmetricFactor
          (fun _ : Unit => (0 : ℝ)) (fun _ : Unit => (0 : ℝ))
          leftResidual rightResidual =
        systCov unitCoupling (fun _ : Unit => (0 : ℝ)) (fun _ : Unit => (0 : ℝ)) := by
    apply factorReturnCovarianceWithResidual_eq_systCov_of_cross_orthogonality
      unitCoupling symmetricFactor
      (fun _ : Unit => (0 : ℝ)) (fun _ : Unit => (0 : ℝ))
      leftResidual rightResidual orthogonalResiduals
    simp [factorResidualCrossCovariance, unitCoupling, symmetricFactor,
      leftResidual, rightResidual]
  simpa [systCov, unitCoupling] using h

example :
    residualCrossCovariance unitCoupling symmetricFactor leftResidual rightResidual = 0 := by
  exact residualCrossCovariance_eq_zero_of_cross_orthogonality
    unitCoupling symmetricFactor leftResidual rightResidual orthogonalResiduals

private noncomputable def twoAssetLaw : JointExposureLaw (Fin 2) (Fin 2) ℝ :=
  { weight := fun _ => 1 / 2
    exposure := fun i s =>
      if i = 0 then (if s = 0 then 1 else -1) else (if s = 0 then 2 else 0)
    nonneg := by intro; norm_num
    mass_one := by simp }

example (i j : Fin 2) :
    (∀ s, ∑ t, (twoAssetLaw.diagonalCoupling i j).w s t = twoAssetLaw.weight s) ∧
      (∀ t, ∑ s, (twoAssetLaw.diagonalCoupling i j).w s t = twoAssetLaw.weight t) := by
  constructor
  · intro s
    exact (twoAssetLaw.diagonalCoupling i j).marginal_fst s
  · intro t
    exact (twoAssetLaw.diagonalCoupling i j).marginal_snd t

example (i j : Fin 2) :
    jointCovariance twoAssetLaw i j =
      systCov (twoAssetLaw.diagonalCoupling i j)
        (fun s => twoAssetLaw.exposure i s)
        (fun t => twoAssetLaw.exposure j t) := by
  exact jointCovariance_eq_systCov twoAssetLaw i j

example (i : Fin 2) :
    (twoAssetLaw.marginal i).weight = twoAssetLaw.weight ∧
      (twoAssetLaw.marginal i).point = twoAssetLaw.exposure i := by
  constructor <;> rfl

example :
    (cloudA.exposure sharedMap).point ()
      = sharedMap (cloudA.characteristic.point ()) + cloudA.slack () := by
  rfl

example :
    (cloudB.exposure sharedMap).point ()
      = sharedMap (cloudB.characteristic.point ()) + cloudB.slack () := by
  rfl

example :
    transportCost unitCoupling (cloudA.loading sharedMap) (cloudB.loading sharedMap)
      ≤ transportCost unitCoupling
          (fun a => sharedMap (cloudA.characteristic.point a))
          (fun b => sharedMap (cloudB.characteristic.point b))
        + 2 * (cloudA.radius + cloudB.radius) * 6
        + (cloudA.radius + cloudB.radius) ^ 2 := by
  apply transportCost_le_of_cloud_slack cloudA cloudB unitCoupling sharedMap
  intro a b
  simp [cloudA, cloudB, sharedMap]
  norm_num

example (x y : ℝ) :
    systCov unitCoupling (fun _ : Unit => x) (fun _ : Unit => y) = x * y := by
  simp [systCov, unitCoupling]
  ring

example (x y : ℝ) :
    transportCost unitCoupling (fun _ : Unit => x) (fun _ : Unit => y) = (x - y) ^ 2 := by
  simp [transportCost, unitCoupling]

example (x y : ℝ) :
    transportCost unitCoupling (fun _ : Unit => x) (fun _ : Unit => y)
      = secondMoment (fun _ : Unit => (1 : ℝ)) (fun _ => x)
        + secondMoment (fun _ : Unit => (1 : ℝ)) (fun _ => y)
        - 2 * systCov unitCoupling (fun _ => x) (fun _ => y) := by
  exact transportCost_polarization unitCoupling (fun _ => x) (fun _ => y)

end PricingPerspective.RandomExposure
