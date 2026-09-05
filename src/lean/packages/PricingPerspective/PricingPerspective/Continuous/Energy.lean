import PricingPerspective.Continuous.Energy.Core
import PricingPerspective.Continuous.Energy.OperationalEstimator
import PricingPerspective.Continuous.Energy.Population
import PricingPerspective.Continuous.Energy.Portfolio
import PricingPerspective.Continuous.Energy.SignedPopulation

/-!
# Energy-distance bounds for continuous asset pricing

This stable roll-up exposes the core energy geometry, the Paper 1 population
return decomposition, its signed-measure realization, the
projective/portfolio theorem family, and the bias decomposition of the
operational plug-in estimator Paper 3 actually deploys.  The implementation is
split so each proof layer remains small enough to audit independently.
-/
