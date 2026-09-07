import Lake
open Lake DSL

package «PricingPerspective» where

require «PaperReconstructions» from "../PaperReconstructions"
require «EnergyStatistics» from "../EnergyStatistics"
require «WassersteinGeometry» from "../WassersteinGeometry"

@[default_target]
lean_lib «PricingPerspective» where
  roots := #[`«PricingPerspective»]
