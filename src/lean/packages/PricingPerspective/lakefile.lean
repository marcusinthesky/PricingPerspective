import Lake
open Lake DSL

package «PricingPerspective» where

require LeanArchitect from git
  "https://github.com/hanwenzhu/LeanArchitect" @
  "9f3bb53916b9d6fa05b7777389e1844f82b1c9a6"

require «EnergyStatistics» from "../EnergyStatistics"
require «WassersteinGeometry» from "../WassersteinGeometry"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @
  "v4.35.0"

@[default_target]
lean_lib «PricingPerspective» where
  roots := #[`«PricingPerspective»]
  globs := #[.andSubmodules `PricingPerspective]
