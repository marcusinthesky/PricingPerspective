import Lake
open Lake DSL

package «pricing-perspective» where
  lintDriver := "batteries/runLinter"
  lintDriverArgs := #["EnergyStatistics", "WassersteinGeometry",
    "PricingPerspective", "Publication"]

-- Publication annotations stay downstream of all mathematical libraries.
require LeanArchitect from git
  "https://github.com/hanwenzhu/LeanArchitect" @
  "9f3bb53916b9d6fa05b7777389e1844f82b1c9a6"

require «PricingPerspective» from "./packages/PricingPerspective"
require «EnergyStatistics» from "./packages/EnergyStatistics"
require «WassersteinGeometry» from "./packages/WassersteinGeometry"

require proofs from git
  "https://github.com/marcusinthesky/statadic.git" @
  "213a32399705af3463a3c1d8835409f284025ca1" / "src/lean"

-- mathlib also declared at root so `lake exe cache get` resolves against the workspace manifest.
-- All sub-packages must pin the same mathlib version. Keep it last.
require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @
  "v4.35.0"

@[default_target]
lean_lib «Main» where
  roots := #[`«Main»]

-- Native LeanArchitect facets: Publication:blueprint and Publication:blueprintJson.
lean_lib Publication where
  roots := #[`Publication]
