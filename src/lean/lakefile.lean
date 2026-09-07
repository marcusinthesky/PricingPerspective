import Lake
open Lake DSL

package «pricing-perspective» where

require «doc-gen4» from git
  "https://github.com/leanprover/doc-gen4" @
  "v4.31.0"

-- last mdgen commit on lean4 v4.31.0 (main is already on v4.32.0-rc1)
require «mdgen» from git
  "https://github.com/Seasawher/mdgen" @
  "94463fabecb27dc443266207b578ca834e94b595"

require «PricingPerspective» from "./packages/PricingPerspective"
require «EnergyStatistics» from "./packages/EnergyStatistics"
require «WassersteinGeometry» from "./packages/WassersteinGeometry"
require «PaperReconstructions» from "./packages/PaperReconstructions"

-- mathlib also declared at root so `lake exe cache get` resolves against the workspace manifest.
-- All sub-packages must pin the same mathlib version.
require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @
  "v4.31.0"

@[default_target]
lean_lib «Main» where
  roots := #[`«Main»]

-- Declaration dumper for the DuckDB harness's Lean plane (`just lean::decl-dump`).
-- NOT a default_target: it is operator tooling, so `lake build` / `just lean::check`
-- (the proof gate) must not depend on it. supportInterpreter is required because
-- it calls importModules at run time rather than importing Mathlib at compile time
-- — see tools/DeclDump.lean for why.
lean_exe «decl_dump» where
  root := `tools.DeclDump
  supportInterpreter := true
