import Lake
open Lake DSL

package «pricing-perspective-blueprint-checkdecls» where
  -- Reuse the parent workspace's pinned dependency cache rather than cloning a
  -- second mathlib checkout for this tiny declaration-checker adapter.
  packagesDir := "../../.lake/packages"

require checkdecls from git
  "https://github.com/PatrickMassot/checkdecls.git" @
  "3d425859e73fcfbef85b9638c2a91708ef4a22d4"

require «PricingPerspective» from "../../packages/PricingPerspective"

@[default_target]
lean_lib «BlueprintCanaryImports» where
  roots := #[`«BlueprintCanaryImports»]
