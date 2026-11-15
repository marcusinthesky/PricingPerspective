#!/usr/bin/env bash
# Workspace orchestration only; Lean, Batteries and Mathlib own the analysis.
set -euo pipefail
cd "$(dirname "$0")/.."

package_roots=(EnergyStatistics WassersteinGeometry PricingPerspective)
roots=("${package_roots[@]}" Publication)
shopt -s globstar nullglob
files=(Main.lean Publication.lean Publication/**/*.lean lakefile.lean)
for pkg in "${package_roots[@]}"; do
  files+=("packages/$pkg/$pkg.lean" "packages/$pkg/lakefile.lean" packages/"$pkg"/"$pkg"/**/*.lean)
done

check_boundaries() {
  uv run --no-project python -m unittest discover -s tools/import-boundaries -v
  uv run --no-project python tools/import-boundaries/check.py
}

case "${1:-lint}" in
  indexes|indexes-fix)
    lake build mathlib/mk_all
    # mk_all returns the number of changed indexes. Fix mode verifies the result too.
    for pkg in "${package_roots[@]}"; do
      if [[ "$1" == indexes-fix ]]; then
        (cd "packages/$pkg" && ../../.lake/packages/mathlib/.lake/build/bin/mk_all --lib "$pkg") || true
      fi
      (cd "packages/$pkg" && ../../.lake/packages/mathlib/.lake/build/bin/mk_all --lib "$pkg" --check)
    done
    ;;
  lint)
    bash scripts/quality.sh indexes
    lake build Main Publication
    check_boundaries
    # Release each imported environment before loading the next package. Keeping
    # all roots in Batteries' single process can exhaust memory.
    for root in "${roots[@]}"; do
      lake exe batteries/runLinter --no-build "$root"
    done
    lake env bash scripts/quality.sh text-style
    ;;
  boundaries)
    lake build Main Publication
    check_boundaries
    ;;
  fmt) lake env bash scripts/quality.sh text-style --fix ;;
  text-style)
    adapter="$PWD/tools/quality/Style.lean"
    flags=()
    if [[ "${2:-}" == --fix ]]; then flags=(--fix); fi
    lean --run "$adapter" "${flags[@]}" "${files[@]}"
    # Mathlib interprets paths as module names. Hyphenated tool directories are
    # not valid module identifiers, so lint each directory from its own cwd.
    for dir in tools tools/*; do
      [[ -d "$dir" ]] || continue
      sources=("$dir"/*.lean)
      ((${#sources[@]})) || continue
      (cd "$dir" && lean --run "$adapter" "${flags[@]}" "${sources[@]##*/}")
    done
    ;;
  builtin) lake lint --builtin-only "${roots[@]}" ;;
  report) lake lint --code-quality "${roots[@]}" ;;
  imports)
    lake build +ImportGraph.Lean.WithImportModules +ImportGraph.Imports.Unused
    lake env lean --run .lake/packages/importGraph/MainUnusedTransitiveImports.lean "${roots[@]}"
    ;;
  *) echo "unknown quality action: $1" >&2; exit 2 ;;
esac
