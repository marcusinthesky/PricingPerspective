#!/usr/bin/env bash
# Sorry/axiom inventory gate.
#
# Fails when the per-file `sorry`/`admit`/`axiom` counts in src/lean/packages drift from
# src/lean/sorry-allowlist.txt — the canonical inventory. A new sorry (or a fill) must
# land together with the matching update to sorry-allowlist.txt in the same PR.
#
# `admit` IS a tactic in Lean 4 v4.31.0 — `theorem t : True := by admit` compiles and
# emits "declaration uses `sorry`" — so it is counted alongside `sorry`. It is matched as
# a bare token rather than only in tactic position, which also catches the two prose uses
# ("Signed measures admit negative values"); those get allowlist rows with a comment, the
# same convention the docstring `sorry` mentions already use. A heuristic that tried to
# match only tactic position would be an evasion surface for no benefit.
#
# This gate is textual and covers every declaration, mapped or not. The semantic gate is
# tools/claim-manifest/extract.py, which runs `#print axioms` against the built workspace
# per manifest entry and rejects sorryAx or any axiom outside {propext, Classical.choice,
# Quot.sound}; it catches macro-expanded sorries and imported axioms that no grep can see.
set -euo pipefail
cd "$(dirname "$0")/.."

# Zero-tolerance: `debug.skipKernelTC` disables kernel type-checking and `debug.byAsSorry`
# elaborates every `by` block as a sorry. Neither leaves a `sorry` token behind, and
# skipKernelTC in particular is invisible to `#print axioms` too, because the extractor
# reads an already-built environment rather than re-checking it. No allowlist.
if debug_opts=$(grep -rn --include='*.lean' --exclude-dir='.lake' -E 'set_option[[:space:]]+debug\.' packages 2>/dev/null); then
  echo "ERROR: soundness-affecting debug options are not permitted in src/lean/packages:"
  printf '%s\n' "$debug_opts"
  exit 1
fi

actual=$(
  {
    grep -rEc --include='*.lean' --exclude-dir='.lake' -e '\b(sorry|admit)\b' packages 2>/dev/null || true
    grep -rEc --include='*.lean' --exclude-dir='.lake' \
      -e '^\s*(@\[[^]]*\]\s*)?(private |protected |noncomputable |unsafe |scoped |local )*axiom\b' \
      packages 2>/dev/null | sed 's/:\([0-9]*\)$/:axiom \1/' || true
  } | grep -Ev ':(axiom )?0$' | sort
)

expected=$(grep -Ev '^\s*(#|$)' sorry-allowlist.txt | sort)

if ! diff <(printf '%s\n' "$expected") <(printf '%s\n' "$actual"); then
  echo ""
  echo "ERROR: sorry/axiom inventory drifted from src/lean/sorry-allowlist.txt."
  echo "Left = allowlist (canonical), right = working tree."
  echo "Update src/lean/sorry-allowlist.txt in the same change."
  exit 1
fi
echo "sorry_gate: inventory matches allowlist ($(printf '%s\n' "$actual" | wc -l) files)."
