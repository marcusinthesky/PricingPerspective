#!/usr/bin/env bash
# Fixture suite for the src/latex quality tools:
#   - tools/check-log        (log policy: glyphs, unresolved objects, overfull)
#   - tools/interpretation-contract (semantic digest and numeric-binding parity)
#   - tools/bundle-audit     (submission-parity clean-room gate, t24.5)
# A "hostile" bundle fixture PASSES when bundle-audit correctly REJECTS it.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."   # -> src/latex/tools
latex_root="$(cd .. && pwd)"             # -> src/latex

# Every compile below must resolve against the same pinned bundle the builder
# uses. Omitting it would fall back to Tectonic's default bundle, which this
# repository never populates — `--only-cached` would then fail, and a hostile
# fixture would appear to be "rejected" for the wrong reason. Tectonic.toml
# states the vendored bundle relative to the project directory and these
# fixtures compile from `mktemp -d`, so resolve it to an absolute path.
bundle_url="$(rg -o '^bundle = "([^"]+)"' -r '$1' "$latex_root/projects/01_continuous_bounds/Tectonic.toml")"
if [[ "$bundle_url" != *://* ]]; then
  bundle_url="$(cd "$latex_root/projects/01_continuous_bounds" && realpath "$bundle_url")"
fi

fail=0
pass() { echo "PASS: $1"; }
bad()  { echo "FAIL: $1" >&2; fail=1; }

# --- check-log --------------------------------------------------------------
if bash check-log tests/missing-glyph.log >/dev/null 2>&1; then
  bad "check-log should exit 1 on tests/missing-glyph.log"
else
  pass "check-log exits 1 on tests/missing-glyph.log"
fi
if bash check-log tests/clean.log >/dev/null 2>&1; then
  pass "check-log exits 0 on tests/clean.log"
else
  bad "check-log should exit 0 on tests/clean.log"
fi

# --- bundle-audit subprocess bytes -----------------------------------------
if uv run --script tests/bundle-audit-bytes.py >/dev/null 2>&1; then
  pass "bundle-audit preserves raw compiler bytes and nonzero failures"
else
  bad "bundle-audit raw compiler byte handling"
fi
if uv run --script tests/stage-images.py >/dev/null 2>&1; then
  pass "reachable-image staging includes references, excludes orphans, and fails missing"
else
  bad "reachable-image staging fixtures"
fi
if uv run --script tests/paper6-bundle.py >/dev/null 2>&1; then
  pass "Paper 6 stages retained generated inputs without unpublished research material"
else
  bad "Paper 6 publication-source staging fixtures"
fi
if uv run --script tests/interpretation-contract.py >/dev/null 2>&1; then
  pass "interpretation-contract positive and hostile fixtures"
else
  bad "interpretation-contract fixtures"
fi

# --- bundle-audit hostile fixtures (must be REJECTED) -----------------------
expect_reject() {  # stage target label
  local extension out; out=$(mktemp -d)
  mkdir "$out/source"
  if ! cp -R "$1/." "$out/source/"; then
    bad "could not copy $3 fixture"
    rm -rf "$out"
    return
  fi
  extension=tar.gz
  [[ "$2" == quantitative-finance ]] && extension=zip
  if uv run --script bundle-audit "$out/source" "$2" "$out/b.$extension" "$bundle_url" >/dev/null 2>&1; then
    bad "bundle-audit should reject $3"
  else
    pass "bundle-audit rejects $3"
  fi
  rm -rf "$out"
}
expect_reject tests/fixtures/hostile-missing-image        arxiv                "omitted \\includegraphics image"
expect_reject tests/fixtures/hostile-missing-bst          quantitative-finance "omitted bibliography style (tfcad.bst)"

# A clean target plus a manually injected image must fail the recorder-closure
# gate even though TeX itself never opens the extra file.
if ( cd "$latex_root" && just _stage 01_continuous_bounds arxiv >/dev/null 2>&1 ); then
  unused_stage="$latex_root/projects/01_continuous_bounds/build/arxiv/bundle"
  mkdir -p "$unused_stage/images"
  printf 'not a referenced PDF\n' > "$unused_stage/images/unreferenced.pdf"
  expect_reject "$unused_stage" arxiv "unreferenced staged image"
else
  bad "could not stage Paper 1 for unreferenced-image fixture"
fi

# External absolute-path fixture: the recorded dependency closure (the
# `--makefile-rules` dump) must reject an \input that resolves outside the
# staging root. main.tex carries a @@ABS_TARGET@@ marker;
# substitute a real absolute path (to a file OUTSIDE the staging copy) at test
# time so the closure check — not a missing-file compile error — is exercised.
absdir=$(mktemp -d)
cp -R tests/fixtures/hostile-absolute-path/. "$absdir/"
abs_target="$(cd tests/fixtures && pwd)/absolute-target.tex"
sed -i "s#@@ABS_TARGET@@#${abs_target}#" "$absdir/main.tex"
if uv run --script bundle-audit "$absdir" arxiv "$absdir/b.tar.gz" "$bundle_url" >/dev/null 2>&1; then
  bad "bundle-audit should reject external absolute-path \\input"
else
  pass "bundle-audit rejects external absolute-path \\input"
fi
rm -rf "$absdir"

# --- cleveref theorem-name regression guard ---------------------------------
crdir=$(mktemp -d)
cp tests/fixtures/cleveref-corollary/main.tex "$crdir/"
cp "$latex_root/templates/pp-manuscript.sty" "$crdir/"
crtext=""
if ( cd "$crdir" && tectonic -X compile --untrusted --only-cached --keep-logs \
      --bundle "$bundle_url" -o "$crdir" main.tex >/dev/null 2>&1 ); then
  crtext=$(pdftotext "$crdir/main.pdf" - 2>/dev/null | tr -s ' \n' ' ')
fi
# The reference must name the object a corollary (independent per-type counters),
# NOT mislabel it "theorem" (the shared-counter regression). Case-insensitive:
# \cref renders lowercase mid-sentence, \Cref capitalises — both are correct.
if echo "$crtext" | grep -qiE "CREFCHECK: corollary" \
   && ! echo "$crtext" | grep -qiE "CREFCHECK: theorem"; then
  pass "cleveref renders a corollary reference as a corollary (not 'Theorem')"
else
  bad "cleveref theorem-name regression: corollary \\cref did not render as a corollary"
fi
rm -rf "$crdir"

# --- bundle manifest determinism --------------------------------------------
run_bundle() { ( cd "$latex_root" && just _bundle 01_continuous_bounds arxiv >/dev/null 2>&1 ); }
archive="$latex_root/projects/01_continuous_bounds/build/arxiv/arxiv.tar.gz"
d1=$(mktemp -d); d2=$(mktemp -d)
if run_bundle && cp "$archive" "$d1/a.tgz" \
   && run_bundle && cp "$archive" "$d2/b.tgz"; then
  tar -xzf "$d1/a.tgz" -C "$d1" MANIFEST.json 2>/dev/null
  tar -xzf "$d2/b.tgz" -C "$d2" MANIFEST.json 2>/dev/null
  if cmp -s "$d1/MANIFEST.json" "$d2/MANIFEST.json"; then
    pass "bundle MANIFEST.json is byte-deterministic across runs"
  else
    bad "bundle MANIFEST.json differs between runs (non-deterministic)"
  fi
  if tar tzf "$d1/a.tgz" | grep -qE '\.(aux|log|fls|fdb_latexmk|mk|out|bbl|blg)$|(^|/)main\.pdf$'; then
    bad "bundle archive contains build intermediates or main.pdf"
  else
    pass "bundle archive contains no build intermediates or compiled PDF"
  fi
else
  bad "determinism check could not build 01_continuous_bounds arxiv bundle twice"
fi
rm -rf "$d1" "$d2"

# Quantitative Finance requires a ZIP upload. Exercise its deterministic writer
# separately from the arXiv tarball path and reject compiled intermediates.
run_qf_bundle() { ( cd "$latex_root" && just _bundle 01_continuous_bounds quantitative-finance >/dev/null 2>&1 ); }
qf_archive="$latex_root/projects/01_continuous_bounds/build/quantitative-finance/quantitative-finance.zip"
q1=$(mktemp -d); q2=$(mktemp -d)
if run_qf_bundle && cp "$qf_archive" "$q1/a.zip" \
   && run_qf_bundle && cp "$qf_archive" "$q2/b.zip"; then
  unzip -p "$q1/a.zip" MANIFEST.json > "$q1/MANIFEST.json" 2>/dev/null
  unzip -p "$q2/b.zip" MANIFEST.json > "$q2/MANIFEST.json" 2>/dev/null
  if cmp -s "$q1/a.zip" "$q2/b.zip" \
     && cmp -s "$q1/MANIFEST.json" "$q2/MANIFEST.json"; then
    pass "QF ZIP and MANIFEST.json are byte-deterministic across runs"
  else
    bad "QF ZIP or MANIFEST.json differs between runs (non-deterministic)"
  fi
  if unzip -Z1 "$q1/a.zip" | grep -qE '\.(aux|log|fls|fdb_latexmk|mk|out|bbl|blg)$|(^|/)main\.pdf$'; then
    bad "QF ZIP contains build intermediates or main.pdf"
  else
    pass "QF ZIP contains no build intermediates or compiled PDF"
  fi
else
  bad "QF ZIP determinism check could not build the template bundle twice"
fi
rm -rf "$q1" "$q2"

exit "$fail"
