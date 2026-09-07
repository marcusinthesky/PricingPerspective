# Lean claim manifest (t24.4)

Renderer-neutral record of which manuscript claims are backed by which
successfully machine-checked Lean declarations, plus their axiom status.

**No claim may be marked "Lean-verified" in a manuscript until a mapping
entry for it has been authored here AND `extract.py` has validated it with
`build_result: "checked"`.** Nothing is inferred from name or notation
similarity — see t24.1 blocker B1. Every entry is author-approved before it is
added; see [Adding a claim](#adding-a-claim).

## Files

- `schema.json` — JSON Schema for a manifest claim record.
- `mapping.toml` — human-authored input (one `[[claims]]` row per approved
  claim; commented template row inside).
- `extract.py` — for each mapping entry, runs `lake env lean` (from
  `src/lean`) on a generated `#check @<decl>` / `#print axioms <decl>`
  snippet, classifies the result, and writes `generated/manifest.json`.
  Fails a claim (and the process exit code) if the declaration is missing,
  its namespace changed, it depends on `sorryAx`, or it uses an axiom outside
  the allowlist `{propext, Classical.choice, Quot.sound}`.
- `render.py` — turns a validated `generated/manifest.json` into thin
  `generated/claim_manifest.tex` (shared), each paper's
  `src/latex/projects/<paper>/src/generated/lean_status.tex` (value bindings +
  `\ppLeanStatusList`), and `../../blueprint/src/generated/claim_bindings.tex`
  bindings. All are stamped
  with the manifest's `content_hash`. The Blueprint projection is the only
  place that emits literal `\lean` / `\leanok` commands, and only checked
  claims receive them.
- `tests/test_manifest.py` — positive + negative tests against the pre-built
  `src/lean` Lake workspace (no full `lake build`; single-declaration
  `lake env lean` checks are fast).

## Run

```bash
uv run src/lean/tools/claim-manifest/extract.py -v
uv run src/lean/tools/claim-manifest/render.py
uv run src/lean/tools/claim-manifest/tests/test_manifest.py
```

## Adding a claim

1. Confirm the exact fully-qualified declaration name with `rg` and a manual
   `lake env lean` probe first — do not guess from paper notation.
2. Uncomment/copy the template row in `mapping.toml`, fill in `paper`,
   `claim_id`, `statement_summary`, `declaration`, `package`, `file`,
   `assumptions`, `disclosure`. Do not author `build_result` or `axioms` —
   `extract.py` fills those in from the actual build.
3. Re-run `extract.py`; a nonzero exit means the entry did not validate.
4. Re-run `render.py`. Blueprint prose may consume a claim by its approved
   manuscript label with `\ppClaimBinding{<claim-id>}`; it must never hand-copy
   a Lean declaration or `\leanok`. Publication prose consumes only the
   human-facing status/disclosure projection and never a Lean identifier
   (`src/latex/AGENTS.md`).
