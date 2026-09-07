# LeanBlueprint proof-development companion

This directory is a proof-development companion. Its PDF and web outputs are
not arXiv, journal, or canonical manuscript sources.

The production claim mapping lives in `../tools/claim-manifest/mapping.toml`
(author-approved entries — see that directory's `README.md`). Authored Blueprint
prose may name an approved manuscript claim ID through `\ppClaimBinding{<claim-id>}`
and may author mathematical `\uses{...}` edges. It must not copy declaration names
or write `\leanok`; `../tools/claim-manifest/render.py` generates those bindings
only for entries that passed declaration, placeholder, and axiom validation.

`content.tex` assembles four authored chapters covering random functional
exposures, information-certified portfolio variance, robust measurement slack,
and spatial interaction. The generated web companion currently exposes those
chapters as a table of contents with 13 section pages; approved declaration
bindings are generated from the claim manifest. This remains independent of the
manuscripts, which consume verification status directly
(`../tools/claim-manifest/README.md`).

## Commands

Run from the repository's activated development shell:

```bash
just lean::build          # after changing Lean declarations or namespaces
just lean::blueprint-check
just lean::blueprint-pdf
just lean::blueprint-web
just lean::blueprint-all
```

The web and PDF targets read the compiled Lean environment while regenerating
the claim manifest. Rebuild first when declaration sources have changed; a
stale Lake environment can otherwise report valid mappings as unknown names.

`blueprint-check` builds the main Lean workspace and the `checkdecls` helper
root, regenerates and tests the claim manifest, checks authored/generated
boundaries, builds the web extraction, and runs the pinned upstream declaration
checker. Generated output lives in `print/`, `web/`, and `lean_decls`; all three
are ignored.

On a cold checkout — a fresh worktree, or after a nix GC — run
`just lean::warm` once. `checkdecls/` is a **second Lake root** that `lake build`
at `src/lean` never touches, and its git dependency is fetched on first build,
so without warming the only symptom is `blueprint-check` failing at its final
step for a provisioning reason. `just agents::worktree <name> <ref> 1`
provisions it at worktree-creation time.

The manuscript-facing half of the claim manifest is DAG-coupled: the DVC
`lean_claims` stage owns `claim_bindings.tex` and each paper's
`src/generated/lean_status.tex` and re-derives them from the proofs, and every
`render` target depends on its paper's status file. A manuscript render on the
`dvc repro` path therefore cannot outrun the proofs it cites. The `blueprint-*`
recipes remain the interactive entry point to the same work; the inner
`just latex::build`/`watch` loop is deliberately not gated on a Lake build.

## Validation layers

1. Lean kernel validation is the main workspace build and sorry gate.
2. Claim-manifest validation checks approved mappings, declaration existence,
   placeholders, and the axiom allowlist.
3. LeanBlueprint's declaration check confirms every generated `\lean` target
   is present in the imported workspace environment.
4. Humans own the informal mathematical exposition and `\uses` dependency
   graph. Neither is inferred from Lean syntax.

LeanBlueprint 0.0.20 and its Python dependency closure are locked by `uv.lock`.
Graphviz, XeLaTeX, and the measured dynamic-library closure needed by the
pinned pygraphviz wheel remain owned by the repository's Nix shell. Upstream
LeanBlueprint assumes the Git and Lake roots coincide; `run_leanblueprint.py`
supplies the repository's nested `src/lean` root and delegates `checkdecls` to
the pinned helper project under `checkdecls/`. It does not run `leanblueprint
new`, alter Git state, create deployment files, or change repository settings.
