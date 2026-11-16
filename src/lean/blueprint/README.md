# Native LeanArchitect and LeanBlueprint companion

This proof-development companion is separate from the canonical arXiv
manuscripts. Project statements live in `@[blueprint]` annotations at their original
Lean declarations beside the proofs. The `Publication` assembly library retains only
the external mathlib annotation; it does not duplicate project statements or proofs.

The companion includes native nodes with `\inputleannode{<claim-id>}`. LeanArchitect
owns declaration links, generated verification markers and inferred proof dependencies.
Authored chapters retain informal explanations and their reviewed dependency edges.
Every approved publication node appears once; missing inputs, duplicate labels and
unresolved authored or exported dependency references fail validation.

The `Publication`, `PricingPerspective` and `WassersteinGeometry` library facets each
export their own modules. `just lean::certify` builds all three native LaTeX/JSON exports,
runs the independent global integrity policy, rechecks mapped statement types and axioms,
and renders identifier-free publication disclosures. Native readiness does not replace
that policy or establish that an informal statement matches a formal theorem.

## Commands

```bash
just lean::certify
just lean::claims-publication-check
just lean::blueprint-check
just lean::blueprint-pdf
just lean::blueprint-web
just lean::blueprint-all
```

Every companion render first runs fresh certification. `blueprint-check` also validates
native JSON/LaTeX against the manifests, builds the web graph and runs the pinned upstream
LeanBlueprint declaration checker. `blueprint-pdf` compiles native LaTeX with the repository's
existing Tectonic engine and vendored bundle; it does not require a separate latexmk/XeLaTeX
installation. Generated native exports, `print/`, `web/` and `lean_decls` are ignored.

`claims-publication-check` is intentionally cheap: it uses committed manifests, checks
manuscript labels and boundaries, and compares each status file exactly with the renderer,
including its owning manifest hash. It needs neither Lake nor generated native exports.
`claims-check` also runs the checker regressions and checks every retained manuscript.
The DVC `lean_claims` and `lean_bindings_check` stages preserve this proof/prose separation.
arXiv source bundles run shared proof certification and publication checks before staging;
interactive LaTeX build/watch remain independent of a Lean build.

## Provisioning and scope

On a cold checkout, run `just lean::warm`. `checkdecls/` remains a second Lake root needed
by upstream LeanBlueprint; the main workspace build does not provision it. The small
`run_leanblueprint.py` adapter supplies the nested Git/Lake root and delegates declaration
checking to this pinned helper. Its `BlueprintPublication` library reuses the canonical
`Publication` assembly and the parent dependency cache, keeping the checked declaration
scope aligned with native exports. LeanArchitect complements this renderer rather than
replacing it. LeanBlueprint 0.0.20 and its Python closure remain locked by `uv.lock`.

The native library set is explicit. Adding annotations to another package requires adding
that package's facets to certification and the adapter's native-library set; it is not an
implicit claim of complete repository coverage. New publication registries must also be
listed as DVC dependencies. Human review owns mathematical prose equivalence and the
scope of assumptions/disclosures.
