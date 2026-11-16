# LaTeX manuscript contract

Read `README.md` for commands and `ARCHITECTURE.md` before changing structure,
templates, engines, or packages.

## Boundaries

- `chapters/` is content-only: no document classes, packages, command or
  environment definitions, layout primitives, bibliography setup, or raw font
  changes.
- `targets/` alone owns document classes, package selection, layout, front
  matter, and bibliography style. Keep target adapters thin.
- `templates/pp-manuscript.sty` owns shared semantic interfaces. Add a macro only
  when at least two manuscripts need the same meaning.
- `generated/` is pipeline-owned. Never hand-edit it or copy its values into
  prose; consume values through `\ppvalue{key}`.
- A project may depend on `templates/`; templates must never depend on a project.
  Manuscript sources must never import from another project.
- Every article project has exactly one target, `arxiv`. Keep its adapter thin
  and the shared body independent of layout.

## Manuscript rules

- Write publication prose, not lab notes: paragraphs over bullet inventories;
  no micro-headings, enthusiasm adjectives, meta-commentary, or bold body prose.
- Cite only keys present in the project's generated `refs.bib`. Never invent or
  silently repair a citation.
- Use semantic theorem environments and label every numbered object with
  `sec:`, `eq:`, `fig:`, `tab:`, `thm:`, `lem:`, `prop:`, `cor:`, `def:`,
  `ass:`, `rem:`, or `app:`.
- Use `booktabs` tables; no vertical rules. Generated figures remain external
  DVC-built assets with self-contained captions, included as
  `\ppfigure{<frac>}{<name>}` — never a bare `\input` or `\includegraphics`, and
  never hand-authored in-document TikZ. `<frac>` is the fraction of `\linewidth`
  the figure was authored at; it scales nothing (PGF is placed at natural size)
  and exists so the manuscript stays an independent source of the intended width
  for the print-size guard. Keep it truthful. See `ARCHITECTURE.md`, §Figure
  transport.
- Keep Lean identifiers and source snippets out of the paper; surface any
  machine-checked backing only through the generated values in §Lean verification
  status below.
- Use semantic line breaks: one sentence per source line where practical. Never
  use `\\`, `\vspace`, or `\hspace` to repair prose layout.

## Lean verification status

Machine-checked Lean 4 backing reaches a manuscript only as generated values —
never Lean identifiers, paths, or snippets. Each target
`\input{generated/lean_status}`, the pipeline projection of
`src/lean/tools/claim-manifest/`. Consume a result by its manuscript label:
`\ppvalue{lean-verified-<label>}` (`checked`/`failed`) and
`\ppvalue{lean-disclosure-<label>}` (the human-facing scope). `\ppLeanStatusList`
renders the paper's "Formal verification status" appendix over all checked claims.
The disclosure is identifier-free by construction; author or change publication ownership/disclosures in
`src/lean/tools/claim-manifest/`. The corresponding mathematical statement lives in its
native annotation beside the original Lean proof. Source bundles run the shared
`just lean::certify` path and exact publication checks before staging.

## Workflow

1. Edit the smallest owning file; do not mix content and template changes.
2. Run `just latex::fmt`, then `just latex::check`.
3. For a release candidate, run `just latex::bundle <paper> arxiv` and inspect
   the PDF. A successful compile alone is not release evidence: the bundle audit
   checks the complete source closure and recompiles the unpacked archive.
4. A font that will not load — `Font OT1/lmr/m/n/5=rm-lmr5 at 5.0pt not
   loadable: Metric (TFM) file or installed font not found`, typically from
   a target — is a bundle-subset gap, not a defect in the `.tex`. The
   accompanying `accessing absolute path .../DejaVu*.ttf` warnings are a
   symptom: once a TFM lookup fails, XeTeX falls through to fontconfig and
   scans every system font. Run `just latex::bundle-refresh` and commit the
   regenerated zip. Do not "fix" the manuscript, and do not repoint a
   `Tectonic.toml` at the upstream URL — that host rate-limits (HTTP 429),
   which is what this vendoring exists to avoid.
