---
description: Why generated LaTeX binds every numeric cell to the value catalog by key, and why Great Tables was not adopted.
title: Pipeline IO
---

## IO Architecture

`io/` is the sole LaTeX side-effect layer (see the layer contract in the package
`ARCHITECTURE.md`). Two emitters own everything a manuscript consumes from the
pipeline:

| Emitter | Writes | Consumed as |
|---|---|---|
| `values.py` | `<project>/src/generated/numbers.tex` — one `\ppDeclareValue{key}{value}` per line | `\ppvalue{key}` in prose, `\ppnum[N]{key}` in tables |
| `tables.py` | `booktabs` table bodies under the same `generated/` directory | `\input` from the owning chapter |

### The binding invariant

A generated number is a **reference**, never a literal. `NumCell` emits
`\ppnum[3]{key}`, not `0.403`, so a table cell and the sentence discussing it
resolve the same catalog entry. Three properties follow, and they are why this
layer is hand-rolled rather than delegated to a table library:

1. **Display precision is centralized.** `values.py` stores at 6 dp; rounding
   happens once in `\ppnum` (`siunitx` `round-mode=places` plus the file-level
   `round-pad=false`) in `templates/pp-manuscript.sty`. A call site carries a
   precision, never a rounding policy.
2. **An undefined key fails the compile.** `\ppvalue` raises a `PackageError`
   rather than expanding to nothing, so a renamed or dropped binding cannot
   reach a PDF as a silent blank.
3. **Uncomputed is distinguishable from zero.** The `MISSING` sentinel renders
   `\ppmissing` (an em dash) instead of a fabricated `0.000`.

`values.py` validates the rest at construction: kebab-case ASCII keys, duplicate
public keys rejected, non-finite non-NaN floats rejected, and input order
preserved — the emitter never re-sorts, so a regenerated catalog diffs cleanly.

### Alternatives considered

**Great Tables** (`GT.as_latex()`, evaluated 2026-08-12) — rejected.

| Requirement | This layer | `GT.as_latex()` |
|---|---|---|
| Cell as catalog reference | `\ppnum[N]{key}` | Not expressible — formatters resolve to text in Python |
| Missing-value marker | `\ppmissing` | `sub_missing` unsupported |
| House styling (no vertical rules, panel headings) | Emitter invariant | Most `tab_options` unsupported |
| Environments in use | `tabular`, `tabularx`, `longtable` | `tabular*`, `longtable` |
| Footnotes, stub and row-group labels | Chapter-owned or emitter-owned | Unsupported |

Its LaTeX backend is documented as experimental and omits precisely the features
the manuscripts depend on; its strong half is HTML and image export, which
nothing in this repository ships. Adopting it would also give a second engine
ownership of manuscript markup, which the 2026-07-23 t24.12 cutover to LaTeX as
the sole canonical source deliberately removed. Reconsider only if an HTML or
image deliverable appears, and then only outside the `.tex` path.

<https://posit-dev.github.io/great-tables/reference/GT.as_latex.html>

### Residual gap

`figures/paper1/_tables/` (~1150 lines) predates this layer and bakes literals
through its own `_fmt`/`_escape` helpers, so a Paper 1 table number need not
exist in `numbers.tex`. Paper-specific figure modules are the reference callers of
the current path. Migrating Paper 1 needs the emitter extended with composite
estimate-over-interval cells, panel-heading records, and `tabularx`/`longtable`
environments, plus a catalog key minted for every currently-baked interval
bound — roughly two to three days, touching the owning DVC stage outputs and
`tests/papers/test_paper1_tables_layout.py`. Output will not be byte-identical:
`round-pad=false` drops trailing zeros, so a baked `0.610` becomes `0.61`.
Verify on rendered pages, not on the `.tex`.
