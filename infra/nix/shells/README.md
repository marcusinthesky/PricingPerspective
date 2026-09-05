---
description: The flake's devShells — an interactive default plus workload-minimal Lean, prek, and replication CI slices.
---

# infra/nix/shells/ — dev shell composition

```text
shells/
├── ci/
│   ├── lean.nix  # Lean build + claim-manifest gate
│   ├── prek.nix  # setup + diff-scoped prek gates
│   └── replication.nix  # audited three-paper source release
├── default.nix   # assembles devShells.{default,lean,prek,replication}
├── env.nix       # shared environment policy used by default and prek
└── groups/       # { description, packages } per interactive-shell group
```

## The groups

| Group | Owns |
|---|---|
| `shell` | Interactive shell, version control, the GitHub CLI |
| `tools` | Things that *do* something: secrets, the DuckDB/dbt plane, DVC, hooks, watch loops, transformation, rendering |
| `languages` | Every language in the repo and its toolchain — see below |
| `search` | Discovery — where is / what exists / what is this made of |
| `documents` | Native PDF/raster primitives for visual-document-analysis |
| `diagnostics` | Runtime cost, measured |
| `cloud` | The GCP client — ADC credentials and resource export |

Three boundaries are easy to get wrong, so they are stated as rules:

- **`search`** holds tools that answer *where is / what exists / what is this
  made of* — `rg`, `fd`, `semble-rs`, `tokei`. A tool that *changes* something
  belongs in `tools` even when you reached it through a search, which is why `sd`
  is not here. `ast-grep` is the deliberate exception: one binary that both
  queries and rewrites cannot sit in two groups, and its query modality is one
  the group would otherwise lack.
- **`diagnostics`** is runtime cost only — `hyperfine` and `py-spy`, whose sole
  output is a number about time. Sizing the codebase (`tokei`) is discovery, not
  cost.
- **`tools`** is everything that *does* something: renders, transforms, runs.
  `merman-cli` renders the DVC Mermaid DAG, while `graphviz` backs DOT call
  graphs and PyGraphviz; `watchexec` reruns a gate rather than measuring one.

These boundaries are the source of truth. The `toolbox` skill used to restate
them and was retired: it was a menu of tools rather than a procedure, and its
routing already existed in `../README.md`'s generated table and in `AGENTS.md`.
The handful of non-obvious flags it carried now sit inline on the packages
themselves, which is where they are read.

The rendered inventory, with every tool named, is generated into
[../README.md](../README.md) by Insitu's `tool_inventory` projector, using
`just nix::inventory` as its source.

`languages` is deliberately the large one. Its axis is the **language**, not the
tool's job: `rumdl` sits with `marksman` because both answer questions about
Markdown, not with `statix` because both happen to be linters. So the question
it answers is "what do we have for language X?", and a new language arrives as
one new section rather than as edits scattered across a linters file, a
formatters file, and an LSP file. It covers general-purpose languages (Python,
TypeScript, Lean), markup and typesetting (LaTeX, Markdown), the configuration
DSLs the repo's own build is written in (just, Nix, YAML, TOML), hand-edited
serialization (JSON, JSON Schema), shell, and HCL.

## Adding a tool

If it reads, writes, checks, or formats a language, it belongs in that
language's section of `groups/languages.nix` — add a section if the language is
new. Otherwise put it in the group that owns it, or add a group file. A group
earns its own file when it owns several packages *or* one paragraph of rationale
a reader would want without the rest; `cloud` is a single package kept separate
because it is the one piece of the infrastructure toolchain that is not a
language tool.

Give it a one-line descriptor naming the binary it puts on `PATH` when that
differs from the attribute name — `opentofu` provides `tofu`, `graphviz`
provides `dot`. Do **not** append to the end of `default.nix`'s list.

## Why order matters

`mkShell` hashes `nativeBuildInputs` in list order, so a permutation of the same
package set is a different derivation. Order also decides which package wins on
`PATH` when two ship the same binary — `chktex` and `pgf-metrics` are both
`texliveInfraOnly` closures and both carry kpathsea-class helpers, so their
relative order is load-bearing and documented in `groups/languages.nix`.

Two different equivalence checks apply, depending on what you are doing:

```bash
# A pure move (no reordering) must not change the derivation at all:
nix eval --raw ./infra/nix#devShells.x86_64-linux.default.drvPath

# A regrouping legitimately changes it. What must hold instead is that no
# package was added, dropped, or duplicated:
nix eval --json ./infra/nix#devShells.x86_64-linux.default \
  --apply 'd: builtins.sort (a: b: a < b) (map (p: p.outPath) d.nativeBuildInputs)'
```

The extraction of this directory out of `flake.nix` was verified with the first;
the later regrouping into `languages` was verified with the second.

## The `description` field

Every group carries one, and `toolInventory` reads them: `just nix::tools` renders
the group name, its description, and its resolved tool names into `../README.md`
through the `insitu-projections` hook. Edit a description and the table follows. It replaced
a hand-written table that had drifted for months, still advertising a `tinymist`
input and Typst tooling the flake had already dropped.

## CI shells are workload boundaries

`ci/lean.nix`, `ci/prek.nix`, and `ci/replication.nix` are exposed as
`devShells.lean`, `devShells.prek`, and `devShells.replication`. They serve the
Lean build, diff-scoped repository gate, and audited source-release workflow,
respectively. Local development continues to use `devShells.default`.

No CI shell composes `groups/`. They are defined by what their workflows reach,
so their packages are listed outright and each addition must be justified
against a job command. The prek slice is wider because an arbitrary PR diff may
activate Python, SQL, LaTeX, Nix, Markdown, or OpenTofu gates. Its own
toolchain-provenance hook also audits agent-declared commands such as `lake`,
`gh`, `fd`, and `dotenvx`; those are real gate dependencies even when a given PR
does not invoke them directly.
