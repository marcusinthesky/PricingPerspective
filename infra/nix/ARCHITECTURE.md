---
description: Why nix for reproducible dev environments, and why each contested tool choice in the flake went the way it did.
title: Nix Flake
---

## infra/nix/

The dev environment is a nix flake that pins every tool to a specific version
via `flake.lock`. `direnv` auto-activates it on `cd`.

### Why nix over alternatives?

| Alternative | Problem |
|---|---|
| **Docker** | Slow builds; requires daemon; non-native filesystem; hard to compose |
| **conda/mamba** | Only covers Python/R; doesn't handle non-Python tools |
| **System-level installs** | Unpinned; OS-specific; pollutes user environment |
| **devbox / devenv** | Layers on nix but adds complexity; nix alone is sufficient |

The same argument rules out `flake-parts` and `haumea` inside the flake: they
would earn their keep across many outputs and systems, and this flake has four
shells and one package set.

### How the directory is laid out, and where prose goes

`flake.nix` was a single 702-line file, of which 54% was comment. It is now
inputs and output wiring only; derivations live one-per-file in `pkgs/`, and the
dev shell is composed from per-domain groups in `shells/groups/`.

CI is split by workload under `shells/ci/`: `lean.nix` contains only the Lean
build and manifest surface, `prek.nix` contains setup plus every possible
diff-scoped system hook, and `replication.nix` contains only the Git/LFS,
uv/DVC, and task-graph tools reached by the source-release audit. None imports
the interactive groups, preventing an LSP, document-analysis tool, profiler, or
cloud CLI from silently enlarging a runner closure.

Prose is tiered by *when it is read*, which is the part worth stating:

| Tier | Lives in | Read when |
|---|---|---|
| One-line descriptor per package | the group file, inline | scanning "what do we have?" |
| `BUMPING:` / `LOCKSTEP:` runbooks | beside the derivation | editing that derivation |
| "Why this and not the alternative" | this file | deciding, or auditing a decision |

The middle tier deliberately does **not** move here. A bump runbook is read at
the moment its code is edited, and this repo already pays for that kind of drift
with dedicated lockstep hooks.

The tool table in `README.md` is generated from the group definitions
(`just nix::tools`) rather than maintained beside them. That is not tidiness: the
hand-written table drifted for months, advertising a `tinymist` input and Typst
tooling that the flake had already dropped.

### Why Tectonic is the sole renderer, yet a second TeX engine still ships

Tectonic supplies the fast, self-resolving authoring loop and is the only
renderer for build, watch, release, and submission-bundle audit. Manuscript
packages resolve from its pinned, content-addressable bundle at compile time, so
a new `\usepackage` is a manuscript edit rather than a flake edit. The former
`latexRelease` closure — texliveSmall plus latexmk/xetex plus a hand-maintained
list of every package the manuscripts load — existed to reproduce publisher
pdfLaTeX builds alongside Tectonic, and has been deleted. `src/latex/ARCHITECTURE.md`
records the publisher-parity trade-off that accepts.

Three narrow TeX closures survive, and all are `texliveInfraOnly` rather than a
scheme:

**ChkTeX** (`pkgs/chktex.nix`) is `infraOnly` by measurement, not by taste. The
standalone `texlivePackages.chktex` finds no `texmf.cnf`, and therefore none of
its global `chktexrc` macro tables — so it does not know that `\left` takes an
argument, and fires Warning 37 ("space after parenthesis") on ordinary display
math. Passing `-l src/latex/.chktexrc` supplies the CmdLine suppression list
only, not those tables, so the divergence broke `just latex::lint`, which runs
under `set -e` against a linter that exits nonzero on any warning. `infraOnly`
adds the kpathsea/texmf scaffolding and nothing else — chktex/chkweb/deweb plus
kpsewhich-class helpers, with no latexmk, pdflatex, pdftex, xelatex, or bibtex.

**pgf-metrics** (`pkgs/pgf-metrics.nix`) is a `pdflatex` used only to measure
text extents, never to render an artifact. An engine is unavoidable here:
`savefig(*.pgf)` needs text extents to size axes margins and honour
`bbox_inches="tight"`, and matplotlib obtains them by holding a *persistent
interactive* TeX process open and querying it per text box (`backend_pgf.py:293`
opens it with stdin/stdout pipes; `:334` reads the metrics typeout).
`RendererPgf.get_text_width_height_descent` (`:734-738`) calls into it
unconditionally — `text.usetex = False` does not avoid it, verified empirically
on matplotlib 3.11. Tectonic is a one-shot batch compiler with no stdin REPL and
cannot serve this role; a shim installed under the name `xelatex` is rejected on
its first flag.

Its package list is derived by measurement: 398 MB against texliveSmall's
595 MB, and the smallest set that compiles matplotlib's own PGF header plus every
math label the five manuscripts use. texliveBasic (433 MB) fails. `underscore`
and `epstopdf-pkg` are pulled in by that header, not by us. TinyTeX is not in
nixpkgs, and its `tlmgr install`-on-demand model is exactly the ambient mutable
state this flake exists to rule out.

**latexdiff** (`pkgs/latexdiff.nix`) marks up two revisions of a manuscript into
a single change-tracked `.tex` — additions underlined, deletions struck through —
which Tectonic then renders like any other target. It is `infraOnly` for the
chktex reason: the standalone attribute is a bare Perl script, and `latexdiff-vc`
shells out to kpsewhich-class helpers. It renders nothing itself, so Tectonic
remains the sole renderer and the markup packages (`ulem`, `color`) are resolved
from the pinned bundle at compile time, not declared here. Its incremental cost
over the chktex closure is ~1.5 MB, because the TeX Live infrastructure is
shared. It is the semantic counterpart to `diff-pdf` in `groups/documents.nix`:
`latexdiff` when the source is ours and the question is *what changed in the
prose*, `diff-pdf` when only the PDF exists or the question is *did the artifact
move at all*.

Because all three are TeX Live environments, all three ship kpathsea-class
helpers, so their **relative order in the shell decides which wins on `PATH`**.
They are kept adjacent and ordered deliberately in `shells/groups/languages.nix`.

### Why the Lean toolchain is a store derivation

`lean`/`lake` used to come from `elan`, which installs into `~/.elan` — outside
the Nix store. `nix-collect-garbage` then deletes the glibc/bash/gcc paths the
toolchain was patchelf'd against, and every Lean binary breaks with
`command failed: 'lake' … No such file`. Provisioning the toolchain as a store
derivation makes it a GC root by construction: the devShell references it, so it
cannot be collected.

The current route reuses lean4-nix's `fetchBinaryLean`, which fetches the
official prebuilt tarball, autoPatchelfs it, swaps in nixpkgs clang/lld, and
wraps `lean`/`leanc`/`lake` with the C compiler on `PATH`. Upstream has no
v4.31.0 manifest (it tops out at v4.30.0, as does nixpkgs' own `lean4`), so
`pkgs/lean-toolchain.nix` hands it an inline manifest.

**The alternative we did not take.** lean4-nix is a thin wrapper over a
`fetchurl` + `autoPatchelfHook` pattern, so the input can be dropped entirely in
favour of a self-contained derivation: fetch the release tarball, `autoPatchelfHook`
to fix the ELF interpreter for NixOS, `stdenv.cc.cc.lib` for the libstdc++ the
binaries link against, then replace the bundled `clang`/`ld.lld`/`llvm-ar` with
nixpkgs' (the bundled ones are unpatched and reference dead store paths after a
GC) and `wrapProgram` each of `lean`/`leanmake`/`leanc`/`lake` with `lld` on
`PATH`. The trade-off: zero external dependencies and full control, but you own
the clang/lld wiring if a future Lean release changes how it bundles its
compiler — which is exactly the churn the lean4-nix maintainer currently absorbs.

Bumping either route means moving `leanManifest.tag` and the per-platform hashes
in lockstep with `src/lean/lean-toolchain`.

### Why several ways to search code?

They answer different questions, and the overlap is deliberate rather than
accidental:

| Reach for | When the question is |
|---|---|
| `rg` | literal or regex — a string, in file contents |
| `fd` | by filename, not content |
| `fzf -f` | approximately the right characters — you know roughly what it is called |
| `semble_rs search` | relevance — which file is about X |
| the DuckDB harness | a repo-wide aggregate or transitive fact — which symbols exist, how they connect |
| `ast-grep` | *syntactic* — match a syntax pattern, and especially **transform** it |
| `sd` | the same transform when it is literal or regex rather than syntactic |

The last two are rewriters and ship in the `tools` group, not `search` — the
routing question and the group boundary are different questions. `search` holds
the tools that answer "where is / what exists / what is this made of";
`ast-grep` sits there anyway because it is one binary and its query modality is
one the group would otherwise lack.

semble-rs does semantic/BM25 retrieval; the DuckDB AST plane (`enable_ast`,
sitting_duck, `ast_language: "python"` → `stg_ast`) makes AST facts queryable in
SQL. Neither can express a structural rewrite, and that is the only reason
ast-grep is present. Its intended users are the `architecture-layout` skill's
`python-symbol-move` runbook and the t57 module-split wave, whose recipe
otherwise carves by line range and proves AST identity with a throwaway script.

semble-rs itself is the Rust port of MinishLab/semble, chosen over the Python
upstream after an A/B on `src/python` (2026-07-25): the top-relevant target landed
in the top 3 on every probe query and was the identical #1 on 6 of 8, at ~162 MB
peak RSS against 247 MB, as one 28 MB static binary instead of a uv-Python runtime
dependency.

### Why a native visual-document stack?

Paginated documents are both structured files and visual artifacts. pdf-inspector
classifies a file as text-based or scanned before anything else runs; Poppler
provides the primary probe, text, and raster path; qpdf checks container
integrity; PDFannots retains review geometry; OCRmyPDF and Tesseract create
explicit searchable derivatives for scans; headless MuPDF is an independent
renderer for disputed output; and ImageMagick handles raster inspection and
comparison. Pinning these complementary command-line primitives in Nix keeps the
visual-document-analysis skill deterministic without embedding repository policy
in scripts.

The classifier earns its slot by removing an inference. Deciding whether to run
OCR previously meant running `pdftotext` and judging the output — empty could
mean a scan, or a CID-font encoding problem, or a broken extraction. `detect-pdf`
answers it directly and reports a confidence, so the expensive OCR path is
entered on a measurement rather than a guess. Unlike the other vendored
derivations, it also obliges us to maintain a
`Cargo.lock` upstream does not ship — see `pkgs/pdf-inspector.nix`.

### Why Rumdl and Lychee?

The two Markdown gates own different failure classes:

| Tool | Ownership | Deliberate boundary |
|---|---|---|
| **Rumdl** | Markdown structure, same-file and cross-file anchors, relative file links | Offline; it does not establish that an HTTP(S) destination still responds |
| **Lychee** | HTTP(S) reachability in authored Markdown | External schemes only; it does not replace Rumdl or the repository's Git/DVC-aware relative-link validator |

Lychee sweeps the authored documentation corpus whenever an in-scope Markdown file
changes. A seven-day disk cache stores successful responses, excludes HTTP failures,
and is persisted by PR CI; without that cache the network gate would repeat hundreds
of unchanged requests during ordinary pre-commit runs. Archived chats and plans,
extracted references, generated API documentation, and changelogs are excluded because
their historical URLs are evidence, not maintained navigation.

### Why py-spy and hyperfine, and not a profiler suite?

Performance work here has two distinct shapes, and only one of them belongs in
the flake:

| Tool | Ownership | Deliberate boundary |
|---|---|---|
| **`dotell`** (Python package, not a flake tool) | Continuous per-stage cost ledger — one JSON row per DVC stage per run, keyed by `git_sha`, always on | Shallow by design: wall, RSS, CPU. It attributes cost to a *stage*, never to a line |
| **py-spy** | On-demand sampling of a live process tree, including `--subprocesses` | Per-run; produces a flamegraph, not a record. Cannot answer "since when" |
| **hyperfine** | Statistical benchmarking of whole commands | Command-level only; blind to what happens inside a run |

The ledger is what the DuckDB harness aggregates (`fct_stage_sha`,
`telemetry_regression`, `optimization_backlog`); no sampling profiler emits a
commit-keyed row, so none can replace it. Conversely a 2 s-interval sampler
cannot attribute a hot loop, so py-spy is the documented escalation once the
ledger has named the stage. `PP_TELEMETRY_PROFILE=1` adds a per-stage `cProfile`
capture in between, main-process only.

Deeper profilers were considered and left out. **scalene** separates Python from
native time, but for JAX/XLA stages the interesting time is inside XLA, where it
reports "native" without line attribution. **Fil** and **memray** attribute peak
memory in batch jobs — relevant to the multi-GB stages, but per-run and better
reached ad hoc than pinned. **probing** targets distributed training (NCCL wait
decomposition, cross-node federation); this repo is single-node
`JAX_PLATFORMS=cpu`, and its SQL-over-telemetry premise is already served by the
harness. **Basilisk**'s profiler has no standalone CLI — it runs only from an
editor LSP session and shells out to py-spy for external processes, so it cannot
attach to a non-interactive `dvc repro` stage.

One gap the ledger cannot close by construction: it measures in-process stage
time only, so DVC's own hashing and checkout are invisible to it. On a recent
full run that was ~17 min of an ~89 min elapsed repro. DVC has hidden
development flags for this (`dvc --instrument <cmd>`, `dvc --cprofile-dump <file>
<cmd>` — accepted by 3.67.1, where an unknown flag exits 254), but neither
emitted an artifact in this nix build; treat closing that gap as open work rather
than a working recipe.

### Trade-offs

Nix has a steep learning curve, and `flake.lock` can drift across machines if it
is not committed. Some packages lag upstream, which is why several derivations are
vendored in `pkgs/` rather than taken from nixpkgs — each one is a maintenance
obligation with its own bump runbook.

For researchers who only need Python, `uv` alone may suffice. Nix earns its cost
here because a single `nix develop` has to serve Python, LaTeX, Lean, SQL, and
the document toolchain simultaneously, and because the alternative — an ambient
`PATH` answered by a maintainer's home-manager profile — has already produced
silent CI-only failures (`rumdl` and `gh` were provisioned that way by accident
until 2026-08-01, which is why the `toolchain-provenance` contract now exists).
