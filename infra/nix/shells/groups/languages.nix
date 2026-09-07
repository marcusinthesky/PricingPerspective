# Every language the repo is written in, and the tooling that reads it. The axis
# is the LANGUAGE, not the tool's job: `rumdl` sits with `marksman` because both
# answer questions about Markdown, not with `statix` because both are linters.
# A new language arrives as one new section.
#
# Descriptor convention: see groups/shell.nix. Rationale for contested choices
# (Tectonic, pgf-metrics, chktex) is in ../../ARCHITECTURE.md, not here.
{ pkgs }:
{
  description = "Every language in the repo and its toolchain — runtime, type checker, formatter, linter, LSP — from Python and Lean through LaTeX, Markdown, Nix, JSON, and HCL.";
  packages = with pkgs; [
    ## Python
    uv # `uv`: fast Python package manager — workspace sync, locks, PEP 723 scripts
    pyrefly # `pyrefly`: fast Python type checker

    ## TypeScript
    bun # `bun`: JavaScript runtime, package manager, bundler, and test runner
    deno # `deno`: JS/TS runtime with built-in lint+fmt; the workflow-lint/-fmt prek gates need it

    ## Lean
    ## GOTCHA: with elan gone, a rare `lake update` can warn about a missing
    ## elan. `lake exe cache get` / `lake build` are unaffected; if it bites, put
    ## a trivial `elan` shim on PATH. Keep src/lean/lean-toolchain in step with
    ## `leanManifest.tag` in ../../pkgs/lean-toolchain.nix.
    lean-toolchain.lean-all # `lean`, `lake`, `leanc`: the Lean 4 compiler and its build tool

    ## LaTeX
    ## ORDER: chktex, pgf-metrics, and latexdiff are all texliveInfraOnly
    ## closures, so all three ship kpathsea-class helpers and their relative
    ## order decides which wins on PATH. Keep them adjacent, in this order.
    harper # `harper-ls`: offline grammar/prose language server
    tectonic # `tectonic`: self-resolving (La)TeX engine — the sole manuscript renderer
    tex-fmt # `tex-fmt`: LaTeX formatter
    texlab # `texlab`: LaTeX language server
    chktex # `chktex`: LaTeX semantic linter (typography, spacing, math)
    pgf-metrics # `pdflatex`: TeX engine used ONLY for matplotlib PGF text extents
    latexdiff # `latexdiff`, `latexdiff-vc`: mark up two .tex revisions into a change-tracked .tex Tectonic then renders
    ## Just
    just # `just`: the polyglot command runner this repo drives everything through
    just-lsp # `just-lsp`: language server for justfiles

    ## Nix
    nixfmt-rfc-style # `nixfmt`: RFC 166 Nix formatter
    statix # `statix check`: Nix anti-pattern linter, with suggested rewrites
    deadnix # `deadnix`: finds unused bindings and arguments in .nix files
    nixd # `nixd`: Nix language server

    ## YAML
    yaml-language-server # `yaml-language-server --stdio`: YAML LSP with JSON Schema validation

    ## Markdown
    ## LOCKSTEP: the rumdl pin must equal the `rev:` in all three prek configs
    ## (root, .context/, .agents/) — upstream tags the hook repo
    ## `v<rumdl-version>`. Skew is silent: an ad-hoc `rumdl check` would measure a
    ## different ruleset than the gate enforces. `just nix::rumdl-lockstep` fails
    ## the commit if you bump one side only.
    marksman # `marksman`: Markdown language server (references, anchors, rename)
    rumdl # `rumdl check`: fast Markdown structure linter — pinned 0.2.31, see LOCKSTEP above
    schematter # `schematter`: document-schema validator for Markdown body shape
    lychee # `lychee`: async link checker for HTTP(S) reachability; config in lychee.toml
    frontmatter-cli # `frontmatter get|set|delete FIELD FILE`: read/write YAML frontmatter

    ## TOML
    tombi # `tombi format|lint`: TOML toolkit (formatter, linter, LSP)

    ## SQL / dbt
    dbt-language-server # `dbt-language-server`: dbt-jinja LSP — navigation only; sqlfluff+sqlfmt remain the gate

    ## JSON and JSON Schema
    jsonschema-cli # `jsonschema validate|bundle`: native JSON Schema validation and external-reference bundling
    vscode-langservers-extracted # `vscode-json-language-server --stdio`: JSON LSP; validates against each file's own $schema

    ## Shell
    bash-language-server # `bash-language-server start`: Bash LSP; shellcheck diagnostics included

    ## HCL — see infra/terraform/
    opentofu # `tofu`: open-source terraform fork (MPL-2.0, not BSL)
    terragrunt # `terragrunt`: opentofu orchestrator — DRY backend, per-component state, deps
    tflint # `tflint --chdir=DIR`: terraform/opentofu linter
    terraform-ls # `terraform-ls serve`: Terraform/HCL language server
    trivy # `trivy config`: scanner for IaC misconfiguration, vulnerabilities, and hard-coded secrets
  ];
}
