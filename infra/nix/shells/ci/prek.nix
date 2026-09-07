# Minimal shell for ci-pipeline.yml. Unlike the interactive default shell, this
# contains only binaries reached by `just setup`, diff-scoped prek hooks, or the
# toolchain-provenance hook that audits agent-declared commands.
{ pkgs }:
pkgs.mkShell {
  name = "pricing-perspective-prek-ci";
  packages = with pkgs; [
    ## Repository and declared-tool provenance
    git
    gh
    git-lfs
    fd
    ripgrep
    dotenvx

    ## Hook runner and repository orchestration
    prek
    just
    just-lsp
    jq
    # `yq` (python-yq) emits JSON for the reference/bibliography gates. Without it
    # the runner image's Go yq is picked up instead and pipes YAML into jq.
    yq
    dvc-with-remotes
    pandoc

    ## Python, SQL, and agent gates
    uv
    python313
    bun
    deno

    ## The provenance contract audits `lake` even though this job does not build Lean
    lean-toolchain.lean-all

    ## Nix, document, and configuration gates
    nixfmt
    statix
    deadnix
    tombi
    rumdl
    schematter
    frontmatter-cli
    lychee

    ## LaTeX setup and diff-scoped gates
    tectonic
    tex-fmt
    chktex

    ## OpenTofu/Terragrunt diff-scoped gates
    opentofu
    terragrunt
    tflint
    trivy

    ## TLS roots for uv, prek, Lychee, dbt, and Tectonic downloads
    cacert
  ];

  UV_PYTHON_DOWNLOADS = "never";
  shellHook = import ../env.nix { inherit pkgs; };
}
