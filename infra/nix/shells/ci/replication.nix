# Minimal shell for the audited replication-source release. It deliberately
# excludes manuscript renderers and analysis runtimes: the workflow validates
# source closure and DVC structure without executing numerical stages.
{ pkgs }:
pkgs.mkShell {
  name = "pricing-perspective-replication-ci";
  packages = with pkgs; [
    git # `git`: archive projection, provenance, and isolated repository setup
    git-lfs # `git lfs`: materialize and filter tracked release assets
    dvc-with-remotes # `dvc`: parse the DAG and dry-run the paper terminals
    just # `just`: prove the projected task graph still parses
    uv # `uv`: execute the PEP 723 auditor and check the frozen workspace lock
    python313 # Python pinned for uv so a cold runner downloads no interpreter
    cacert # CA bundle for uv's locked PEP 723 dependency fetch (no binary)
  ];
  UV_PYTHON_DOWNLOADS = "never";
}
