# Minimal shell for the Lean PR gate. It deliberately does not compose the
# default-shell groups: additions must be justified by lean.yml's commands.
{ pkgs }:
pkgs.mkShell {
  name = "pricing-perspective-lean-ci";
  packages = with pkgs; [
    git # `git`: manifest freshness gate and repository metadata
    git-lfs # `git lfs`: the checkout contains LFS-managed inputs
    curl # `curl`: `lake exe cache get` downloads mathlib oleans
    cacert # CA bundle for HTTPS fetches (no binary)
    lean-toolchain.lean-all # `lean`, `lake`, `leanc`: compiler and build tool
    uv # `uv`: runs the claim-manifest PEP 723 script
    python313 # Python pinned for uv so a cold runner downloads no interpreter
  ];
  UV_PYTHON_DOWNLOADS = "never";
}
