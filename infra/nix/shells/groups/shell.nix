# Each package carries a one-line descriptor: what it is, and the binary it puts
# on PATH when that differs from the attribute name. Run `<binary> --help` for
# the rest — descriptors are deliberately short and are not a substitute.
{ pkgs }:
{
  description = "Interactive shell, version control, and the GitHub CLI.";
  packages = with pkgs; [
    zsh # `zsh`: the interactive shell this environment assumes
    git # `git`: version control
    git-lfs # `git lfs`: large-file storage for binaries DVC does not own
    gh # `gh`: GitHub from the command line — PRs, releases, API
  ];
}
