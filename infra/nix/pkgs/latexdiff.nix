# latexdiff from `texliveInfraOnly`, for the same reason as chktex.nix: the
# standalone `texlivePackages.latexdiff` is a bare Perl script with no kpathsea
# scaffolding, and `latexdiff-vc` shells out to kpsewhich-class helpers.
#
# This closure renders nothing. `latexdiff` emits a marked-up *.tex*; the diff
# PDF is produced by Tectonic like any other manuscript, so Tectonic remains the
# sole renderer and the markup packages (ulem, color) come from the pinned
# bundle rather than from here. See ../ARCHITECTURE.md § "Why Tectonic is the
# sole renderer".
#
# Incremental cost over the chktex closure already in the shell: ~1.5 MB — the
# Perl scripts and latexdiff's own texmf files; everything else is shared.
#
# `meta` is attached with `//`, not `overrideAttrs` — see chktex.nix.
{ texliveInfraOnly }:
let
  env = texliveInfraOnly.withPackages (ps: [ ps.latexdiff ]);
in
env
// {
  meta = (env.meta or { }) // {
    mainProgram = "latexdiff";
  };
}
