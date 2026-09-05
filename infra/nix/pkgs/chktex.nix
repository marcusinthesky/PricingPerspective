# ChkTeX built from `texliveInfraOnly`, NOT the standalone
# `texlivePackages.chktex`: the standalone binary finds no texmf.cnf, so it
# misses the chktexrc macro tables and fires Warning 37 on ordinary display
# math. See ../ARCHITECTURE.md § "Why Tectonic is the sole renderer".
#
# `meta` is attached with `//`, not `overrideAttrs` — overrideAttrs re-runs the
# env builder and was measured to move the derivation hash.
{ texliveInfraOnly }:
let
  env = texliveInfraOnly.withPackages (ps: [ ps.chktex ]);
in
env
// {
  meta = (env.meta or { }) // {
    mainProgram = "chktex";
  };
}
