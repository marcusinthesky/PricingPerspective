# ChkTeX needs its minimal TeX tree and global macro tables. The standalone
# texlivePackages.chktex binary misses that configuration and reports Warning 37
# for ordinary display math. Keep this tree ahead of pgf-metrics on PATH.
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
