# Matplotlib's PGF backend uses pdflatex to measure text interactively.
# Tectonic renders the manuscripts; this small tree supplies PGF metrics only.
#
# CHANGING THE LIST: this is the smallest set that compiles matplotlib's own PGF
#   header plus every math label the manuscripts use — texliveBasic fails.
#   Re-measure against a real `savefig(*.pgf)` before trimming.
#
# `meta` is attached with `//` so the derivation hash is untouched — see chktex.nix.
{ texliveInfraOnly }:
let
  env = texliveInfraOnly.withPackages (ps: [
    ps.latex-bin
    ps.graphics
    ps.epstopdf-pkg # pulled in by matplotlib's PGF header, not by us
    ps.underscore # likewise
  ]);
in
env
// {
  meta = (env.meta or { }) // {
    mainProgram = "pdflatex";
  };
}
