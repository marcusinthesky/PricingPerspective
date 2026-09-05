# The default shell's `shellHook` — environment the shell carries, as opposed to
# packages it contains. Separate from ./groups so that adding a tool never
# touches this file, and changing a thread cap never touches a group.
{ pkgs }:
''
  echo "pricing-perspective dev shell"
  echo "  run 'just' to list available tasks"
  # pygraphviz (LeanBlueprint canary) is a manylinux wheel that dlopen()s
  # Graphviz's font/rendering stack, which is not on any default search path.
  export LD_LIBRARY_PATH=${
    pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      pkgs.expat
      pkgs.zlib
      pkgs.glib
      pkgs.libx11
      pkgs.libxext
      pkgs.libxrender
    ]
  }:$LD_LIBRARY_PATH
  # Cap BLAS/OpenMP threads: the pipeline already parallelises at the process
  # level, and unbounded per-process BLAS threads can freeze the host. Set here
  # rather than in .envrc so `nix develop --command …` and worktree shells
  # inherit it too. Override per-command for a single-process stage:
  # `OMP_NUM_THREADS=8 just …`.
  export OMP_NUM_THREADS="''${OMP_NUM_THREADS:-2}"
  export OPENBLAS_NUM_THREADS="''${OPENBLAS_NUM_THREADS:-2}"
  export MKL_NUM_THREADS="''${MKL_NUM_THREADS:-2}"
  export NUMEXPR_NUM_THREADS="''${NUMEXPR_NUM_THREADS:-2}"
  # CPU-only host: skip JAX CUDA probing here too (mirrors .envrc) so
  # wrapped `nix develop --command` shells also avoid the cuInit traceback.
  export JAX_PLATFORMS="''${JAX_PLATFORMS:-cpu}"
  # phonemizer dlopen()s libespeak-ng by walking /usr/lib and friends, none of
  # which exist here, so point it at the store path directly.
  export PHONEMIZER_ESPEAK_LIBRARY="''${PHONEMIZER_ESPEAK_LIBRARY:-${pkgs.espeak-ng}/lib/libespeak-ng.so}"
  # Nix's Python setup hook leaks every Python input's site-packages into a
  # global PYTHONPATH, while repository Python runs are uv-managed. Scrub the
  # ambient path so a store dependency cannot shadow a locked project package.
  unset PYTHONPATH
''
