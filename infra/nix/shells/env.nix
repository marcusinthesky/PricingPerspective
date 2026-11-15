{ pkgs }:
''
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
  }:''${LD_LIBRARY_PATH:-}
  export OMP_NUM_THREADS="''${OMP_NUM_THREADS:-4}"
  export OPENBLAS_NUM_THREADS="''${OPENBLAS_NUM_THREADS:-4}"
  export MKL_NUM_THREADS="''${MKL_NUM_THREADS:-4}"
  export NUMEXPR_NUM_THREADS="''${NUMEXPR_NUM_THREADS:-4}"
  export JAX_PLATFORMS="''${JAX_PLATFORMS:-cpu}"
  export UV_PYTHON_DOWNLOADS=never
  unset PYTHONPATH
''
