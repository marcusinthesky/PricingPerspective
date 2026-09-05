# ── dbt/duckdb SQL toolchain, built from the harnessme workspace member ──
#
# A hermetic, GC-rooted Python env exposing sqlfmt, sqlfluff, dbt, and
# check-script-* on PATH the moment the shell activates, built from the same
# uv.lock the developer edits. Why it is built this way rather than fetched
# lazily by `uv run`: ../ARCHITECTURE.md § "Why the SQL toolchain is built from
# uv.lock".
#
# BUMPING: edit src/python/apps/harnessme/pyproject.toml, run `uv lock` from
# src/python, and re-lock the `python-workspace` flake input. Keep python313 in
# step with the workspace's ~=3.13.0 requirement.
{
  lib,
  callPackage,
  python313,
  python-workspace,
  pyproject-build-systems,
  pyproject-nix,
  uv2nix,
}:
let
  duckdbPython = python313;
  pythonWorkspace = uv2nix.lib.workspace.loadWorkspace {
    workspaceRoot = python-workspace;
  };
  # Prefer prebuilt wheels: duckdb / pydantic-core / ruamel.yaml.clib etc. ship
  # manylinux wheels, so we autoPatchelf them instead of compiling from source.
  duckdbOverlay = pythonWorkspace.mkPyprojectOverlay {
    sourcePreference = "wheel";
  };
  # Build fixups for packages whose sdist under-declares its build backend.
  # dbt-checkpoint (dist name `pre-commit-hooks`) builds from a git sdist and
  # needs setuptools, which its legacy setup.cfg packaging doesn't declare.
  duckdbOverrides = final: prev: {
    pre-commit-hooks = prev.pre-commit-hooks.overrideAttrs (old: {
      nativeBuildInputs =
        (old.nativeBuildInputs or [ ]) ++ final.resolveBuildSystem { setuptools = [ ]; };
    });
  };
  duckdbPythonSet =
    (callPackage pyproject-nix.build.packages {
      python = duckdbPython;
    }).overrideScope
      (
        lib.composeManyExtensions [
          pyproject-build-systems.overlays.default
          duckdbOverlay
          duckdbOverrides
        ]
      );
in
# Env exposing every entry point on PATH (sqlfmt, sqlfluff, dbt, check-*).
# dbt splits the `dbt` namespace across dbt-core/-adapters/-duckdb, each
# wheel shipping an identical namespace-stub dbt/__init__.py (and nested
# dbt/adapters/__init__.py). Those are benign collisions, so we tell the
# venv builder to tolerate them rather than fight PEP-420 namespace stubs.
(duckdbPythonSet.mkVirtualEnv "harnessme-sql-gates" { harnessme = [ ]; }).overrideAttrs (_: {
  venvIgnoreCollisions = [ "*" ];
})
