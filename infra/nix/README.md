# Pinned development tools

Enter the default environment with `devenv shell`, or use
`nix develop ./infra/nix`. The `lean`, `prek`, and `replication` profiles provide
smaller closures for proof checks, root Python hooks, and paper reproduction.
Python dependencies are installed separately with `just python::setup`.

Use the default environment for `prek run --all-files`: its nested hooks also
need the manuscript and website tools. The `prek` profile supplies the root
analysis gates and Python tests only.

`just nix::parity` verifies matching flake/devenv input pins, the Lean toolchain,
and the three retained Python workspace members. `just nix::inventory` exposes
the package inventory as JSON.

`just nix::tools` prints the same package inventory as a Markdown table.
