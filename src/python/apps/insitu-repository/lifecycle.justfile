set shell := ["bash", "-uc"]

_run command:
    cd "$(git -C '{{ justfile_directory() }}' rev-parse --show-toplevel)" && uv run --locked --project src/python insitu {{ command }}

check: (_run "check")

sync: (_run "sync")

watch: (_run "watch")
