#!/usr/bin/env bash
# Compiled declaration inventory and trust gate; see tools/sorry-audit/README.md.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run --no-project python tools/sorry-audit/audit.py "$@"
