#!/usr/bin/env bash
# Run JAX training from any cwd: resolves repo root and uses this repo's .venv.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
PY="${REPO}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing ${PY}. From the repo root run one of:" >&2
  echo "  uv sync" >&2
  echo "  python3 -m venv .venv && .venv/bin/pip install -e ." >&2
  exit 1
fi
exec "$PY" scripts/train.py "$@"
