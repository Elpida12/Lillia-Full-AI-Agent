#!/usr/bin/env bash
set -euo pipefail
lillia_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lillia_config="$lillia_root/config.toml"
if [[ -f "$lillia_root/config.local.toml" ]]; then
  lillia_config="$lillia_root/config.local.toml"
fi
if [[ ! -x "$lillia_root/.venv/bin/python" ]]; then
  echo "Install first: python3 -m venv .venv && .venv/bin/python -m pip install -e '.[dev]'" >&2
  exit 1
fi
exec "$lillia_root/.venv/bin/python" -m lillia --config "$lillia_config" "$@"
