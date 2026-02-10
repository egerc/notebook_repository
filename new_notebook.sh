#!/usr/bin/env bash
set -euo pipefail

# Timestamp command as defined in README.md:
#   date -u +%Y-%m-%dT%H-%M-%SZ
ts="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
dir="$ts"

if [[ -e "$dir" ]]; then
  echo "Error: path already exists: $dir" >&2
  exit 1
fi

mkdir "$dir"
cp "TEMPLATE.ipynb" "$dir/$ts.ipynb"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: 'uv' not found on PATH (needed for: uv init --bare --no-workspace)" >&2
  exit 1
fi

(
  cd "$dir"
  uv init --bare --no-workspace
)

echo "$dir"
