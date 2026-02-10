#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./new_notebook.sh [template_notebook.ipynb]

If no template is provided, TEMPLATE.ipynb is used.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi

template="${1:-TEMPLATE.ipynb}"
if [[ ! -f "$template" ]]; then
  echo "Error: template notebook not found: $template" >&2
  exit 1
fi

# Timestamp command as defined in README.md:
#   date -u +%Y-%m-%dT%H-%M-%SZ
ts="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
dir="$ts"

if [[ -e "$dir" ]]; then
  echo "Error: path already exists: $dir" >&2
  exit 1
fi

mkdir "$dir"
cp "$template" "$dir/$ts.ipynb"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: 'uv' not found on PATH (needed for: uv init --bare --no-workspace)" >&2
  exit 1
fi

(
  cd "$dir"
  uv init --bare --no-workspace
)

echo "$dir"
