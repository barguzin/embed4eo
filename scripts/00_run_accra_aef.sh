#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mamba run -n diss python "$SCRIPT_DIR/download_accra_aef.py" --download-tiffs "$@"
