#!/usr/bin/env bash
set -euo pipefail

RAIZ_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DIRECTORIO_ISAACLAB="$RAIZ_REPO/isaac_lab/IsaacLab"
RUTA_SCRIPT="$RAIZ_REPO/codigo/entornos/spawn_box_top.py"

source "$RAIZ_REPO/.venv/bin/activate"
export PYTHONPATH="$RAIZ_REPO/codigo:${PYTHONPATH:-}"
export TERM="xterm"
cd "$DIRECTORIO_ISAACLAB"

echo "[INFO] Lanzando: $RUTA_SCRIPT"
./isaaclab.sh -p "$RUTA_SCRIPT" "$@"
