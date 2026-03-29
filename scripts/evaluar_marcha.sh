#!/usr/bin/env bash
set -euo pipefail

RAIZ_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DIRECTORIO_ISAACLAB="$RAIZ_REPO/isaac_lab/IsaacLab"
RUTA_SCRIPT="$RAIZ_REPO/codigo/entornos/marcha_basica.py"

source "$RAIZ_REPO/.venv/bin/activate"
export PYTHONPATH="$RAIZ_REPO/codigo:${PYTHONPATH:-}"
export TERM="xterm"
cd "$DIRECTORIO_ISAACLAB"

echo "[INFO] Evaluacion de la demo actual de marcha."

if [ "$#" -eq 0 ]; then
    ARGUMENTOS_SCRIPT=(--headless --num-pasos 1000)
else
    ARGUMENTOS_SCRIPT=("$@")
fi

echo "[INFO] Lanzando: $RUTA_SCRIPT"
./isaaclab.sh -p "$RUTA_SCRIPT" "${ARGUMENTOS_SCRIPT[@]}"
