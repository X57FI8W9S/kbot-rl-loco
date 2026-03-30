#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo "[ERROR] No ejecutes este script con 'source' o '.'."
    echo "[ERROR] Usalo como proceso independiente: bash scripts/ver_checkpoint_ppo_marcha_vectorizado.sh"
    return 1
fi

RAIZ_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DIRECTORIO_ISAACLAB="$RAIZ_REPO/isaac_lab/IsaacLab"
RUTA_SCRIPT="$RAIZ_REPO/codigo/evaluacion/ver_checkpoint_ppo_marcha_vectorizado.py"

source "$RAIZ_REPO/.venv/bin/activate"
export PYTHONPATH="$RAIZ_REPO/codigo:${PYTHONPATH:-}"
export TERM="xterm"

cd "$DIRECTORIO_ISAACLAB"

echo "[INFO] Lanzando visualizacion de checkpoint PPO vectorizado."
echo "[INFO] Script: $RUTA_SCRIPT"

./isaaclab.sh -p "$RUTA_SCRIPT" "$@"
