#!/usr/bin/env bash
set -euo pipefail

RAIZ_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$RAIZ_REPO/.venv"
ISAAC_DIR="$RAIZ_REPO/isaac_lab/IsaacLab"

if [ ! -d "$VENV_DIR" ]; then
    echo "[ERROR] No existe el entorno virtual en $VENV_DIR"
    echo "[ERROR] Ejecuta primero: setup/instalar_isaac_lab.sh"
    exit 1
fi

source "$VENV_DIR/bin/activate"
export ISAACLAB_REPO_DIR="$ISAAC_DIR"
export PYTHONPATH="$RAIZ_REPO/codigo:${PYTHONPATH:-}"

echo "[INFO] Entorno activado"
echo "[INFO] Repo: $RAIZ_REPO"
echo "[INFO] Isaac Lab: $ISAAC_DIR"
