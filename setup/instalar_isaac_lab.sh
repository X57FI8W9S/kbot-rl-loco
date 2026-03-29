#!/usr/bin/env bash
set -euo pipefail

RAIZ_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RAIZ_ISAAC="$RAIZ_REPO/isaac_lab"
DIRECTORIO_CLON="$RAIZ_ISAAC/IsaacLab"
VENV_DIR="$RAIZ_REPO/.venv"

if [ -d "$DIRECTORIO_CLON/.git" ]; then
    echo "[INFO] Isaac Lab ya existe en: $DIRECTORIO_CLON"
else
    echo "[INFO] Clonando Isaac Lab en: $DIRECTORIO_CLON"
    git clone https://github.com/isaac-sim/IsaacLab.git "$DIRECTORIO_CLON"
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creando entorno virtual en: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

echo "[INFO] Activando entorno virtual"
source "$VENV_DIR/bin/activate"

echo "[INFO] Actualizando pip"
python -m pip install --upgrade pip

echo "[INFO] Instalando dependencias base de Isaac Lab"
cd "$DIRECTORIO_CLON"
./isaaclab.sh --install

echo "[INFO] Instalacion finalizada"
