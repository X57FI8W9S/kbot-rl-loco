#!/usr/bin/env bash
set -euo pipefail

RAIZ_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DIRECTORIO_ISAACLAB="$RAIZ_REPO/isaac_lab/IsaacLab"
RUTA_SCRIPT="$RAIZ_REPO/codigo/entornos/marcha_basica.py"

# --- Intentar dejar el gobernador de CPU en performance ---
if command -v cpupower >/dev/null 2>&1; then
    GOBERNADOR_ACTUAL="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true)"
    if [ "$GOBERNADOR_ACTUAL" != "performance" ]; then
        echo "[INFO] Ajustando gobernador de CPU a performance..."
        sudo cpupower frequency-set -g performance || true
    else
        echo "[INFO] El gobernador de CPU ya está en performance."
    fi
else
    echo "[WARN] cpupower no está instalado (instalar con: sudo apt install linux-tools-common linux-tools-generic)"
fi

source "$RAIZ_REPO/.venv/bin/activate"
export PYTHONPATH="$RAIZ_REPO/codigo:${PYTHONPATH:-}"
export TERM="xterm"
cd "$DIRECTORIO_ISAACLAB"

if [ "$#" -eq 0 ]; then
    ARGUMENTOS_SCRIPT=(--num-pasos 0 --mantener-abierto)
else
    ARGUMENTOS_SCRIPT=("$@")
fi

echo "[INFO] Lanzando: $RUTA_SCRIPT"
./isaaclab.sh -p "$RUTA_SCRIPT" "${ARGUMENTOS_SCRIPT[@]}"
